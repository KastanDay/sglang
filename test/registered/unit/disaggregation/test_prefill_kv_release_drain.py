"""Prefill must not free transfer buffers while a worker is still sending them.

A request can be observed as failed -- an abort notification from the decode
side, or a failure propagated from another rank through poll_and_all_reduce --
while a transfer worker on this rank is mid-send for it. On the unfixed code
the scheduler freed the request's KV pages and metadata row immediately; a
request allocated the same storage next had its contents silently read by the
in-flight transfer and sent to the failed request's decode destination.

The fix counts chunks a worker is executing per room, and the scheduler parks
the release of a failed request's pages until that count drains.
"""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.utils import FastQueue, TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOM = 7


def make_prefill_manager():
    """A prefill manager with just enough wired up to run transfer_worker."""
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.request_status = {ROOM: KVPoll.WaitingForInput}
    mgr.check_status = lambda room: mgr.request_status[room]
    mgr._inflight_chunk_counts = {}
    mgr._inflight_chunk_lock = threading.Lock()
    mgr.enable_trace = False
    mgr.enable_staging = False
    mgr.bootstrap_port = 0
    mgr.session_lock = threading.Lock()
    mgr.failed_sessions = set()
    mgr.session_failures = {}
    mgr.attn_tp_rank = mgr.attn_cp_rank = mgr.pp_rank = 0
    mgr.attn_tp_size = mgr.attn_cp_size = mgr.pp_size = 1
    return mgr


def make_sender(mgr, room=ROOM):
    sender = MooncakeKVSender.__new__(MooncakeKVSender)
    sender.kv_mgr = mgr
    sender.bootstrap_room = room
    return sender


class TestWorkerChunksAreCounted(CustomTestCase):
    def test_sender_reports_inflight_while_worker_is_mid_send(self):
        # Drive the real transfer_worker with a send that blocks inside the
        # engine write, exactly where an abort would race it.
        mgr = make_prefill_manager()
        entered_send = threading.Event()
        release_send = threading.Event()

        def blocking_send(*_args, **_kwargs):
            entered_send.set()
            release_send.wait(5)
            return 0

        mgr.transfer_infos = {
            ROOM: {
                "session:1": SimpleNamespace(
                    room=ROOM,
                    is_dummy=False,
                    mooncake_session_id="session:1",
                    dst_kv_indices=np.array([0], dtype=np.int32),
                    required_dst_info_num=1,
                )
            }
        }
        mgr.decode_kv_args_table = {
            "session:1": SimpleNamespace(
                dst_attn_tp_size=1, dst_kv_ptrs=[0], dst_kv_layer_ids=[0]
            )
        }
        mgr._get_dsa_cache_transfer_skip_flags = MagicMock(return_value=(False, False))
        mgr.kv_args = SimpleNamespace(kv_data_ptrs=[0x1000])
        mgr.is_mla_backend = True
        mgr.is_hybrid_mla_backend = False
        mgr.send_kvcache = blocking_send
        sender = make_sender(mgr)
        worker_drained = threading.Event()
        end_inflight_chunk = mgr._end_inflight_chunk

        def record_worker_drain(room):
            end_inflight_chunk(room)
            worker_drained.set()

        mgr._end_inflight_chunk = record_worker_drain

        queue = FastQueue()
        worker = threading.Thread(
            target=mgr.transfer_worker,
            args=(queue, None, None, 0),
            daemon=True,
        )
        worker.start()
        self.assertFalse(sender.has_inflight_transfers())

        queue.put(
            TransferKVChunk(
                room=ROOM,
                prefill_kv_indices=np.array([0], dtype=np.int32),
                index_slice=slice(0, 1),
                is_last_chunk=False,
                prefill_aux_index=None,
                state_indices=None,
            )
        )
        self.assertTrue(entered_send.wait(5))
        self.assertTrue(
            sender.has_inflight_transfers(),
            "a chunk is mid-send; the scheduler must not free its pages",
        )

        # The failed-room race: the room fails while the send is in flight.
        mgr.request_status[ROOM] = KVPoll.Failed
        self.assertTrue(sender.has_inflight_transfers())

        release_send.set()
        self.assertTrue(worker_drained.wait(5))
        self.assertFalse(
            sender.has_inflight_transfers(),
            "the counter must drain once the send returns",
        )


class TestSchedulerDefersRelease(CustomTestCase):
    def _scheduler_stub(self):
        stub = SimpleNamespace(
            tree_cache=MagicMock(),
            req_to_metadata_buffer_idx_allocator=MagicMock(),
            disagg_prefill_pending_kv_releases=[],
        )
        stub.maybe_defer_kv_release = (
            SchedulerDisaggregationPrefillMixin.maybe_defer_kv_release.__get__(stub)
        )
        stub.process_pending_kv_releases = (
            SchedulerDisaggregationPrefillMixin.process_pending_kv_releases.__get__(
                stub
            )
        )
        return stub

    def _req(self, inflight: bool):
        return SimpleNamespace(
            rid="r1",
            metadata_buffer_index=3,
            disagg_kv_sender=SimpleNamespace(has_inflight_transfers=lambda: inflight),
        )

    def test_release_is_parked_while_a_chunk_is_in_flight(self):
        stub = self._scheduler_stub()
        req = self._req(inflight=True)
        with patch("sglang.srt.disaggregation.prefill.release_kv_cache") as release:
            stub.maybe_defer_kv_release(req, is_insert=False)
            release.assert_not_called()
            stub.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
            self.assertEqual(len(stub.disagg_prefill_pending_kv_releases), 1)

            # Still draining: the sweep must keep it parked.
            stub.process_pending_kv_releases()
            release.assert_not_called()
            stub.req_to_metadata_buffer_idx_allocator.free.assert_not_called()

            # Drained: the sweep releases KV and metadata together.
            req.disagg_kv_sender.has_inflight_transfers = lambda: False
            stub.process_pending_kv_releases()
            release.assert_called_once_with(req, stub.tree_cache, is_insert=False)
            stub.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
            self.assertEqual(req.metadata_buffer_index, -1)
            self.assertEqual(stub.disagg_prefill_pending_kv_releases, [])

    def test_release_is_immediate_when_nothing_is_in_flight(self):
        stub = self._scheduler_stub()
        req = self._req(inflight=False)
        with patch("sglang.srt.disaggregation.prefill.release_kv_cache") as release:
            stub.maybe_defer_kv_release(req)
            release.assert_called_once_with(req, stub.tree_cache, is_insert=True)
            stub.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
            self.assertEqual(req.metadata_buffer_index, -1)

    def test_active_transfer_keeps_pages_pinned(self):
        stub = self._scheduler_stub()
        req = self._req(inflight=True)
        with patch("sglang.srt.disaggregation.prefill.release_kv_cache") as release:
            stub.maybe_defer_kv_release(req)
            stub.process_pending_kv_releases()
            stub.process_pending_kv_releases()
            release.assert_not_called()
            stub.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
            self.assertEqual(stub.disagg_prefill_pending_kv_releases, [(req, True)])

    def test_pending_release_keeps_scheduler_non_idle(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.enable_overlap = False
        scheduler.ps = ParallelState.trivial()
        scheduler.running_mbs = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.disagg_prefill_inflight_queue = []
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        scheduler.disagg_prefill_pending_kv_releases = [(object(), True)]
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False

        self.assertFalse(scheduler.is_fully_idle())


if __name__ == "__main__":
    unittest.main()
