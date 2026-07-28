import concurrent.futures
import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import FastQueue
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMooncakeTransferQuiescence(unittest.TestCase):
    @staticmethod
    def _manager():
        manager = object.__new__(MooncakeKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.request_status = {7: KVPoll.Transferring}
        manager.transfer_infos = {7: {"session:1": object()}}
        manager.transfer_queues = [FastQueue()]
        manager._pending_transfer_counts = defaultdict(int)
        manager._pending_transfer_counts_lock = threading.Lock()
        return manager

    def test_queued_chunk_keeps_source_pages_live(self):
        manager = self._manager()
        sender = object.__new__(MooncakeKVSender)
        sender.kv_mgr = manager
        sender.bootstrap_room = 7

        manager.add_transfer_request(
            7,
            np.array([3], dtype=np.int32),
            slice(0, 1),
            is_last_chunk=False,
        )

        self.assertFalse(sender.is_transfer_quiesced())
        chunk = manager.transfer_queues[0].get()
        manager._finish_transfer(chunk.room)
        self.assertTrue(sender.is_transfer_quiesced())

    @patch(
        "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
        return_value=[KVPoll.Failed],
    )
    def test_failed_request_stays_in_sweep_until_local_reads_finish(self, _poll):
        sender = MagicMock()
        sender.is_transfer_quiesced.return_value = False
        req = SimpleNamespace(
            rid="request-1",
            pending_bootstrap=False,
            disagg_kv_sender=sender,
        )
        scheduler = SimpleNamespace(
            disagg_prefill_inflight_queue=[req],
            attn_cp_cpu_group=None,
            attn_tp_cpu_group=None,
            handle_inflight_transfer_failure=MagicMock(),
            output_streamer=MagicMock(),
        )

        done = (
            SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
                scheduler
            )
        )

        self.assertEqual(done, [])
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [req])
        scheduler.handle_inflight_transfer_failure.assert_not_called()

    def test_layer_failure_waits_for_sibling_native_transfer(self):
        manager = object.__new__(MooncakeKVManager)
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = False
        manager.enable_custom_mem_pool = True

        slow_started = threading.Event()
        allow_slow_finish = threading.Event()
        failure_returned = threading.Event()
        call_returned = threading.Event()
        result = []

        def transfer(_session_id, blocks):
            if blocks[0][0] == 100:
                self.assertTrue(slow_started.wait(timeout=1))
                failure_returned.set()
                return -1
            slow_started.set()
            self.assertTrue(allow_slow_finish.wait(timeout=1))
            return 0

        manager._transfer_data = transfer

        def run_transfer(executor):
            result.append(
                manager._send_kvcache_generic(
                    mooncake_session_id="session",
                    src_data_ptrs=[100, 200],
                    dst_data_ptrs=[300, 400],
                    item_lens=[4, 4],
                    prefill_data_indices=np.array([0], dtype=np.int32),
                    dst_data_indices=np.array([0], dtype=np.int32),
                    executor=executor,
                    force_flat=True,
                )
            )
            call_returned.set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            thread = threading.Thread(target=run_transfer, args=(executor,))
            thread.start()
            self.assertTrue(failure_returned.wait(timeout=1))
            self.assertFalse(call_returned.wait(timeout=0.1))
            allow_slow_finish.set()
            thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [-1])


if __name__ == "__main__":
    unittest.main()
