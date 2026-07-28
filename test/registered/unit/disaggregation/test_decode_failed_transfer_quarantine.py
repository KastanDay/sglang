"""Decode must not free a failed transfer's KV pages while prefill may write.

Decode's pre-allocated pages are the *destination* of the prefill side's RDMA
writes. When a transfer fails or is aborted, the failure/abort notification
stops the prefill from starting new chunks, but a chunk already in flight
keeps writing until it drains. On the unfixed code `pop_transferred` freed the
pages (and the metadata buffer slot, which is the aux-data write destination)
immediately, so the next request could be allocated pages that a straggling
write still lands in -- silently corrupting its KV, with no error anywhere.

There is no local timeout that proves a remote write is done. The fix
quarantines failed destinations until process restart, and additionally sends
the abort notification when a HiCache restore fails while the transfer itself
is still live (that branch previously released without telling prefill to
stop).
"""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.decode import (
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVReceiver
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeReceiver:
    def __init__(self):
        self.abort_called = False
        self.clear_called = False

    def abort(self):
        self.abort_called = True

    def clear(self):
        self.clear_called = True

    def failure_exception(self):
        return None


def make_transfer_queue(decode_req):
    queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    queue.queue = [decode_req]
    queue._failed_transfer_quarantine = []
    queue.enable_staging = False
    queue.tp_rank = 0
    queue.tree_cache = MagicMock()
    queue.metadata_buffers = MagicMock()
    queue.req_to_metadata_buffer_idx_allocator = MagicMock()
    queue._clean_hicache_prefetch_resources = MagicMock()

    scheduler = MagicMock()
    scheduler.enable_decode_hicache = False
    scheduler.enable_hisparse = False
    scheduler.metrics_reporter.enable_metrics = False
    queue.scheduler = scheduler
    return queue


def make_decode_req(receiver, restore_status=None):
    req = SimpleNamespace(
        rid="r1",
        bootstrap_room=7,
        return_logprob=False,
        finished_reason=None,
    )
    return SimpleNamespace(
        req=req,
        kv_receiver=receiver,
        hicache_restore_status=restore_status,
        metadata_buffer_index=3,
    )


class TestFailedTransferReleaseIsQuarantined(CustomTestCase):
    def test_failed_transfer_destinations_remain_quarantined(self):
        receiver = FakeReceiver()
        decode_req = make_decode_req(receiver)
        queue = make_transfer_queue(decode_req)
        queue._poll_with_metadata_gate = MagicMock(return_value=[KVPoll.Failed])

        with patch("sglang.srt.disaggregation.decode.release_kv_cache") as release:
            queue.pop_transferred()

            # The request concluded and left the queue, but its pages and its
            # metadata buffer slot -- both in-flight write destinations --
            # must still be allocated.
            self.assertEqual(queue.queue, [])
            self.assertTrue(receiver.clear_called)
            release.assert_not_called()
            queue.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
            self.assertEqual(queue._failed_transfer_quarantine, [decode_req])

            # Polling again cannot turn elapsed wall time into proof that a
            # remote write is quiescent.
            queue.pop_transferred()
            release.assert_not_called()
            queue.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
            self.assertEqual(queue._failed_transfer_quarantine, [decode_req])

    def test_restore_failure_aborts_the_live_transfer(self):
        # A failed HiCache restore says nothing about the KV transfer: the
        # prefill side may not even have started writing. It must be told to
        # stop, and the destination buffers must still be quarantined.
        receiver = FakeReceiver()
        decode_req = make_decode_req(
            receiver, restore_status=HiCacheRestoreResult.FAILED
        )
        queue = make_transfer_queue(decode_req)
        queue._poll_with_metadata_gate = MagicMock(
            return_value=[KVPoll.WaitingForInput]
        )

        with patch("sglang.srt.disaggregation.decode.release_kv_cache") as release:
            queue.pop_transferred()

        self.assertTrue(
            receiver.abort_called,
            "prefill must be told to stop before its writes land in freed pages",
        )
        release.assert_not_called()
        self.assertEqual(queue._failed_transfer_quarantine, [decode_req])

    def test_completed_transfer_can_release_after_restore_failure(self):
        receiver = FakeReceiver()
        decode_req = make_decode_req(
            receiver, restore_status=HiCacheRestoreResult.FAILED
        )
        queue = make_transfer_queue(decode_req)
        queue._poll_with_metadata_gate = MagicMock(return_value=[KVPoll.Success])

        with patch("sglang.srt.disaggregation.decode.release_kv_cache") as release:
            queue.pop_transferred()

        self.assertFalse(receiver.abort_called)
        release.assert_called_once_with(
            decode_req.req, queue.tree_cache, is_insert=False
        )
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        self.assertEqual(queue._failed_transfer_quarantine, [])

    def test_all_reduced_mooncake_failure_notifies_prefill(self):
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.bootstrap_room = 7
        receiver.conclude_state = None
        receiver.abort_notified = False
        receiver.bootstrap_infos = [{"rank_ip": "127.0.0.1", "rank_port": 1}]
        receiver.kv_mgr = SimpleNamespace(
            request_status={7: KVPoll.Failed},
            required_prefill_response_num_table={7: 1},
            prefill_response_tracker={7: set()},
            failure_lock=threading.Lock(),
            failure_records={},
        )

        with patch.object(receiver, "_send_abort_notification") as notify:
            with self.assertRaises(KVTransferError) as raised:
                receiver.failure_exception()

        notify.assert_called_once_with()
        self.assertTrue(receiver.abort_notified)
        self.assertTrue(raised.exception.is_from_another_rank)


if __name__ == "__main__":
    unittest.main()
