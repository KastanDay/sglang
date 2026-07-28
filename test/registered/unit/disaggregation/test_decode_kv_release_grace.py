"""Decode must not free a failed transfer's KV pages while prefill may write.

Decode's pre-allocated pages are the *destination* of the prefill side's RDMA
writes. When a transfer fails or is aborted, the failure/abort notification
stops the prefill from starting new chunks, but a chunk already in flight
keeps writing until it drains. On the unfixed code `pop_transferred` freed the
pages (and the metadata buffer slot, which is the aux-data write destination)
immediately, so the next request could be allocated pages that a straggling
write still lands in -- silently corrupting its KV, with no error anywhere.

The fix holds the release for KV_RELEASE_GRACE_PERIOD_S, and additionally
sends the abort notification when a HiCache restore fails while the transfer
itself is still live (that branch previously released without telling the
prefill side to stop).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
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
    queue._deferred_kv_releases = []
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


class TestFailedTransferReleaseIsDeferred(CustomTestCase):
    def test_pages_are_held_for_the_grace_period_then_freed(self):
        receiver = FakeReceiver()
        decode_req = make_decode_req(receiver)
        queue = make_transfer_queue(decode_req)
        queue._poll_with_metadata_gate = MagicMock(return_value=[KVPoll.Failed])

        with patch(
            "sglang.srt.disaggregation.decode.release_kv_cache"
        ) as release, patch(
            "sglang.srt.disaggregation.decode.KV_RELEASE_GRACE_PERIOD_S", 0.05
        ):
            queue.pop_transferred()

            # The request concluded and left the queue, but its pages and its
            # metadata buffer slot -- both in-flight write destinations --
            # must still be allocated.
            self.assertEqual(queue.queue, [])
            self.assertTrue(receiver.clear_called)
            release.assert_not_called()
            queue.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
            self.assertEqual(len(queue._deferred_kv_releases), 1)

            # Before the grace period ends, the sweep keeps holding them.
            queue.pop_transferred()
            release.assert_not_called()

            # After it, the sweep frees pages and the metadata slot -- even
            # with an empty queue.
            with patch(
                "sglang.srt.disaggregation.decode.time.monotonic",
                return_value=queue._deferred_kv_releases[0][1] + 1,
            ):
                queue.pop_transferred()
            release.assert_called_once_with(
                decode_req.req, queue.tree_cache, is_insert=False
            )
            queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
            self.assertEqual(queue._deferred_kv_releases, [])

    def test_restore_failure_aborts_the_live_transfer(self):
        # A failed HiCache restore says nothing about the KV transfer: the
        # prefill side may not even have started writing. It must be told to
        # stop, and the pages must still go through the grace period.
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
        self.assertEqual(len(queue._deferred_kv_releases), 1)


if __name__ == "__main__":
    unittest.main()
