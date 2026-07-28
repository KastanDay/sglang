"""A failed KV send must not return while sibling layer transfers still run.

`_send_kvcache_generic` fans layers out to an executor and returns on the
first failing future after cancelling the rest. `Future.cancel()` is a no-op
for futures that are already running, so the caller could report failure --
letting the scheduler free the request's KV pages -- while sibling transfers
were still reading and writing those pages. A request that was then allocated
the same pages had its KV silently overwritten.
"""

import concurrent.futures
import threading
import unittest
from unittest.mock import patch

import numpy as np

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FAILING_LAYER_PTR = 0x1000
BLOCKED_LAYER_PTR = 0x2000


class TestSendFailureDrainsSiblings(CustomTestCase):
    def test_failed_send_waits_for_running_sibling_layer(self):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.is_mla_backend = True
        mgr.is_hybrid_mla_backend = False
        mgr.enable_custom_mem_pool = True
        mgr.pp_size = 1

        sibling_started = threading.Event()
        release_sibling = threading.Event()
        sibling_finished = threading.Event()

        def fake_transfer(session_id, transfer_blocks):
            src_addr = transfer_blocks[0][0]
            if src_addr == FAILING_LAYER_PTR:
                # Fail only once the sibling is inside its write, so cancel()
                # is guaranteed to hit a running -- uncancellable -- future.
                sibling_started.wait(5)
                return 1
            sibling_started.set()
            release_sibling.wait(5)
            sibling_finished.set()
            return 0

        mgr._transfer_data = fake_transfer

        result = []
        executor = concurrent.futures.ThreadPoolExecutor(2)

        def send():
            result.append(
                mgr._send_kvcache_generic(
                    mooncake_session_id="session",
                    src_data_ptrs=[FAILING_LAYER_PTR, BLOCKED_LAYER_PTR],
                    dst_data_ptrs=[0x3000, 0x4000],
                    item_lens=[16, 16],
                    prefill_data_indices=np.array([0], dtype=np.int32),
                    dst_data_indices=np.array([0], dtype=np.int32),
                    executor=executor,
                    src_layer_ids=[0, 1],
                    dst_layer_ids=[0, 1],
                )
            )

        sender = threading.Thread(target=send)
        real_wait = concurrent.futures.wait
        wait_entered = threading.Event()

        def recording_wait(*args, **kwargs):
            wait_entered.set()
            return real_wait(*args, **kwargs)

        with patch(
            "sglang.srt.disaggregation.mooncake.conn.concurrent.futures.wait",
            side_effect=recording_wait,
        ):
            sender.start()
            try:
                self.assertTrue(sibling_started.wait(5))
                self.assertTrue(
                    wait_entered.wait(5),
                    "the failed send did not start draining its sibling futures",
                )
                # wait() has been reached while the sibling remains blocked, so
                # returning now would release KV pages under the transfer.
                self.assertTrue(sender.is_alive())
            finally:
                release_sibling.set()
                sender.join(timeout=5)

        executor.shutdown(wait=True)

        self.assertEqual(result, [1], "the failure status must still be reported")
        self.assertTrue(sibling_finished.is_set())

    def test_exception_waits_for_running_sibling_layer(self):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.is_mla_backend = True
        mgr.is_hybrid_mla_backend = False
        mgr.enable_custom_mem_pool = True
        mgr.pp_size = 1

        sibling_started = threading.Event()
        release_sibling = threading.Event()
        wait_entered = threading.Event()

        def fake_transfer(_session_id, transfer_blocks):
            if transfer_blocks[0][0] == FAILING_LAYER_PTR:
                self.assertTrue(sibling_started.wait(5))
                raise RuntimeError("transfer failed")
            sibling_started.set()
            self.assertTrue(release_sibling.wait(5))
            return 0

        mgr._transfer_data = fake_transfer
        caught = []
        executor = concurrent.futures.ThreadPoolExecutor(2)
        real_wait = concurrent.futures.wait

        def recording_wait(*args, **kwargs):
            wait_entered.set()
            return real_wait(*args, **kwargs)

        def send():
            try:
                mgr._send_kvcache_generic(
                    mooncake_session_id="session",
                    src_data_ptrs=[FAILING_LAYER_PTR, BLOCKED_LAYER_PTR],
                    dst_data_ptrs=[0x3000, 0x4000],
                    item_lens=[16, 16],
                    prefill_data_indices=np.array([0], dtype=np.int32),
                    dst_data_indices=np.array([0], dtype=np.int32),
                    executor=executor,
                    src_layer_ids=[0, 1],
                    dst_layer_ids=[0, 1],
                )
            except RuntimeError as exc:
                caught.append(exc)

        sender = threading.Thread(target=send)
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.concurrent.futures.wait",
            side_effect=recording_wait,
        ):
            sender.start()
            try:
                self.assertTrue(sibling_started.wait(5))
                self.assertTrue(
                    wait_entered.wait(5),
                    "an exceptional send did not drain its sibling futures",
                )
                self.assertTrue(sender.is_alive())
            finally:
                release_sibling.set()
                sender.join(timeout=5)

        executor.shutdown(wait=True)

        self.assertEqual(len(caught), 1)
        self.assertEqual(str(caught[0]), "transfer failed")

    def test_submission_exception_drains_accepted_work(self):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.is_mla_backend = True
        mgr.is_hybrid_mla_backend = False
        mgr.enable_custom_mem_pool = True
        mgr.pp_size = 1

        work_started = threading.Event()
        release_work = threading.Event()
        wait_entered = threading.Event()

        def blocking_transfer(_session_id, _transfer_blocks):
            work_started.set()
            self.assertTrue(release_work.wait(5))
            return 0

        class FailingExecutor:
            def __init__(self):
                self.executor = concurrent.futures.ThreadPoolExecutor(1)
                self.submit_count = 0

            def submit(self, *args, **kwargs):
                self.submit_count += 1
                if self.submit_count == 2:
                    raise RuntimeError("submit failed")
                return self.executor.submit(*args, **kwargs)

            def shutdown(self):
                self.executor.shutdown(wait=True)

        mgr._transfer_data = blocking_transfer
        executor = FailingExecutor()
        caught = []
        real_wait = concurrent.futures.wait

        def recording_wait(*args, **kwargs):
            wait_entered.set()
            return real_wait(*args, **kwargs)

        def send():
            try:
                mgr._send_kvcache_generic(
                    mooncake_session_id="session",
                    src_data_ptrs=[FAILING_LAYER_PTR, BLOCKED_LAYER_PTR],
                    dst_data_ptrs=[0x3000, 0x4000],
                    item_lens=[16, 16],
                    prefill_data_indices=np.array([0], dtype=np.int32),
                    dst_data_indices=np.array([0], dtype=np.int32),
                    executor=executor,
                    src_layer_ids=[0, 1],
                    dst_layer_ids=[0, 1],
                )
            except RuntimeError as exc:
                caught.append(exc)

        sender = threading.Thread(target=send)
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.concurrent.futures.wait",
            side_effect=recording_wait,
        ):
            sender.start()
            try:
                self.assertTrue(work_started.wait(5))
                self.assertTrue(
                    wait_entered.wait(5),
                    "submission failure did not drain accepted transfer work",
                )
                self.assertTrue(sender.is_alive())
            finally:
                release_work.set()
                sender.join(timeout=5)

        executor.shutdown()

        self.assertEqual(len(caught), 1)
        self.assertEqual(str(caught[0]), "submit failed")


if __name__ == "__main__":
    unittest.main()
