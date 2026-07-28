"""CPU unit tests for the transfer-room status state machine in CommonKVManager.

A transfer room aggregates status reports from several concurrent sources
(per-rank transfer workers, the heartbeat checker, abort handling). These
tests pin the invariant that a reported failure is terminal for the room
until clear(), and that clear() fully resets the room for reuse.
"""

import threading
import unittest

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVSender
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCommonKVStatus(unittest.TestCase):
    @staticmethod
    def _manager(status=None):
        manager = object.__new__(CommonKVManager)
        manager.request_status = {}
        manager._request_status_lock = threading.Lock()
        if status is not None:
            manager.request_status[7] = status
        return manager

    @staticmethod
    def _sender(manager):
        sender = object.__new__(CommonKVSender)
        sender.kv_mgr = manager
        sender.bootstrap_room = 7
        return sender

    def test_failed_room_is_not_promoted_by_late_updates(self):
        """A late completion must not promote a failed room back to Success.

        KVPoll.Failed is the smallest enum value, so ordering the states with
        max() silently promotes a failed room when another participant reports
        Transferring or Success afterwards. Decode treats Success as permission
        to commit the transfer, so the promotion lets it read partially written
        KV pages.
        """
        manager = self._manager(KVPoll.Failed)

        for late_status in (KVPoll.Transferring, KVPoll.Success):
            with self.subTest(late_status=late_status):
                manager.update_status(7, late_status)
                self.assertEqual(manager.request_status[7], KVPoll.Failed)

    def test_cleared_room_advances_normally_on_reuse(self):
        """clear() ends the failure; a reused room must progress to Success."""
        manager = self._manager(KVPoll.Failed)

        self._sender(manager).clear()
        manager.update_status(7, KVPoll.Bootstrapping)
        manager.update_status(7, KVPoll.WaitingForInput)
        manager.update_status(7, KVPoll.Success)

        self.assertEqual(manager.request_status[7], KVPoll.Success)

    def test_late_failed_update_does_not_resurrect_cleared_room(self):
        """A late Failed after clear() must not seed a reused room with failure."""
        manager = self._manager(KVPoll.Failed)

        self._sender(manager).clear()
        manager.update_status(7, KVPoll.Failed)

        self.assertNotIn(7, manager.request_status)

    def test_status_updates_are_serialized_with_the_room_lock(self):
        """update_status must hold the room lock across its read-modify-write.

        Without the lock, a transfer worker's Success can interleave with the
        heartbeat checker's Failed (read old state, write after), losing the
        failure. The Success reporter here must block until the lock is
        released, then observe the failure and keep it.
        """
        manager = self._manager(KVPoll.WaitingForInput)
        success_started = threading.Event()

        def report_success():
            success_started.set()
            manager.update_status(7, KVPoll.Success)

        with manager._request_status_lock:
            manager.request_status[7] = KVPoll.Failed
            thread = threading.Thread(target=report_success)
            thread.start()
            self.assertTrue(success_started.wait(timeout=1))
            thread.join(timeout=0.2)
            self.assertTrue(thread.is_alive())

        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(manager.request_status[7], KVPoll.Failed)


if __name__ == "__main__":
    unittest.main()
