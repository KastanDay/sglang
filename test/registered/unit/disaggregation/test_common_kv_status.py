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
        if status is not None:
            manager.request_status[7] = status
        return manager

    def test_failed_status_is_absorbing_until_clear(self):
        manager = self._manager(KVPoll.Failed)

        for late_status in (KVPoll.Transferring, KVPoll.Success):
            with self.subTest(late_status=late_status):
                manager.update_status(7, late_status)
                self.assertEqual(manager.request_status[7], KVPoll.Failed)

    def test_cleared_room_can_start_a_new_generation(self):
        manager = self._manager(KVPoll.Failed)

        manager.request_status.pop(7)
        manager.update_status(7, KVPoll.Bootstrapping)
        manager.update_status(7, KVPoll.Success)

        self.assertEqual(manager.request_status[7], KVPoll.Success)

    def test_late_success_after_clear_does_not_resurrect_room(self):
        manager = self._manager()
        manager.is_dummy_cp_rank = True
        sender = CommonKVSender(manager, "127.0.0.1:30000", 7, [], 0)

        sender.clear()

        for late_status in (
            KVPoll.WaitingForInput,
            KVPoll.Transferring,
            KVPoll.Success,
            KVPoll.Failed,
        ):
            with self.subTest(late_status=late_status):
                manager.update_status(7, late_status)
                self.assertNotIn(7, manager.request_status)

    def test_dummy_cp_rank_starts_at_bootstrapping(self):
        manager = self._manager()
        manager.is_dummy_cp_rank = True

        CommonKVSender(manager, "127.0.0.1:30000", 7, [], 0)

        self.assertEqual(manager.request_status[7], KVPoll.WaitingForInput)


if __name__ == "__main__":
    unittest.main()
