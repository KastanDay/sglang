import threading
import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    CommonKVSender,
    PrefillRankInfo,
)
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

        manager.clear_status(7)
        manager.update_status(7, KVPoll.Bootstrapping)
        manager.update_status(7, KVPoll.Success)

        self.assertEqual(manager.request_status[7], KVPoll.Success)

    def test_late_success_after_clear_does_not_resurrect_room(self):
        manager = self._manager()
        manager.is_dummy_cp_rank = True
        sender = CommonKVSender(manager, "127.0.0.1:30000", 7, [], 0, request_id="old")

        sender.clear()
        manager.mark_metadata_ready(7, "old")

        for late_status in (
            KVPoll.Transferring,
            KVPoll.Success,
            KVPoll.Failed,
        ):
            with self.subTest(late_status=late_status):
                manager.update_status(7, late_status)
                self.assertNotIn(7, manager.request_status)

        manager.is_dummy_cp_rank = False
        manager.server_args = SimpleNamespace(dp_size=1)
        CommonKVSender(manager, "127.0.0.1:30000", 7, [], 0, request_id="new")
        self.assertEqual(manager.request_status[7], KVPoll.Bootstrapping)

    def test_dummy_cp_rank_initializes_waiting(self):
        manager = self._manager()
        manager.is_dummy_cp_rank = True

        CommonKVSender(manager, "127.0.0.1:30000", 7, [], 0)

        self.assertEqual(manager.request_status[7], KVPoll.WaitingForInput)

    def test_metadata_readiness_can_arrive_before_sender(self):
        manager = self._manager()
        manager.is_dummy_cp_rank = False
        manager.server_args = SimpleNamespace(dp_size=1)
        manager.mark_metadata_ready(7, "new")

        CommonKVSender(manager, "127.0.0.1:30000", 7, [], 0, request_id="new")

        self.assertEqual(manager.request_status[7], KVPoll.WaitingForInput)

    def test_reused_room_accepts_only_new_generation_metadata(self):
        manager = self._manager()
        manager.is_dummy_cp_rank = True
        sender = CommonKVSender(manager, "127.0.0.1:30000", 7, [], 0, request_id="old")
        sender.clear()

        manager.mark_metadata_ready(7, "old")
        manager.mark_metadata_ready(7, "new")
        manager.is_dummy_cp_rank = False
        manager.server_args = SimpleNamespace(dp_size=1)
        CommonKVSender(manager, "127.0.0.1:30000", 7, [], 0, request_id="new")

        self.assertEqual(manager.request_status[7], KVPoll.WaitingForInput)

    def test_old_sender_cannot_clear_reused_room(self):
        manager = self._manager()
        manager.is_dummy_cp_rank = True
        old_sender = CommonKVSender(
            manager, "127.0.0.1:30000", 7, [], 0, request_id="old"
        )
        old_sender.clear()

        manager.is_dummy_cp_rank = False
        manager.server_args = SimpleNamespace(dp_size=1)
        CommonKVSender(manager, "127.0.0.1:30000", 7, [], 0, request_id="new")
        manager.transfer_infos = {7: {"new-peer": object()}}
        manager.req_to_decode_prefix_len = {7: 5}
        self.assertFalse(old_sender.clear())

        self.assertEqual(manager.request_status[7], KVPoll.Bootstrapping)
        self.assertIn(7, manager.transfer_infos)
        self.assertEqual(manager.req_to_decode_prefix_len[7], 5)

    def test_late_old_generation_status_cannot_complete_reused_room(self):
        manager = self._manager()
        manager.is_dummy_cp_rank = True
        old_sender = CommonKVSender(
            manager, "127.0.0.1:30000", 7, [], 0, request_id="old"
        )
        old_sender.clear()

        manager.is_dummy_cp_rank = False
        manager.server_args = SimpleNamespace(dp_size=1)
        CommonKVSender(manager, "127.0.0.1:30000", 7, [], 0, request_id="new")

        manager.update_status(7, KVPoll.Success, request_id="old")
        manager.update_status(7, KVPoll.Success)

        self.assertEqual(manager.request_status[7], KVPoll.Bootstrapping)

    def test_early_transfer_metadata_is_partitioned_by_generation(self):
        manager = self._manager()
        manager.transfer_infos = {}
        old_info = SimpleNamespace(decode_prefix_len=3)
        new_info = SimpleNamespace(decode_prefix_len=5)

        manager.store_transfer_info(7, "old", "old-peer", old_info)
        manager.store_transfer_info(7, "new", "new-peer", new_info)
        manager.start_generation(7, "new")

        self.assertEqual(manager.transfer_infos[7], {"new-peer": new_info})

    def test_late_metadata_for_retired_generation_is_rejected(self):
        manager = self._manager()
        manager.transfer_infos = {}
        manager.start_generation(7, "old")
        manager.clear_status(7, "old")

        result = manager.store_transfer_info(
            7,
            "old",
            "late-peer",
            SimpleNamespace(decode_prefix_len=3),
        )

        self.assertIsNone(result)
        self.assertNotIn(7, manager.transfer_infos)

    def test_metadata_cannot_mutate_generation_after_bootstrap(self):
        manager = self._manager()
        manager.transfer_infos = {}
        first_info = SimpleNamespace(decode_prefix_len=3)
        manager.start_generation(7, "current")
        manager.store_transfer_info(7, "current", "first-peer", first_info)
        manager.mark_metadata_ready(7, "current")

        result = manager.store_transfer_info(
            7,
            "current",
            "late-peer",
            SimpleNamespace(decode_prefix_len=5),
        )

        self.assertIsNone(result)
        self.assertEqual(manager.transfer_infos[7], {"first-peer": first_info})

    def test_legacy_status_requires_generation_negotiation(self):
        manager = self._manager()
        manager.start_generation(7, "current")

        manager.update_status(7, KVPoll.Success)
        self.assertEqual(manager.request_status[7], KVPoll.Bootstrapping)

        manager.allow_legacy_generation(7)
        manager.update_status(7, KVPoll.Success)
        self.assertEqual(manager.request_status[7], KVPoll.Success)

    def test_legacy_abort_can_negotiate_only_during_bootstrap(self):
        manager = self._manager()
        manager.start_generation(7, "current")

        self.assertTrue(manager.is_current_generation_or_legacy_bootstrap(7, None))
        self.assertIsNone(manager.wire_request_id(7, "current"))

        manager.clear_status(7, "current")
        manager.start_generation(7, "new")
        manager.update_status(7, KVPoll.WaitingForInput, request_id="new")

        self.assertFalse(manager.is_current_generation_or_legacy_bootstrap(7, None))
        self.assertEqual(manager.wire_request_id(7, "new"), "new")

    def test_legacy_transfer_info_negotiates_active_generation(self):
        manager = self._manager()
        manager.transfer_infos = {}
        manager.start_generation(7, "current")
        info = SimpleNamespace(decode_prefix_len=3, request_id=None)

        infos = manager.store_transfer_info(7, None, "legacy-peer", info)
        manager.mark_metadata_ready(7, None)

        self.assertEqual(infos, {"legacy-peer": info})
        self.assertIsNone(info.request_id)
        self.assertEqual(manager.request_status[7], KVPoll.WaitingForInput)

    def test_early_legacy_transfer_info_is_adopted_by_new_sender(self):
        manager = self._manager()
        manager.transfer_infos = {}
        info = SimpleNamespace(decode_prefix_len=3, request_id=None)
        manager.store_transfer_info(7, None, "legacy-peer", info)

        manager.start_generation(7, "current")

        self.assertEqual(manager.transfer_infos[7], {"legacy-peer": info})
        self.assertIsNone(info.request_id)
        self.assertTrue(manager.is_current_generation(7, None))

    def test_new_generation_does_not_inherit_legacy_fallback(self):
        manager = self._manager()
        manager.start_generation(7, "old")
        manager.allow_legacy_generation(7)
        self.assertIsNone(manager.wire_request_id(7, "old"))
        manager.clear_status(7, "old")
        manager.start_generation(7, "new")

        self.assertEqual(manager.wire_request_id(7, "new"), "new")
        manager.update_status(7, KVPoll.Success)

        self.assertEqual(manager.request_status[7], KVPoll.Bootstrapping)

    def test_rank_bootstrap_advertises_generation_capability(self):
        info = PrefillRankInfo(rank_ip="127.0.0.1", rank_port=30000)

        self.assertTrue(info.generation_ids_supported)

    def test_concurrent_success_cannot_overwrite_failure(self):
        manager = self._manager(KVPoll.WaitingForInput)
        success_started = threading.Event()

        def report_success():
            success_started.set()
            manager.update_status(7, KVPoll.Success)

        with manager._get_request_status_lock():
            thread = threading.Thread(target=report_success)
            thread.start()
            self.assertTrue(success_started.wait(timeout=1))
            manager.update_status(7, KVPoll.Failed)

        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(manager.request_status[7], KVPoll.Failed)


if __name__ == "__main__":
    unittest.main()
