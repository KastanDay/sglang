import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.unified_radix_cache import (
    _OngoingPrefetch,
    UnifiedRadixCache,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestUnifiedRadixAbortCleanup(unittest.TestCase):
    RID = "request"

    def _cache(self, *, local_state: int, global_state: int):
        cache = object.__new__(UnifiedRadixCache)
        cache.prefetch_loaded_tokens_by_reqid = {self.RID: 4}
        cache.ongoing_prefetch = {}
        cache._barrier_attn_groups = mock.Mock()
        cache.dec_host_lock_ref = mock.Mock()
        cache.cache_controller = SimpleNamespace(
            terminate_prefetch=mock.Mock(return_value=(3, None)),
            append_host_mem_release=mock.Mock(),
            prefetch_tokens_occupied=4,
        )

        def all_reduce(state, op):
            self.assertEqual(op, torch.distributed.ReduceOp.MAX)
            state.fill_(global_state)

        cache._all_reduce_attn_groups = mock.Mock(side_effect=all_reduce)

        operation = None
        if local_state:
            host_indices = torch.arange(4) if local_state == 2 else None
            operation = SimpleNamespace(host_indices=host_indices)
            cache.ongoing_prefetch[self.RID] = _OngoingPrefetch(
                anchor_node=mock.sentinel.anchor,
                prefetch_key=[0, 1, 2, 3],
                host_indices=host_indices,
                operation=operation,
                anchor_lock_params=mock.sentinel.lock_params,
                comp_xfers={},
            )
        return cache, operation

    def test_missing_rank_joins_barrier_when_peer_started_io(self):
        cache, _ = self._cache(local_state=0, global_state=2)

        cache.release_aborted_request(self.RID)

        cache._all_reduce_attn_groups.assert_called_once()
        cache._barrier_attn_groups.assert_called_once_with()
        cache.cache_controller.terminate_prefetch.assert_not_called()

    def test_pending_rank_revokes_then_joins_active_peer_barrier(self):
        cache, operation = self._cache(local_state=1, global_state=2)

        cache.release_aborted_request(self.RID)

        cache.cache_controller.terminate_prefetch.assert_called_once_with(operation)
        cache.cache_controller.append_host_mem_release.assert_called_once_with(
            extra_pools=[]
        )
        cache._barrier_attn_groups.assert_called_once_with()
        self.assertNotIn(self.RID, cache.ongoing_prefetch)
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)

    def test_active_rank_drains_and_releases_after_collective_barrier(self):
        cache, operation = self._cache(local_state=2, global_state=2)

        cache.release_aborted_request(self.RID)

        cache.cache_controller.terminate_prefetch.assert_called_once_with(operation)
        cache._barrier_attn_groups.assert_called_once_with()
        cache.dec_host_lock_ref.assert_called_once_with(
            mock.sentinel.anchor, mock.sentinel.lock_params
        )
        release = cache.cache_controller.append_host_mem_release.call_args.kwargs
        self.assertTrue(torch.equal(release["host_indices"], torch.arange(3)))
        self.assertEqual(release["extra_pools"], [])
        self.assertNotIn(self.RID, cache.ongoing_prefetch)


if __name__ == "__main__":
    unittest.main()
