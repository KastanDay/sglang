import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    _parallel_sample_bootstrap_room,
    _parallel_sample_request_id,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestParallelSampleRequestId(unittest.TestCase):
    def test_ids_are_stable_across_tokenizer_workers_and_distinct(self):
        first_worker = [
            _parallel_sample_request_id("shared", None),
            _parallel_sample_request_id("shared", 0),
            _parallel_sample_request_id("shared", 1),
        ]
        second_worker = [
            _parallel_sample_request_id("shared", None),
            _parallel_sample_request_id("shared", 0),
            _parallel_sample_request_id("shared", 1),
        ]

        self.assertEqual(first_worker, second_worker)
        self.assertEqual(len(set(first_worker)), len(first_worker))

    def test_rooms_are_stable_and_do_not_reuse_prefix_cache_room(self):
        first_worker = [
            _parallel_sample_bootstrap_room(7, "shared", i) for i in range(3)
        ]
        second_worker = [
            _parallel_sample_bootstrap_room(7, "shared", i) for i in range(3)
        ]

        self.assertEqual(first_worker, second_worker)
        self.assertEqual(len(set(first_worker)), len(first_worker))
        self.assertNotIn(7, first_worker)
        self.assertIsNone(_parallel_sample_bootstrap_room(None, "shared", 0))


if __name__ == "__main__":
    unittest.main()
