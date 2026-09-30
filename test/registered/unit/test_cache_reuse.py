import ast
import importlib.util
import threading
from pathlib import Path

from prometheus_client import CollectorRegistry, generate_latest

source = Path(__file__).parents[3] / "python/sglang/srt/observability/cache_reuse.py"
spec = importlib.util.spec_from_file_location("cache_reuse", source)
cache_reuse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_reuse)


def test_tiers_are_disjoint_and_unknown_is_not_zero():
    result = cache_reuse.committed_reuse(
        100,
        80,
        {
            "device": 10,
            "host": 20,
            "storage": 50,
            "storage_sources": {"store_dram": 20, "store_ssd": 15, "store_mixed": 10},
        },
    )
    assert sum(result.values()) == 100
    assert result["computed"] == 20
    assert result["store_unknown"] == 5
    assert cache_reuse.committed_reuse(100, 80, None)["reuse_unknown"] == 80


def test_promotions_and_invalid_shards_do_not_double_count():
    result = cache_reuse.committed_reuse(
        10, 100, {"device": 10, "host": 10, "storage": 10}
    )
    assert sum(result.values()) == 10
    assert result["gpu"] == 10
    assert result["host"] == result["store_unknown"] == 0


def test_denominator_includes_no_hits_and_each_tier_gets_zero_observations():
    registry = CollectorRegistry()
    collector = cache_reuse.CacheReuseCollector({"model_name": "test"}, registry)
    collector.observe(100, 0, None)
    collector.observe(100, 50, {"storage": 50})
    text = generate_latest(registry).decode()
    assert (
        'sglang:committed_prefill_requests_total{hit="no",model_name="test"} 1.0'
        in text
    )
    assert (
        'sglang:committed_prefill_requests_total{hit="yes",model_name="test"} 1.0'
        in text
    )
    assert (
        'sglang:committed_cache_prefix_tokens_count{model_name="test",source="all"} 2.0'
        in text
    )
    assert (
        'sglang:committed_input_tokens_total{model_name="test",source="computed"} 150.0'
        in text
    )


def test_prefetch_deadline_is_not_consumption():
    assert list(cache_reuse.prefetch_stages(100, 80, 70, 30).values()) == [
        100,
        80,
        70,
        30,
    ]
    assert cache_reuse.prefetch_stages(100, 200, 300, 400)["published"] == 100


def test_receipts_union_components_and_truncate_to_consumed_pages():
    codes = cache_reuse.group_receipts(
        [([10, 10, 10, 10], ["memory", "local_disk", "memory", "memory"])], 2
    )
    assert codes == [3, 1]
    assert cache_reuse.summarize_sources(codes, 3, 2) == {
        "store_dram": 1,
        "store_ssd": 0,
        "store_mixed": 2,
        "store_unknown": 0,
    }


def test_aborted_or_missing_usage_is_coverage_not_completed_consumption():
    registry = CollectorRegistry()
    collector = cache_reuse.CacheReuseCollector({"model_name": "test"}, registry)
    collector.observe(100, 50, None, "aborted")
    collector.observe(None, None, None)
    text = generate_latest(registry).decode()
    assert 'status="aborted"' in text
    assert 'status="usage_unavailable"' in text
    assert "sglang:committed_input_tokens_total{" not in text


def test_absent_failed_malformed_or_incomplete_receipts_are_unknown():
    assert cache_reuse.group_receipts([], 2) == [8, 8]
    assert cache_reuse.group_receipts([([-1, 10], ["memory", "local_disk"])], 2) == [
        8,
        2,
    ]
    assert cache_reuse.group_receipts([([10], ["memory", "local_disk"])], 2) == [8, 8]
    codes = cache_reuse.group_receipts(
        [([10, 10], ["memory", "local_disk"]), ([10, 10], ["unknown", "memory"])], 2
    )
    assert codes == [9, 3]
    assert cache_reuse.summarize_sources(codes, 4, 2)["store_unknown"] == 2


def test_both_radix_implementations_pop_provenance_once():
    from types import SimpleNamespace

    for filename, class_name in (
        ("hiradix_cache.py", "HiRadixCache"),
        ("unified_radix_cache.py", "UnifiedRadixCache"),
    ):
        path = source.parents[1] / "mem_cache" / filename
        tree = ast.parse(path.read_text())
        selected = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        method = next(
            node
            for node in selected.body
            if isinstance(node, ast.FunctionDef) and node.name == "pop_prefetch_sources"
        )
        namespace = {}
        module = ast.Module(body=[method], type_ignores=[])
        eval(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        instance = SimpleNamespace(prefetch_sources_by_reqid={"request": ([1, 2], 2)})
        assert namespace[method.name](instance, "request") == ([1, 2], 2)
        assert namespace[method.name](instance, "request") is None


def connector_methods():
    path = source.parents[1] / "mem_cache/storage/mooncake_store/mooncake_store.py"
    tree = ast.parse(path.read_text())
    methods = {
        "batch_get_v1_with_sources",
        "batch_get_v2_with_sources",
        "_get_batch_zero_copy_impl",
    }
    selected = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MooncakeStore"
    )
    selected.body = [
        node
        for node in selected.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    selected.bases = []
    selected.decorator_list = []
    module = ast.Module(body=[selected], type_ignores=[])
    namespace = {"List": list, "Any": object}
    compiled = compile(ast.fix_missing_locations(module), str(path), "exec")
    eval(compiled, namespace)
    return namespace["MooncakeStore"]


def test_actual_connector_preserves_old_sdk_and_fails_unknown():
    import sys
    from types import SimpleNamespace

    sys.modules["sglang.srt.observability.cache_reuse"] = cache_reuse
    connector = connector_methods()()
    connector._receipt_context = threading.local()
    connector._uses_multi_buffer = lambda buffers: False
    connector.store = SimpleNamespace(
        batch_get_into=lambda keys, buffers, sizes: [10] * len(keys)
    )
    connector.batch_get_v1 = lambda keys, indices, extra: [
        result > 0
        for result in connector._get_batch_zero_copy_impl(
            keys, indices, [10] * len(keys)
        )
    ]
    flags, codes = connector.batch_get_v1_with_sources(["opaque-key"], [1])
    assert flags == [True]
    assert codes == [8]
    assert connector._receipt_context.calls is None
    assert connector._get_batch_zero_copy_impl(["opaque-key"], [1], [10]) == [10]


def test_actual_connector_receipts_flow_to_consumed_export():
    import sys
    from types import SimpleNamespace

    sys.modules["sglang.srt.observability.cache_reuse"] = cache_reuse
    connector = connector_methods()()
    connector._receipt_context = threading.local()
    connector._uses_multi_buffer = lambda buffers: False
    connector.store = SimpleNamespace(
        batch_get_into_with_sources=lambda keys, buffers, sizes: (
            [10, 10],
            ["memory", "local_disk"],
        )
    )
    connector.batch_get_v1 = lambda keys, indices, extra: [
        result > 0
        for result in connector._get_batch_zero_copy_impl(
            keys, indices, [10] * len(keys)
        )
    ]
    flags, codes = connector.batch_get_v1_with_sources(["first", "second"], [1, 2])
    assert flags == [True, True]
    registry = CollectorRegistry()
    collector = cache_reuse.CacheReuseCollector({"model_name": "test"}, registry)
    collector.observe(
        10,
        3,
        {"storage": 3, "storage_sources": cache_reuse.summarize_sources(codes, 3, 2)},
    )
    text = generate_latest(registry).decode()
    assert (
        'sglang:committed_input_tokens_total{model_name="test",source="store_dram"} 2.0'
        in text
    )
    assert (
        'sglang:committed_input_tokens_total{model_name="test",source="store_ssd"} 1.0'
        in text
    )
    assert (
        'sglang:committed_input_tokens_total{model_name="test",source="computed"} 7.0'
        in text
    )
    assert "first" not in text and "second" not in text
