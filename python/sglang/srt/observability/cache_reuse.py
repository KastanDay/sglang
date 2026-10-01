from prometheus_client import Counter, Histogram

SOURCES = (
    "gpu",
    "host",
    "store_dram",
    "store_ssd",
    "store_mixed",
    "store_unknown",
    "reuse_unknown",
    "computed",
)
PREFIX_BUCKETS = (0, 1, 16, 64, 256, 1024, 4096, 16384, 65536, 262144)


def source_code(source):
    return {"memory": 1, "local_disk": 2}.get(source, 8)


def group_receipts(calls, pages):
    if not pages:
        return []
    combined = [0] * pages
    for results, sources in calls:
        if len(results) != len(sources) or len(sources) % pages:
            return [8] * pages
        components = len(sources) // pages
        for page in range(pages):
            for component in range(components):
                offset = page * components + component
                combined[page] |= (
                    source_code(sources[offset]) if results[offset] > 0 else 8
                )
    return [code or 8 for code in combined]


def summarize_sources(codes, tokens, page_size):
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    result = dict.fromkeys(
        ("store_dram", "store_ssd", "store_mixed", "store_unknown"), 0
    )
    for offset in range(0, tokens, page_size):
        index = offset // page_size
        code = codes[index] if index < len(codes) else 8
        source = {1: "store_dram", 2: "store_ssd", 3: "store_mixed"}.get(
            code, "store_unknown"
        )
        result[source] += min(page_size, tokens - offset)
    return result


def committed_reuse(prompt, cached, details):
    """Partition logical input once; never infer a Store's physical tier."""
    prompt = max(0, int(prompt))
    cached = min(prompt, max(0, int(cached)))
    remaining = cached
    result = dict.fromkeys(SOURCES, 0)
    details = details or {}
    for source, field in (("gpu", "device"), ("host", "host")):
        amount = min(remaining, max(0, int(details.get(field, 0))))
        result[source] = amount
        remaining -= amount
    storage = min(remaining, max(0, int(details.get("storage", 0))))
    receipt = details.get("storage_sources") or {}
    for source in ("store_dram", "store_ssd", "store_mixed"):
        amount = min(storage, max(0, int(receipt.get(source, 0))))
        result[source] = amount
        storage -= amount
        remaining -= amount
    result["store_unknown"] = storage
    remaining -= storage
    result["reuse_unknown"] = remaining
    result["computed"] = prompt - cached
    return result


class CacheReuseCollector:
    def __init__(self, labels, registry=None):
        arguments = {} if registry is None else {"registry": registry}
        self.labels = dict(labels)
        self.attempts = Counter(
            "sglang:cache_observability_prefill_attempts",
            "Finished prefill attempts with explicit measurement coverage.",
            [*labels, "status"],
            **arguments,
        )
        self.tokens = Counter(
            "sglang:committed_input_tokens",
            "Logical input consumed at prefill, once.",
            [*labels, "source"],
            **arguments,
        )
        self.requests = Counter(
            "sglang:committed_prefill_requests",
            "Completed prefill denominator including no-hit requests.",
            [*labels, "hit"],
            **arguments,
        )
        self.prefix = Histogram(
            "sglang:committed_cache_prefix_tokens",
            "Consumed reusable prefix including zero-hit requests.",
            [*labels, "source"],
            buckets=PREFIX_BUCKETS,
            **arguments,
        )

    def observe(self, prompt, cached, details, status="observed"):
        if status not in ("observed", "aborted", "usage_unavailable"):
            raise ValueError("unknown observation status")
        if prompt is None or cached is None:
            status = "usage_unavailable"
        self.attempts.labels(**self.labels, status=status).inc()
        if status != "observed":
            return
        partition = committed_reuse(prompt, cached, details)
        reused = sum(
            amount for source, amount in partition.items() if source != "computed"
        )
        self.requests.labels(**self.labels, hit="yes").inc(bool(reused))
        self.requests.labels(**self.labels, hit="no").inc(not reused)
        self.prefix.labels(**self.labels, source="all").observe(reused)
        for source, amount in partition.items():
            self.tokens.labels(**self.labels, source=source).inc(amount)
            if source != "computed":
                self.prefix.labels(**self.labels, source=source).observe(amount)


def prefetch_stages(requested, found, returned, published):
    requested = max(0, requested)
    found = min(requested, max(0, found))
    returned = min(found, max(0, returned))
    published = min(returned, max(0, published))
    return {
        "requested": requested,
        "found": found,
        "returned": returned,
        "published": published,
    }
