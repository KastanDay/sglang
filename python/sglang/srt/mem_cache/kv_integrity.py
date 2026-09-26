import collections
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_VALID_MODES = {"off", "host", "device"}


@dataclass
class IntegrityViolation:
    req: Any
    bad_pages: list[int]


class _NullTracker:

    enabled = False

    def on_alloc(self, req_pool_idx: int, slot_indices: Any) -> None:
        return None

    def on_prefix_hit(self, req_pool_idx: int, slot_indices: Any) -> None:
        return None

    def on_free(self, slot_indices: Any) -> None:
        return None

    def on_req_free(self, req_pool_idx: Optional[int]) -> None:
        return None

    def validate_batch(self, batch: Any) -> Sequence[Any]:
        return []


class KvIntegrityTracker:
    """Host-side per-page request-ownership tracker. See
    `docs/superpowers/specs/2026-04-30-sglang-kv-integrity-tagging-design.md`.
    """

    enabled = True

    def __init__(self, num_pages: int, page_size: int, req_pool_size: int):
        self.num_pages = num_pages
        self.page_size = page_size
        self.req_pool_size = req_pool_size
        self.owner_words = math.ceil((req_pool_size + 1) / 64)
        self.page_owners = np.zeros((num_pages, self.owner_words), dtype=np.uint64)
        self.req_pages: dict[int, set[int]] = {}
        self.transition_log: collections.deque = collections.deque(maxlen=10_000)

    def _pages_from_slots(self, slot_indices: Any) -> np.ndarray:
        if hasattr(slot_indices, "detach"):
            slot_indices = slot_indices.detach().cpu().numpy()
        slots = np.asarray(slot_indices, dtype=np.int64).ravel()
        if slots.size == 0:
            return np.empty(0, dtype=np.int64)
        pages = np.unique(slots // self.page_size)
        in_range = (pages >= 0) & (pages < self.num_pages)
        if not in_range.all():
            bad = pages[~in_range].tolist()
            logger.warning(
                "KV integrity: dropped out-of-range pages %s (num_pages=%d) "
                "from slot tracker input — caller passed invalid slot indices",
                bad,
                self.num_pages,
            )
            pages = pages[in_range]
        return pages

    def _bit_for(self, req_pool_idx: int) -> tuple[int, np.uint64]:
        word_idx = req_pool_idx // 64
        bit = np.uint64(1) << np.uint64(req_pool_idx % 64)
        return word_idx, bit

    def _record(self, kind: str, page: int, req_pool_idx: Optional[int]) -> None:
        self.transition_log.append((kind, page, req_pool_idx))

    def on_alloc(self, req_pool_idx: int, slot_indices: Any) -> None:
        pages = self._pages_from_slots(slot_indices)
        if pages.size == 0:
            return
        word_idx, bit = self._bit_for(req_pool_idx)
        self.page_owners[pages, word_idx] |= bit
        bucket = self.req_pages.setdefault(req_pool_idx, set())
        for p in pages.tolist():
            bucket.add(p)
            self._record("ALLOC", p, req_pool_idx)

    def on_free(self, slot_indices: Any) -> None:
        pages = self._pages_from_slots(slot_indices)
        if pages.size == 0:
            return
        self.page_owners[pages, :] = 0
        for p in pages.tolist():
            self._record("FREE", p, None)

    def on_prefix_hit(self, req_pool_idx: int, slot_indices: Any) -> None:
        pages = self._pages_from_slots(slot_indices)
        if pages.size == 0:
            return
        word_idx, bit = self._bit_for(req_pool_idx)
        self.page_owners[pages, word_idx] |= bit
        bucket = self.req_pages.setdefault(req_pool_idx, set())
        for p in pages.tolist():
            bucket.add(p)
            self._record("PREFIX_HIT", p, req_pool_idx)

    def on_req_free(self, req_pool_idx: Optional[int]) -> None:
        if req_pool_idx is None:
            return
        pages = self.req_pages.pop(req_pool_idx, None)
        if not pages:
            return
        page_arr = np.fromiter(pages, dtype=np.int64, count=len(pages))
        word_idx, bit = self._bit_for(req_pool_idx)
        self.page_owners[page_arr, word_idx] &= ~bit
        for p in pages:
            self._record("REQ_FREE", p, req_pool_idx)

    def validate_batch(self, batch: Any) -> list[IntegrityViolation]:
        """Check each request's *current* read range against page_owners.

        Reads ``batch.req_to_token_pool.req_to_token[idx, :seq_len]`` — the
        actual slot list the forward will read — and verifies every page
        derived from those slots is authorized for ``idx``. Pages no longer
        referenced by req_to_token_pool (spec-decode rejection, tree-cache
        eviction of inactive prefix nodes, decode-offload move-to-CPU) are
        ignored, so legitimate frees do not produce false positives.

        See spec section "Update 2026-04-30: Option A revision" for the
        production data that motivated this semantic.
        """
        violations: list[IntegrityViolation] = []
        if batch is None:
            return violations
        pool = getattr(batch, "req_to_token_pool", None)
        if pool is None:
            return violations
        req_to_token = getattr(pool, "req_to_token", None)
        if req_to_token is None:
            return violations
        seq_lens_cpu = getattr(batch, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            return violations
        seq_lens = (
            seq_lens_cpu.tolist()
            if hasattr(seq_lens_cpu, "tolist")
            else list(seq_lens_cpu)
        )
        reqs = list(getattr(batch, "reqs", ()))
        if len(seq_lens) != len(reqs):
            return violations
        for req, seq_len in zip(reqs, seq_lens):
            idx = getattr(req, "req_pool_idx", None)
            if idx is None or seq_len <= 0:
                continue
            slots = req_to_token[idx, :seq_len]
            if hasattr(slots, "detach"):
                slots = slots.detach().cpu().numpy()
            slots = np.asarray(slots, dtype=np.int64).ravel()
            if slots.size == 0:
                continue
            pages = np.unique(slots // self.page_size)
            in_range = (pages >= 0) & (pages < self.num_pages)
            pages = pages[in_range]
            if pages.size == 0:
                continue
            word_idx, bit = self._bit_for(idx)
            authorized = (self.page_owners[pages, word_idx] & bit) != 0
            if authorized.all():
                continue
            bad_pages = pages[~authorized].tolist()
            self._log_violation(req, idx, bad_pages)
            violations.append(IntegrityViolation(req=req, bad_pages=bad_pages))
        return violations

    def _log_violation(self, req: Any, req_pool_idx: int, bad_pages: list[int]) -> None:
        rid = getattr(req, "rid", "?")
        n = len(bad_pages)
        head = bad_pages[:32]
        suffix = "" if n <= 32 else f" ... +{n - 32} more"
        relevant = [entry for entry in self.transition_log if entry[1] in set(head)]
        logger.warning(
            "KV integrity violation: rid=%s req_pool_idx=%d bad_pages=%s%s "
            "recent_transitions=%s",
            rid,
            req_pool_idx,
            head,
            suffix,
            relevant[-32:],
        )


def record_alloc_per_req(
    tracker,
    req_pool_indices,
    lens_per_req,
    out_cache_loc,
) -> None:
    """Distribute a flat ``out_cache_loc`` across requests by per-req length.

    Calls ``tracker.on_alloc(req_pool_idx, slots)`` for each non-empty req.

    Note: this records allocations only. Validation (catching reads of pages
    not authorized for the requesting req) runs at the start of the next
    ``ScheduleBatch.prepare_for_extend`` or ``prepare_for_decode``. For
    spec-decode call sites, this means a violation is detected one
    scheduler step after it occurs, not in the same step. This lag is
    acceptable because:

    * The bitmap is already correct at the time of the next forward.
    * Aborting mid-spec-decode-verify would corrupt the speculation pipeline
      worse than letting the bad output complete.
    """
    if not getattr(tracker, "enabled", False):
        return
    if hasattr(out_cache_loc, "detach"):
        slots_host = out_cache_loc.detach().cpu().numpy()
    else:
        slots_host = np.asarray(out_cache_loc)
    if hasattr(req_pool_indices, "tolist"):
        indices = req_pool_indices.tolist()
    else:
        indices = list(req_pool_indices)
    if hasattr(lens_per_req, "tolist"):
        lens = lens_per_req.tolist()
    else:
        lens = list(lens_per_req)
    offset = 0
    for idx, n in zip(indices, lens):
        n = int(n)
        if n <= 0:
            continue
        tracker.on_alloc(idx, slots_host[offset : offset + n])
        offset += n


def should_warn_unsupported_spec_decode_shape(
    tracker_enabled: bool,
    speculative_eagle_topk,
    page_size: int,
) -> bool:
    """Whether to warn that the eagle_worker.py topk>1 + page_size>1 verify-phase
    shape (post-alloc cache_loc duplication) is not modeled by this tracker.

    Tolerates ``speculative_eagle_topk`` being ``None`` or missing — when
    spec-decode is disabled, the server_args attribute may be set to ``None``
    rather than absent, so ``getattr(args, name, 1)`` is not enough on its own.
    """
    topk = speculative_eagle_topk or 1
    return bool(tracker_enabled) and topk > 1 and page_size > 1


def make_tracker(num_pages: int, page_size: int, req_pool_size: int):
    mode = os.environ.get("SGLANG_KV_INTEGRITY", "off").lower()
    if mode == "off":
        return _NullTracker()
    if mode == "host":
        return KvIntegrityTracker(
            num_pages=num_pages,
            page_size=page_size,
            req_pool_size=req_pool_size,
        )
    if mode in _VALID_MODES:
        logger.warning(
            "SGLANG_KV_INTEGRITY=%s is reserved but not implemented; falling back to off.",
            mode,
        )
        return _NullTracker()
    logger.warning(
        "SGLANG_KV_INTEGRITY=%s is not a recognized mode (expected off|host|device); "
        "falling back to off.",
        mode,
    )
    return _NullTracker()
