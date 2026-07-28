"""Primitives for the Mooncake KV transfer ownership barrier.

A *bootstrap room* names one request's KV handoff from a prefill rank to its
decode peers. The barrier keeps a failed request's KV pages allocated until no
transfer work can still touch them, and each side of the transfer tracks that
differently:

* prefill owns the *source* pages and counts local transfer leases --
  :class:`RoomTransferLifetime`, tracked and reclaimed per room by
  :class:`RoomLifetimeRegistry`;
* decode owns the *destination* pages and counts peer acknowledgements --
  :class:`PeerAckBarrier`.
"""

from __future__ import annotations

import itertools
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Soft cap on tracked room lifetimes. Reaching it means the periodic sweep is not
# keeping up, so an insert also does a *bounded* scan for reclaimable rooms.
MAX_TRACKED_ROOMS = 4096
# Entries examined per emergency scan, so an insert can never become O(n).
MAX_EMERGENCY_SCAN = 256


class RoomTransferLifetime:
    """Ownership barrier for one bootstrap room's KV pages on this rank.

    Mooncake transfers hand raw KV pointers to native code (RDMA, the transfer
    executor, CUDA staging copies). Those readers and writers outlive the
    Python call that started them, so a request that becomes terminal while
    they run would let the allocator hand the same pages to another request.

    Every piece of native transfer work holds a *lease*. ``close()`` stops new
    leases from being handed out; the room is *quiesced* once it is closed and
    every outstanding lease has been returned. Only then is it safe to release
    the room's KV pages.

    Abort tokens minted by decode peers are recorded here so that a late or
    duplicated abort for a recycled room cannot close a live room.
    """

    __slots__ = ("_cond", "_leases", "_open", "_abort_tokens", "created_at", "_claimed")

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._leases = 0
        self._open = True
        self._abort_tokens: set = set()
        self.created_at = time.monotonic()
        # True once a local sender has taken responsibility for the room. An
        # unclaimed room was created by decode metadata alone, so nothing will
        # ever release it and the sweep must.
        self._claimed = False

    def try_lease(self) -> bool:
        """Take a lease, or return False if the room no longer admits work."""
        with self._cond:
            if not self._open:
                return False
            self._leases += 1
            return True

    def end_lease(self) -> None:
        with self._cond:
            self._leases -= 1
            if self._leases == 0:
                self._cond.notify_all()

    def close(self) -> None:
        """Stop admitting transfer work (idempotent)."""
        with self._cond:
            self._open = False
            if self._leases == 0:
                self._cond.notify_all()

    def is_closed(self) -> bool:
        with self._cond:
            return not self._open

    def is_quiesced(self) -> bool:
        with self._cond:
            return not self._open and self._leases == 0

    def outstanding_leases(self) -> int:
        with self._cond:
            return self._leases

    def claim(self) -> None:
        with self._cond:
            self._claimed = True

    def is_claimed(self) -> bool:
        with self._cond:
            return self._claimed

    def is_reclaimable(self) -> bool:
        """Whether this room has no active local transfer ownership.

        A quiesced room admits no work and has none running. An unclaimed room
        has no sender *yet*, so callers must additionally preserve it for the
        bootstrap grace period in which a metadata-late sender may still arrive.
        """
        with self._cond:
            return (not self._open and self._leases == 0) or not self._claimed

    def wait_quiesced(self, timeout: Optional[float]) -> bool:
        """Block until quiesced. Returns False if *timeout* elapsed first."""
        with self._cond:
            return self._cond.wait_for(
                lambda: not self._open and self._leases == 0, timeout
            )

    def add_abort_token(self, token: bytes) -> None:
        if not token:
            return
        with self._cond:
            self._abort_tokens.add(token)

    def authorizes_abort(self, token: bytes) -> bool:
        """Whether *token* may close this room.

        A tokenless abort comes from a peer that predates the token protocol and
        is honoured unconditionally, as before.

        A room that has not received any decode metadata yet has no tokens to
        compare against, and accepts the abort. That is deliberate, and it is the
        safe direction of the trade: an abort can legitimately arrive before this
        rank has any metadata for the room (the decode gave up during bootstrap),
        and dropping it would leave this rank free to transfer into pages the
        decode has already released -- the corruption this barrier exists to
        prevent. Accepting instead risks one spurious request failure if a
        recycled bootstrap_room draws a delayed abort from its previous occupant,
        which is availability rather than correctness, and requires a collision in
        a 64-bit space. Distinguishing the two cases would need a generation tag
        on every room-scoped message; that is not worth a wire change here.
        """
        if not token:
            return True
        with self._cond:
            return not self._abort_tokens or token in self._abort_tokens


class RoomLifetimeRegistry:
    """All room lifetimes on a prefill rank, with their garbage collection.

    There is one retirement rule: a room may be forgotten once it is older
    than the bootstrap grace period *and* ``is_reclaimable()``. Younger
    entries are always kept -- that is the window in which a sender may still
    appear and claim the room, and in which a tombstone must keep rejecting a
    late abort. The periodic ``sweep()`` applies the rule to the whole table;
    when the table reaches ``MAX_TRACKED_ROOMS``, an insert applies it to a
    bounded prefix of the oldest entries so insertion can never degrade to
    O(tracked rooms).

    ``on_retire`` drops the caller's remaining room-keyed state and runs under
    ``lock``, in the same step that removes the lifetime. Retirement must be
    atomic: dropping the lifetime first and the rest afterwards would let a
    request that draws the same bootstrap_room in between have its own
    destinations and status deleted, and leaving them behind would let it
    inherit the old decode's addresses.

    ``lock`` is public so a caller can make lifetime validation and its own
    room-keyed writes one atomic step (see ``_handle_bootstrap_metadata``);
    use the ``*_locked`` methods while holding it.
    """

    def __init__(self, sweep_ttl: float, on_retire: Callable[[int], None]) -> None:
        self.lock = threading.Lock()
        self._rooms: OrderedDict[int, RoomTransferLifetime] = OrderedDict()
        self._sweep_ttl = sweep_ttl
        self._on_retire = on_retire

    def get(self, room: int) -> Optional[RoomTransferLifetime]:
        with self.lock:
            return self._rooms.get(room)

    def get_or_create(self, room: int) -> RoomTransferLifetime:
        with self.lock:
            return self.get_or_create_locked(room)

    def get_or_create_locked(self, room: int) -> RoomTransferLifetime:
        lifetime = self._rooms.get(room)
        if lifetime is None:
            # Reclaim before inserting, so the entry being created can never be
            # the one that gets evicted.
            if len(self._rooms) >= MAX_TRACKED_ROOMS:
                if not self._reclaim_locked(scan_limit=MAX_EMERGENCY_SCAN):
                    logger.warning_once(
                        "Tracking more than %d Mooncake bootstrap rooms with "
                        "none old enough and reclaimable; allowing the soft cap "
                        "to grow while metadata-first rooms remain inside their "
                        "bootstrap window.",
                        MAX_TRACKED_ROOMS,
                    )
            lifetime = self._rooms[room] = RoomTransferLifetime()
        return lifetime

    def forget(self, room: int) -> None:
        """Drop only the lifetime; the caller owns the rest of its cleanup."""
        with self.lock:
            self._rooms.pop(room, None)

    def sweep(self) -> int:
        """Reclaim every room that no local sender will ever release."""
        with self.lock:
            reclaimed = self._reclaim_locked(scan_limit=None)
        if reclaimed:
            logger.debug("Reclaimed %d abandoned Mooncake rooms", reclaimed)
        return reclaimed

    def rooms(self) -> List[int]:
        """Snapshot of the tracked rooms, oldest first."""
        with self.lock:
            return list(self._rooms)

    def _reclaim_locked(self, scan_limit: Optional[int]) -> int:
        """Retire rooms past the grace period; the one rule both GC paths use."""
        cutoff = time.monotonic() - self._sweep_ttl
        entries = self._rooms.items()
        if scan_limit is not None:
            # islice over the live view keeps this O(scan_limit); the keys are
            # copied out first because the dict cannot be mutated while iterating.
            entries = itertools.islice(entries, scan_limit)
        stale = [
            room
            for room, lifetime in entries
            if lifetime.created_at <= cutoff and lifetime.is_reclaimable()
        ]
        for room in stale:
            del self._rooms[room]
            self._on_retire(room)
        return len(stale)


class PeerAckBarrier:
    """Which prefill peers may write into one room's KV pages, and which have
    proven they stopped.

    Decode-side counterpart of :class:`RoomTransferLifetime`: that one counts
    local transfer leases on the prefill rank, this one counts peer
    acknowledgements on the decode rank, whose pages are the *destination* of
    the prefill's RDMA writes.

    Each peer gets a one-off token, minted together with its abort target. A
    peer is *exposed* immediately before the metadata naming our pages is sent
    to it, and its write ownership ends only when it echoes that token in an
    ABORT_ACK -- which it sends after draining its transfer work. A tokenless
    ACK comes from a peer that predates the token protocol and acknowledges
    *before* draining; it is recorded, but it is never proof.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._targets: List[Tuple[dict, bytes]] = []
        self._exposed: set = set()
        self._acked: set = set()

    def mint_targets(self, bootstrap_infos) -> List[Tuple[dict, bytes]]:
        """Per-peer abort nonces, minted once and reused across retries.

        One nonce per prefill rank lets an ABORT_ACK identify *which* peer has
        drained, and prevents a stale ACK for a recycled room from satisfying
        this request.
        """
        with self._lock:
            if not self._targets and bootstrap_infos:
                self._targets = [
                    (bootstrap_info, os.urandom(16).hex().encode("ascii"))
                    for bootstrap_info in bootstrap_infos
                ]
            return list(self._targets)

    def expose(self, token: bytes) -> None:
        """Require an ACK from a peer that may now write into our pages."""
        with self._lock:
            self._exposed.add(token)

    def withdraw(self, token: bytes) -> None:
        """Undo an exposure whose atomic metadata send did not complete."""
        with self._lock:
            self._exposed.discard(token)
            self._acked.discard(token)

    def any_exposed(self) -> bool:
        with self._lock:
            return bool(self._exposed)

    def record_ack(self, token: Optional[bytes]) -> bool:
        """Record one ABORT_ACK. Returns True if it was tokenless."""
        with self._lock:
            if not token:
                return True
            if token in self._exposed:
                self._acked.add(token)
            return False

    def unacked_targets(self) -> List[Tuple[dict, bytes]]:
        with self._lock:
            return [
                (info, token)
                for info, token in self._targets
                if token not in self._acked
            ]

    def missing_acks(self) -> Tuple[int, int]:
        """(unacknowledged, total) exposed peers, for diagnostics."""
        with self._lock:
            return len(self._exposed - self._acked), len(self._exposed)

    def quiesced(self) -> bool:
        """Whether every peer that saw our page indices has proven it stopped.

        A tokenless legacy ACK is never counted: that peer acknowledges before
        draining its transfers, so the ACK proves nothing about page ownership.
        """
        with self._lock:
            return self._exposed <= self._acked
