"""Room lifetime tracking for the Mooncake KV transfer ownership barrier.

A *bootstrap room* names one request's KV handoff from a prefill rank to its
decode peers. :class:`RoomTransferLifetime` is the ownership barrier for one
room's KV pages; :class:`RoomLifetimeRegistry` tracks every lifetime on a
prefill rank and reclaims the ones no local sender will ever release.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections import OrderedDict
from typing import Callable, List, Optional

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
