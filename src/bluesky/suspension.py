"""A plan's suspension, and the reasons that trip it."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from .utils import Msg

PlanLike = Iterable[Msg] | Callable[[], Iterable[Msg]]
Reasons = Mapping[Hashable, "SuspensionReason"]


@dataclass(frozen=True)
class SuspensionReason:
    """One reason a suspension is tripped, with its pre- and post-plans."""

    justification: str
    pre_plan: PlanLike | None = None
    post_plan: PlanLike | None = None
    # Seconds from its recovery until it is dropped; None until it recovers.
    settle_time: float | None = None


@dataclass(frozen=True)
class SuspensionChange:
    """How the reasons changed during a `Suspension.wait_until`."""

    # Every reason now.
    now: Reasons
    # Tripped since, including a recovering reason tripped again.
    added: Reasons
    # Recovered since, each with its settle time.
    recovered: Reasons

    @classmethod
    def from_reasons(cls, before: Reasons, after: Reasons) -> SuspensionChange:
        """What changed from ``before`` to ``after``."""
        added = {k: r for k, r in after.items() if r.settle_time is None and before.get(k) is not r}
        recovered = {k: r for k, r in after.items() if r.settle_time is not None and before.get(k) is not r}
        # Dropped at once, with no sleep or by a removal, so never seen recovering.
        for k, r in before.items():
            if k not in after and r.settle_time is None:
                recovered[k] = replace(r, settle_time=0)
        return cls(after, MappingProxyType(added), MappingProxyType(recovered))


def join_justifications(reasons: Reasons) -> str:
    """Every reason's justification, one per line, in the order tripped."""
    return "\n".join(reason.justification for reason in reasons.values() if reason.justification)


class Suspension:
    """The reasons a plan may not run, keyed by the suspender that tripped each.

    ``reasons`` is safe on any thread; everything else is loop-only and unchecked.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        # Immutable and swapped whole on every change.
        self._reasons: Reasons = MappingProxyType({})
        # Pending releases of recovered reasons.
        self._releases: dict[Hashable, asyncio.TimerHandle] = {}
        # Set and cleared on every change, to wake waiters.
        self._changed = asyncio.Event()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """The loop this suspension's state lives on."""
        return self._loop

    @property
    def reasons(self) -> Reasons:
        """Every reason, in the order tripped; a new mapping after every change."""
        return self._reasons

    def trip(
        self,
        key: Hashable,
        justification: str,
        *,
        pre_plan: PlanLike | None = None,
        post_plan: PlanLike | None = None,
    ) -> None:
        """Add ``key``'s reason, replacing any it has, recovered or not."""
        self._cancel_release(key)
        self._set_reasons({**self._reasons, key: SuspensionReason(justification, pre_plan, post_plan)})

    def recover(self, key: Hashable, *, after: float = 0) -> None:
        """Drop ``key``'s reason ``after`` seconds from now, marking it with that settle time."""
        reason = self._reasons.get(key)
        if reason is None:
            return
        self._cancel_release(key)
        if after:
            self._releases[key] = self._loop.call_later(after, self._release, key)
            self._set_reasons({**self._reasons, key: replace(reason, settle_time=after)})
        else:
            self._release(key)

    async def wait_until(
        self, predicate: Callable[[SuspensionChange], object], *, since: Reasons | None = None
    ) -> SuspensionChange:
        """Wait until ``predicate`` holds for the change since ``since``, by default now, and return it."""
        before = self.reasons if since is None else since
        while not predicate(change := SuspensionChange.from_reasons(before, self.reasons)):
            await self._changed.wait()
        return change

    def _cancel_release(self, key: Hashable) -> None:
        release = self._releases.pop(key, None)
        if release is not None:
            release.cancel()

    def _release(self, key: Hashable) -> None:
        self._releases.pop(key, None)
        self._set_reasons({k: v for k, v in self._reasons.items() if k != key})

    def _set_reasons(self, reasons: dict[Hashable, SuspensionReason]) -> None:
        self._reasons = MappingProxyType(reasons)
        self._changed.set()
        self._changed.clear()
