"""Tests that PlanRunner is pure asyncio and needs no RunEngine."""

import inspect
import pathlib
import threading

import pytest

import bluesky
from bluesky.plan_runner import PlanRunner


class _RecordingSignal:
    """A Subscribable stand-in: enough for a suspender to be constructed."""

    name = "recording"

    def subscribe_reading(self, function):
        pass

    def clear_sub(self, function):
        pass


THREADING_PRIMITIVES = (
    threading.Event,
    threading.Lock().__class__,
    threading.RLock().__class__,
    threading.Condition,
    threading.Semaphore,
    threading.Barrier,
    threading.Thread,
)


@pytest.mark.parametrize("cls", [PlanRunner])
def test_source_takes_no_locks(cls):
    source = inspect.getsource(cls)
    for forbidden in ("threading.", "_state_lock", ".acquire(", ".join("):
        assert forbidden not in source


def test_a_suspender_holds_no_threading_primitives():
    # Its state is only touched on the loop.
    from bluesky.suspenders import SuspendBoolHigh

    suspender = SuspendBoolHigh(_RecordingSignal())
    offenders = {
        name: type(value).__name__
        for name, value in vars(suspender).items()
        if isinstance(value, THREADING_PRIMITIVES)
    }
    assert offenders == {}


def test_the_runner_never_says_what_to_press():
    source = (pathlib.Path(bluesky.__file__).parent / "plan_runner.py").read_text()

    assert "Ctrl" not in source
