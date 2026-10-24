import gc
import threading
from collections.abc import Callable

import pytest

from bluesky.fsm import _TRANSITIONS, MachineDescriptor, RunEngineState, RunEngineStateMachine


class Owner:
    fsm = MachineDescriptor(RunEngineStateMachine)
    state_hook: Callable[[RunEngineState, RunEngineState], None] | None = None

    def __init__(self) -> None:
        self._state_lock = threading.RLock()


class OwnerNoLock:
    """Raises attribute error because the state machine needs a _state_lock to work."""

    fsm = MachineDescriptor(RunEngineStateMachine)


def test_states():
    assert RunEngineState.states() == [
        "idle",
        "running",
        "pausing",
        "paused",
        "halting",
        "stopping",
        "aborting",
        "suspending",
        "panicked",
    ]


def test_state_values():
    """Each enum member's value matches the expected lowercase string."""
    assert RunEngineState.IDLE.value == "idle"
    assert RunEngineState.RUNNING.value == "running"
    assert RunEngineState.PAUSING.value == "pausing"
    assert RunEngineState.PAUSED.value == "paused"
    assert RunEngineState.HALTING.value == "halting"
    assert RunEngineState.STOPPING.value == "stopping"
    assert RunEngineState.ABORTING.value == "aborting"
    assert RunEngineState.SUSPENDING.value == "suspending"
    assert RunEngineState.PANICKED.value == "panicked"


def test_transitions():

    assert _TRANSITIONS["idle"] == ["running", "panicked"]
    assert _TRANSITIONS["running"] == [
        "idle",
        "pausing",
        "halting",
        "stopping",
        "aborting",
        "suspending",
        "panicked",
    ]
    assert _TRANSITIONS["pausing"] == ["paused", "idle", "halting", "aborting", "panicked"]
    assert _TRANSITIONS["suspending"] == ["running", "halting", "aborting", "panicked"]
    assert _TRANSITIONS["paused"] == ["idle", "running", "halting", "stopping", "aborting", "panicked"]
    assert _TRANSITIONS["halting"] == ["idle", "panicked"]
    assert _TRANSITIONS["stopping"] == ["idle", "panicked"]
    assert _TRANSITIONS["aborting"] == ["idle", "panicked"]
    assert _TRANSITIONS["panicked"] == []


def test_all_states_in_transitions():
    """Every RunEngineState has an entry in _TRANSITIONS."""
    for state in RunEngineState:
        assert state.value in _TRANSITIONS, f"{state.value!r} missing from _TRANSITIONS"


def test_no_lock():
    owner = OwnerNoLock()

    with pytest.raises(AttributeError):
        _ = owner.fsm.is_idle


def test_descriptor_set_calls_state_hook():

    hook_calls: list[tuple[RunEngineState, RunEngineState]] = []

    def hook_cb(new, old):
        return hook_calls.append((new, old))

    owner = Owner()
    owner.state_hook = hook_cb
    owner.fsm = "running"
    assert len(hook_calls) == 1
    new, old = hook_calls[0]
    assert new == RunEngineState.RUNNING
    assert old == RunEngineState.IDLE


def test_weak_reference_cleanup():
    """FSM entry is removed from descriptor memory when owner is GC'd."""
    descriptor = Owner.__dict__["fsm"]  # access via __dict__ to get descriptor directly
    owner = Owner()
    _ = owner.fsm  # force creation
    assert owner in descriptor._memory
    del owner
    gc.collect()
    assert len(descriptor._memory) == 0


@pytest.mark.parametrize(
    "start, end",
    [
        ("idle", "running"),
        ("running", "pausing"),
        ("running", "halting"),
        ("running", "stopping"),
        ("running", "aborting"),
        ("running", "suspending"),
        ("pausing", "paused"),
        ("pausing", "idle"),
        ("paused", "running"),
        ("paused", "idle"),
        ("suspending", "running"),
        ("halting", "idle"),
        ("stopping", "idle"),
        ("aborting", "idle"),
    ],
)
def test_valid_transitions(start, end):
    fsm = RunEngineStateMachine()
    # Fast-forward to the start state
    fsm._state = RunEngineState(start)
    old, new = fsm.set(end)
    assert old == RunEngineState(start)
    assert new == RunEngineState(end)
    assert str(fsm) == end
