from unittest.mock import MagicMock

import pytest

import bluesky.plan_stubs as bps
from bluesky.preprocessors import (
    contingency_decorator,
    contingency_wrapper,
    lazily_stage_decorator,
    msg_mutator,
    stage_decorator,
    stage_wrapper,
)
from bluesky.protocols import HasHints, HasParent, Movable, Stageable
from bluesky.run_engine import RequestStop, RunEngine


class _StageableDevice:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent

    def stage(self):
        return []

    def unstage(self):
        return []


class _SuccessfulStatus:
    def add_callback(self, callback):
        callback(self)

    def exception(self, timeout=0.0):
        return None

    @property
    def done(self):
        return True

    @property
    def success(self):
        return True


def _collect_messages(plan, *, asynchronous_stage=False):
    messages = []
    response = None
    while True:
        try:
            msg = plan.send(response)
        except StopIteration:
            return messages
        messages.append(msg)
        if asynchronous_stage and msg.command in {"stage", "unstage"}:
            response = _SuccessfulStatus()
        else:
            response = None


def test_given_a_plan_that_raises_contigency_will_call_except_plan_with_exception_and_run_engine_errors():
    expected_exception = Exception()

    def except_plan(exception: Exception):
        assert exception == expected_exception
        yield from bps.null()

    # Mock so we can assert called
    except_plan = MagicMock(side_effect=except_plan)

    @contingency_decorator(except_plan=except_plan)
    def raising_plan():
        yield from bps.null()
        raise expected_exception

    RE = RunEngine()

    with pytest.raises(Exception) as exception:
        RE(raising_plan())
        assert exception == expected_exception

    except_plan.assert_called_once()


def test_given_a_plan_that_raises_contigency_with_no_auto_raise_will_call_except_plan_and_RE_does_not_raise():
    expected_exception = Exception()
    expected_return_value = "test"

    def except_plan(exception: Exception):
        assert exception == expected_exception
        yield from bps.null()
        return expected_return_value

    # Mock so we can assert called
    except_plan = MagicMock(side_effect=except_plan)

    @contingency_decorator(except_plan=except_plan, auto_raise=False)
    def raising_plan():
        yield from bps.null()
        raise expected_exception

    RE = RunEngine(call_returns_result=True)

    returned_value = RE(raising_plan())

    except_plan.assert_called_once()
    assert returned_value.plan_result == expected_return_value


def test_given_a_plan_that_raises_contigency_with_no_auto_raise_and_except_plan_that_reraises_run_engine_errors():
    expected_exception = Exception()

    def except_plan(exception: Exception):
        assert exception == expected_exception
        yield from bps.null()
        raise exception

    # Mock so we can assert called
    except_plan = MagicMock(side_effect=except_plan)

    @contingency_decorator(except_plan=except_plan, auto_raise=False)
    def raising_plan():
        yield from bps.null()
        raise expected_exception

    RE = RunEngine()

    with pytest.raises(Exception) as exception:
        RE(raising_plan())
        assert exception == expected_exception

    except_plan.assert_called_once()


def test_exceptions_through_msg_mutator():
    from bluesky import Msg

    def outer():
        for j in range(50):
            yield Msg(f"step {j}")

    def attach(msg):
        cmd = msg.command
        return msg._replace(command=f"{cmd}+")

    def except_plan(e):
        yield Msg("handle it")

    gen = msg_mutator(contingency_wrapper(outer(), except_plan=except_plan), attach)

    msgs = []

    msgs.append(next(gen))
    msgs.append(next(gen))
    msgs.append(next(gen))
    msgs.append(next(gen))
    msgs.append(gen.throw(RequestStop))
    try:
        while True:
            msgs.append(next(gen))
    except RequestStop:
        pass
    else:
        raise False  # noqa: B016
    assert ["step 0+", "step 1+", "step 2+", "step 3+", "handle it+"] == [m.command for m in msgs]


def test_lazily_stage_decorator():
    class Device(Stageable, HasParent, HasHints, Movable): ...

    device1 = MagicMock(spec=Device)
    device2 = MagicMock(spec=Device)
    device1.name = "device1"
    device2.name = "device2"
    device1.parent = None
    device2.parent = None

    @lazily_stage_decorator()
    def plan():
        yield from bps.mv(device1, 1)
        yield from bps.mv(device2, 2)

    commands = [m.command for m in plan()]
    assert commands == ["stage", "set", "wait", "stage", "set", "wait", "unstage", "unstage"]


def test_lazily_stage_decorator_with_nested_devices():
    class Device(Stageable, HasParent, HasHints, Movable): ...

    root_device = MagicMock(spec=Device)
    root_device.name = "root_device"
    root_device.parent = None
    child1 = MagicMock(spec=Device)
    child1.name = "child1"
    child1.parent = root_device
    child2 = MagicMock(spec=Device)
    child2.name = "child2"
    child2.parent = root_device

    @lazily_stage_decorator()
    def plan():
        yield from bps.mv(child1, 1)
        yield from bps.mv(child2, 2)
        yield from bps.mv(root_device, 3)

    commands = [m.command for m in plan()]
    assert commands == ["stage", "set", "wait", "set", "wait", "set", "wait", "unstage"]


def test_stage_wrapper_legacy_devices_waits_for_async_stage_and_unstage():
    device1 = _StageableDevice("device1")
    device2 = _StageableDevice("device2")

    messages = _collect_messages(
        stage_wrapper(bps.null(), [device1, device2]),
        asynchronous_stage=True,
    )

    assert [msg.command for msg in messages] == [
        "stage",
        "stage",
        "wait",
        "null",
        "unstage",
        "unstage",
        "wait",
    ]
    assert [msg.obj for msg in messages if msg.command == "stage"] == [device1, device2]
    assert [msg.obj for msg in messages if msg.command == "unstage"] == [device2, device1]

    stage_messages = messages[:3]
    unstage_messages = messages[-3:]
    assert stage_messages[0].kwargs["group"] is not None
    assert stage_messages[0].kwargs["group"] == stage_messages[1].kwargs["group"]
    assert stage_messages[1].kwargs["group"] == stage_messages[2].kwargs["group"]
    assert unstage_messages[0].kwargs["group"] is not None
    assert unstage_messages[0].kwargs["group"] == unstage_messages[1].kwargs["group"]
    assert unstage_messages[1].kwargs["group"] == unstage_messages[2].kwargs["group"]
    assert stage_messages[0].kwargs["group"] != unstage_messages[0].kwargs["group"]


def test_grouped_stage_wrapper_preserves_groups_and_explicit_wait_positions():
    device1 = _StageableDevice("device1")
    device2 = _StageableDevice("device2")
    stage_group = ("stage", 1)
    unstage_group = frozenset({"unstage"})

    def body():
        yield from bps.wait(group=stage_group)
        yield from bps.null()

    def enclosing_plan():
        yield from stage_wrapper(
            body(),
            [
                (device1, stage_group, None),
                (device2, None, unstage_group),
            ],
        )
        yield from bps.wait(group=unstage_group)

    messages = _collect_messages(enclosing_plan(), asynchronous_stage=True)

    assert [(msg.command, msg.obj, msg.kwargs.get("group")) for msg in messages] == [
        ("stage", device1, stage_group),
        ("stage", device2, None),
        ("wait", None, stage_group),
        ("null", None, None),
        ("unstage", device2, unstage_group),
        ("unstage", device1, None),
        ("wait", None, unstage_group),
    ]


def test_grouped_stage_decorator():
    device = _StageableDevice("device")

    @stage_decorator([(device, "stage-group", "unstage-group")])
    def plan():
        yield from bps.null()

    messages = _collect_messages(plan())

    assert [(msg.command, msg.kwargs.get("group")) for msg in messages] == [
        ("stage", "stage-group"),
        ("null", None),
        ("unstage", "unstage-group"),
    ]


def test_grouped_stage_wrapper_unstages_when_plan_raises():
    device = _StageableDevice("device")
    expected_exception = RuntimeError("failure in wrapped plan")

    def plan():
        yield from bps.null()
        raise expected_exception

    messages = []
    wrapped_plan = stage_wrapper(plan(), [(device, "stage-group", "unstage-group")])
    with pytest.raises(RuntimeError) as exc_info:
        while True:
            messages.append(next(wrapped_plan))

    assert exc_info.value is expected_exception
    assert [(msg.command, msg.kwargs.get("group")) for msg in messages] == [
        ("stage", "stage-group"),
        ("null", None),
        ("unstage", "unstage-group"),
    ]


def test_grouped_stage_wrapper_normalizes_to_unique_roots_with_first_groups():
    root = _StageableDevice("root")
    child = _StageableDevice("child", parent=root)

    messages = _collect_messages(
        stage_wrapper(
            bps.null(),
            [
                (child, "first-stage", "first-unstage"),
                (root, "ignored-stage", "ignored-unstage"),
            ],
        )
    )

    stage_message, _, unstage_message = messages
    assert (stage_message.command, stage_message.obj, stage_message.kwargs["group"]) == (
        "stage",
        root,
        "first-stage",
    )
    assert (unstage_message.command, unstage_message.obj, unstage_message.kwargs["group"]) == (
        "unstage",
        root,
        "first-unstage",
    )
