from unittest.mock import MagicMock

import pytest

import bluesky.plan_stubs as bps
from bluesky.preprocessors import (
    contingency_decorator,
    contingency_wrapper,
    lazily_stage_decorator,
    msg_mutator,
)
from bluesky.protocols import HasHints, HasParent, Movable, Stageable
from bluesky.run_engine import RequestStop, RunEngine


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
