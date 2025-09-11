import uuid
from collections.abc import Generator
from functools import partial
from math import isclose
from time import time
from typing import Any
from unittest.mock import MagicMock

import pytest

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky import Msg
from bluesky.plans import grid_scan, scan
from bluesky.simulators import (
    END,
    RunEngineSimulator,
    assert_message_and_return_remaining,
    check_limits,
    plot_raster_path,
    print_summary,
    print_summary_wrapper,
    summarize_plan,
)


def test_print_summary(hw):
    det = hw.det
    motor = hw.motor
    print_summary(scan([det], motor, -1, 1, 10))  # old name
    summarize_plan(scan([det], motor, -1, 1, 10))  # new name
    list(print_summary_wrapper(scan([det], motor, -1, 1, 10)))


def test_old_module_name(hw):
    det = hw.det
    motor = hw.motor
    motor1 = hw.motor1
    motor2 = hw.motor2
    from bluesky.plan_tools import (
        plot_raster_path,
        print_summary,
        print_summary_wrapper,
    )

    with pytest.warns(UserWarning):
        print_summary(scan([det], motor, -1, 1, 10))
    with pytest.warns(UserWarning):
        list(print_summary_wrapper(scan([det], motor, -1, 1, 10)))
    with pytest.warns(UserWarning):
        plan = grid_scan([det], motor1, -5, 5, 10, motor2, -7, 7, 15, True)
        plot_raster_path(plan, "motor1", "motor2", probe_size=0.3)


def test_check_limits(RE, hw):
    det = hw.det
    motor = hw.motor
    # The motor object does not currently implement limits.
    # Use an assert to help us out if this changes in the future.
    assert not hasattr(motor, "limits")

    # # check_limits should warn if it can't find check_value
    # TODO: Is there _any_ object to test?
    # with pytest.warns(UserWarning):
    #     check_limits(scan([det], motor, -1, 1, 3))

    # monkey-patch some limits
    motor.limits = (-2, 2)
    # check_limits should do nothing here
    check_limits(scan([det], motor, -1, 1, 3))

    # check_limits should error if limits are exceeded only if object raises
    # this object does not raise
    check_limits(scan([det], motor, -3, 3, 3))

    # check_limits should raise if limits are equal only if object raises
    # this object does not raise
    motor.limits = (2, 2)
    check_limits(scan([det], motor, -1, 1, 3))


def test_check_limits_needs_RE():
    with pytest.raises(RuntimeError) as ctx:
        check_limits([])
    assert str(ctx.value) == "Bluesky event loop not running"


def test_plot_raster_path(hw):
    det = hw.det
    motor1 = hw.motor1
    motor2 = hw.motor2
    plan = grid_scan([det], motor1, -5, 5, 10, motor2, -7, 7, 15, True)
    plot_raster_path(plan, "motor1", "motor2", probe_size=0.3)


def test_simulator_simulates_simple_plan():
    def simple_plan() -> Generator[Msg, Any, Any]:
        yield from bps.null()

    sim = RunEngineSimulator()
    messages = sim.simulate_plan(simple_plan())
    assert len(messages) == 1
    assert messages[0] == Msg("null")


def test_simulator_add_handler(hw):
    callback = MagicMock()

    def do_sleep():
        yield from bps.sleep("60")

    sim = RunEngineSimulator()
    sim.add_handler("sleep", lambda msg: callback(msg))
    msgs = sim.simulate_plan(do_sleep())
    callback.assert_called_once()
    msg = callback.mock_calls[0].args[0]
    assert msg.command == "sleep"
    assert len(msgs) == 1
    assert msgs[0].command == "sleep"


@pytest.mark.parametrize(
    "insert_order, expected",
    [([0, 0, 0], 2), ([END, END, END], 0), ([0, 0, 1], 1), ([0, -1, 2], 1)],
)
def test_simulator_add_handler_append_with_index(hw, insert_order, expected):
    sim = RunEngineSimulator()

    def generate_value(id, msg):
        return {"values": {"value": id}}

    for index, id in zip(insert_order, range(0, len(insert_order))):
        sim.add_handler("read", partial(generate_value, id), None, index)

    def read_det():
        value = yield from bps.rd("det_a")
        return value

    sim.simulate_plan(read_det())
    result = sim.return_value
    assert result == expected


def test_simulator_add_read_handler_for(hw):
    def trigger_and_return_position():
        yield from bps.trigger(hw.ab_det)
        pos = yield from bps.rd(hw.ab_det.a)
        return pos

    sim = RunEngineSimulator()
    sim.add_read_handler_for(hw.ab_det.a, 5)
    msgs = sim.simulate_plan(trigger_and_return_position())
    assert sim.return_value == 5
    msgs = assert_message_and_return_remaining(
        msgs, lambda msg: msg.command == "trigger" and msg.obj.name == "det"
    )
    assert_message_and_return_remaining(msgs, lambda msg: msg.command == "read" and msg.obj.name == "det_a")


def test_simulator_add_read_handler_for_with_dict(hw):
    def trigger_and_return_position():
        yield from bps.trigger(hw.ab_det)
        pos = yield from bps.rd(hw.ab_det.a)
        return pos

    sim = RunEngineSimulator()
    sim.add_read_handler_for(hw.ab_det.a, {"value": 5})
    msgs = sim.simulate_plan(trigger_and_return_position())
    assert sim.return_value == 5
    msgs = assert_message_and_return_remaining(
        msgs, lambda msg: msg.command == "trigger" and msg.obj.name == "det"
    )
    assert_message_and_return_remaining(msgs, lambda msg: msg.command == "read" and msg.obj.name == "det_a")


def test_simulator_add_read_handler_for_multiple(hw):
    def read_all_values():
        data = yield from bps.read(hw.ab_det)
        return data

    sim = RunEngineSimulator()
    sim.add_read_handler_for_multiple(hw.ab_det, a=11, b={"value": 12, "timestamp": 1717071719})
    sim.simulate_plan(read_all_values())
    assert sim.return_value["a"]["value"] == 11
    assert sim.return_value["b"]["value"] == 12
    assert isclose(sim.return_value["a"]["timestamp"], time(), abs_tol=1)
    assert sim.return_value["b"]["timestamp"] == 1717071719


def test_simulator_add_wait_handler(hw):
    def trigger_and_return_position():
        pos = yield from bps.rd(hw.ab_det.a)
        assert pos == 0
        yield from bps.trigger(hw.ab_det, wait=True)
        pos = yield from bps.rd(hw.ab_det.a)
        return pos

    sim = RunEngineSimulator()
    sim.add_read_handler_for(hw.ab_det.a, 0)
    sim.add_wait_handler(lambda _: sim.add_read_handler_for(hw.ab_det.a, 5))
    msgs = sim.simulate_plan(trigger_and_return_position())
    assert sim.return_value == 5
    msgs = assert_message_and_return_remaining(msgs, lambda msg: msg.command == "read" and msg.obj.name == "det_a")
    msgs = assert_message_and_return_remaining(
        msgs, lambda msg: msg.command == "trigger" and msg.obj.name == "det"
    )
    msgs = assert_message_and_return_remaining(msgs, lambda msg: msg.command == "wait")
    assert_message_and_return_remaining(msgs, lambda msg: msg.command == "read" and msg.obj.name == "det_a")


def test_fire_callback(hw):
    callback = MagicMock()
    run_start_uid = uuid.uuid4()
    descriptor_uid = uuid.uuid4()
    event_uid = uuid.uuid4()

    def take_a_reading():
        yield from bps.subscribe("all", callback)
        yield from bp.count([hw.det], num=1)

    start_doc = {"plan_name": "count", "uid": run_start_uid}
    descriptor_doc = {"run_start": run_start_uid, "uid": descriptor_uid}
    event_doc = {"descriptor": descriptor_uid, "uid": event_uid}

    sim = RunEngineSimulator()
    sim.add_handler_for_callback_subscribes()

    sim.add_callback_handler_for(
        "open_run",
        "start",
        start_doc,
        lambda msg: msg.kwargs["plan_name"] == "count",
    )
    sim.add_callback_handler_for_multiple("save", [[("descriptor", descriptor_doc), ("event", event_doc)]])

    sim.simulate_plan(take_a_reading())
    calls = callback.mock_calls
    assert len(calls) == 3
    assert calls[0].args == ("start", start_doc)
    assert calls[1].args == ("descriptor", descriptor_doc)
    assert calls[2].args == ("event", event_doc)


def test_assert_message_and_return_remaining(hw):
    def take_three_readings():
        yield from bp.count([hw.det], num=3)

    sim = RunEngineSimulator()
    msgs = sim.simulate_plan(take_three_readings())
    msgs = assert_message_and_return_remaining(msgs, lambda msg: msg.command == "stage" and msg.obj.name == "det")
    msgs = assert_message_and_return_remaining(
        msgs,
        lambda msg: msg.command == "open_run"
        and msg.kwargs["plan_name"] == "count"
        and msg.kwargs["num_points"] == 3,
    )
    for _ in range(0, 3):
        msgs = assert_message_and_return_remaining(msgs, lambda msg: msg.command == "checkpoint")
        msgs = assert_message_and_return_remaining(
            msgs, lambda msg: msg.command == "trigger" and msg.obj.name == "det"
        )
        trigger_group = msgs[0].kwargs["group"]
        msgs = assert_message_and_return_remaining(
            msgs,
            lambda msg: msg.command == "wait" and msg.kwargs["group"] == trigger_group,  # noqa: B023
        )
        msgs = assert_message_and_return_remaining(msgs, lambda msg: msg.command == "create")
        msgs = assert_message_and_return_remaining(
            msgs, lambda msg: msg.command == "read" and msg.obj.name == "det"
        )
        msgs = assert_message_and_return_remaining(msgs, lambda msg: msg.command == "save")
    assert msgs[1].command == "close_run"
