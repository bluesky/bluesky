from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from bluesky import Msg
from bluesky.plans import grid_scan, scan
from bluesky.protocols import Triggerable
from bluesky.simulators import check_limits, plot_raster_path, print_summary, print_summary_wrapper, summarize_plan, \
    RunEngineSimulator
import bluesky.plan_stubs as bps

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
    from bluesky.plan_tools import plot_raster_path, print_summary, print_summary_wrapper

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
    def simple_plan() -> Generator[Msg, object, object]:
        yield from bps.null()

    sim = RunEngineSimulator()
    messages = sim.simulate_plan(simple_plan())
    assert len(messages) == 1
    assert messages[0] == Msg('null')

def test_simulator_add_handler(hw):
    callback = MagicMock()
    def do_sleep():
        yield from bps.sleep("60")

    sim = RunEngineSimulator()
    sim.add_handler(lambda msg: callback(msg), "sleep")
    msgs = sim.simulate_plan(do_sleep())
    callback.assert_called_once()
    msg = callback.mock_calls[0].args[0]
    assert msg.command == "sleep"

def test_simulator_add_read_handler(hw):
    def trigger_and_return_position():
        yield from bps.trigger(hw.ab_det)
        pos = yield from bps.rd(hw.ab_det.a)
        return pos

    sim = RunEngineSimulator()
    sim.add_read_handler("det_a", 5, "det_a")
    msgs = sim.simulate_plan(trigger_and_return_position())
    assert sim.return_value == 5

def test_simulator_add_wait_handler(hw):
    def trigger_and_return_position():
        pos = yield from bps.rd(hw.ab_det.a)
        assert pos == 0
        yield from bps.trigger(hw.ab_det, wait=True)
        pos = yield from bps.rd(hw.ab_det.a)
        return pos

    sim = RunEngineSimulator()
    sim.add_read_handler("det_a", 0, "det_a")
    sim.add_wait_handler(lambda _: sim.add_read_handler("det_a", 5, "det_a"))
    msgs = sim.simulate_plan(trigger_and_return_position())
    assert sim.return_value == 5