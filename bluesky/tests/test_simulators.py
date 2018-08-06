from bluesky.simulators import (print_summary, print_summary_wrapper,
                                summarize_plan,
                                check_limits, LimitsExceeded,
                                plot_raster_path, _TimeStats, est_delta_time,
                                est_time, est_time_per_group, est_time_run,
                                est_time_per_run)
import pytest
from bluesky.plans import grid_scan, scan
from ophyd.telemetry import TelemetryUI
import numpy


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
    from bluesky.plan_tools import (print_summary, print_summary_wrapper,
                                    plot_raster_path)
    with pytest.warns(UserWarning):
        print_summary(scan([det], motor, -1, 1, 10))
    with pytest.warns(UserWarning):
        list(print_summary_wrapper(scan([det], motor, -1, 1, 10)))
    with pytest.warns(UserWarning):
        plan = grid_scan([det], motor1, -5, 5, 10, motor2, -7, 7, 15,
                         True)
        plot_raster_path(plan, 'motor1', 'motor2', probe_size=.3)


def test_check_limits(hw):
    det = hw.det
    motor = hw.motor
    # The motor object does not currently implement limits.
    # Use an assert to help us out if this changes in the future.
    assert not hasattr(motor, 'limits')

    # check_limits should warn if it can't find limits
    with pytest.warns(UserWarning):
        check_limits(scan([det], motor, -1, 1, 3))

    # monkey-patch some limits
    motor.limits = (-2, 2)
    # check_limits should do nothing here
    check_limits(scan([det], motor, -1, 1, 3))

    # check_limits should error if limits are exceeded
    with pytest.raises(LimitsExceeded):
        check_limits(scan([det], motor, -3, 3, 3))

    # check_limits should warn if limits are equal
    motor.limits = (2, 2)
    with pytest.warns(UserWarning):
        check_limits(scan([det], motor, -1, 1, 3))


def test_plot_raster_path(hw):
    det = hw.det
    motor1 = hw.motor1
    motor2 = hw.motor2
    plan = grid_scan([det], motor1, -5, 5, 10, motor2, -7, 7, 15, True)
    plot_raster_path(plan, 'motor1', 'motor2', probe_size=.3)


# set up the telemetry dictionary with some known values.
@pytest.fixture
def setup_telemetry():
    TelemetryUI.telemetry = []  # clear out the telemetry dictionary.
    TelemetryUI.telemetry.extend([{'action': 'set',
                                   'estimation': {'std_dev': float('nan'),
                                                  'time': 3.0},
                                   'object_name': 'motor1',
                                   'position': {'start': 0, 'stop': 3},
                                   'settle_time': {'setpoint': 0},
                                   'time': {'start': 10, 'stop': 12.85},
                                   'velocity': {'setpoint': 1}},
                                  {'action': 'set',
                                   'estimation': {'std_dev': float('nan'),
                                                  'time': 0.95},
                                   'object_name': 'motor1',
                                   'position': {'start': 3, 'stop': 2},
                                   'settle_time': {'setpoint': 0},
                                   'time': {'start': 15, 'stop': 16.05},
                                   'velocity': {'setpoint': 1}}])

    TelemetryUI.telemetry.extend([{'action': 'stage',
                                   'estimation': {'std_dev': float('nan'),
                                                  'time': float('nan')},
                                   'object_name': 'motor1',
                                   'time': {'start': 20, 'stop': 21.05}},
                                  {'action': 'stage',
                                   'estimation': {'std_dev': float('nan'),
                                                  'time': float('nan')},
                                   'object_name': 'motor1',
                                   'time': {'start': 22, 'stop': 22.95}},
                                  {'action': 'unstage',
                                   'estimation': {'std_dev': float('nan'),
                                                  'time': float('nan')},
                                   'object_name': 'motor1',
                                   'time': {'start': 20, 'stop': 21.05}},
                                  {'action': 'unstage',
                                   'estimation': {'std_dev': float('nan'),
                                                  'time': float('nan')},
                                   'object_name': 'motor1',
                                   'time': {'start': 22, 'stop': 22.95}}])

    TelemetryUI.telemetry.extend([{'acquire_period': {'setpoint': 1},
                                   'acquire_time': {'setpoint': 1},
                                   'action': 'trigger',
                                   'estimation': {'std_dev': float('nan'),
                                                  'time': 1.0},
                                   'num_images': {'setpoint': 1},
                                   'object_name': 'det1',
                                   'settle_time': {'setpoint': 0},
                                   'time': {'start': 20, 'stop': 21.05},
                                   'trigger_mode': {'setpoint': 1}},
                                  {'acquire_period': {'setpoint': 1},
                                   'acquire_time': {'setpoint': 1.0},
                                   'action': 'trigger',
                                   'estimation': {'std_dev': float('nan'),
                                                  'time': 1.05},
                                   'num_images': {'setpoint': 1},
                                   'object_name': 'det1',
                                   'settle_time': {'setpoint': 0},
                                   'time': {'start': 22, 'stop': 22.95},
                                   'trigger_mode': {'setpoint': 1}}])

    TelemetryUI.telemetry.extend([{'action': 'stage',
                                   'estimation': {'std_dev': float('nan'),
                                                  'time': float('nan')},
                                   'object_name': 'det1',
                                   'time': {'start': 20, 'stop': 21.05}},
                                 {'action': 'stage',
                                  'estimation': {'std_dev': float('nan'),
                                                 'time': float('nan')},
                                  'object_name': 'det1',
                                  'time': {'start': 22, 'stop': 22.95}},
                                 {'action': 'unstage',
                                  'estimation': {'std_dev': float('nan'),
                                                 'time': float('nan')},
                                  'object_name': 'det1',
                                  'time': {'start': 20, 'stop': 21.05}},
                                 {'action': 'unstage',
                                  'estimation': {'std_dev': float('nan'),
                                                 'time': float('nan')},
                                  'object_name': 'det1',
                                  'time': {'start': 22, 'stop': 22.95}}])


# define some generators for the expected returns from various est_delta_time
# functions
def expected_delta_time():
    return [(_TimeStats(est_time=1.0, std_dev=0.07071067811865576), None),
            (_TimeStats(est_time=1.0, std_dev=0.07071067811865576), None),
            (_TimeStats(est_time=0, std_dev=0), None),
            (_TimeStats(est_time=0, std_dev=0), None),
            (_TimeStats(est_time=0, std_dev=0),
             _TimeStats(est_time=0.0, std_dev=0.0)),
            (_TimeStats(est_time=0, std_dev=0), None),
            (_TimeStats(est_time=0, std_dev=0),
             _TimeStats(est_time=1.0, std_dev=0.07071067811865572)),
            (_TimeStats(est_time=0, std_dev=0),
             _TimeStats(est_time=float(float('nan')), std_dev=float('nan'))),
            (_TimeStats(est_time=1.0, std_dev=1.0), None),
            (_TimeStats(est_time=0, std_dev=0), None),
            (_TimeStats(est_time=0, std_dev=0), None),
            (_TimeStats(est_time=0, std_dev=0), None),
            (_TimeStats(est_time=0, std_dev=0), None),
            (_TimeStats(est_time=0, std_dev=0), None),
            (_TimeStats(est_time=0, std_dev=0),
             _TimeStats(est_time=0.0, std_dev=0.0)),
            (_TimeStats(est_time=0, std_dev=0), None),
            (_TimeStats(est_time=0, std_dev=0),
             _TimeStats(est_time=1.0, std_dev=0.07071067811865572)),
            (_TimeStats(est_time=0, std_dev=0),
             _TimeStats(est_time=float('nan'), std_dev=float('nan'))),
            (_TimeStats(est_time=1.0, std_dev=1.0), None),
            (_TimeStats(est_time=0, std_dev=0), None),
            (_TimeStats(est_time=0, std_dev=0), None),
            (_TimeStats(est_time=0, std_dev=0), None),
            (_TimeStats(est_time=0, std_dev=0), None),
            (_TimeStats(est_time=0, std_dev=0), None),
            (_TimeStats(est_time=1.0, std_dev=0.07071067811865572), None),
            (_TimeStats(est_time=1.0, std_dev=0.07071067811865576), None)]


def expected_time():
    return [(_TimeStats(est_time=1.0, std_dev=0.07071067811865576),
             _TimeStats(est_time=0, std_dev=0)),
            (_TimeStats(est_time=2.0, std_dev=0.10000000000000143),
             _TimeStats(est_time=1.0, std_dev=0.07071067811865576)),
            (_TimeStats(est_time=2.0, std_dev=0.10000000000000143),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000143)),
            (_TimeStats(est_time=2.0, std_dev=0.10000000000000143),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000143)),
            (_TimeStats(est_time=2.0, std_dev=0.10000000000000143),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000143)),
            (_TimeStats(est_time=2.0, std_dev=0.10000000000000143),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000143)),
            (_TimeStats(est_time=3.0, std_dev=0.12247448713916063),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000142)),
            (_TimeStats(est_time=float('nan'), std_dev=float('nan')),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000142)),
            (_TimeStats(est_time=3.0, std_dev=1.0049875621120892),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000142)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=1.1022703842524315),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=float('nan'), std_dev=float('nan')),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=1.4866068747318515),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=2.1000000000000014),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=2.1000000000000014),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=2.1000000000000014),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=2.1000000000000014),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=2.1000000000000014),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=5.0, std_dev=2.101190138945071),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=6.0, std_dev=2.1023796041628655),
             _TimeStats(est_time=5.0, std_dev=2.101190138945071))]


def expected_time_per_group():
    return [(_TimeStats(est_time=1.0, std_dev=0.07071067811865576),
             _TimeStats(est_time=0, std_dev=0)),
            (_TimeStats(est_time=2.0, std_dev=0.10000000000000143),
             _TimeStats(est_time=1.0, std_dev=0.07071067811865576)),
            (_TimeStats(est_time=2.0, std_dev=0.10000000000000143),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000143)),
            (_TimeStats(est_time=2.0, std_dev=0.10000000000000143),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000143)),
            (_TimeStats(est_time=2.0, std_dev=0.10000000000000143),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000143)),
            (_TimeStats(est_time=3.0, std_dev=1.0049875621120892),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000142)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=1.4866068747318515),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=2.1000000000000014),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=2.1000000000000014),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=2.1000000000000014),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=2.1000000000000014),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=2.1000000000000014),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=5.0, std_dev=2.101190138945071),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=6.0, std_dev=2.1023796041628655),
             _TimeStats(est_time=5.0, std_dev=2.101190138945071))]


def expected_time_run():
    return [(_TimeStats(est_time=1.0, std_dev=0.07071067811865576),
             _TimeStats(est_time=0, std_dev=0)),
            (_TimeStats(est_time=2.0, std_dev=0.10000000000000143),
             _TimeStats(est_time=1.0, std_dev=0.07071067811865576)),
            (_TimeStats(est_time=2.0, std_dev=0.10000000000000143),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000143)),
            (_TimeStats(est_time=2.0, std_dev=0.10000000000000143),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000143)),
            (_TimeStats(est_time=2.0, std_dev=0.10000000000000143),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000143)),
            (_TimeStats(est_time=2.0, std_dev=0.10000000000000143),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000143)),
            (_TimeStats(est_time=3.0, std_dev=0.12247448713916063),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000142)),
            (_TimeStats(est_time=float('nan'), std_dev=float('nan')),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000142)),
            (_TimeStats(est_time=3.0, std_dev=1.0049875621120892),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000142)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=3.0, std_dev=1.1000000000000014),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=1.1022703842524315),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=float('nan'), std_dev=float('nan')),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=1.4866068747318515),
             _TimeStats(est_time=3.0, std_dev=1.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=2.1000000000000014),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=2.1000000000000014),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=2.1000000000000014),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=2.1000000000000014),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=4.0, std_dev=2.1000000000000014),
             _TimeStats(est_time=2.0, std_dev=0.1000000000000014)),
            (_TimeStats(est_time=5.0, std_dev=2.101190138945071),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=6.0, std_dev=2.1023796041628655),
             _TimeStats(est_time=5.0, std_dev=2.101190138945071))]


def expected_time_per_run():
    return [(_TimeStats(est_time=1.0, std_dev=0.07071067811865576),
             _TimeStats(est_time=0, std_dev=0)),
            (_TimeStats(est_time=2.0, std_dev=0.10000000000000143),
             _TimeStats(est_time=1.0, std_dev=0.07071067811865576)),
            (_TimeStats(est_time=2.0, std_dev=0.10000000000000143),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000143)),
            (_TimeStats(est_time=4.0, std_dev=2.1000000000000014),
             _TimeStats(est_time=2.0, std_dev=0.10000000000000143)),
            (_TimeStats(est_time=5.0, std_dev=2.101190138945071),
             _TimeStats(est_time=4.0, std_dev=2.1000000000000014)),
            (_TimeStats(est_time=6.0, std_dev=2.1023796041628655),
             _TimeStats(est_time=5.0, std_dev=2.101190138945071))]


def expected_generator(raw_data_function):
    for output in raw_data_function():
        yield output


# define a dictionary that contains expected values for the est_time functions.
est_time_functions = {est_delta_time: expected_delta_time,
                      est_time: expected_time,
                      est_time_per_group: expected_time_per_group,
                      est_time_run: expected_time_run,
                      est_time_per_run: expected_time_per_run}


def test_est_delta_time(setup_telemetry, hw):
    det1 = hw.det1
    motor1 = hw.motor1
    for est_time_gen in est_time_functions:
        plan = scan([det1], motor1, 0, 2, 2)
        raw_data = est_time_functions[est_time_gen]
        for result, expected in zip(est_time_gen(plan),
                                    expected_generator(raw_data)):
            err_string = 'failed test of {}'.format(str(est_time_gen))
            if result[1] is None and expected[0] is None:
                pass
            else:
                numpy.testing.assert_almost_equal(result[1], expected[0],
                                                  decimal=4,
                                                  err_msg=err_string)

            if result[2] is None and expected[1] is None:
                pass
            else:
                numpy.testing.assert_almost_equal(result[2], expected[1],
                                                  decimal=4,
                                                  err_msg=err_string)
