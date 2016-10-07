import warnings
import pytest
from bluesky.callbacks import collector, CallbackCounter
from bluesky.plans import (AbsListScanPlan, AbsScanPlan, LogAbsScanPlan,
                           DeltaListScanPlan, DeltaScanPlan, LogDeltaScanPlan,
                           AdaptiveAbsScanPlan, AdaptiveDeltaScanPlan, Count,
                           OuterProductAbsScanPlan, InnerProductAbsScanPlan,
                           OuterProductDeltaScanPlan,
                           InnerProductDeltaScanPlan, SpiralScan,
                           SpiralFermatScan, RelativeSpiralScan,
                           RelativeSpiralFermatScan)

from bluesky import Msg
from bluesky.examples import motor, det, SynGauss, motor1, motor2, Mover
from bluesky.tests.utils import setup_test_run_engine
import asyncio
import time as ttime
import numpy as np
import numpy.testing

loop = asyncio.get_event_loop()

RE = setup_test_run_engine()


def traj_checker(scan, expected_traj):
    actual_traj = []
    callback = collector('motor', actual_traj)
    RE(scan, subs={'event': callback})
    assert actual_traj == list(expected_traj)


def multi_traj_checker(scan, expected_data):
    actual_data = []

    def collect_data(name, event):
        actual_data.append(event['data'])

    RE(scan, subs={'event': collect_data})
    assert actual_data == expected_data


def approx_multi_traj_checker(RE, scan, expected_data, *, decimal=2):
    actual_data = []

    def collect_data(name, event):
        actual_data.append(event['data'])

    RE(scan, subs={'event': collect_data})
    keys = sorted(expected_data[0].keys())
    actual_values = [[row[key] for key in keys]
                     for row in actual_data]
    expected_values = [[row[key] for key in keys]
                       for row in expected_data]
    err_msg = 'Trajectory differs. Data keys: {}'.format(', '.join(keys))
    numpy.testing.assert_almost_equal(actual_values, expected_values,
                                      decimal=decimal,
                                      err_msg=err_msg)


def test_outer_product_ascan():
    motor.set(0)
    scan = OuterProductAbsScanPlan([det], motor1, 1, 3, 3, motor2, 10, 20, 2,
                                   False)
    # Note: motor1 is the first motor specified, and so it is the "slow"
    # axis, matching the numpy convention.
    expected_data = [
        {'motor2': 10.0, 'det': 1.0, 'motor1': 1.0},
        {'motor2': 20.0, 'det': 1.0, 'motor1': 1.0},
        {'motor2': 10.0, 'det': 1.0, 'motor1': 2.0},
        {'motor2': 20.0, 'det': 1.0, 'motor1': 2.0},
        {'motor2': 10.0, 'det': 1.0, 'motor1': 3.0},
        {'motor2': 20.0, 'det': 1.0, 'motor1': 3.0}]
    for d in expected_data:
        d.update({'motor1_setpoint': d['motor1']})
        d.update({'motor2_setpoint': d['motor2']})
    multi_traj_checker(scan, expected_data)


def test_outer_product_ascan_snaked():
    motor.set(0)
    scan = OuterProductAbsScanPlan([det], motor1, 1, 3, 3, motor2, 10, 20, 2, True)
    # Note: motor1 is the first motor specified, and so it is the "slow"
    # axis, matching the numpy convention.
    expected_data = [
        {'motor2': 10.0, 'det': 1.0, 'motor1': 1.0},
        {'motor2': 20.0, 'det': 1.0, 'motor1': 1.0},
        {'motor2': 20.0, 'det': 1.0, 'motor1': 2.0},
        {'motor2': 10.0, 'det': 1.0, 'motor1': 2.0},
        {'motor2': 10.0, 'det': 1.0, 'motor1': 3.0},
        {'motor2': 20.0, 'det': 1.0, 'motor1': 3.0}]
    for d in expected_data:
        d.update({'motor1_setpoint': d['motor1']})
        d.update({'motor2_setpoint': d['motor2']})
    multi_traj_checker(scan, expected_data)


def test_inner_product_ascan():
    motor.set(0)
    scan = InnerProductAbsScanPlan([det], 3, motor1, 1, 3, motor2, 10, 30)
    # Note: motor1 is the first motor specified, and so it is the "slow"
    # axis, matching the numpy convention.
    expected_data = [
        {'motor2': 10.0, 'det': 1.0, 'motor1': 1.0},
        {'motor2': 20.0, 'det': 1.0, 'motor1': 2.0},
        {'motor2': 30.0, 'det': 1.0, 'motor1': 3.0}]
    for d in expected_data:
        d.update({'motor1_setpoint': d['motor1']})
        d.update({'motor2_setpoint': d['motor2']})
    multi_traj_checker(scan, expected_data)


def test_outer_product_dscan():
    scan = OuterProductDeltaScanPlan([det], motor1, 1, 3, 3, motor2, 10, 20, 2,
                                 False)
    # Note: motor1 is the first motor specified, and so it is the "slow"
    # axis, matching the numpy convention.
    motor.set(0)
    motor1.set(5)
    motor2.set(8)
    expected_data = [
        {'motor2': 18.0, 'det': 1.0, 'motor1': 6.0},
        {'motor2': 28.0, 'det': 1.0, 'motor1': 6.0},
        {'motor2': 18.0, 'det': 1.0, 'motor1': 7.0},
        {'motor2': 28.0, 'det': 1.0, 'motor1': 7.0},
        {'motor2': 18.0, 'det': 1.0, 'motor1': 8.0},
        {'motor2': 28.0, 'det': 1.0, 'motor1': 8.0}]
    for d in expected_data:
        d.update({'motor1_setpoint': d['motor1']})
        d.update({'motor2_setpoint': d['motor2']})
    multi_traj_checker(scan, expected_data)


def test_outer_product_dscan_snaked():
    scan = OuterProductDeltaScanPlan([det], motor1, 1, 3, 3, motor2, 10, 20, 2,
                                 True)
    # Note: motor1 is the first motor specified, and so it is the "slow"
    # axis, matching the numpy convention.
    motor.set(0)
    motor1.set(5)
    motor2.set(8)
    expected_data = [
        {'motor2': 18.0, 'det': 1.0, 'motor1': 6.0},
        {'motor2': 28.0, 'det': 1.0, 'motor1': 6.0},
        {'motor2': 28.0, 'det': 1.0, 'motor1': 7.0},
        {'motor2': 18.0, 'det': 1.0, 'motor1': 7.0},
        {'motor2': 18.0, 'det': 1.0, 'motor1': 8.0},
        {'motor2': 28.0, 'det': 1.0, 'motor1': 8.0}]
    for d in expected_data:
        d.update({'motor1_setpoint': d['motor1']})
        d.update({'motor2_setpoint': d['motor2']})
    multi_traj_checker(scan, expected_data)


def test_inner_product_dscan():
    motor.set(0)
    motor1.set(5)
    motor2.set(8)
    scan = InnerProductDeltaScanPlan([det], 3, motor1, 1, 3, motor2, 10, 30)
    # Note: motor1 is the first motor specified, and so it is the "slow"
    # axis, matching the numpy convention.
    expected_data = [
        {'motor2': 18.0, 'det': 1.0, 'motor1': 6.0},
        {'motor2': 28.0, 'det': 1.0, 'motor1': 7.0},
        {'motor2': 38.0, 'det': 1.0, 'motor1': 8.0}]
    for d in expected_data:
        d.update({'motor1_setpoint': d['motor1']})
        d.update({'motor2_setpoint': d['motor2']})
    multi_traj_checker(scan, expected_data)


def test_ascan():
    traj = [1, 2, 3]
    scan = AbsListScanPlan([det], motor, traj)
    traj_checker(scan, traj)


def test_dscan():
    traj = np.array([1, 2, 3])
    motor.set(-4)
    print(motor.read())
    scan = DeltaListScanPlan([det], motor, traj)
    traj_checker(scan, traj - 4)


def test_dscan_list_input():
    # GH225
    traj = [1, 2, 3]
    motor.set(-4)
    scan = DeltaListScanPlan([det], motor, traj)
    traj_checker(scan, np.array(traj) - 4)


def test_lin_ascan():
    traj = np.linspace(0, 10, 5)
    scan = AbsScanPlan([det], motor, 0, 10, 5)
    traj_checker(scan, traj)


def test_log_ascan():
    traj = np.logspace(0, 10, 5)
    scan = LogAbsScanPlan([det], motor, 0, 10, 5)
    traj_checker(scan, traj)


def test_lin_dscan():
    traj = np.linspace(0, 10, 5) + 6
    motor.set(6)
    scan = DeltaScanPlan([det], motor, 0, 10, 5)
    traj_checker(scan, traj)


def test_log_dscan():
    traj = np.logspace(0, 10, 5) + 6
    motor.set(6)
    scan = LogDeltaScanPlan([det], motor, 0, 10, 5)
    traj_checker(scan, traj)


def test_adaptive_ascan():
    scan1 = AdaptiveAbsScanPlan([det], 'det', motor, 0, 5, 0.1, 1, 0.1, True)
    scan2 = AdaptiveAbsScanPlan([det], 'det', motor, 0, 5, 0.1, 1, 0.2, True)
    scan3 = AdaptiveAbsScanPlan([det], 'det', motor, 0, 5, 0.1, 1, 0.1, False)

    actual_traj = []
    col = collector('motor', actual_traj)
    counter1 = CallbackCounter()
    counter2 = CallbackCounter()

    RE(scan1, subs={'event': [col, counter1]})
    RE(scan2, subs={'event': counter2})
    assert counter1.value > counter2.value
    assert actual_traj[0] == 0

    actual_traj = []
    col = collector('motor', actual_traj)
    RE(scan3, {'event': col})
    monotonic_increasing = np.all(np.diff(actual_traj) > 0)
    assert monotonic_increasing


def test_adaptive_dscan():
    scan1 = AdaptiveDeltaScanPlan([det], 'det', motor, 0, 5, 0.1, 1, 0.1, True)
    scan2 = AdaptiveDeltaScanPlan([det], 'det', motor, 0, 5, 0.1, 1, 0.2, True)
    scan3 = AdaptiveDeltaScanPlan([det], 'det', motor, 0, 5, 0.1, 1, 0.1, False)

    actual_traj = []
    col = collector('motor', actual_traj)
    counter1 = CallbackCounter()
    counter2 = CallbackCounter()

    motor.set(1)
    RE(scan1, subs={'event': [col, counter1]})
    RE(scan2, subs={'event': counter2})
    assert counter1.value > counter2.value
    assert actual_traj[0] == 1

    actual_traj = []
    col = collector('motor', actual_traj)
    RE(scan3, {'event': col})
    monotonic_increasing = np.all(np.diff(actual_traj) > 0)
    assert monotonic_increasing


def test_count():
    actual_intensity = []
    col = collector('det', actual_intensity)
    motor.set(0)
    scan = Count([det])
    RE(scan, subs={'event': col})
    assert actual_intensity[0] == 1.

    # multiple counts, via updating attribute
    actual_intensity = []
    col = collector('det', actual_intensity)
    scan = Count([det], num=3, delay=0.05)
    RE(scan, subs={'event': col})
    assert scan.num == 3
    assert actual_intensity == [1., 1., 1.]

    # multiple counts, via passing arts to __call__
    actual_intensity = []
    col = collector('det', actual_intensity)
    scan = Count([det], num=3, delay=0.05)
    RE(scan(num=2), subs={'event': col})
    assert actual_intensity == [1., 1.]
    # attribute should still be 3
    assert scan.num == 3
    actual_intensity = []
    col = collector('det', actual_intensity)
    RE(scan, subs={'event': col})
    assert actual_intensity == [1., 1., 1.]


def test_set():
    scan = AbsScanPlan([det], motor, 1, 5, 3)
    assert scan.start == 1
    assert scan.stop == 5
    assert scan.num == 3
    scan.set(start=2)
    assert scan.start == 2
    scan.set(num=4)
    assert scan.num == 4
    assert scan.start == 2


def test_wait_for():
    ev = asyncio.Event(loop=loop)

    def done():
        ev.set()
    scan = [Msg('wait_for', None, [ev.wait(), ]), ]
    loop.call_later(2, done)
    start = ttime.time()
    RE(scan)
    stop = ttime.time()
    assert stop - start >= 2


def test_pre_run_post_run():
    c = Count([])

    def f(x):
        yield Msg('HEY', None)
    c.pre_run = f
    list(c)[0].command == 'HEY'

    c = Count([])

    def f(x):
        yield Msg('HEY', None)
    c.pre_run = f
    list(c)[-1].command == 'HEY'


def _get_spiral_data(start_x, start_y):
    return [{'motor2': start_y + 0.000, 'det': 1.0, 'motor1': start_x + 0.100},
            {'motor2': start_y + 0.000, 'det': 1.0, 'motor1': start_x + 0.200},
            {'motor2': start_y + 0.000, 'det': 1.0, 'motor1': start_x - 0.200},
            {'motor2': start_y + 0.000, 'det': 1.0, 'motor1': start_x + 0.300},
            {'motor2': start_y + 0.260, 'det': 1.0, 'motor1': start_x - 0.150},
            {'motor2': start_y - 0.260, 'det': 1.0, 'motor1': start_x - 0.150},
            {'motor2': start_y + 0.000, 'det': 1.0, 'motor1': start_x + 0.400},
            {'motor2': start_y + 0.400, 'det': 1.0, 'motor1': start_x + 0.000},
            {'motor2': start_y + 0.000, 'det': 1.0, 'motor1': start_x - 0.400},
            {'motor2': start_y - 0.400, 'det': 1.0, 'motor1': start_x - 0.000},
            {'motor2': start_y + 0.000, 'det': 1.0, 'motor1': start_x + 0.500},
            {'motor2': start_y + 0.476, 'det': 1.0, 'motor1': start_x + 0.155},
            {'motor2': start_y + 0.294, 'det': 1.0, 'motor1': start_x - 0.405},
            {'motor2': start_y - 0.294, 'det': 1.0, 'motor1': start_x - 0.405},
            {'motor2': start_y - 0.476, 'det': 1.0, 'motor1': start_x + 0.155},
            ]


def test_absolute_spiral(fresh_RE):
    motor = Mover('motor', {'motor': lambda x: x}, {'x': 0})
    motor1 = Mover('motor1', {'motor1': lambda x: x}, {'x': 0})
    motor2 = Mover('motor2', {'motor2': lambda x: x}, {'x': 0})

    det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)
    motor1.set(1.0)
    motor2.set(1.0)
    scan = SpiralScan([det], motor1, motor2, 0.0, 0.0, 1.0, 1.0, 0.1, 1.0, 0.0)
    approx_multi_traj_checker(fresh_RE, scan, _get_spiral_data(0.0, 0.0),
                              decimal=2)

    scan = SpiralScan([det], motor1, motor2, 0.5, 0.5, 1.0, 1.0, 0.1, 1.0, 0.0)
    approx_multi_traj_checker(fresh_RE, scan, _get_spiral_data(0.5, 0.5),
                              decimal=2)


def test_relative_spiral(fresh_RE):
    motor = Mover('motor', {'motor': lambda x: x}, {'x': 0})
    motor1 = Mover('motor1', {'motor1': lambda x: x}, {'x': 0})
    motor2 = Mover('motor2', {'motor2': lambda x: x}, {'x': 0})
    det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)

    start_x = 1.0
    start_y = 1.0

    motor1.set(start_x)
    motor2.set(start_y)
    scan = RelativeSpiralScan([det], motor1, motor2, 1.0, 1.0, 0.1, 1.0, 0.0)

    approx_multi_traj_checker(fresh_RE, scan,
                              _get_spiral_data(start_x, start_y),
                              decimal=2)


def _get_fermat_data(x_start, y_start):
    return [
        {'motor2': y_start + 0.068, 'det': 1.0, 'motor1': x_start - 0.074},
        {'motor2': y_start - 0.141, 'det': 1.0, 'motor1': x_start + 0.012},
        {'motor2': y_start + 0.137, 'det': 1.0, 'motor1': x_start + 0.105},
        {'motor2': y_start - 0.035, 'det': 1.0, 'motor1': x_start - 0.197},
        {'motor2': y_start - 0.120, 'det': 1.0, 'motor1': x_start + 0.189},
        {'motor2': y_start + 0.237, 'det': 1.0, 'motor1': x_start - 0.064},
        {'motor2': y_start - 0.235, 'det': 1.0, 'motor1': x_start - 0.122},
        {'motor2': y_start + 0.097, 'det': 1.0, 'motor1': x_start + 0.266},
        {'motor2': y_start + 0.114, 'det': 1.0, 'motor1': x_start - 0.277},
        {'motor2': y_start - 0.286, 'det': 1.0, 'motor1': x_start + 0.134},
        {'motor2': y_start + 0.316, 'det': 1.0, 'motor1': x_start + 0.099},
        {'motor2': y_start - 0.174, 'det': 1.0, 'motor1': x_start - 0.300},
        {'motor2': y_start - 0.077, 'det': 1.0, 'motor1': x_start + 0.352},
        {'motor2': y_start + 0.306, 'det': 1.0, 'motor1': x_start - 0.215},
        {'motor2': y_start - 0.384, 'det': 1.0, 'motor1': x_start - 0.050},
        {'motor2': y_start + 0.258, 'det': 1.0, 'motor1': x_start + 0.306},
        {'motor2': y_start + 0.017, 'det': 1.0, 'motor1': x_start - 0.412},
        {'motor2': y_start - 0.299, 'det': 1.0, 'motor1': x_start + 0.301},
        {'motor2': y_start + 0.435, 'det': 1.0, 'motor1': x_start - 0.020},
        {'motor2': y_start - 0.343, 'det': 1.0, 'motor1': x_start - 0.287},
        {'motor2': y_start + 0.061, 'det': 1.0, 'motor1': x_start + 0.454},
        {'motor2': y_start + 0.268, 'det': 1.0, 'motor1': x_start - 0.385},
        {'motor2': y_start - 0.468, 'det': 1.0, 'motor1': x_start + 0.105},
        {'motor2': y_start + 0.425, 'det': 1.0, 'motor1': x_start + 0.244},
        {'motor2': y_start - 0.152, 'det': 1.0, 'motor1': x_start - 0.476},
        {'motor2': y_start - 0.214, 'det': 1.0, 'motor1': x_start + 0.463},
        {'motor2': y_start + 0.479, 'det': 1.0, 'motor1': x_start - 0.201},
        {'motor2': y_start - 0.498, 'det': 1.0, 'motor1': x_start - 0.179},
        {'motor2': y_start + 0.251, 'det': 1.0, 'motor1': x_start + 0.477},
        {'motor2': y_start - 0.468, 'det': 1.0, 'motor1': x_start + 0.301},
        {'motor2': y_start - 0.352, 'det': 1.0, 'motor1': x_start - 0.454},
        {'motor2': y_start + 0.434, 'det': 1.0, 'motor1': x_start - 0.402},
        {'motor2': y_start + 0.451, 'det': 1.0, 'motor1': x_start + 0.409},
        {'motor2': y_start - 0.377, 'det': 1.0, 'motor1': x_start + 0.498},
    ]


def test_absolute_fermat_spiral(fresh_RE):
    motor = Mover('motor', {'motor': lambda x: x}, {'x': 0})
    motor1 = Mover('motor1', {'motor1': lambda x: x}, {'x': 0})
    motor2 = Mover('motor2', {'motor2': lambda x: x}, {'x': 0})

    det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)

    motor1.set(1.0)
    motor2.set(1.0)
    scan = SpiralFermatScan([det], motor1, motor2, 0.0, 0.0, 1.0, 1.0, 0.1,
                            1.0, 0.0)
    approx_multi_traj_checker(fresh_RE, scan, _get_fermat_data(0.0, 0.0),
                              decimal=2)

    scan = SpiralFermatScan([det], motor1, motor2, 0.5, 0.5, 1.0, 1.0, 0.1,
                            1.0, 0.0)
    approx_multi_traj_checker(fresh_RE, scan, _get_fermat_data(0.5, 0.5),
                              decimal=2)


def test_relative_fermat_spiral(fresh_RE):
    start_x = 1.0
    start_y = 1.0

    motor = Mover('motor', {'motor': lambda x: x}, {'x': 0})
    motor1 = Mover('motor1', {'motor1': lambda x: x}, {'x': 0})
    motor2 = Mover('motor2', {'motor2': lambda x: x}, {'x': 0})

    det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)

    motor1.set(start_x)
    motor2.set(start_y)
    scan = RelativeSpiralFermatScan([det], motor1, motor2, 1.0, 1.0, 0.1, 1.0,
                                    0.0)

    approx_multi_traj_checker(fresh_RE, scan,
                              _get_fermat_data(start_x, start_y),
                              decimal=2)
