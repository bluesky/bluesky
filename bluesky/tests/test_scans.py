from bluesky.callbacks import collector, CallbackCounter
import bluesky.plans as bp
from bluesky import Msg
import asyncio
import time as ttime
import numpy as np
import numpy.testing
import pytest


def traj_checker(RE, scan, expected_traj):
    actual_traj = []
    callback = collector('motor', actual_traj)
    RE(scan, {'event': callback})
    assert actual_traj == list(expected_traj)


def multi_traj_checker(RE, scan, expected_data):
    actual_data = []

    def collect_data(name, event):
        actual_data.append(event['data'])

    RE(scan, {'event': collect_data})
    assert actual_data == expected_data


def approx_multi_traj_checker(RE, scan, expected_data, *, decimal=2):
    actual_data = []

    def collect_data(name, event):
        actual_data.append(event['data'])

    RE(scan, {'event': collect_data})
    keys = sorted(expected_data[0].keys())

    actual_values = [[row[key] for key in keys]
                     for row in actual_data]

    expected_values = [[row[key] for key in keys]
                       for row in expected_data]

    err_msg = 'Trajectory differs. Data keys: {}'.format(', '.join(keys))
    numpy.testing.assert_almost_equal(actual_values, expected_values,
                                      decimal=decimal,
                                      err_msg=err_msg)


def test_outer_product_ascan(RE, hw):
    scan = bp.grid_scan([hw.det],
                                 hw.motor1, 1, 3, 3,
                                 hw.motor2, 10, 20, 2, False)
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
    multi_traj_checker(RE, scan, expected_data)


def test_outer_product_ascan_snaked(RE, hw):
    scan = bp.grid_scan([hw.det],
                                 hw.motor1, 1, 3, 3,
                                 hw.motor2, 10, 20, 2, True)
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
    multi_traj_checker(RE, scan, expected_data)


def test_inner_product_ascan(RE, hw):
    scan = bp.inner_product_scan([hw.det], 3,
                                 hw.motor1, 1, 3,
                                 hw.motor2, 10, 30)
    # Note: motor1 is the first motor specified, and so it is the "slow"
    # axis, matching the numpy convention.
    expected_data = [
        {'motor2': 10.0, 'det': 1.0, 'motor1': 1.0},
        {'motor2': 20.0, 'det': 1.0, 'motor1': 2.0},
        {'motor2': 30.0, 'det': 1.0, 'motor1': 3.0}]
    for d in expected_data:
        d.update({'motor1_setpoint': d['motor1']})
        d.update({'motor2_setpoint': d['motor2']})
    multi_traj_checker(RE, scan, expected_data)


def test_outer_product_dscan(RE, hw):
    scan = bp.rel_grid_scan([hw.det],
                                          hw.motor1, 1, 3, 3,
                                          hw.motor2, 10, 20, 2, False)
    # Note: motor1 is the first motor specified, and so it is the "slow"
    # axis, matching the numpy convention.
    hw.motor1.set(5)
    hw.motor2.set(8)
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
    multi_traj_checker(RE, scan, expected_data)


def test_outer_product_dscan_snaked(RE, hw):
    scan = bp.rel_grid_scan([hw.det],
                                          hw.motor1, 1, 3, 3,
                                          hw.motor2, 10, 20, 2, True)
    # Note: motor1 is the first motor specified, and so it is the "slow"
    # axis, matching the numpy convention.
    hw.motor1.set(5)
    hw.motor2.set(8)
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
    multi_traj_checker(RE, scan, expected_data)


def test_inner_product_dscan(RE, hw):
    scan = bp.relative_inner_product_scan([hw.det], 3,
                                          hw.motor1, 1, 3,
                                          hw.motor2, 10, 30)
    # Note: motor1 is the first motor specified, and so it is the "slow"
    # axis, matching the numpy convention.
    hw.motor1.set(5)
    hw.motor2.set(8)
    expected_data = [
        {'motor2': 18.0, 'det': 1.0, 'motor1': 6.0},
        {'motor2': 28.0, 'det': 1.0, 'motor1': 7.0},
        {'motor2': 38.0, 'det': 1.0, 'motor1': 8.0}]
    for d in expected_data:
        d.update({'motor1_setpoint': d['motor1']})
        d.update({'motor2_setpoint': d['motor2']})
    multi_traj_checker(RE, scan, expected_data)


def test_ascan(RE, hw):
    traj = [1, 2, 3]
    scan = bp.list_scan([hw.det], hw.motor, traj)
    traj_checker(RE, scan, traj)


def test_dscan(RE, hw):
    traj = np.array([1, 2, 3])
    hw.motor.set(-4)
    scan = bp.rel_list_scan([hw.det], hw.motor, traj)
    traj_checker(RE, scan, traj - 4)


def test_dscan_list_input(RE, hw):
    # GH225
    traj = [1, 2, 3]
    hw.motor.set(-4)
    scan = bp.rel_list_scan([hw.det], hw.motor, traj)
    traj_checker(RE, scan, np.array(traj) - 4)


def test_lin_ascan(RE, hw):
    traj = np.linspace(0, 10, 5)
    scan = bp.scan([hw.det], hw.motor, 0, 10, 5)
    traj_checker(RE, scan, traj)


def test_log_ascan(RE, hw):
    traj = np.logspace(0, 10, 5)
    scan = bp.log_scan([hw.det], hw.motor, 0, 10, 5)
    traj_checker(RE, scan, traj)


def test_lin_dscan(RE, hw):
    traj = np.linspace(0, 10, 5) + 6
    hw.motor.set(6)
    scan = bp.rel_scan([hw.det], hw.motor, 0, 10, 5)
    traj_checker(RE, scan, traj)


def test_log_dscan(RE, hw):
    traj = np.logspace(0, 10, 5) + 6
    hw.motor.set(6)
    scan = bp.rel_log_scan([hw.det], hw.motor, 0, 10, 5)
    traj_checker(RE, scan, traj)


def test_adaptive_ascan(RE, hw):
    scan1 = bp.adaptive_scan([hw.det], 'det', hw.motor, 0, 5, 0.1, 1, 0.1, True)
    scan2 = bp.adaptive_scan([hw.det], 'det', hw.motor, 0, 5, 0.1, 1, 0.2, True)
    scan3 = bp.adaptive_scan([hw.det], 'det', hw.motor, 0, 5, 0.1, 1, 0.1, False)
    scan4 = bp.adaptive_scan([hw.det], 'det', hw.motor, 5, 0, 0.1, 1, 0.1, False)

    actual_traj = []
    col = collector('motor', actual_traj)
    counter1 = CallbackCounter()
    counter2 = CallbackCounter()

    RE(scan1, {'event': [col, counter1]})
    RE(scan2, {'event': counter2})
    assert counter1.value > counter2.value
    assert actual_traj[0] == 0

    actual_traj = []
    col = collector('motor', actual_traj)
    RE(scan3, {'event': col})
    monotonic_increasing = np.all(np.diff(actual_traj) > 0)
    assert monotonic_increasing

    actual_traj = []
    col = collector('motor', actual_traj)
    RE(scan4, {'event': col})
    monotonic_decreasing = np.all(np.diff(actual_traj) < 0)
    assert monotonic_decreasing

    with pytest.raises(ValueError):  # min step < 0
        scan5 = bp.adaptive_scan([hw.det], 'det', hw.motor,
                                 5, 0, -0.1, 1.0, 0.1, False)
        RE(scan5)


def test_adaptive_dscan(RE, hw):
    scan1 = bp.rel_adaptive_scan(
        [hw.det], 'det', hw.motor, 0, 5, 0.1, 1, 0.1, True)
    scan2 = bp.rel_adaptive_scan(
        [hw.det], 'det', hw.motor, 0, 5, 0.1, 1, 0.2, True)
    scan3 = bp.rel_adaptive_scan(
        [hw.det], 'det', hw.motor, 0, 5, 0.1, 1, 0.1, False)
    scan4 = bp.rel_adaptive_scan(
        [hw.det], 'det', hw.motor, 5, 0, 0.1, 1, 0.1, False)

    actual_traj = []
    col = collector('motor', actual_traj)
    counter1 = CallbackCounter()
    counter2 = CallbackCounter()

    hw.motor.set(1)
    RE(scan1, {'event': [col, counter1]})
    RE(scan2, {'event': counter2})
    assert counter1.value > counter2.value
    assert actual_traj[0] == 1

    actual_traj = []
    col = collector('motor', actual_traj)
    RE(scan3, {'event': col})
    monotonic_increasing = np.all(np.diff(actual_traj) > 0)
    assert monotonic_increasing

    actual_traj = []
    col = collector('motor', actual_traj)
    RE(scan4, {'event': col})
    monotonic_decreasing = np.all(np.diff(actual_traj) < 0)
    assert monotonic_decreasing

    with pytest.raises(ValueError):  # min step > max step
        scan5 = bp.rel_adaptive_scan(
            [hw.det], 'det', hw.motor, 5, 0, 1, 0.1, 0.1, False)
        RE(scan5)


def test_tune_centroid(RE, hw):
    det = hw.det
    motor = hw.motor
    scan1 = bp.tune_centroid([det], 'det', motor, 0, 5, 0.1, 10, snake=True)
    scan2 = bp.tune_centroid([det], 'det', motor, 0, 5, 0.01, 10, snake=True)
    scan3 = bp.tune_centroid([det], 'det', motor, 0, 5, 0.1, 10, snake=False)
    scan4 = bp.tune_centroid([det], 'det', motor, 5, 0, 0.1, 10, snake=False)

    actual_traj = []
    col = collector('motor', actual_traj)
    counter1 = CallbackCounter()
    counter2 = CallbackCounter()

    RE(scan1, {'event': [col, counter1]})
    RE(scan2, {'event': counter2})
    #assert counter1.value > counter2.value
    assert actual_traj[0] == 0

    actual_traj = []
    col = collector('motor', actual_traj)
    RE(scan3, {'event': col})
    monotonic_increasing = np.all(np.diff(actual_traj) > 0)
    assert not monotonic_increasing

    actual_traj = []
    col = collector('motor', actual_traj)
    RE(scan4, {'event': col})
    monotonic_decreasing = np.all(np.diff(actual_traj) < 0)
    assert not monotonic_decreasing

    with pytest.raises(ValueError):  # min step < 0
        scan5 = bp.tune_centroid([det], 'det', motor, 5, 0, -0.1, 10, snake=False)
        RE(scan5)


def test_count(RE, hw):
    det = hw.det
    motor = hw.motor
    actual_intensity = []
    col = collector('det', actual_intensity)
    motor.set(0)
    plan = bp.count([det])
    RE(plan, {'event': col})
    assert actual_intensity[0] == 1.
    # multiple counts, via updating attribute
    actual_intensity = []
    col = collector('det', actual_intensity)
    plan = bp.count([det], num=3, delay=0.05)
    RE(plan, {'event': col})
    assert actual_intensity == [1., 1., 1.]


def test_wait_for(RE):
    ev = asyncio.Event(loop=RE.loop)

    def done():
        ev.set()
    scan = [Msg('wait_for', None, [ev.wait(), ]), ]
    RE.loop.call_later(2, done)
    start = ttime.time()
    RE(scan)
    stop = ttime.time()
    assert stop - start >= 2


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


def test_absolute_spiral(RE, hw):
    motor1 = hw.motor1
    motor2 = hw.motor2
    det = hw.det
    motor1.set(1.0)
    motor2.set(1.0)
    scan = bp.spiral([det], motor1, motor2, 0.0, 0.0, 1.0, 1.0, 0.1, 1.0,
                     tilt=0.0)
    approx_multi_traj_checker(RE, scan, _get_spiral_data(0.0, 0.0),
                              decimal=2)

    scan = bp.spiral([det], motor1, motor2, 0.5, 0.5, 1.0, 1.0, 0.1, 1.0,
                     tilt=0.0)
    approx_multi_traj_checker(RE, scan, _get_spiral_data(0.5, 0.5),
                              decimal=2)


def test_rel_spiral(RE, hw):
    motor1 = hw.motor1
    motor2 = hw.motor2
    det = hw.det

    start_x = 1.0
    start_y = 1.0

    motor1.set(start_x)
    motor2.set(start_y)
    scan = bp.rel_spiral([det],
                              motor1, motor2,
                              1.0, 1.0,
                              0.1, 1.0,
                              tilt=0.0)

    approx_multi_traj_checker(RE, scan,
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


def test_absolute_fermat_spiral(RE, hw):
    motor1 = hw.motor1
    motor2 = hw.motor2
    det = hw.det

    motor1.set(1.0)
    motor2.set(1.0)
    scan = bp.spiral_fermat([det], motor1, motor2, 0.0, 0.0, 1.0, 1.0, 0.1,
                            1.0, tilt=0.0)
    approx_multi_traj_checker(RE, scan, _get_fermat_data(0.0, 0.0),
                              decimal=2)

    scan = bp.spiral_fermat([det],
                            motor1, motor2,
                            0.5, 0.5,
                            1.0, 1.0,
                            0.1, 1.0,
                            tilt=0.0)
    approx_multi_traj_checker(RE, scan, _get_fermat_data(0.5, 0.5),
                              decimal=2)


def test_relative_fermat_spiral(RE, hw):
    start_x = 1.0
    start_y = 1.0

    motor1 = hw.motor1
    motor2 = hw.motor2
    det = hw.det

    motor1.set(start_x)
    motor2.set(start_y)
    scan = bp.rel_spiral_fermat([det],
                                     motor1, motor2,
                                     1.0, 1.0,
                                     0.1, 1.0,
                                     tilt=0.0)

    approx_multi_traj_checker(RE, scan,
                              _get_fermat_data(start_x, start_y),
                              decimal=2)


def test_x2x_scan(RE, hw):
    y_start = -1
    y_stop = 4
    y_num = 11
    expected_traj = []

    for i in range(y_num):
        y_val = y_start + i * (y_stop - y_start) / (y_num - 1)
        expected_traj.append({'motor1': y_val, 'det': 1.0, 'motor2': y_val / 2})

    for d in expected_traj:
        d.update({'motor1_setpoint': d['motor1']})
        d.update({'motor2_setpoint': d['motor2']})

    scan = bp.x2x_scan([hw.det], hw.motor1, hw.motor2, y_start, y_stop, y_num )

    multi_traj_checker(RE, scan, expected_traj)


square_spiral_data = [
    {'det': 1.0, 'motor1': 0.0, 'motor2': 0.0},
    {'det': 1.0, 'motor1': 0.5, 'motor2': 0.0},
    {'det': 1.0, 'motor1': 0.5, 'motor2': -0.5},
    {'det': 1.0, 'motor1': 0.0, 'motor2': -0.5},
    {'det': 1.0, 'motor1': -0.5, 'motor2': -0.5},
    {'det': 1.0, 'motor1': -0.5, 'motor2': 0.0},
    {'det': 1.0, 'motor1': -0.5, 'motor2': 0.5},
    {'det': 1.0, 'motor1': 0.0, 'motor2': 0.5},
    {'det': 1.0, 'motor1': 0.5, 'motor2': 0.5},
    {'det': 1.0, 'motor1': 1.0, 'motor2': 0.5},
    {'det': 1.0, 'motor1': 1.0, 'motor2': 0.0},
    {'det': 1.0, 'motor1': 1.0, 'motor2': -0.5},
    {'det': 1.0, 'motor1': 1.0, 'motor2': -1.0},
    {'det': 1.0, 'motor1': 0.5, 'motor2': -1.0},
    {'det': 1.0, 'motor1': 0.0, 'motor2': -1.0},
    {'det': 1.0, 'motor1': -0.5, 'motor2': -1.0},
    {'det': 1.0, 'motor1': -1.0, 'motor2': -1.0},
    {'det': 1.0, 'motor1': -1.0, 'motor2': -0.5},
    {'det': 1.0, 'motor1': -1.0, 'motor2': 0.0},
    {'det': 1.0, 'motor1': -1.0, 'motor2': 0.5},
    {'det': 1.0, 'motor1': -1.0, 'motor2': 1.0},
    {'det': 1.0, 'motor1': -0.5, 'motor2': 1.0},
    {'det': 1.0, 'motor1': 0.0, 'motor2': 1.0},
    {'det': 1.0, 'motor1': 0.5, 'motor2': 1.0},
    {'det': 1.0, 'motor1': 1.0, 'motor2': 1.0},
    {'det': 1.0, 'motor1': 1.5, 'motor2': 1.0},
    {'det': 1.0, 'motor1': 1.5, 'motor2': 0.5},
    {'det': 1.0, 'motor1': 1.5, 'motor2': 0.0},
    {'det': 1.0, 'motor1': 1.5, 'motor2': -0.5},
    {'det': 1.0, 'motor1': 1.5, 'motor2': -1.0},
    {'det': 1.0, 'motor1': 1.5, 'motor2': -1.5},
    {'det': 1.0, 'motor1': 1.0, 'motor2': -1.5},
    {'det': 1.0, 'motor1': 0.5, 'motor2': -1.5},
    {'det': 1.0, 'motor1': 0.0, 'motor2': -1.5},
    {'det': 1.0, 'motor1': -0.5, 'motor2': -1.5},
    {'det': 1.0, 'motor1': -1.0, 'motor2': -1.5},
    {'det': 1.0, 'motor1': -1.5, 'motor2': -1.5},
    {'det': 1.0, 'motor1': -1.5, 'motor2': -1.0},
    {'det': 1.0, 'motor1': -1.5, 'motor2': -0.5},
    {'det': 1.0, 'motor1': -1.5, 'motor2': 0.0},
    {'det': 1.0, 'motor1': -1.5, 'motor2': 0.5},
    {'det': 1.0, 'motor1': -1.5, 'motor2': 1.0},
    {'det': 1.0, 'motor1': -1.5, 'motor2': 1.5},
    {'det': 1.0, 'motor1': -1.0, 'motor2': 1.5},
    {'det': 1.0, 'motor1': -0.5, 'motor2': 1.5},
    {'det': 1.0, 'motor1': 0.0, 'motor2': 1.5},
    {'det': 1.0, 'motor1': 0.5, 'motor2': 1.5},
    {'det': 1.0, 'motor1': 1.0, 'motor2': 1.5},
    {'det': 1.0, 'motor1': 1.5, 'motor2': 1.5},
    {'det': 1.0, 'motor1': 1.5, 'motor2': -2.0},
    {'det': 1.0, 'motor1': 1.0, 'motor2': -2.0},
    {'det': 1.0, 'motor1': 0.5, 'motor2': -2.0},
    {'det': 1.0, 'motor1': 0.0, 'motor2': -2.0},
    {'det': 1.0, 'motor1': -0.5, 'motor2': -2.0},
    {'det': 1.0, 'motor1': -1.0, 'motor2': -2.0},
    {'det': 1.0, 'motor1': -1.5, 'motor2': -2.0},
    {'det': 1.0, 'motor1': -1.5, 'motor2': 2.0},
    {'det': 1.0, 'motor1': -1.0, 'motor2': 2.0},
    {'det': 1.0, 'motor1': -0.5, 'motor2': 2.0},
    {'det': 1.0, 'motor1': 0.0, 'motor2': 2.0},
    {'det': 1.0, 'motor1': 0.5, 'motor2': 2.0},
    {'det': 1.0, 'motor1': 1.0, 'motor2': 2.0},
    {'det': 1.0, 'motor1': 1.5, 'motor2': 2.0},
]


def test_spiral_square(RE, hw):
    motor1 = hw.motor1
    motor2 = hw.motor2
    det = hw.det

    plan = bp.spiral_square(
        [det], motor1, motor2,
        x_center=0, y_center=0,
        x_range=3, y_range=4,
        x_num=3*2+1, y_num=4*2+1)

    approx_multi_traj_checker(RE, plan,
                              square_spiral_data, decimal=2)
                              

def test_rel_spiral_square(RE, hw):
    motor1 = hw.motor1
    motor2 = hw.motor2
    det = hw.det

    plan = bp.rel_spiral_square(
        [det], motor1, motor2,
        x_range=3, y_range=4,
        x_num=3*2+1, y_num=4*2+1)

    approx_multi_traj_checker(RE, plan,
                              square_spiral_data, decimal=2)
