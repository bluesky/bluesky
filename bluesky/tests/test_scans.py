import warnings
from nose.tools import (assert_equal, assert_greater, assert_in, assert_true,
                        assert_less, assert_is)

from bluesky.callbacks import collector, CallbackCounter
from bluesky.scans import (AbsListScan, AbsScan, LogAbsScan,
                           DeltaListScan, DeltaScan, LogDeltaScan,
                           AdaptiveAbsScan, AdaptiveDeltaScan, Count, Center,
                           OuterProductAbsScan, InnerProductAbsScan,
                           OuterProductDeltaScan, InnerProductDeltaScan)

from bluesky.standard_config import ascan, dscan, ct
from bluesky import Msg
from bluesky.examples import motor, det, SynGauss, motor1, motor2
from bluesky.tests.utils import setup_test_run_engine
import asyncio
import time as ttime
import numpy as np
loop = asyncio.get_event_loop()

RE = setup_test_run_engine()


def traj_checker(scan, expected_traj):
    actual_traj = []
    callback = collector('motor', actual_traj)
    RE(scan, subs={'event': callback})
    assert_equal(actual_traj, list(expected_traj))


def multi_traj_checker(scan, expected_data):
    actual_data = []

    def collect_data(event):
        actual_data.append(event['data'])

    RE(scan, subs={'event': collect_data})
    assert_equal(actual_data, expected_data)


def test_outer_product_ascan():
    motor.set(0)
    scan = OuterProductAbsScan([det], motor1, 1, 3, 3, motor2, 10, 20, 2)
    # Note: motor1 is the first motor specified, and so it is the "slow"
    # axis, matching the numpy convention.
    expected_data = [
        {'motor2': 10.0, 'det': 1.0, 'motor1': 1.0},
        {'motor2': 20.0, 'det': 1.0, 'motor1': 1.0},
        {'motor2': 10.0, 'det': 1.0, 'motor1': 2.0},
        {'motor2': 20.0, 'det': 1.0, 'motor1': 2.0},
        {'motor2': 10.0, 'det': 1.0, 'motor1': 3.0},
        {'motor2': 20.0, 'det': 1.0, 'motor1': 3.0}]
    yield multi_traj_checker, scan, expected_data


def test_inner_product_ascan():
    motor.set(0)
    scan = InnerProductAbsScan([det], 3, motor1, 1, 3, motor2, 10, 30)
    # Note: motor1 is the first motor specified, and so it is the "slow"
    # axis, matching the numpy convention.
    expected_data = [
        {'motor2': 10.0, 'det': 1.0, 'motor1': 1.0},
        {'motor2': 20.0, 'det': 1.0, 'motor1': 2.0},
        {'motor2': 30.0, 'det': 1.0, 'motor1': 3.0}]
    yield multi_traj_checker, scan, expected_data


def test_outer_product_dscan():
    scan = OuterProductDeltaScan([det], motor1, 1, 3, 3, motor2, 10, 20, 2)
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
    yield multi_traj_checker, scan, expected_data


def test_inner_product_dscan():
    motor.set(0)
    motor1.set(5)
    motor2.set(8)
    scan = InnerProductDeltaScan([det], 3, motor1, 1, 3, motor2, 10, 30)
    # Note: motor1 is the first motor specified, and so it is the "slow"
    # axis, matching the numpy convention.
    expected_data = [
        {'motor2': 18.0, 'det': 1.0, 'motor1': 6.0},
        {'motor2': 28.0, 'det': 1.0, 'motor1': 7.0},
        {'motor2': 38.0, 'det': 1.0, 'motor1': 8.0}]
    yield multi_traj_checker, scan, expected_data


def test_ascan():
    traj = [1, 2, 3]
    scan = AbsListScan([det], motor, traj)
    yield traj_checker, scan, traj


def test_dscan():
    traj = np.array([1, 2, 3])
    motor.set(-4)
    scan = DeltaListScan([det], motor, traj)
    yield traj_checker, scan, traj - 4


def test_lin_ascan():
    traj = np.linspace(0, 10, 5)
    scan = AbsScan([det], motor, 0, 10, 5)
    yield traj_checker, scan, traj


def test_log_ascan():
    traj = np.logspace(0, 10, 5)
    scan = LogAbsScan([det], motor, 0, 10, 5)
    yield traj_checker, scan, traj


def test_lin_dscan():
    traj = np.linspace(0, 10, 5) + 6
    motor.set(6)
    scan = DeltaScan([det], motor, 0, 10, 5)
    yield traj_checker, scan, traj


def test_log_dscan():
    traj = np.logspace(0, 10, 5) + 6
    motor.set(6)
    scan = LogDeltaScan([det], motor, 0, 10, 5)
    yield traj_checker, scan, traj


def test_adaptive_ascan():
    scan1 = AdaptiveAbsScan([det], 'det', motor, 0, 5, 0.1, 1, 0.1, True)
    scan2 = AdaptiveAbsScan([det], 'det', motor, 0, 5, 0.1, 1, 0.2, True)
    scan3 = AdaptiveAbsScan([det], 'det', motor, 0, 5, 0.1, 1, 0.1, False)

    actual_traj = []
    col = collector('motor', actual_traj)
    counter1 = CallbackCounter()
    counter2 = CallbackCounter()

    RE(scan1, subs={'event': [col, counter1]})
    RE(scan2, subs={'event': counter2})
    assert_greater(counter1.value, counter2.value)
    assert_equal(actual_traj[0], 0)

    actual_traj = []
    col = collector('motor', actual_traj)
    RE(scan3, {'event': col})
    monotonic_increasing = np.all(np.diff(actual_traj) > 0)
    assert_true(monotonic_increasing)


def test_adaptive_dscan():
    scan1 = AdaptiveDeltaScan([det], 'det', motor, 0, 5, 0.1, 1, 0.1, True)
    scan2 = AdaptiveDeltaScan([det], 'det', motor, 0, 5, 0.1, 1, 0.2, True)
    scan3 = AdaptiveDeltaScan([det], 'det', motor, 0, 5, 0.1, 1, 0.1, False)

    actual_traj = []
    col = collector('motor', actual_traj)
    counter1 = CallbackCounter()
    counter2 = CallbackCounter()

    motor.set(1)
    RE(scan1, subs={'event': [col, counter1]})
    RE(scan2, subs={'event': counter2})
    assert_greater(counter1.value, counter2.value)
    assert_equal(actual_traj[0], 1)

    actual_traj = []
    col = collector('motor', actual_traj)
    RE(scan3, {'event': col})
    monotonic_increasing = np.all(np.diff(actual_traj) > 0)
    assert_true(monotonic_increasing)


def test_count():
    actual_intensity = []
    col = collector('det', actual_intensity)
    motor.set(0)
    scan = Count([det])
    RE(scan, subs={'event': col})
    assert_equal(actual_intensity[0], 1.)

    # multiple counts
    actual_intensity = []
    col = collector('det', actual_intensity)
    scan = Count([det], num=3, delay=0.05)
    RE(scan, subs={'event': col})
    assert_equal(scan.num, 3)
    assert_equal(actual_intensity, [1., 1., 1.])


def test_center():
    assert_true(not RE._run_is_open)
    det = SynGauss('det', motor, 'motor', 0, 1000, 1, 'poisson', True)
    d = {}
    cen = Center([det], 'det', motor, 0.1, 1.1, 0.01, d)
    RE(cen)
    assert_less(abs(d['center']), 0.1)


def test_legacy_scans():
    # smoke tests
    ascan.detectors.append(det)
    for _re in [ascan.RE, dscan.RE, ct.RE]:
        _re.md['owner'] = 'test_owner'
        _re.md['group'] = 'Grant No. 12345'
        _re.md['config'] = {'detector_model': 'XYZ', 'pixel_size': 10}
        _re.md['beamline_id'] = 'test_beamline'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ascan(motor, 0, 5, 5)
        dscan(motor, 0, 5, 5)
        ct()

    # test that metadata is passed
    # notice that we can pass subs to the RE as well

    def assert_lion(doc):
        assert_in('animal', doc)
        assert_equal(doc['animal'], 'lion')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ct(animal='lion', subs={'start': assert_lion})


def test_legacy_scan_state():
    assert_is(ascan.RE, dscan.RE)
    assert_is(ascan.RE, ct.RE)
    assert_is(ascan.detectors, dscan.detectors)
    assert_is(ascan.detectors, ct.detectors)


def test_set():
    scan = AbsScan([det], motor, 1, 5, 3)
    assert_equal(scan.start, 1)
    assert_equal(scan.stop, 5)
    assert_equal(scan.num, 3)
    scan.set(start=2)
    assert_equal(scan.start, 2)
    scan.set(num=4)
    assert_equal(scan.num, 4)
    assert_equal(scan.start, 2)


def test_wait_for():
    ev = asyncio.Event()

    def done():
        ev.set()
    scan = [Msg('wait_for', [ev.wait(), ]), ]
    loop.call_later(2, done)
    start = ttime.time()
    RE(scan)
    stop = ttime.time()
    assert stop - start >= 2
