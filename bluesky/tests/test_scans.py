import warnings
from nose.tools import assert_equal, assert_greater, assert_in, assert_true
from bluesky.scans import *
from bluesky.callbacks import *
from bluesky.standard_config import ascan, dscan, ct
from bluesky import RunEngine
from bluesky.examples import motor, det
from bluesky.tests.utils import setup_test_run_engine


RE = setup_test_run_engine()


def traj_checker(scan, expected_traj):
    actual_traj = []
    callback = collector('motor', actual_traj)
    RE(scan, subs={'event': callback})
    assert_equal(actual_traj, list(expected_traj))


def test_ascan():
    traj = [1, 2, 3]
    scan = Ascan(motor, [det], traj)
    yield traj_checker, scan, traj


def test_dscan():
    traj = np.array([1, 2, 3]) - 4
    motor.set(-4)
    scan = Ascan(motor, [det], traj)
    yield traj_checker, scan, traj


def test_lin_ascan():
    traj = np.linspace(0, 10, 5)
    scan = LinAscan(motor, [det], 0, 10, 5)
    yield traj_checker, scan, traj


def test_log_ascan():
    traj = np.logspace(0, 10, 5)
    scan = LogAscan(motor, [det], 0, 10, 5)
    yield traj_checker, scan, traj


def test_lin_dscan():
    traj = np.linspace(0, 10, 5) + 6
    motor.set(6)
    scan = LinDscan(motor, [det], 0, 10, 5)
    yield traj_checker, scan, traj


def test_log_dscan():
    traj = np.logspace(0, 10, 5) + 6
    motor.set(6)
    scan = LogDscan(motor, [det], 0, 10, 5)
    yield traj_checker, scan, traj


def test_adaptive_ascan():
    scan1 = AdaptiveAscan(motor, [det], 'det', 0, 5, 0.1, 1, 0.1, True)
    scan2 = AdaptiveAscan(motor, [det], 'det', 0, 5, 0.1, 1, 0.2, True)
    scan3 = AdaptiveAscan(motor, [det], 'det', 0, 5, 0.1, 1, 0.1, False)

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
    scan1 = AdaptiveDscan(motor, [det], 'det', 0, 5, 0.1, 1, 0.1, True)
    scan2 = AdaptiveDscan(motor, [det], 'det', 0, 5, 0.1, 1, 0.2, True)
    scan3 = AdaptiveDscan(motor, [det], 'det', 0, 5, 0.1, 1, 0.1, False)

    actual_traj = []
    col = collector('motor', actual_traj)
    counter1 = CallbackCounter()
    counter2 = CallbackCounter()

    motor.set(1)
    RE.verbose = False
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


def test_legacy_scans():
    # smoke tests
    ascan.detectors.append(det)
    ascan.RE = RE
    ascan.RE.md['owner'] = 'test_owner'
    ascan.RE.md['group'] = 'test_group'
    ascan.RE.md['config'] = {}
    ascan.RE.md['beamline_id'] = 'test_beamline'
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

    # cleanup
    ascan.RE.md.clear()
