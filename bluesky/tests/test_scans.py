from nose.tools import assert_equal
from bluesky.scans import *
from bluesky.callbacks import *
from bluesky import RunEngine
from bluesky.examples import motor, det


RE = None


def setup():
    global RE
    RE = RunEngine()


def traj_checker(scan, expected_traj):
    actual_traj = []
    callback = collector('pos', actual_traj)
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


def test_linascan():
    traj = np.linspace(0, 10, 5)
    scan = LinAscan(motor, [det], 0, 10, 5)
    yield traj_checker, scan, traj


def test_logascan():
    traj = np.logspace(0, 10, 5)
    scan = LogAscan(motor, [det], 0, 10, 5)
    yield traj_checker, scan, traj


def test_lindscan():
    traj = np.linspace(0, 10, 5) + 6
    motor.set(6)
    scan = LinDscan(motor, [det], 0, 10, 5)
    yield traj_checker, scan, traj


def test_logdscan():
    traj = np.logspace(0, 10, 5) + 6
    motor.set(6)
    scan = LogDscan(motor, [det], 0, 10, 5)
    yield traj_checker, scan, traj

def test_linascan():
    traj = np.linspace(0, 10, 5)
    scan = LinAscan(motor, [det], 0, 10, 5)
    yield traj_checker, scan, traj


def test_logascan():
    traj = np.logspace(0, 10, 5)
    scan = LogAscan(motor, [det], 0, 10, 5)
    yield traj_checker, scan, traj


def test_lindscan():
    traj = np.linspace(0, 10, 5) + 6
    motor.set(6)
    scan = LinDscan(motor, [det], 0, 10, 5)
    yield traj_checker, scan, traj


def test_logdscan():
    traj = np.logspace(0, 10, 5) + 6
    motor.set(6)
    scan = LogDscan(motor, [det], 0, 10, 5)
    yield traj_checker, scan, traj
