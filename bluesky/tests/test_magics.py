from bluesky.magics import BlueskyMagics
import bluesky.plans as bp
from bluesky.examples import det, motor1, motor2, det1, det2
import os
import pytest
import signal


class FakeIPython:
    def __init__(self, user_ns):
        self.user_ns = user_ns


def compare_msgs(actual, expected):
    for a, e in zip(actual, expected):
        # Strip off randomized stuff that cannot be compared.
        a.kwargs.pop('group', None)
        e.kwargs.pop('group', None)
        assert a == e

dets = [det]
default_dets = [det1, det2]

@pytest.mark.parametrize('pln,magic,line', [
    (bp.mv(motor1, 2), 'mov', 'motor1 2'),
    (bp.mv(motor1, 2, motor2, 3), 'mov', 'motor1 2 motor2 3'),
    (bp.mvr(motor1, 2), 'movr', 'motor1 2'),
    (bp.mvr(motor1, 2, motor2, 3), 'movr', 'motor1 2 motor2 3'),
    (bp.count(dets), 'ct', 'dets'),
    (bp.count(default_dets), 'ct', ''),
    ])
def test_bluesky_magics(pln, line, magic, fresh_RE):
    RE = fresh_RE

    # Build a FakeIPython instance to use the magics with.

    dets = [det]
    ip = FakeIPython({'motor1': motor1, 'motor2': motor2, 'dets': dets})
    sm = BlueskyMagics(ip)
    sm.detectors = default_dets

    # Spy on all msgs processed by RE.
    msgs = []

    def collect(msg):
        msgs.append(msg)

    RE.msg_hook = collect
    sm.RE.msg_hook = collect

    # Test magics cause the RunEngine to execute the messages we expect.
    RE(bp.mv(motor1, 10, motor2, 10))  # ensure known initial state
    RE(pln)
    expected = msgs.copy()
    msgs.clear()
    RE(bp.mv(motor1, 10, motor2, 10))  # ensure known initial state
    getattr(sm, magic)(line)
    actual = msgs.copy()
    msgs.clear()
    compare_msgs(actual, expected)


# The %wa magic doesn't use a RunEngine or a plan.
def test_wa():
    from bluesky.examples import motor
    ip = FakeIPython({'motor': motor})
    sm = BlueskyMagics(ip)
    # Test an empty list.
    sm.wa('')

    sm.positioners.extend([motor])
    sm.wa('')

    sm.wa('[motor]')


def test_magics_missing_ns_key(fresh_RE):
    RE = fresh_RE
    ip = FakeIPython({})
    sm = BlueskyMagics(ip)
    with pytest.raises(NameError):
        sm.mov('motor1 5')
    ip.user_ns['motor1'] = motor1
    sm.mov('motor1 5')


def test_interrupted(motor_det):
    motor, det = motor_det
    motor._fake_sleep = 10

    ip = FakeIPython({})
    sm = BlueskyMagics(ip)
    ip.user_ns['motor'] = motor

    pid = os.getpid()

    def sim_kill(n=1):
        for j in range(n):
            print('KILL')
            os.kill(pid, signal.SIGINT)

    motor.loop = sm.RE.loop
    sm.RE.loop.call_later(1, sim_kill, 2)
    sm.mov('motor 1')
    assert sm.RE.state == 'idle'
