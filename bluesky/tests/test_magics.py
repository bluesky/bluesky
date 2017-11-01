from bluesky.magics import BlueskyMagics
import bluesky.plans as bp
import bluesky.plan_stubs as bps
import os
import pytest
import signal
from types import SimpleNamespace


class FakeIPython:
    def __init__(self, user_ns):
        self.user_ns = user_ns


def compare_msgs(actual, expected):
    for a, e in zip(actual, expected):
        # Strip off randomized stuff that cannot be compared.
        a.kwargs.pop('group', None)
        e.kwargs.pop('group', None)
        assert a == e


@pytest.mark.parametrize('pln,plnargs,magic,line', [
    (bps.mv, lambda hw: (hw.motor1, 2),
     'mov', 'motor1 2'),
    (bps.mv, lambda hw: (hw.motor1, 2, hw.motor2, 3),
     'mov', 'motor1 2 motor2 3'),
    (bps.mvr, lambda hw: (hw.motor1, 2),
     'movr', 'motor1 2'),
    (bps.mvr, lambda hw: (hw.motor1, 2, hw.motor2, 3),
     'movr', 'motor1 2 motor2 3'),
    (bp.count, lambda hw: ([hw.invariant1],),
     'ct', 'dets'),
    (bp.count, lambda hw: ([hw.invariant1, hw.invariant2],),
     'ct', ''),
    ])
def test_bluesky_magics(pln, plnargs, line, magic, RE, hw):
    # Build a FakeIPython instance to use the magics with.
    dets = [hw.invariant1]
    ip = FakeIPython({'motor1': hw.motor1, 'motor2': hw.motor2, 'dets': dets})
    sm = BlueskyMagics(ip)
    sm.detectors = [hw.invariant1, hw.invariant2]

    # Spy on all msgs processed by RE.
    msgs = []

    def collect(msg):
        msgs.append(msg)

    RE.msg_hook = collect
    sm.RE.msg_hook = collect

    # Test magics cause the RunEngine to execute the messages we expect.
    RE(bps.mv(hw.motor1, 10, hw.motor2, 10))  # ensure known initial state
    RE(pln(*plnargs(hw)))
    expected = msgs.copy()
    msgs.clear()
    RE(bps.mv(hw.motor1, 10, hw.motor2, 10))  # ensure known initial state
    getattr(sm, magic)(line)
    actual = msgs.copy()
    msgs.clear()
    compare_msgs(actual, expected)


# The %wa magic doesn't use a RunEngine or a plan.
def test_wa(hw):
    motor = hw.motor
    ip = FakeIPython({'motor': motor})
    sm = BlueskyMagics(ip)
    # Test an empty list.
    sm.wa('')

    sm.positioners.extend([motor])
    sm.wa('')

    # Make motor support more attributes.
    motor.limits = (-1, 1)
    sm.wa('')
    motor.user_offset = SimpleNamespace(get=lambda: 0)

    sm.wa('[motor]')


def test_magics_missing_ns_key(RE, hw):
    ip = FakeIPython({})
    sm = BlueskyMagics(ip)
    with pytest.raises(NameError):
        sm.mov('motor1 5')
    ip.user_ns['motor1'] = hw.motor1
    sm.mov('motor1 5')


def test_interrupted(RE, hw):
    motor = hw.motor
    motor.delay = 10

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
