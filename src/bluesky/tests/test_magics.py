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


@pytest.mark.parametrize('pln,plnargs,magic,line,detectors_factory', [
    (bps.mv, lambda hw: (hw.motor1, 2),
     'mov', 'motor1 2', lambda hw: []),
    (bps.mv, lambda hw: (hw.motor1, 2, hw.motor2, 3),
     'mov', 'motor1 2 motor2 3', lambda hw: []),
    (bps.mvr, lambda hw: (hw.motor1, 2),
     'movr', 'motor1 2', lambda hw: []),
    (bps.mvr, lambda hw: (hw.motor1, 2, hw.motor2, 3),
     'movr', 'motor1 2 motor2 3', lambda hw: []),
    (bp.count, lambda hw: ([hw.invariant1],),
     'ct', 'favorite_detectors', lambda hw: []),
    (bp.count, lambda hw: ([hw.invariant1, hw.invariant2],),
     'ct', '', lambda hw: []),
    (bp.count, lambda hw: ([hw.invariant1],),
     'ct', 'dets', lambda hw: [hw.invariant1, hw.invariant2]),
    (bp.count, lambda hw: ([hw.invariant1, hw.invariant2],),
     'ct', '', lambda hw: [hw.invariant1, hw.invariant2]),
    ])
def test_bluesky_magics(pln, plnargs, magic, line, detectors_factory,
                        RE, hw):
    # Build a FakeIPython instance to use the magics with.
    dets = [hw.invariant1]
    hw.invariant1._ophyd_labels_ = set(['detectors', 'favorite_detectors'])
    hw.invariant2._ophyd_labels_ = set(['detectors'])
    ip = FakeIPython({'motor1': hw.motor1, 'motor2': hw.motor2,
                      'invariant1': hw.invariant1, 'invariant2': hw.invariant2,
                      'dets': dets}
                     )
    sm = BlueskyMagics(ip)
    detectors = detectors_factory(hw)
    if detectors:
        # Test deprecated usage of %ct.
        with pytest.warns(UserWarning):
            BlueskyMagics.detectors = detectors

    # Spy on all msgs processed by RE.
    msgs = []

    def collect(msg):
        msgs.append(msg)

    RE.msg_hook = collect
    BlueskyMagics.RE.msg_hook = collect

    # Test magics cause the RunEngine to execute the messages we expect.
    RE(bps.mv(hw.motor1, 10, hw.motor2, 10))  # ensure known initial state
    msgs.clear()
    RE(pln(*plnargs(hw)))
    expected = msgs.copy()
    RE(bps.mv(hw.motor1, 10, hw.motor2, 10))  # ensure known initial state
    msgs.clear()
    if detectors:
        # Test deprecated usage of %ct. Must catch warning.
        with pytest.warns(UserWarning):
            getattr(sm, magic)(line)
    else:
        # Normal usage, no warning.
        getattr(sm, magic)(line)
    actual = msgs.copy()
    msgs.clear()
    compare_msgs(actual, expected)
    if detectors:
        with pytest.warns(UserWarning):
            BlueskyMagics.detectors.clear()


def test_wa(hw):
    motor = hw.motor
    det = hw.motor
    ip = FakeIPython({'motor': motor, 'det': det})
    sm = BlueskyMagics(ip)
    # Test an empty list with no labels set.
    sm.wa('')

    # Test again with labeled objects.
    motor._ophyd_labels_ = ['motors']
    motor._ophyd_labels_ = ['detectors']

    # Test an empty list.
    sm.wa('')

    # Test with a label whitelist
    sm.wa('motors')
    sm.wa('motors detectors')
    sm.wa('motors typo')

    with pytest.raises(ValueError):
        sm.wa('[motors, detectors]')


# The %wa magic doesn't use a RunEngine or a plan.
def test_wa_legacy(hw):
    motor = hw.motor
    ip = FakeIPython({'motor': motor})
    sm = BlueskyMagics(ip)
    BlueskyMagics.positioners.extend([motor])
    with pytest.warns(UserWarning):
        sm.wa('')

    # Make motor support more attributes.
    motor.limits = (-1, 1)
    with pytest.warns(UserWarning):
        sm.wa('')
    motor.user_offset = SimpleNamespace(get=lambda: 0)

    with pytest.warns(UserWarning):
        sm.wa('[motor]')

    with pytest.warns(UserWarning):
        BlueskyMagics.positioners.clear()


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
