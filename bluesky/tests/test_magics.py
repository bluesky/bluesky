from bluesky.magics import SPECMagics, BlueskyMagics
import bluesky.plans as bp
from bluesky.examples import det, motor1, motor2
import pytest


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

@pytest.mark.parametrize('pln,magic,line', [
    (bp.mv(motor1, 2), 'mov', 'motor1 2'),
    (bp.mv(motor1, 2, motor2, 3), 'mov', 'motor1 2 motor2 3'),
    (bp.mvr(motor1, 2), 'movr', 'motor1 2'),
    (bp.mvr(motor1, 2, motor2, 3), 'movr', 'motor1 2 motor2 3'),
    ])
def test_bluesky_magics(pln, line, magic, fresh_RE):
    RE = fresh_RE

    # Spy on all msgs processed by RE.
    msgs = []

    def collect(msg):
        msgs.append(msg)

    RE.msg_hook = collect

    # Build a FakeIPython instance to use the magics with.

    ip = FakeIPython({'RE': RE, 'dets': dets, 'motor1': motor1, 'motor2':
                      motor2})
    sm = BlueskyMagics(ip)

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

@pytest.mark.parametrize('pln,magic,line', [
    (bp.count(dets), 'ct', ''),
    (bp.scan(dets, motor1, 1, 2, 1 + 3), 'ascan', 'motor1 1 2 3'),
    (bp.relative_scan(dets, motor1, 1, 2, 1 + 3), 'dscan', 'motor1 1 2 3'),
    (bp.inner_product_scan(dets, 1 + 2, motor1, 1, 3, motor2, 4, 6), 'a2scan',
     'motor1 1 3 motor2 4 6 2'),
    (bp.relative_inner_product_scan(dets, 1 + 2, motor1, 1, 3, motor2, 4, 6),
     'd2scan', 'motor1 1 3 motor2 4 6 2'),
    (bp.outer_product_scan(dets, motor1, 1, 3, 1 + 2, motor2, 4, 6, 1 + 5,
                           False),
     'mesh', 'motor1 1 3 2 motor2 4 6 5'),
    ])
def test_spec_magics(pln, line, magic, fresh_RE):
    RE = fresh_RE

    # Spy on all msgs processed by RE.
    msgs = []

    def collect(msg):
        msgs.append(msg)

    RE.msg_hook = collect

    # Build a FakeIPython instance to use the magics with.

    ip = FakeIPython({'RE': RE, 'dets': dets, 'motor1': motor1, 'motor2':
                      motor2})
    sm = SPECMagics(ip)

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


@pytest.mark.parametrize('pln,magic,line', [
    (bp.count(dets), 'ct', 'foo'),
    (bp.scan(dets, motor1, 1, 2, 1 + 3), 'ascan', 'motor1 1 2 3 wrong'),
    (bp.relative_scan(dets, motor1, 1, 2, 1 + 3), 'dscan', 'motor1 1 2 3 wrong'),
    (bp.inner_product_scan(dets, 2, motor1, 1, 3, motor2, 4, 6), 'a2scan',
     'motor1 1 3 motor2 4 wrong'),
    (bp.relative_inner_product_scan(dets, 2, motor1, 1, 3, motor2, 4, 6),
     'd2scan', 'motor1 1 3 motor2 4 wrong'),
    (bp.outer_product_scan(dets, motor1, 1, 3, 2, motor2, 4, 6, 5, False),
     'mesh', 'motor1 1 3 2 motor2 4 6 5 wrong'),
    ])
def test_spec_magics_bad_args(pln, line, magic, fresh_RE):
    RE = fresh_RE

    # Spy on all msgs processed by RE.
    msgs = []

    def collect(msg):
        msgs.append(msg)

    RE.msg_hook = collect

    # Build a FakeIPython instance to use the magics with.

    ip = FakeIPython({'RE': RE, 'dets': dets, 'motor1': motor1, 'motor2':
                      motor2})
    sm = SPECMagics(ip)

    # Test magics cause the RunEngine to execute the messages we expect.
    with pytest.raises(TypeError):
        getattr(sm, magic)(line)


# The %wa magic doesn't use a RunEngine or a plan.
def test_wa():
    from bluesky.examples import motor
    ip = FakeIPython({})
    sm = SPECMagics(ip)
    # Test an empty list.
    sm.wa('')

    sm.positioners.extend([motor])
    sm.wa('')


def test_magics_missing_ns_key(fresh_RE):
    RE = fresh_RE
    ip = FakeIPython({})
    sm = BlueskyMagics(ip)
    with pytest.raises(KeyError):
        sm.mov('motor1 5')
    ip.user_ns['RE'] = RE
    with pytest.raises(KeyError):
        sm.mov('motor1 5')
    ip.user_ns['motor1'] = motor1
    sm.mov('motor1 5')
