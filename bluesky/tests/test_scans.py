from nose.tools import assert_equal
from bluesky.scans import *
from bluesky.callbacks import *
from bluesky import RunEngine
from bluesky.examples import motor, det


RE = None


def setup():
    global RE
    RE = RunEngine()


def test_linascan():
    counter = CallbackCounter()
    s = LinAscan(motor, [det], 0, 10, 5)
    RE(s, subs={'event': counter})
    assert_equal(counter.value, 5)
    s.num = 3
    counter = CallbackCounter()
    RE(s, subs={'event': counter})
    assert_equal(counter.value, 3)
