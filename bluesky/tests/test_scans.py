from itertools import count
from nose.tools import assert_equal
from bluesky.scans import *
from bluesky import RunEngine
from bluesky.examples import motor, det


RE = None


def setup():
    global RE
    RE = RunEngine()


class CallbackCounter:
    "As simple as it sounds: count how many times a callback is called."
    # Wrap itertools.count in something we can use as a callback.
    def __init__(self):
        self.counter = count()
        self(None)  # Start counting at 1.

    def __call__(self, doc):
        self.value = next(self.counter)


def test_linascan():
    counter = CallbackCounter()
    s = LinAscan(motor, [det], 0, 10, 5)
    RE(s, subs={'event': counter})
    assert_equal(counter.value, 5)
    s.num = 3
    counter = CallbackCounter()
    RE(s, subs={'event': counter})
    assert_equal(counter.value, 3)
