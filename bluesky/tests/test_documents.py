from nose.tools import assert_in, assert_equal
from history import History
from bluesky.run_engine import RunEngine
from bluesky.examples import simple_scan, motor


RE = None


def setup():
    global RE
    h = History(':memory:')
    RE = RunEngine(h)
    RE.memory.put('owner', 'test_owner')
    RE.memory.put('beamline_id', 'test_beamline')


def test_custom_metadata():
    def assert_lion(doc):
        assert_in('animal', doc)
        assert_equal(doc['animal'], 'lion')

    RE(simple_scan(motor), animal='lion', subs={'start': assert_lion})
    # Note: Because assert_lion is processed on the main thread, it can
    # fail the test. I checked by writing a failing version of it.  - D.A.
