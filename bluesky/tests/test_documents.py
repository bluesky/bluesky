from bluesky.run_engine import RunEngine
from bluesky.tests.utils import setup_test_run_engine
from bluesky.examples import simple_scan, motor


RE = setup_test_run_engine()


def test_custom_metadata():
    def assert_lion(name, doc):
        assert 'animal' in doc
        assert doc['animal'] == 'lion'

    RE(simple_scan(motor), animal='lion', subs={'start': assert_lion})
    # Note: Because assert_lion is processed on the main thread, it can
    # fail the test. I checked by writing a failing version of it.  - D.A.
