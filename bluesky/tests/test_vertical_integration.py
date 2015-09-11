from nose import SkipTest
from dataportal import DataBroker as db
from bluesky.examples import stepscan, det, motor
from bluesky.standard_config import gs

def setup():
    try:
        from metadatastore.utils.testing import mds_setup
    except ImportError:
        pass  # test will be skipped
    else:
        mds_setup()

def teardown():
    try:
        from metadatastore.utils.testing import mds_teardown
    except ImportError:
        pass  # test will be skipped
    else:
        mds_teardown()


def test_scan_and_get_data():
    try:
        import metadatastore
    except ImportError:
        raise SkipTest("metadatastore is not installed")
    uid = gs.RE(stepscan(det, motor), group='foo', beamline_id='testing',
             config={})

    hdr = db[uid]
    db.fetch_events(hdr)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
