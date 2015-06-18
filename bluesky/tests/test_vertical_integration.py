
from metadatastore.utils.testing import mds_setup, mds_teardown
from dataportal import DataBroker as db
from bluesky.examples import *
from bluesky.standard_config import RE

def setup():
    mds_setup()

def teardown():
    mds_teardown()


def test_scan_and_get_data():
    uid = RE(stepscan(det, motor), group='foo', beamline_id='testing',
             config={})

    hdr = db[uid]
    ev = db.fetch_events(hdr)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
