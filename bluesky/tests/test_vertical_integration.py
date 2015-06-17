
import time as ttime
from metadatastore.utils.testing import mds_setup, mds_teardown
from metadatastore.api import (insert_run_start, insert_beamline_config,
                               find_run_stops)
from dataportal import DataBroker as db
from dataportal.examples.sample_data.image_and_scalar import run
from bluesky.examples import *
from bluesky.standard_config import RE
from bluesky.broker_callbacks import (validate_all_events,
                                      validate_all_resources, post_run)


def setup():
    mds_setup()


def teardown():
    mds_teardown()


def _generate_run_start():
    """Helper function to generate a run_start document because it is rather
    cumbersome. This should get moved to metadatastore testing utils eventually
    """
    blc_uid = insert_beamline_config(config_params={}, time=ttime.time())
    run_start_uid = insert_run_start(time=ttime.time(),
                                     beamline_id='bluesky-testing',
                                     beamline_config=blc_uid,
                                     scan_id=42)
    return blc_uid, run_start_uid


def test_scan_and_get_data():
    uid = RE(stepscan(motor, det), group='foo', beamline_id='testing',
             config={})

    hdr = db[uid]
    ev = db.fetch_events(hdr)


def _caller(function, document):
    function(document)

def test_dataportal_callbacks():
    blc_uid, run_start_uid = _generate_run_start()
    # insert some data that pokes filestore
    run(run_start_uid=run_start_uid)
    run_stop, = find_run_stops(run_start=run_start_uid)
    funcs = validate_all_resources, validate_all_events, post_run
    for func in funcs:
        yield _caller, func, run_stop


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
