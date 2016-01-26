from collections import defaultdict
from bluesky.examples import stepscan, det, motor
import pytest


def setup_module():
    try:
        from metadatastore.utils.testing import mds_setup
    except ImportError:
        pass  # test will be skipped
    else:
        mds_setup()

def teardown_module():
    try:
        from metadatastore.utils.testing import mds_teardown
    except ImportError:
        pass  # test will be skipped
    else:
        mds_teardown()


def test_scan_and_get_data():
    try:
        import metadatastore
        del metadatastore
    except ImportError as ie:
        raise pytest.skip('skipping because metadatastore is not available\nMessage is: {}'.format(ie))
    try:
        from databroker import DataBroker as db
    except ImportError as ie:
        raise pytest.skip('skipping because databroker is not available\nMessage is:{}'.format(ie))
    from bluesky.standard_config import gs
    uid = gs.RE(stepscan(det, motor), group='foo', beamline_id='testing',
             config={})

    hdr = db[uid]
    db.fetch_events(hdr)


def test_post_run():
    try:
        import databroker
        del databroker
    except ImportError as ie:
        raise pytest.skip('skipping because databroker is not available\nMessage is:{}'.format(ie))
    from bluesky.standard_config import gs
    from bluesky.broker_callbacks import post_run
    output = defaultdict(list)
    def do_nothing(doctype, doc):
        output[doctype].append(doc)

    gs.RE.ignore_callback_exceptions = False

    gs.RE(stepscan(det, motor), subs={'stop': [post_run(do_nothing)]})
    assert len(output)
    assert len(output['start']) == 1
    assert len(output['stop']) == 1
    assert len(output['descriptor']) == 1
    assert len(output['event']) == 10
