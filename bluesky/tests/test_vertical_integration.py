from collections import defaultdict
from bluesky.global_state import gs
from bluesky.examples import stepscan, det, motor
import pytest


def setup_module(module):
    try:
        import metadatastore
    except ImportError as ie:
        raise pytest.skip('ImportError: {0}'.format(ie))
    else:
        from metadatastore.test.utils import mds_setup
        mds_setup()
        gs.RE.subscribe_lossless('all', metadatastore.commands.insert)

def teardown_module(module):
    from metadatastore.test.utils import mds_teardown
    mds_teardown()


def test_scan_and_get_data():
    try:
        from databroker import DataBroker as db
        from bluesky.global_state import gs
    except ImportError as ie:
        raise pytest.skip('skipping because some libary is unavailable\n'
                          'ImportError:  is:{}'.format(ie))
    uid, = gs.RE(stepscan(det, motor), group='foo', beamline_id='testing',
             config={})

    hdr = db[uid]
    db.fetch_events(hdr)


def test_post_run():
    try:
        import databroker
        from bluesky.global_state import gs
        from bluesky.broker_callbacks import post_run
    except ImportError as ie:
        raise pytest.skip('skipping because some libary is unavailable\n'
                          'ImportError:  is:{}'.format(ie))
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


def test_verify_files_saved():
    try:
        import databroker
        from bluesky.global_state import gs
        from bluesky.broker_callbacks import verify_files_saved
    except ImportError as ie:
        raise pytest.skip('skipping because some libary is unavailable\n'
                          'ImportError:  is:{}'.format(ie))
    gs.RE(stepscan(det, motor), subs={'stop': verify_files_saved})
