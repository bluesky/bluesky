from collections import defaultdict
from nose import SkipTest
from nose.tools import assert_equal
from bluesky.examples import stepscan, det, motor

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
        del metadatastore
    except ImportError:
        raise SkipTest
    try:
        from databroker import DataBroker as db
    except ImportError:
        raise SkipTest
    from bluesky.standard_config import gs
    uid = gs.RE(stepscan(det, motor), group='foo', beamline_id='testing',
             config={})

    hdr = db[uid]
    db.fetch_events(hdr)


def test_post_run():
    try:
        import databroker
        del databroker
    except ImportError:
        raise SkipTest('requires databroker')
    from bluesky.standard_config import gs
    from bluesky.broker_callbacks import post_run
    output = defaultdict(list)
    def do_nothing(doctype, doc):
        output[doctype].append(doc)

    gs.RE.ignore_callback_exceptions = False

    gs.RE(stepscan(det, motor), subs={'stop': [post_run(do_nothing)]})
    assert len(output)
    assert_equal(len(output['start']), 1)
    assert_equal(len(output['stop']), 1)
    assert_equal(len(output['descriptor']), 1)
    assert_equal(len(output['event']), 10)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
