from collections import defaultdict
from bluesky.examples import stepscan, det, motor
from bluesky.callbacks.broker import post_run, verify_files_saved
from functools import partial


def test_scan_and_get_data(fresh_RE, db):
    RE = fresh_RE
    RE.subscribe('all', db.mds.insert)
    uid, = RE(stepscan(det, motor), group='foo', beamline_id='testing',
              config={})

    hdr = db[uid]
    db.fetch_events(hdr)


def test_post_run(fresh_RE, db):
    RE = fresh_RE
    RE.subscribe('all', db.mds.insert)
    output = defaultdict(list)

    def do_nothing(doctype, doc):
        output[doctype].append(doc)

    RE.ignore_callback_exceptions = False

    RE(stepscan(det, motor), subs={'stop': [post_run(do_nothing, db=db)]})
    assert len(output)
    assert len(output['start']) == 1
    assert len(output['stop']) == 1
    assert len(output['descriptor']) == 1
    assert len(output['event']) == 10


def test_verify_files_saved(fresh_RE, db):
    RE = fresh_RE
    RE.subscribe('all', db.mds.insert)

    vfs = partial(verify_files_saved, db=db)
    RE(stepscan(det, motor), subs={'stop': vfs})
