from collections import defaultdict
from functools import partial

from bluesky.callbacks.broker import post_run, verify_files_saved
from bluesky.plans import list_scan


def stepscan(det, motor):
    yield from list_scan([det], motor, list(range(-5, 5)))


def test_scan_and_get_data(RE, hw, db):
    RE.subscribe(db.insert)
    rs = RE(stepscan(hw.det, hw.motor), group="foo", beamline_id="testing", config={})
    if RE.call_returns_result:
        uid = rs.run_start_uids[0]
    else:
        uid = rs[0]
    hdr = db[uid]
    list(hdr.events())


def test_post_run(RE, hw, db):
    RE.subscribe(db.insert)
    output = defaultdict(list)

    def do_nothing(doctype, doc):
        output[doctype].append(doc)

    RE(stepscan(hw.det, hw.motor), {"stop": [post_run(do_nothing, db=db)]})
    assert len(output)
    assert len(output["start"]) == 1
    assert len(output["stop"]) == 1
    assert len(output["descriptor"]) == 1
    assert len(output["event"]) == 10


def test_verify_files_saved(RE, hw, db):
    RE.subscribe(db.insert)

    vfs = partial(verify_files_saved, db=db)
    RE(stepscan(hw.det, hw.motor), {"stop": vfs})
