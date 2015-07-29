from nose.tools import assert_true
from bluesky.scans import DeltaScan
from bluesky.examples import motor, det
from bluesky.tests.utils import setup_test_run_engine


def test_dscan_logbook():
    result = {}
    def logbook(m, d):
        result['msg'] = m

    RE = setup_test_run_engine()
    RE.logbook = logbook
    d = DeltaScan([det], motor, 1, 2, 2)
    RE(d)
    assert_true(result['msg'].startswith(EXPECTED_FORMAT_STR))
    # order of the rest of the msg (metadata) is not deterministic


EXPECTED_FORMAT_STR = 'Header uid: {uid}\n\nScan Plan\n---------\nScan Class: {scn_cls}\n\ndetectors: {detectors!r}\nmotor: {motor!r}\nstart: {start!r}\nstop: {stop!r}\nnum: {num!r}\n\nTo call:\n{motor!r}.set({init_pos})\nRE({scn_cls}(detectors={detectors!r}, motor={motor!r}, start={start!r}, stop={stop!r}, num={num!r}))\n\n'
