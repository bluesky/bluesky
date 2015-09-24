import nose
from nose.tools import assert_raises
from traitlets import TraitError
from bluesky.examples import motor, motor1, motor2, det


def test_basic_usage():
    try:
        import metadatastore
    except ImportError:
        raise nose.SkipTest
    from bluesky.global_state import gs
    from bluesky.spec_api import (ct, ascan, a2scan, a3scan, dscan, d2scan,
                                  d3scan, mesh, tscan, dtscan, th2th)
    gs.DETS = [det]
    with assert_raises(TraitError):
        gs.DETS = [det, det]  # no duplicate data keys
    gs.TEMP_CONTROLLER = motor
    gs.TH_MOTOR = motor1
    gs.TTH_MOTOR = motor2
    gs.RE.md['group'] = 'test'
    gs.RE.md['beamline_id'] = 'test'
    gs.RE.md['config'] = {}
    # without count time specified
    ct()
    ct(num=3)  # passing kwargs through to scan
    ascan(motor, 1, 2, 2)
    a2scan(motor, 1, 2, 2)
    a3scan(motor, 1, 2, 2)
    dscan(motor, 1, 2, 2)
    d2scan(motor, 1, 2, 2)
    d3scan(motor, 1, 2, 2)
    mesh(motor1, 1, 2, 2, motor2, 1, 2, 3)
    tscan(1, 2, 2)
    dtscan(1, 2, 2)
    th2th(1, 2, 2)
    # with count time specified as positional arg
    ct()
    ascan(motor, 1, 2, 2, 0.1)
    a2scan(motor, 1, 2, 2, 0.1)
    a3scan(motor, 1, 2, 2, 0.1)
    dscan(motor, 1, 2, 2, 0.1)
    d2scan(motor, 1, 2, 2, 0.1)
    d3scan(motor, 1, 2, 2, 0.1)
    mesh(motor1, 1, 2, 2, motor2, 1, 2, 3, 0.1)
    tscan(1, 2, 2, 0.1)
    dtscan(1, 2, 2, 0.1)
    th2th(1, 2, 2, 0.1)
    # with count time specified as keyword arg
    ct()
    ascan(motor, 1, 2, 2, time=0.1)
    a2scan(motor, 1, 2, 2, time=0.1)
    a3scan(motor, 1, 2, 2, time=0.1)
    dscan(motor, 1, 2, 2, time=0.1)
    d2scan(motor, 1, 2, 2, time=0.1)
    d3scan(motor, 1, 2, 2, time=0.1)
    mesh(motor1, 1, 2, 2, motor2, 1, 2, 3, time=0.1)
    tscan(1, 2, 2, time=0.1)
    dtscan(1, 2, 2, time=0.1)
    th2th(1, 2, 2, time=0.1)
