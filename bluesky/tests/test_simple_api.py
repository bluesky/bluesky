from traitlets import TraitError
from bluesky.examples import motor, motor1, motor2, det, det1, det2, FlyMagic
import pytest

def test_basic_usage_for_smoke():
    try:
        import metadatastore
    except ImportError as ie:
        pytest.skip('metadatastore is not installed. Cannot run basic usage '
                    'test. ImportError: {}'.format(ie))
    from bluesky.global_state import gs
    from bluesky.spec_api import (ct, ascan, a2scan, a3scan, dscan, d2scan,
                                  d3scan, mesh, th2th, afermat, fermat, spiral,
                                  aspiral)
    gs.DETS = [det]
    with pytest.raises(TraitError):
        gs.DETS = [det, det]  # no duplicate data keys
    gs.TH_MOTOR = motor1
    gs.TTH_MOTOR = motor2
    gs.RE.md['owner'] = 'test'
    gs.RE.md['group'] = 'test'
    gs.RE.md['beamline_id'] = 'test'
    gs.RE.md['config'] = {}
    # without count time specified
    RE = gs.RE
    RE(ct())
    RE(ct(num=3))  # passing kwargs through to scan
    RE(ascan(motor, 1, 2, 2))
    RE(a2scan(motor, 1, 2, 2))
    RE(a3scan(motor, 1, 2, 2))
    RE(dscan(motor, 1, 2, 2))
    RE(d2scan(motor, 1, 2, 2))
    RE(d3scan(motor, 1, 2, 2))
    RE(mesh(motor1, 1, 2, 2, motor2, 1, 2, 3))
    RE(th2th(1, 2, 2))
    RE(aspiral(motor1, motor2, 0.0, 0.0, 0.1, 0.1, 0.05, 1.0))
    RE(spiral(motor1, motor2, 0.1, 0.1, 0.05, 1.0))
    RE(afermat(motor1, motor2, 0.0, 0.0, 0.1, 0.1, 0.05, 1.0))
    RE(fermat(motor1, motor2, 0.1, 0.1, 0.05, 1.0))
    # with count time specified as keyword arg
    RE(ct())
    RE(ascan(motor, 1, 2, 2, time=0.1))
    RE(a2scan(motor, 1, 2, 2, time=0.1))
    RE(a3scan(motor, 1, 2, 2, time=0.1))
    RE(dscan(motor, 1, 2, 2, time=0.1))
    RE(d2scan(motor, 1, 2, 2, time=0.1))
    RE(d3scan(motor, 1, 2, 2, time=0.1))
    RE(mesh(motor1, 1, 2, 2, motor2, 1, 2, 3, time=0.1))
    RE(th2th(1, 2, 2, time=0.1))
    RE(aspiral(motor1, motor2, 0.0, 0.0, 0.1, 0.1, 0.05, 1.0, time=0.1))
    RE(spiral(motor1, motor2, 0.1, 0.1, 0.05, 1.0, time=0.1))
    RE(afermat(motor1, motor2, 0.0, 0.0, 0.1, 0.1, 0.05, 1.0, time=0.1))
    RE(fermat(motor1, motor2, 0.1, 0.1, 0.05, 1.0, time=0.1))
    flyer = FlyMagic('wheee', motor, det1, det2)
    gs.FLYERS = [flyer]
    RE(ct())
