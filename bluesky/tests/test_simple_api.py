import bson
from traitlets import TraitError
from bluesky.examples import motor, motor1, motor2, det, det1, det2, MockFlyer
import pytest
from bluesky.spec_api import (ct, ascan, a2scan, a3scan, dscan,
                              d2scan, d3scan, mesh, th2th, afermat,
                              fermat, spiral, aspiral,
                              get_factory_input,
                              plan_sub_factory_input, InvalidFactory)


@pytest.mark.parametrize('pln,name,args,kwargs', [
    (ct, 'ct', (), {}),
    (ct, 'ct', (), {'num': 3}),
    (ascan, 'ascan', (motor, 1, 2, 2), {}),
    (a2scan, 'a2scan', (motor, 1, 2, 2), {}),
    (a3scan, 'a3scan', (motor, 1, 2, 2), {}),
    (dscan, 'dscan', (motor, 1, 2, 2), {}),
    (d2scan, 'd2scan', (motor, 1, 2, 2), {}),
    (d3scan, 'd3scan', (motor, 1, 2, 2), {}),
    (mesh, 'mesh', (motor1, 1, 2, 2, motor2, 1, 2, 3), {}),
    (th2th, 'th2th', (1, 2, 2), {}),
    (aspiral, 'aspiral', (motor1, motor2, 0.0, 0.0, 0.1, 0.1, 0.05, 1.0), {}),
    (spiral, 'spiral', (motor1, motor2, 0.1, 0.1, 0.05, 1.0), {}),
    (afermat, 'afermat', (motor1, motor2, 0.0, 0.0, 0.1, 0.1, 0.05, 1.0), {}),
    (fermat, 'fermat', (motor1, motor2, 0.1, 0.1, 0.05, 1.0), {}),
    # with count time specified as keyword arg
    (ct, 'ct', (), {'time': 0.1}),
    (ascan, 'ascan', (motor, 1, 2, 2), {'time': 0.1}),
    (a2scan, 'a2scan', (motor, 1, 2, 2), {'time': 0.1}),
    (a3scan, 'a3scan', (motor, 1, 2, 2), {'time': 0.1}),
    (dscan, 'dscan', (motor, 1, 2, 2), {'time': 0.1}),
    (d2scan, 'd2scan', (motor, 1, 2, 2), {'time': 0.1}),
    (d3scan, 'd3scan', (motor, 1, 2, 2), {'time': 0.1}),
    (mesh, 'mesh', (motor1, 1, 2, 2, motor2, 1, 2, 3), {'time': 0.1}),
    (th2th, 'th2th', (1, 2, 2), {'time': 0.1}),
    (aspiral, 'aspiral',
     (motor1, motor2, 0.0, 0.0, 0.1, 0.1, 0.05, 1.0), {'time': 0.1}),
    (spiral, 'spiral', (motor1, motor2, 0.1, 0.1, 0.05, 1.0), {'time': 0.1}),
    (afermat, 'afermat',
     (motor1, motor2, 0.0, 0.0, 0.1, 0.1, 0.05, 1.0), {'time': 0.1}),
    (fermat, 'fermat',
     (motor1, motor2, 0.1, 0.1, 0.05, 1.0), {'time': 0.1})])
def test_spec_plans(fresh_RE, pln, name, args, kwargs):
    from bluesky.global_state import gs

    gs.DETS = [det]
    gs.TH_MOTOR = motor1
    gs.TTH_MOTOR = motor2
    run_start = None
    gs.MASTER_DET_FIELD = list(det._fields)[0]
    gs.MD_TIME_KEY = 'exposure_time'

    def capture_run_start(name, doc):
        nonlocal run_start
        if name == 'start':
            run_start = doc
    gs.SUB_FACTORIES[pln.__name__].append(lambda: capture_run_start)
    fresh_RE(pln(*args, **kwargs))
    gs.SUB_FACTORIES[pln.__name__].pop()

    assert run_start['plan_name'] == name
    assert gs.MD_TIME_KEY in run_start
    if 'time' in kwargs:
        assert run_start[gs.MD_TIME_KEY] == kwargs['time']

    # Ensure that the runstart document can be stored
    bson.BSON.encode(run_start)


def test_flyers(fresh_RE):
    from bluesky.global_state import gs
    RE = fresh_RE
    flyer = MockFlyer('wheee', det1, motor, -1, 1, 15, RE.loop)
    gs.FLYERS = [flyer]
    RE(ct())


def test_gs_validation():
    from bluesky.global_state import gs
    with pytest.raises(TraitError):
        gs.DETS = [det, det]  # no duplicate data keys


def test_factory_tools_smoke():
    from bluesky.global_state import gs
    from inspect import Signature, Parameter

    def failer(*args, **kwargs):
        ...

    failer.__signature__ = Signature((Parameter('fail',
                                                Parameter.POSITIONAL_ONLY), ))
    with pytest.raises(InvalidFactory):
        get_factory_input(failer)

    for pln in ['ct', 'ascan', 'dscan']:
        merge, by_fac = plan_sub_factory_input(pln)
        opt = set()
        req = set()
        for k, v in by_fac.items():
            opt.update(v.opt)
            req.update(v.req)

        assert opt == merge.opt
        assert req == merge.req

    gs.SUB_FACTORIES['TST_DUMMY'] = [failer]

    merge, by_fac = plan_sub_factory_input('TST_DUMMY')
    for k, v in by_fac.items():
        assert isinstance(v, InvalidFactory)
