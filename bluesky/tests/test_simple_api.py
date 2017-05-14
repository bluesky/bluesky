import bson
from traitlets import TraitError
from bluesky.examples import motor, motor1, motor2, det, det1, det2, MockFlyer
import pytest
from bluesky.spec_api import (ct, ascan, a2scan, a3scan, dscan,
                              d2scan, d3scan, mesh, th2th, afermat,
                              fermat, spiral, aspiral)
import bluesky.plans as bp
import bluesky.callbacks as bc
from bluesky.callbacks.scientific import PeakStats


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
    fresh_RE(pln(*args, **kwargs), capture_run_start)

    assert run_start['plan_name'] == name
    assert gs.MD_TIME_KEY in run_start
    if 'time' in kwargs:
        assert run_start[gs.MD_TIME_KEY] == kwargs['time']

    # Ensure that the runstart document can be stored
    bson.BSON.encode(run_start)


def test_smart_dispatcher(fresh_RE):
    RE = fresh_RE

    D = bc.SmartDispatcher()
    RE(bp.count([det]), D)
    assert any(isinstance(cb, bc.LiveTable) for cb in D.active_cbs)
    for cb in D.active_cbs:
        if isinstance(cb, bc.LiveTable):
            table = cb
    assert not table._fields
    del table

    # Include D.ys in table
    D = bc.SmartDispatcher()
    D.ys.append('det')
    RE(bp.count([det]), D)
    assert any(isinstance(cb, bc.LiveTable) for cb in D.active_cbs)
    for cb in D.active_cbs:
        if isinstance(cb, bc.LiveTable):
            table = cb
    assert 'det' in table._fields
    # 1-point count plan should not get a plot
    assert not any(isinstance(cb, bc.LivePlot) for cb in D.active_cbs)
    del table

    # show a plot for count with > 1 point
    D = bc.SmartDispatcher()
    D.ys.append('det')
    RE(bp.count([det], num=2), D)
    assert any(isinstance(cb, bc.LiveTable) for cb in D.active_cbs)
    assert any(isinstance(cb, bc.LivePlot) for cb in D.active_cbs)
    for cb in D.active_cbs:
        if isinstance(cb, bc.LivePlot):
            plot = cb
    assert plot.x == bc.ExogenousVars.seq_num
    assert plot.y == 'det'
    del plot

    # plot against first field in motor by default
    D = bc.SmartDispatcher()
    D.ys.append('det')
    RE(bp.scan([det], motor, 1, 5, 5), D)
    assert any(isinstance(cb, bc.LiveTable) for cb in D.active_cbs)
    assert any(isinstance(cb, bc.LivePlot) for cb in D.active_cbs)
    for cb in D.active_cbs:
        if isinstance(cb, bc.LivePlot):
            plot = cb
        if isinstance(cb, bc.LiveTable):
            table = cb
    assert plot.x in ('motor_setpoint', 'motor')  # deterministic in 3.6
    assert plot.y == 'det'
    assert 'motor' in table._fields
    assert 'det' in table._fields
    del table, plot

    # plot against seq_num if specified
    D = bc.SmartDispatcher()
    D.ys.append('det')
    D.x = bc.ExogenousVars.seq_num
    RE(bp.scan([det], motor, 1, 5, 5), D)
    assert any(isinstance(cb, bc.LiveTable) for cb in D.active_cbs)
    assert any(isinstance(cb, bc.LivePlot) for cb in D.active_cbs)
    for cb in D.active_cbs:
        if isinstance(cb, bc.LivePlot):
            plot = cb
        if isinstance(cb, bc.LiveTable):
            table = cb
    assert plot.x == bc.ExogenousVars.seq_num
    assert plot.y == 'det'
    del table, plot

    # plot against time if specified
    D = bc.SmartDispatcher()
    D.ys.append('det')
    D.x = bc.ExogenousVars.time
    RE(bp.scan([det], motor, 1, 5, 5), D)
    assert any(isinstance(cb, bc.LiveTable) for cb in D.active_cbs)
    assert any(isinstance(cb, bc.LivePlot) for cb in D.active_cbs)
    for cb in D.active_cbs:
        if isinstance(cb, bc.LivePlot):
            plot = cb
        if isinstance(cb, bc.LiveTable):
            table = cb
    assert plot.x == bc.ExogenousVars.time
    assert plot.y == 'det'
    del table, plot

    # plot grid with axes guessed from first field in each motor
    D = bc.SmartDispatcher()
    D.ys.append('det')
    RE(bp.outer_product_scan([det], motor1, 1, 5, 5, motor2, 1, 3, 3, True), D)
    assert any(isinstance(cb, bc.LiveTable) for cb in D.active_cbs)
    assert not any(isinstance(cb, bc.LivePlot) for cb in D.active_cbs)
    assert any(isinstance(cb, bc.LiveGrid) for cb in D.active_cbs)
    for cb in D.active_cbs:
        if isinstance(cb, bc.LiveGrid):
            grid = cb
        if isinstance(cb, bc.LiveTable):
            table = cb
    assert grid.I == 'det'
    assert grid.ax.get_ylabel() in ('motor1_setpoint', 'motor1')  # as above
    assert grid.ax.get_xlabel() in ('motor2_setpoint', 'motor2')
    assert 'motor1' in table._fields
    assert 'motor2' in table._fields
    assert 'det' in table._fields
    del table, grid

    # peak stats
    D = bc.SmartDispatcher()
    D.ps_field = 'det'
    RE(bp.scan([det], motor, 1, 5, 5), D)
    assert any(isinstance(cb, PeakStats) for cb in D.active_cbs)
    assert D.ps.max

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
