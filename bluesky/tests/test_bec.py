import ast
import pytest
import jsonschema
from bluesky.plans import scan, grid_scan
import bluesky.preprocessors as bpp
import bluesky.plan_stubs as bps
from bluesky.preprocessors import SupplementalData
from bluesky.callbacks.best_effort import BestEffortCallback


def test_hints(RE, hw):
    motor = hw.motor
    expected_hint = {'fields': [motor.name]}
    assert motor.hints == expected_hint
    collector = []

    def collect(*args):
        collector.append(args)

    RE(scan([], motor, 1, 2, 2), {'descriptor': collect})
    name, doc = collector.pop()
    assert doc['hints'][motor.name] == expected_hint


def test_simple(RE, hw):
    bec = BestEffortCallback()
    RE.subscribe(bec)
    RE(scan([hw.ab_det], hw.motor, 1, 5, 5))


def test_disable(RE, hw):
    det, motor = hw.ab_det, hw.motor
    bec = BestEffortCallback()
    RE.subscribe(bec)

    bec.disable_table()

    RE(scan([det], motor, 1, 5, 5))
    assert bec._table is None

    bec.enable_table()

    RE(scan([det], motor, 1, 5, 5))
    assert bec._table is not None

    bec.peaks.com
    bec.peaks['com']
    assert ast.literal_eval(repr(bec.peaks)) == vars(bec.peaks)

    bec.clear()
    assert bec._table is None

    # smoke test
    bec.disable_plots()
    bec.enable_plots()
    bec.disable_baseline()
    bec.enable_baseline()
    bec.disable_heading()
    bec.enable_heading()


def test_blank_hints(RE, hw):
    bec = BestEffortCallback()
    RE.subscribe(bec)
    RE(scan([hw.ab_det], hw.motor, 1, 5, 5, md={'hints': {}}))


def test_with_baseline(RE, hw):
    bec = BestEffortCallback()
    RE.subscribe(bec)
    sd = SupplementalData(baseline=[hw.det])
    RE.preprocessors.append(sd)
    RE(scan([hw.ab_det], hw.motor, 1, 5, 5))


def test_underhinted_plan(RE, hw):
    bec = BestEffortCallback()
    RE.subscribe(bec)

    @bpp.run_decorator()
    def broken_plan(dets):
        yield from bps.trigger_and_read(dets)

    RE(broken_plan([hw.det]))


def test_live_grid(RE, hw):
    bec = BestEffortCallback()
    RE.subscribe(bec)
    RE(grid_scan([hw.det4], hw.motor1, 0, 1, 1, hw.motor2, 0, 1, 2, True))


@pytest.mark.xfail(not (jsonschema.__version__.split('.') < ['3', ]),
                   reason='Deprecations in jsonschema')
def test_plot_ints(RE):
    from ophyd import Signal
    from bluesky.callbacks.best_effort import BestEffortCallback
    from bluesky.plans import count
    import bluesky.plan_stubs as bps

    bec = BestEffortCallback()
    RE.subscribe(bec)

    s = Signal(name='s')
    RE(bps.mov(s, int(0)))
    assert s.describe()['s']['dtype'] == 'integer'
    s.kind = 'hinted'
    with pytest.warns(None) as record:
        RE(count([s], num=35))

    assert len(record) == 0
