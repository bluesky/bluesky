from bluesky import preprocessors as bpp
from bluesky import plans as bp
from bluesky import plan_stubs as bps
from bluesky.preprocessors import define_run_wrapper as drw
from ophyd.sim import motor, det
from bluesky.tests.utils import DocCollector


def test_multirun_smoke(RE, hw):
    dc = DocCollector()

    def interlaced_plan(dets, motor):
        to_read = (motor, *dets)
        run_ids = list("abc")
        for rid in run_ids:
            yield from drw(bps.open_run(md={rid: rid}), run_id=rid)

        for j in range(5):
            for i, rid in enumerate(run_ids):
                yield from bps.mov(motor, j + 0.1 * i)
                yield from drw(bps.trigger_and_read(to_read), run_id=rid)

        for rid in run_ids:
            yield from drw(bps.close_run(), run_id=rid)

    RE(interlaced_plan([hw.det], hw.motor), dc.insert)

    assert len(dc.start) == 3
    for start in dc.start:
        desc, = dc.descriptor[start["uid"]]
        assert len(dc.event[desc["uid"]]) == 5

    for stop in dc.stop.values():
        for start in dc.start:
            assert start["time"] < stop["time"]


def test_multirun_smoke(RE, hw):
    dc = DocCollector()

    def interlaced_plan(dets, motor):
        to_read = (motor, *dets)
        run_ids = list("abc")
        for rid in run_ids:
            yield from drw(bps.open_run(md={rid: rid}), run_id=rid)
        raise Exception('womp womp')

    with pytest.raises(Exception):
        RE(interlaced_plan([hw.det], hw.motor), dc.insert)


    assert len(dc.start) == len(dc.stop)
    assert len(dc.start) == 3

    for v in dc.stop.values():
        assert v['exit_stats'] == 'fail'
        assert v['reason'] == 'womp womp'
