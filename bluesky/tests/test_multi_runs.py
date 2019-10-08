from bluesky import preprocessors as bpp
from bluesky import plans as bp
from bluesky import plan_stubs as bps
from bluesky.preprocessors import define_run_wrapper as drw
from bluesky.tests.utils import DocCollector
import pytest


def test_multirun_smoke(RE, hw):
    dc = DocCollector()

    def interlaced_plan(dets, motor):
        to_read = (motor, *dets)
        run_ids = list("abc")
        for rid in run_ids:
            yield from drw(bps.open_run(md={rid: rid}), run=rid)

        for j in range(5):
            for i, rid in enumerate(run_ids):
                yield from bps.mov(motor, j + 0.1 * i)
                yield from drw(bps.trigger_and_read(to_read), run=rid)

        for rid in run_ids:
            yield from drw(bps.close_run(), run=rid)

    RE(interlaced_plan([hw.det], hw.motor), dc.insert)

    assert len(dc.start) == 3
    for start in dc.start:
        desc, = dc.descriptor[start["uid"]]
        assert len(dc.event[desc["uid"]]) == 5

    for stop in dc.stop.values():
        for start in dc.start:
            assert start["time"] < stop["time"]


def test_multirun_smoke_fail(RE, hw):
    dc = DocCollector()

    def interlaced_plan(dets, motor):
        run_ids = list("abc")
        for rid in run_ids:
            yield from drw(bps.open_run(md={rid: rid}), run=rid)
        raise Exception("womp womp")

    with pytest.raises(Exception):
        RE(interlaced_plan([hw.det], hw.motor), dc.insert)

    assert len(dc.start) == len(dc.stop)
    assert len(dc.start) == 3

    for v in dc.stop.values():
        assert v["exit_status"] == "fail"
        assert v["reason"] == "womp womp"
