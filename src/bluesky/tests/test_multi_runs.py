import pytest

import bluesky.preprocessors as bsp
from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from bluesky.preprocessors import set_run_key_wrapper as srkw
from bluesky.tests.utils import DocCollector


def test_multirun_smoke(RE, hw):
    """Test on interlaced runs (using wrapper on each command)"""
    dc = DocCollector()

    def interlaced_plan(dets, motor):
        to_read = (motor, *dets)
        run_names = ["run_one", "run_two", "run_three"]
        for rid in run_names:
            yield from srkw(bps.open_run(md={rid: rid}), run=rid)
            yield from srkw(bps.declare_stream(*to_read, name="primary"), run=rid)
        for j in range(5):
            for i, rid in enumerate(run_names):
                yield from bps.mov(motor, j + 0.1 * i)
                yield from srkw(bps.trigger_and_read(to_read), run=rid)

        for rid in run_names:
            yield from srkw(bps.close_run(), run=rid)

    RE(interlaced_plan([hw.det], hw.motor), dc.insert)

    assert len(dc.start) == 3
    for start in dc.start:
        (desc,) = dc.descriptor[start["uid"]]
        assert len(dc.event[desc["uid"]]) == 5

    for stop in dc.stop.values():
        for start in dc.start:
            assert start["time"] < stop["time"]


def test_multirun_smoke_nested(RE, hw):
    """Test on nested runs (using decorator on each plan)"""
    dc = DocCollector()
    to_read = (hw.motor, hw.det)

    def some_plan():
        """This plan is called on each level of nesting"""
        for j in range(5):
            yield from bps.mov(hw.motor, j)
            yield from bps.trigger_and_read(to_read)

    @bsp.set_run_key_decorator("run_one")
    @bsp.run_decorator(md={})
    def plan_inner():
        yield from bps.declare_stream(*to_read, name="primary")
        yield from some_plan()

    @bsp.set_run_key_decorator("run_two")
    @bsp.run_decorator(md={})
    def plan_middle():
        yield from bps.declare_stream(*to_read, name="primary")
        yield from some_plan()
        yield from plan_inner()

    @bsp.set_run_key_decorator(run="run_three")  # Try kwarg
    @bsp.run_decorator(md={})
    def plan_outer():
        yield from bps.declare_stream(*to_read, name="primary")
        yield from some_plan()
        yield from plan_middle()

    RE(plan_outer(), dc.insert)

    assert len(dc.start) == 3
    for start in dc.start:
        (desc,) = dc.descriptor[start["uid"]]
        assert len(dc.event[desc["uid"]]) == 5

    for stop in dc.stop.values():
        for start in dc.start:
            assert start["time"] < stop["time"]


def test_multirun_run_key_type(RE, hw):
    """Test calls to wrapper with run key set to different types"""

    dc = DocCollector()

    @bsp.run_decorator(md={})
    def empty_plan():
        yield from bps.mov(hw.motor, 5)

    # The wrapper is expected to raise an exception if called with run ID = None
    with pytest.raises(ValueError, match="run ID can not be None"):

        def plan1():
            yield from srkw(empty_plan(), None)

        RE(plan1(), dc.insert)

    # Check with run ID of type reference
    def plan2():
        yield from srkw(empty_plan(), object())

    RE(plan2(), dc.insert)

    # Check with run ID of type 'int'
    def plan3():
        yield from srkw(empty_plan(), 10)

    RE(plan3(), dc.insert)

    # Check if call with correct parameter type are successful
    def plan4():
        yield from srkw(empty_plan(), "run_name")

    RE(plan4(), dc.insert)

    def plan5():
        yield from srkw(empty_plan(), run="run_name")

    RE(plan5(), dc.insert)


def test_multirun_smoke_fail(RE, hw):
    dc = DocCollector()

    def interlaced_plan(dets, motor):
        run_names = ["run_one", "run_two", "run_three"]
        for rid in run_names:
            yield from srkw(bps.open_run(md={rid: rid}), run=rid)
        raise Exception("womp womp")

    with pytest.raises(Exception):  # noqa: B017
        RE(interlaced_plan([hw.det], hw.motor), dc.insert)

    assert len(dc.start) == len(dc.stop)
    assert len(dc.start) == 3

    for v in dc.stop.values():
        assert v["exit_status"] == "fail"
        assert v["reason"] == "womp womp"


def test_multirun_baseline(RE, hw):
    det1, det2, det3 = hw.det1, hw.det2, hw.det3

    @bpp.set_run_key_decorator("run_2")
    @bpp.run_decorator(md={})
    def sim_plan_inner(npts):
        for j in range(npts):
            yield from bps.mov(hw.motor1, j * 0.1 + 1, hw.motor2, j * 0.2 - 2)
            yield from bps.trigger_and_read([hw.motor1, hw.motor2, hw.det2])

    @bpp.set_run_key_decorator("run_1")
    @bpp.run_decorator(md={})
    def sim_plan_outer(npts):
        for j in range(int(npts / 2)):
            yield from bps.mov(hw.motor, j * 0.2)
            yield from bps.trigger_and_read([hw.motor, hw.det])

        yield from sim_plan_inner(npts + 1)

        for j in range(int(npts / 2), npts):
            yield from bps.mov(hw.motor, j * 0.2)
            yield from bps.trigger_and_read([hw.motor, hw.det])

    # add baseline to RE
    baseline = [det1, det2, det3]
    sd = bpp.SupplementalData(baseline=baseline)
    RE.preprocessors.append(sd)  # comment out this line to avoid error

    # run the plan
    RE(sim_plan_outer(10))
