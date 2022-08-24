from bluesky import plan_stubs as bps
from bluesky.preprocessors import set_run_key_wrapper as srkw
import bluesky.preprocessors as bsp
from bluesky.tests.utils import DocCollector
import pytest


def test_multirun_smoke(RE, hw):
    """Test on interlaced runs (using wrapper on each command)"""
    dc = DocCollector()

    def interlaced_plan(dets, motor):
        to_read = (motor, *dets)
        run_names = ["run_one", "run_two", "run_three"]
        for rid in run_names:
            yield from srkw(bps.open_run(md={rid: rid}), run=rid)
            yield from srkw(bps.declare_stream(*to_read, name='primary'), run=rid)
        for j in range(5):
            for i, rid in enumerate(run_names):
                yield from bps.mov(motor, j + 0.1 * i)
                yield from srkw(bps.trigger_and_read(to_read), run=rid)

        for rid in run_names:
            yield from srkw(bps.close_run(), run=rid)

    RE(interlaced_plan([hw.det], hw.motor), dc.insert)

    assert len(dc.start) == 3
    for start in dc.start:
        desc, = dc.descriptor[start["uid"]]
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
        yield from bps.declare_stream(*to_read, name='primary')
        yield from some_plan()

    @bsp.set_run_key_decorator("run_two")
    @bsp.run_decorator(md={})
    def plan_middle():
        yield from bps.declare_stream(*to_read, name='primary')
        yield from some_plan()
        yield from plan_inner()

    @bsp.set_run_key_decorator(run="run_three")  # Try kwarg
    @bsp.run_decorator(md={})
    def plan_outer():
        yield from bps.declare_stream(*to_read, name='primary')
        yield from some_plan()
        yield from plan_middle()

    RE(plan_outer(), dc.insert)

    assert len(dc.start) == 3
    for start in dc.start:
        desc, = dc.descriptor[start["uid"]]
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

    with pytest.raises(Exception):
        RE(interlaced_plan([hw.det], hw.motor), dc.insert)

    assert len(dc.start) == len(dc.stop)
    assert len(dc.start) == 3

    for v in dc.stop.values():
        assert v["exit_status"] == "fail"
        assert v["reason"] == "womp womp"
