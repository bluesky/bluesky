from collections.abc import Generator

from opentelemetry.trace import get_current_span

from bluesky.plan_stubs import sleep
from bluesky.tracing import trace_plan, tracer


@trace_plan(tracer, "test_plan")
def _test_plan():
    yield from sleep(0)


def test_trace_plan_wrapper_gives_back_generator():
    assert isinstance(_test_plan(), Generator)


def test_trace_plan_wrapper_opens_span_but_doesnt_do_anything_without_trace_provider(RE):
    @trace_plan(tracer, "test_plan_2")
    def test_plan_2():
        yield from sleep(0)
        assert not get_current_span().is_recording()

    RE(test_plan_2())
