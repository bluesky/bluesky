from collections.abc import Generator

import pytest
from opentelemetry.trace import get_current_span

from bluesky import plans as bp
from bluesky.plan_stubs import sleep
from bluesky.tests import requires_ophyd
from bluesky.tracing import trace_plan, tracer


@pytest.fixture(scope="session")
def with_otl_instrumentation():
    pytest.importorskip("opentelemetry")

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    provider = TracerProvider()
    processor = SimpleSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)

    # Sets the global default tracer provider
    trace.set_tracer_provider(provider)


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


@requires_ophyd
def test_trace_plan_correctly_sends_sanitized_args(RE, caplog, with_otl_instrumentation):
    from ophyd.sim import SynAxis

    _det = SynAxis(name="test_det")

    @trace_plan(tracer, "test_plan_np")
    def test_plan_np():
        return (yield from bp.scan([_det], _det, -1, 1, num=3))

    RE(test_plan_np())

    assert "Invalid type" not in caplog.text
