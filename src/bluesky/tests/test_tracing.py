import json
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
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Sets the global default tracer provider
    trace.set_tracer_provider(provider)
    return exporter


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


@requires_ophyd
@pytest.mark.parametrize("trace_msg_payloads", [True, False])
def test_trace_msg_payloads_controls_payload_attributes(RE, hw, with_otl_instrumentation, trace_msg_payloads):
    with_otl_instrumentation.clear()
    RE.trace_msg_payloads = trace_msg_payloads

    RE(bp.scan([hw.det], hw.motor, -1, 1, num=2, md={"sample": "md-marker"}))

    spans = [s for s in with_otl_instrumentation.get_finished_spans() if s.name.startswith("Bluesky RunEngine")]
    by_command = {s.attributes["msg.command"]: s.attributes for s in spans}
    assert set(by_command) == {"open_run", "set", "wait"}
    assert "exit_status" in by_command["open_run"]
    assert "group" in by_command["wait"]
    for span in spans:
        assert ("msg.args" in span.attributes) is trace_msg_payloads
        assert ("msg.kwargs" in span.attributes) is trace_msg_payloads
    recorded = json.dumps([dict(s.attributes) for s in spans], default=repr)
    assert ("md-marker" in recorded) is trace_msg_payloads
