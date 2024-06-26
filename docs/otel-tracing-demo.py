from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from ophyd.sim import det1

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky.run_engine import RunEngine
from bluesky.tracing import trace_plan, tracer

# Service name is required for most backends,
# and although it's not necessary for console export,
# it's good to set service name anyways.
resource = Resource(attributes={SERVICE_NAME: "bluesky-docs-example"})

traceProvider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(ConsoleSpanExporter())
traceProvider.add_span_processor(processor)
trace.set_tracer_provider(traceProvider)

RE = RunEngine()


@bpp.run_decorator()
@trace_plan(tracer, "demo plan")
def test_tracing_plan():
    yield from bps.abs_set(det1._motor, 5)


RE(test_tracing_plan())
