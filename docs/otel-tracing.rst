Tracing with Opentelemetry
==========================

Bluesky is instrumented with [OpenTelemetry](https://opentelemetry.io/) tracing span hooks on runs, and on potentially long-running RunEngine methods such as `wait()` and `collect()`. This can allow you to analyze where your plans are spending a lot of time.

By itself, this doesn't do anything. To collect tracing output you should configure an exporter in your application or plan, and a collector to send the data to. You can find an example of how to do that [here](https://opentelemetry.io/docs/languages/python/exporters/#usage). Since the `start_as_current_span` decorator from the opentelemetry library doesn't work for generators, we also provide a `trace_plan` decorator in `bluesky.tracing`

Tracing messages from the `RunEngine` are named `"Bluesky RunEngine <method name>"`, e.g. `"Bluesky RunEngine wait"`. Traces for runs are tagged with the `success`, `exit_status` as attributes, as well as the `reason` if one is available. Traces for methods are tagged with the content of the message (`msg.command`, `msg.args`, `msg.kwargs`, and `msg.obj`). Traces for `wait()` also log the `group` if one was given, or set `no_group_given` true if none was. Ophyd also has traces on `Status` objects to easily record how long they live for.

Examples:
---------

Using the following script, which runs a simple scan and sends traces to the console:

.. literalinclude:: otel-tracing-demo.py
   :language: python

We obtain the following trace data. Note that the innermost spans are closed first, and therefore printed to the console first.

```json
{
    "name": "Ophyd Status",
    "context": {
        "trace_id": "0x6a5abd70e7f74967975ecf64a863a12d",
        "span_id": "0x0f94466a988f72ea",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": "0x4d6e0e523cfdfec8",
    "start_time": "2024-06-18T08:40:56.378644Z",
    "end_time": "2024-06-18T08:40:56.378896Z",
    "status": {
        "status_code": "UNSET"
    },
    "attributes": {
        "status_type": "MoveStatus",
        "settle_time": 0,
        "no_timeout_given": true,
        "object_repr": "MoveStatus(done=False, pos=motor1, elapsed=0.0, success=False, settle_time=0.0)",
        "device_name": "motor1",
        "device_type": "SynAxis",
        "kwargs": "{}",
        "target": 5,
        "start_time": 1718700056.37859,
        "start_pos ": 0,
        "unit": "mm",
        "positioner_name": "motor1",
        "positioner": "SynAxis(prefix='', name='motor1', read_attrs=['readback', 'setpoint'], configuration_attrs=['velocity', 'acceleration'])",
        "finish_time": 1718700056.3788693,
        "finish_pos": 5.0
    },
    "events": [],
    "links": [],
    "resource": {
        "attributes": {
            "service.name": "bluesky-docs-example"
        },
        "schema_url": ""
    }
}
{
    "name": "Bluesky RunEngine set",
    "context": {
        "trace_id": "0x6a5abd70e7f74967975ecf64a863a12d",
        "span_id": "0x4d6e0e523cfdfec8",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": "0x6d4272c02f6990f5",
    "start_time": "2024-06-18T08:40:56.378468Z",
    "end_time": "2024-06-18T08:40:56.378995Z",
    "status": {
        "status_code": "UNSET"
    },
    "attributes": {
        "msg.command": "set",
        "msg.args": [
            5
        ],
        "msg.kwargs": "{\"group\": null}",
        "msg.obj": "SynAxis(prefix='', name='motor1', read_attrs=['readback', 'setpoint'], configuration_attrs=['velocity', 'acceleration'])"
    },
    "events": [],
    "links": [],
    "resource": {
        "attributes": {
            "service.name": "bluesky-docs-example"
        },
        "schema_url": ""
    }
}
{
    "name": "demo plan",
    "context": {
        "trace_id": "0x6a5abd70e7f74967975ecf64a863a12d",
        "span_id": "0x6d4272c02f6990f5",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": null,
    "start_time": "2024-06-18T08:40:56.378255Z",
    "end_time": "2024-06-18T08:40:56.379044Z",
    "status": {
        "status_code": "UNSET"
    },
    "attributes": {},
    "events": [],
    "links": [],
    "resource": {
        "attributes": {
            "service.name": "bluesky-docs-example"
        },
        "schema_url": ""
    }
}
{
    "name": "Bluesky RunEngine run",
    "context": {
        "trace_id": "0xedc3dae09fb45d4982e0af29f053141c",
        "span_id": "0xb54c64dcdebaa81a",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": null,
    "start_time": "2024-06-18T08:40:56.377673Z",
    "end_time": "2024-06-18T08:40:56.379335Z",
    "status": {
        "status_code": "UNSET"
    },
    "attributes": {
        "msg.command": "open_run",
        "msg.args": [],
        "msg.kwargs": "{}",
        "msg.no_obj_given": true,
        "exit_status": "success",
        "reason": ""
    },
    "events": [],
    "links": [],
    "resource": {
        "attributes": {
            "service.name": "bluesky-docs-example"
        },
        "schema_url": ""
    }
}
```