Tracing with Opentelemetry
==========================

Bluesky is instrumented with [OpenTelemetry](https://opentelemetry.io/) tracing span hooks on runs, and on potentially long-running RunEngine methods such as `wait()` and `collect()`. This can allow you to analyze where your plans are spending a lot of time.

By itself, this doesn't do anything. To collect tracing output you should configure an exporter in your application or plan, and a collector to send the data to. You can find an example of how to do that [here](https://opentelemetry.io/docs/languages/python/exporters/#usage).

Tracing messages from the `RunEngine` are named `"Bluesky RunEngine <method name>"`, e.g. `"Bluesky RunEngine wait"`. Traces for runs are tagged with the `success`, `exit_status` as attributes, as well as the `reason` if one is available. Traces for methods are tagged with the content of the message (`msg.command`, `msg.args`, `msg.kwargs`, and `msg.obj`). Traces for `wait()` also log the `group` if one was given, or set `no_group_given` true if none was. Ophyd also has traces on `Status` objects to easily record how long they live for.

Examples: