Multi-Run Plans
===============

Bluesky v1.6.0 is adding support for multi-run plans. Multi-run plans contain multiple runs started
within the plan.

Definition of a 'Run'
---------------------

A run is defined as a block of code which starts with `open_run` and
ends with `close_run` message. Each run is assigned distinct Scan ID and UID and result in a separate
set of documents to be emitted (and saved by Databroker if enabled), including 'start' and 'stop' documents.
In the plan, the run may be defined by explicitely putting `bps.open_run()` and `bps.close_run()` in the plan:

.. code-block:: python

    import bluesky.plan_stubs as bps
    from bluesky import RunEngine

    RE = RunEngine({})

    def sample_plan():
        ...
        bps.open_run(md={})  # 'md' - metadata to be added to the 'start' document
        ...
        < code that controls execution of the scan >
        ...
        bps.close_run()

    RE(sample_plan())

or using `@bpp.run_decorator`, which inserts the function between the `open_run` and `close_run` commands:

.. code-block:: python

    import bluesky.preprocessors as bpp
    from bluesky import RunEngine

    RE = RunEngine({})

    @bpp.run_decorator(md={})  # 'md' - metadata to be added to the 'start' document
    def sample_plan():
        ...
        < code that controls execution of the scan >
        ...

    RE(sample_plan())

Prepackaged plans, such as `bluesky.plans.count` or `bluesky.plans.list_scan`, are complete single-run
plans, which already contain in `open_run` and `close_run` commands. If a prepackaged plan is called
from a custom plan that defines another run, then the custom plan becomes multi-run plan and the rules
for writing multi-run plans must be used.

Bluesky Features for Support of Multi-run Plans
-----------------------------------------------

In order to handle multiple runs within a plan, Run Engine needs to distinguish messages originated from
different runs. This is achieved by assigning unique run keys to blocks of code, which defines the runs.
The run key is used by Run Engine to

* maintain the state of each run independently from other open runs;

* include run metadata, including scan ID and UID, into the emitted documents. (Metadata is then used
  to route the documents to the appropriate callbacks. If documents are saved using Databroker, the metadata
  allows to associate documents with scans and retrieve scan data.

Run key is assigned to a block of code using `bpp.set_run_key_wrapper` or `@bpp.set_run_key_decorator`:

.. code-block:: python

    import bluesky.preprocessors as bpp
    from bluesky import RunEngine

    # Using decorator
    @bpp.set_run_key_decorator("run_key_example_1")
    @bpp.run_decorator(md={})
    def sample_plan():
        ...
        < code that controls execution of the scan >
        ...

    RE(sample_plan())

    from bluesky.plans import scan
    from ophyd.sim import hw
    det, motor = hw().det, hw().motor

    # Using wrapper
    s = scan([det], motor, -1, 1, 10)
    s_wrapped = bpp.set_run_key_wrapper(s, "run_key_example_2")
    RE(s_wrapped)

The implementations of `@bpp.set_run_key_decorator` and `bpp.set_run_key_wrapper` are
replacing the default value `None` of the attribute `run` of each message inside the block with
the run key passed as an argument. The decorator/wrapper are primarily intended to be applied to
a function that contain implementation of the complete run, but may be used on a function that
implements any sequence of plan instructions.

Plans with Sequential Runs
---------------------------

Runs may be call from a plan sequentially without assigning run keys. This approach works
if the runs are not overlapping, i.e. the next run is opened after the previous run is closed.
In the following example, a sequence two prepackaged (single-run) plans are called in sequence.
Run Engine is subscribed to a single instance of BestEffortCallback, which is
set up at the opening of each run.

.. code-block:: python

    # Example: consecutive execution of single-run plans

    from bluesky import RunEngine
    from bluesky.callbacks.best_effort import BestEffortCallback
    from databroker import Broker
    from bluesky.plans import scan, rel_scan

    from ophyd.sim import hw
    hw = hw()

    RE = RunEngine({})

    db = Broker.named("temp")
    RE.subscribe(db.insert)

    bec = BestEffortCallback()
    RE.subscribe(bec)

    def plan_sequential_runs(npts):
        # Single-run plans may be called consecutively. No special handling is required
        #   as long as the previous scan is closed before the next one is opened
        yield from scan([hw.det1], hw.motor1, -1, 1, npts)
        yield from rel_scan([hw.det1, hw.det2], hw.motor1, -1, 1, npts)

.. ipython:: python
    :suppress:

    %run -m multi_run_plans_sequential

.. ipython:: python

    RE(plan_sequential_runs(10))

Plans with Nested Runs
----------------------

.. code-block:: python

    # Example: nested runs

    from bluesky import RunEngine
    from bluesky.callbacks.best_effort import BestEffortCallback
    import bluesky.preprocessors as bpp
    import bluesky.plan_stubs as bps
    from databroker import Broker
    from event_model import RunRouter

    from ophyd.sim import hw
    hw = hw()

    RE = RunEngine({})

    db = Broker.named("temp")
    RE.subscribe(db.insert)

    def factory(name, doc):
        # Documents from each run is routed to an independent
        #   instance of BestEffortCallback
        bec = BestEffortCallback()
        return [bec], []

    rr = RunRouter([factory])
    RE.subscribe(rr)

    @bpp.set_run_key_decorator("run_2")
    @bpp.run_decorator(md={})
    def sim_plan_inner(npts):
        for j in range(npts):
            yield from bps.mov(hw.motor1, j * 0.1 + 1, hw.motor2, j * 0.2 - 2)
            yield from bps.trigger_and_read([hw.motor1, hw.motor2, hw.det2])

    @bpp.set_run_key_decorator("run_1")
    @bpp.run_decorator(md={})
    def sim_plan_outer(npts):
        for j in range(int(npts/2)):
            yield from bps.mov(hw.motor, j * 0.2)
            yield from bps.trigger_and_read([hw.motor, hw.det])

        yield from sim_plan_inner(npts + 1)

        for j in range(int(npts/2), npts):
            yield from bps.mov(hw.motor, j * 0.2)
            yield from bps.trigger_and_read([hw.motor, hw.det])

.. ipython:: python
    :suppress:

    %run -m multi_run_plans_nested

.. ipython:: python

    RE(sim_plan_outer(10))

The wrapper `bpp.set_run_key_wrapper` can be used instead of the decorator. For example
the run `sim_plan_inner` from the previous example can be rewritten as follows:

.. code-block:: python

    def sim_plan_inner(npts):
        def f():
            for j in range(npts):
                yield from bps.mov(hw.motor1, j * 0.1 + 1, hw.motor2, j * 0.2 - 2)
                yield from bps.trigger_and_read([hw.motor1, hw.motor2, hw.det2])
        f = bpp.run_wrapper(f(), md={})
        return bpp.set_run_key_wrapper(f, "run_2")

Subscription to callbacks via RunRouter is more flexible and allows to subscribe each run
to a separate set of callbacks. In the following example `run_key` is added to the start
document metadata and used to distinguish between two runs in the function factory that
performs callback subscriptions.

.. code-block:: python

    # Example: subscribing runs to individual sets of callbacks

    from bluesky import RunEngine
    from bluesky.callbacks import LiveTable, LivePlot
    import bluesky.preprocessors as bpp
    import bluesky.plan_stubs as bps
    from databroker import Broker
    from event_model import RunRouter

    from ophyd.sim import hw
    hw = hw()

    RE = RunEngine({})

    db = Broker.named("temp")
    RE.subscribe(db.insert)

    def factory(name, doc):
        # Runs may be subscribed to different sets of callbacks. Metadata from start
        #   document may be used to identify, which run is currently being started.
        #   In this example, the run key is explicitely added to the start document
        #   and used to identify runs, but other data can be similarly used.
        cb_list = []
        if doc["run_key"] == "run_1":
            cb_list.append(LiveTable([hw.motor1, hw.det1]))
            cb_list.append(LivePlot('det1', x='motor1'))
        elif doc["run_key"] == "run_2":
            cb_list.append(LiveTable([hw.motor1, hw.motor2, hw.det2]))
        return cb_list, []

    rr = RunRouter([factory])
    RE.subscribe(rr)

    @bpp.set_run_key_decorator("run_2")
    @bpp.run_decorator(md={"run_key": "run_2"})
    def sim_plan_inner(npts):
        for j in range(npts):
            yield from bps.mov(hw.motor1, j * 0.1 + 1, hw.motor2, j * 0.2 - 2)
            yield from bps.trigger_and_read([hw.motor1, hw.motor2, hw.det2])

    @bpp.set_run_key_decorator("run_1")
    @bpp.run_decorator(md={"run_key": "run_1"})
    def sim_plan_outer(npts):
        for j in range(int(npts/2)):
            yield from bps.mov(hw.motor1, j)
            yield from bps.trigger_and_read([hw.motor1, hw.det1])

        yield from sim_plan_inner(npts + 1)

        for j in range(int(npts/2), npts):
            yield from bps.mov(hw.motor1, j)
            yield from bps.trigger_and_read([hw.motor1, hw.det1])

.. ipython:: python
    :suppress:

    %run -m multi_run_plans_select_cb

.. ipython:: python

    RE(sim_plan_outer(10))

In some cases it may be necessary to implement a run that may be interrupted in the middle of execution
and a new instance of the same run started by RunEngine. For example, the suspender pre- or post-plan
may implement a complete run, which may be suspended before it is closed if the suspender is triggered again.
This functionality may be implemented by generating unique run key for each instance of the run.

The following example illustrates dynamic generation of run keys. The plan has no practical purpose
besides demonstration of the principle. The plan that calls itself recursively multiple times until
the global counter variable `n_calls` reaches the value of `n_calls_max`. The unique run key is generated at
each function call.

.. code-block:: python

    # Example: recursive runs

    from bluesky import RunEngine
    from bluesky.callbacks.best_effort import BestEffortCallback
    import bluesky.preprocessors as bpp
    import bluesky.plan_stubs as bps
    from databroker import Broker
    from event_model import RunRouter

    from ophyd.sim import hw
    hw = hw()

    RE = RunEngine({})

    db = Broker.named("temp")
    RE.subscribe(db.insert)

    def factory(name, doc):
        # Each run is subscribed to independent instance of BEC
        bec = BestEffortCallback()
        return [bec], []

    rr = RunRouter([factory])
    RE.subscribe(rr)

    # Call counter and the maximum number calls
    n_calls, n_calls_max = 0, 3

    def sim_plan_recursive(npts):
        global n_calls, n_calls_max

        n_calls += 1  # Increment counter
        if n_calls <= n_calls_max:
            # Generate unique key for each run. The key generation algorithm
            #   must only guarantee that execution of the runs that are assigned
            #   the same key will never overlap in time.
            run_key = f"run_key_{n_calls}"

            @bpp.set_run_key_decorator(run_key)
            @bpp.run_decorator(md={})
            def plan(npts):

                for j in range(int(npts/2)):
                    yield from bps.mov(hw.motor1, j * 0.2)
                    yield from bps.trigger_and_read([hw.motor1, hw.det1])

                # Different parameter values may be passed to the recursively called plans
                yield from sim_plan_recursive(npts + 2)

                for j in range(int(npts/2), npts):
                    yield from bps.mov(hw.motor1, j * 0.2)
                    yield from bps.trigger_and_read([hw.motor1, hw.det1])

            yield from plan(npts)

.. ipython:: python
    :suppress:

    %run -m multi_run_plans_recursive

.. ipython:: python

    RE(sim_plan_recursive(4))


The same can be achieved by using `bpp.set_run_key_wrapper()`:

.. code-block:: python

            @bpp.run_decorator(md={})
            def plan(npts):

                for j in range(int(npts/2)):
                    yield from bps.mov(hw.motor1, j * 0.2)
                    yield from bps.trigger_and_read([hw.motor1, hw.det1])

                # Different parameter values may be passed to the recursively called plans
                yield from sim_plan_recursive(npts + 2)

                for j in range(int(npts/2), npts):
                    yield from bps.mov(hw.motor1, j * 0.2)
                    yield from bps.trigger_and_read([hw.motor1, hw.det1])

            yield from bpp.set_run_key_wrapper(plan(npts), run_key)
