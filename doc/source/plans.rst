.. currentmodule:: bluesky.plans

Plans
=====

An experimental procedure is represented as a sequence of granular
instructions. In bluesky, each instruction is called a *message* and a sequence
of instructions is called a *plan*.

The plans are organized like a burger menu. A variety of fully-assembled plans
are provided --- just as you might order the All-American Burger or The Hog
Wild Burger. But you can also build your own plan by mixing the ingredients
to make something original.

(For developers: what we are calling a "plan" is not actually a special data
structure. It is typically a generator, but it can be any iterable or iterator
--- a list, for example. The messages in a plan are ``Msg`` objects, a
namedtuple.)

Standard Plans (ready to use)
-----------------------------

The names below are links. Click for details, and see below for examples.

.. autosummary::
   :nosignatures:
   :toctree:

    count
    scan
    relative_scan
    list_scan
    relative_list_scan
    log_scan
    relative_log_scan
    inner_product_scan
    outer_product_scan
    scan_nd
    adaptive_scan
    relative_adaptive_scan
    tweak

Basic Usage
-----------

Before we begin, we'll make a RunEngine to execute our plans.

**This RunEngine is not set up to save any data.** You may already have a
RunEngine instance, ``RE``, defined in an IPython profile. If ``RE`` is already
defined, do not redefine it. Just skip to the next step.

.. ipython:: python

    from bluesky import RunEngine
    RE = RunEngine({})


Execute the a ``count`` plan, which reads one or more detectors.

.. ipython:: python

    from bluesky.plans import count
    from bluesky.examples import det  # a simulated detector
    RE(count([det]))

It worked, but the data was not displayed. This time, send the output
``LiveTable``.

.. ipython:: python

    from bluesky.callbacks import LiveTable
    RE(count([det]), LiveTable([det]))

Stub Plans (ingredients for remixing)
-------------------------------------

.. autosummary::
   :nosignatures:
   :toctree:

    trigger_and_read
    abs_set
    rel_set
    wait
    sleep
    checkpoint
    clear_checkpoint
    pause
    deferred_pause
    open_run
    close_run
    create
    save
    trigger
    read
    monitor
    unmonitor
    kickoff
    collect
    configure
    stage
    unstage
    subscribe
    unsubscribe
    wait_for
    null
    one_1d_step
    one_nd_step

Concatenating Plans
-------------------

Plans are iterables (roughly speaking, lists) and the Python language has nice
facilities for handling them. For example to join to plans together, use

.. code-block:: python

    from bluesky.plans import bschain

    plan1 = scan([det1, det2], motor, 1, 5, 3)  # 1 to 5 in 3 steps
    plan2 = scan([det1], motor, 5, 10, 2)  # 5 to 10 in 2 steps

    # Do this.
    master_plan = bschain(plan1, plan2)
    RE(master_plan)

This has advantages over executing them in sequence like so:

.. code-block:: python

    # Don't do this.
    RE(plan1); RE(plan2)

If there are no interruptions or errors, these two methods are equivalent. But
in the event of a problem, the RunEngine can do more to recover if it
maintains control. In the second example, it loses control between completing
``plan1`` and beginning ``plan2``.

What if want to ``print`` or do other activities between executing the plans?
There is another way to combine plans to accomodate this.

.. code-block:: python


    def make_master_plan():
        plan1 = scan([det1, det2], motor, 1, 5, 10)
        plan2 = relative_scan([det1], motor, 5, 10, 10)

        yield from plan1
        print('plan1 is finished -- moving onto plan2')
        yield from plan2

    RE(make_master_plan())  # note the ()

Arbitrary Python code can go inside ``master_plan``. It could employ ``if``
blocks, ``for`` loops -- anything except a ``return`` statement. If you want to
know more about what is happening here, structures like this in Python are
called *generators*.

Here are a couple more useful recipes:

.. code-block:: python

    "Run plan1, wait for user confirmation, then run plan2."

    def make_master_plan():
        plan1 = scan([det1, det2], motor, 1, 5, 10)
        plan2 = relative_scan([det1], motor, 5, 10, 10)

        yield from plan1
        # pause and consult the user
        if input('continue? (y/n)') != 'y':
            raise StopIteration
        yield from plan2

.. code-block:: python

    "Run a plan several times, changing the step size each time."

    def make_master_plan():
        for num in range(5, 10):
            # Change the number of steps in the plan in each loop
            plan1 = scan([det1, det2], motor, 1, 5, num)
            yield from plan1

Plan Context Managers
---------------------

These context managers provide a sunninct, readable syntax for inserting plans
before and after other plans.

.. autosummary::
   :nosignatures:
   :toctree:

    baseline_context
    monitor_context
    subs_context
    run_context
    event_context
    stage_context

For example, the ``baseline_context`` reads a list of detectors at the
beginning and end of an experiment.

.. code-block:: python

    from bluesky.plans import baseline_context, count
    from bluesky.examples import det

    plans = []
    with baseline_context(plans, [det]):
        plans.append(count([det]))

Use with the ``planify`` decorator to join the list of plans into one plan.

.. code-block:: python

    from bluesky.plans import planify

    @planify
    def count_with_baseline_readings():
        plans = []
        with baseline_context(plans, [det]):
            plans.append(count([det]))
        return plans

Plan Preprocessors
------------------

These "preprocessors" take in a plan and modify its contents on the fly.

.. autosummary::
   :nosignatures:
   :toctree:

    relative_set
    reset_positions
    lazily_stage
    fly_during
    finalize

For example, ``relative_set`` rewrites all positions to be relative to the
initial position.

.. code-block:: python

    def relative_scan(detectors, motor, start, stop, num):
        absolute = scan(detectors, motor, start, stop, num)
        relative = relative_set(absolute, [motor])
        yield from relative    

Or, equivalently, using the ``planify`` decorator:

.. code-block:: python

    @planify
    def relative_scan(detectors, motor, start, stop, num):
        absolute = scan(detectors, motor, start, stop, num)
        relative = relative_set(absolute, [motor])
        return [relative]

Plan Utilities
--------------

.. autosummary::

    planify
    msg_mutator
    plan_mutator
    bschain
    single_gen
    broadcast_msg
    repeater
    caching_repeater

Object-Oriented Standard Plans
------------------------------

These provide a different way of using the standard plans. The plan becomes
a reusable object, whose parameters can be adjusted interactively between
uses.

.. ipython:: python

    from bluesky.plans import Scan
    from bluesky.examples import motor, det3
    plan = Scan([det], motor, 1, 5, 10)
    RE(plan)
    RE(plan)

Any of the plan's parameters can be updated individually.

.. ipython:: python

    plan.num = 4  # change number of data points from 10 to 4
    RE(plan)
    plan.detectors.append(det3)  # add another detector
    RE(plan)

The ``set`` method is a convenient way to update multiple parameters at once.

.. ipython:: python

    plan.set(start=20, stop=25)

.. autosummary::
   :nosignatures:
   :toctree:

    Count
    Scan
    RelativeScan
    ListScan
    RelativeListScan
    LogScan
    RelativeLogScan
    InnerProductScan
    RelativeInnerProductScan
    OuterProductScan
    RelativeOuterProductScan
    ScanND
    AdaptiveScan
    RelativeAdaptiveScan
    Tweak
