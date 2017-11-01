.. currentmodule:: bluesky.plans

=======
 Plans
=======

A *plan* is bluesky's concept of an experimental procedure. A
:doc:`previous section <plans_intro>` introduced some built-in plans like
:func:`count`, :func:`scan`, and :func:`rel_scan`. This section covers all
of the plans and plan-related tools in bluesky with examples showing how to
combine and customize them.

A variety of pre-assembled plans are provided. Like sandwiches on a deli menu,
you can use our pre-assembled plans or assemble your own from the same
ingredients, catalogued under the heading :ref:`stub_plans` below.

Built-in Plans
==============

.. _preassembled_plans:

Pre-assembled Plans
-------------------

Below this summary table, we break the down the plans by category and show
examples with figures.

Summary
~~~~~~~

Notice that the names in the left column are links to detailed API
documentation.

.. autosummary::
   :toctree: generated
   :nosignatures:

   count
   scan
   rel_scan
   list_scan
   rel_list_scan
   log_scan
   rel_log_scan
   inner_product_scan
   grid_scan
   relative_inner_product_scan
   rel_grid_scan
   scan_nd
   spiral
   spiral_fermat
   rel_spiral
   rel_spiral_fermat
   adaptive_scan
   rel_adaptive_scan
   tune_centroid
   tweak
   fly


Time series ("count")
~~~~~~~~~~~~~~~~~~~~~

Examples:

.. code-block:: python

    from bluesky.examples import det
    from bluesky.plans import count

    # a single reading of the detector 'det'
    RE(count([det]))

    # five consecutive readings
    RE(count([det], num=5))

    # five sequential readings separated by a 1-second delay
    RE(count([det], num=5, delay=1))

    # a variable delay
    RE(count([det], num=5, delay=[1, 2, 3, 4])

    # Take readings forever, until interrupted (e.g., with Ctrl+C)
    RE(count([det], num=None))

We can use ``LivePlot`` to visualize this data. It is documented in the
:ref:`next section <liveplot>`.

.. code-block:: python

    from bluesky.callbacks import LivePlot

    # We'll use the 'noisy_det' example detector for a more interesting plot.
    from bluesky.examples import noisy_det

    RE(count([noisy_det], num=5), LivePlot('noisy_det'))


.. plot::

    from bluesky import RunEngine
    from bluesky.plans import count
    from bluesky.examples import noisy_det
    from bluesky.callbacks import LivePlot
    RE = RunEngine({})
    RE(count([noisy_det], num=5), LivePlot('noisy_det'))

.. autosummary::
   :toctree: generated
   :nosignatures:

   count

Scans over one dimension
~~~~~~~~~~~~~~~~~~~~~~~~

The "dimension" might be a physical motor position, a temperature, or a
pseudo-axis. It's all the same to the plans. Examples:

.. code-block:: python

    from bluesky.examples import det, motor
    from bluesky.plans import scan, rel_scan, list_scan

    # scan a motor from 1 to 5, taking 5 equally-spaced readings of 'det'
    RE(scan([det], motor, 1, 5, 5))

    # scan a motor from 1 to 5 *relative to its current position*
    RE(rel_scan([det], motor, 1, 5, 5))

    # scan a motor through a list of user-specified positions
    RE(list_scan([det], motor, [1, 1, 2, 3, 5, 8]))

Again, we can use ``LivePlot`` to visualize this data. It is documented in the
:ref:`next section <liveplot>`.

.. code-block:: python

    from bluesky.callbacks import LivePlot

    RE(scan([det], motor, 1, 5, 5), LivePlot('det', 'motor'))

Or, again, to save some typing for repeated use,
:ref:`define a custom plan with the plot incorporated <subs_decorator>`.
(LivePlot itself is documented :ref:`here <liveplot>`.)

.. plot::

    from bluesky import RunEngine
    from bluesky.plans import scan
    from bluesky.examples import det, motor
    from bluesky.callbacks import LivePlot
    RE = RunEngine({})
    RE(scan([det], motor, 1, 5, 5), LivePlot('det', 'motor'))

.. autosummary::
   :toctree: generated
   :nosignatures:

   scan
   rel_scan
   list_scan
   rel_list_scan
   log_scan
   rel_log_scan

.. _multi-dimensional_scans:

Multi-dimensional scans
~~~~~~~~~~~~~~~~~~~~~~~

Here, "dimensions" are things independently scanned. They may be physical
position (stepping motor), temperature, etc.

We introduce jargon for two different kinds of a multi-dimensional
(multi-"motor") scan. Moving motors together in a joint trajectory is an "inner
product scan." This is like moving an object along a diagonal by moving the x
and y motors simultaneously.

.. code-block:: python

    from bluesky.examples import det, motor1, motor2
    from bluesky.plans import inner_product_scan

    # Inner product: move motors together.
    # Move motor1 from 1-5 while moving motor2 from 10-50 -- both in 5 steps.
    RE(inner_product_scan([det], 5, motor1, 1, 5, motor2, 10, 50))

Demo:

.. ipython:: python
    :suppress:

    from bluesky.examples import det, motor1, motor2
    from bluesky.callbacks import LiveTable
    from bluesky import RunEngine
    from bluesky.plans import grid_scan, inner_product_scan
    RE = RunEngine({})

.. ipython:: python

    RE(inner_product_scan([det], 5, motor1, 1, 5, motor2, 10, 50),
       LiveTable(['det', 'motor1', 'motor2']))

.. plot::

    from bluesky.simulators import plot_raster_path
    from bluesky.examples import motor1, motor2, det
    from bluesky.plans import inner_product_scan
    import matplotlib.pyplot as plt

    plan = inner_product_scan([det], 5, motor1, 1, 5, motor2, 10, 50)
    plot_raster_path(plan, 'motor1', 'motor2', probe_size=.3)

Notice that, in an inner product scan, each motor moves the same number
of steps (in the example above, 5).

Moving motors separately, exploring every combination, is an "outer product
scan". This is like moving x and y to draw a mesh. The mesh does not have to be
square: each motor can move a different number of steps.

.. code-block:: python

    from bluesky.examples import det, motor1, motor2
    from bluesky.plans import grid_scan

    # Outer product: move motors in a mesh.
    # Move motor1 from 1-3 in 3 steps and motor2 from 10-50 in 5 steps.
    RE(grid_scan([det], motor1, 1, 3, 3, motor2, 10, 50, 5, False))

Demo:

.. ipython:: python

    RE(grid_scan([det], motor1, 1, 3, 3, motor2, 10, 50, 5, False),
       LiveTable(['det', 'motor1', 'motor2']))

The final parameter designates whether motor2 should "snake" back and forth
along motor1's trajectory (``True``) or retread its positions in the same
direction each time (``False``), as illustrated.

.. plot::

    from bluesky.simulators import plot_raster_path
    from bluesky.examples import motor1, motor2, det
    from bluesky.plans import grid_scan
    import matplotlib.pyplot as plt

    true_plan = grid_scan([det], motor1, -5, 5, 10, motor2, -7, 7, 15, True)
    false_plan = grid_scan([det], motor1, -5, 5, 10, motor2, -7, 7, 15, False)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    plot_raster_path(true_plan, 'motor1', 'motor2', probe_size=.3, ax=ax1)
    plot_raster_path(false_plan, 'motor1', 'motor2', probe_size=.3, ax=ax2)
    ax1.set_title('True')
    ax2.set_title('False')
    ax1.set_xlim(-6, 6)
    ax2.set_xlim(-6, 6)

Both :func:`inner_product_scan` and :func:`grid_scan` support an
unlimited number of motors/dimensions.

To visualize 2-dimensional data, we can use ``LiveGrid``, which is documented
in :ref:`in the next section <liveraster>`. In previous examples we used
``LivePlot`` to visualize readings as a function of one variable;
``LiveGrid`` is appropriate for functions of two variables.

.. code-block:: python

    from bluesky.callbacks import LiveGrid

    # The 'det4' example detector a 2D Gaussian function of motor1, motor2.
    from bluesky.examples import det4

    RE(grid_scan([det4], motor1, -3, 3, 6, motor2, -5, 5, 10, False),
       LiveGrid((6, 10), 'det4'))

.. plot::

    from bluesky import RunEngine
    from bluesky.plans import grid_scan
    from bluesky.examples import det4, motor1, motor2
    from bluesky.callbacks import LiveGrid
    motor1.delay = 0
    motor2.delay = 0
    RE = RunEngine({})
    RE(grid_scan([det4], motor1, -3, 3, 6, motor2, -5, 5, 10, False),
       LiveGrid((6, 10), 'det4'))

The general case, moving some motors together in an "inner product" against
another (or motors) in an "outer product," can be addressed using a ``cycler``.
Notice what happens when we add or multiply ``cycler`` objects.

.. ipython:: python

    from cycler import cycler
    from bluesky.examples import motor1, motor2, motor3

    traj1 = cycler(motor1, [1, 2, 3])
    traj2 = cycler(motor2, [10, 20, 30])
    list(traj1)  # a trajectory for motor1
    list(traj1 + traj2)  # an "inner product" trajectory
    list(traj1 * traj2)  # an "outer product" trajectory

We have reproduced inner product and outer product. The real power comes in
when we combine them, like so. Here, motor1 and motor2 together in a mesh
against motor3.

.. ipython:: python

    traj3 = cycler(motor3, [100, 200, 300])
    list((traj1 + traj2) * traj3)

For more on cycler, we refer you to the
`cycler documentation <http://matplotlib.org/cycler/>`_. To build a plan
incorporating these trajectories, use our general N-dimensional scan plan,
:func:`scan_nd`.

.. code-block:: python

    RE(scan_nd([det], (traj1 + traj2) * traj3))

.. autosummary::
   :toctree:
   :nosignatures:

   inner_product_scan
   grid_scan
   relative_inner_product_scan
   rel_grid_scan
   scan_nd

Spiral trajectories
~~~~~~~~~~~~~~~~~~~

We provide two-dimensional scans that trace out spiral trajectories.

A simple spiral:

.. plot::
   :include-source:

    from bluesky.simulators import plot_raster_path
    from bluesky.examples import motor1, motor2, det
    from bluesky.plans import spiral

    plan = spiral([det], motor1, motor2, x_start=0.0, y_start=0.0, x_range=1.,
                  y_range=1.0, dr=0.1, nth=10)
    plot_raster_path(plan, 'motor1', 'motor2', probe_size=.01)


A fermat spiral:

.. plot::
   :include-source:

    from bluesky.simulators import plot_raster_path
    from bluesky.examples import motor1, motor2, det
    from bluesky.plans import spiral_fermat

    plan = spiral_fermat([det], motor1, motor2, x_start=0.0, y_start=0.0,
                         x_range=2.0, y_range=2.0, dr=0.1, factor=2.0, tilt=0.0)
    plot_raster_path(plan, 'motor1', 'motor2', probe_size=.01, lw=0.1)


.. autosummary::
   :toctree:
   :nosignatures:

   spiral
   spiral_fermat
   rel_spiral
   rel_spiral_fermat

Adaptive scans
~~~~~~~~~~~~~~

These are one-dimension scans with an adaptive step size tuned to move quickly
over flat regions can concentrate readings in areas of high variation by
computing the local slope aiming for a target delta y between consecutive
points.

This is a basic example of the power of adaptive plan logic.

.. code-block:: python

    from bluesky.plans import adaptive_scan
    from bluesky.callbacks import LivePlot
    from bluesky.examples import motor, det

    RE(adaptive_scan([det], 'det', motor,
                     start=-15,
                     stop=10,
                     min_step=0.01,
                     max_step=5,
                     target_delta=.05,
                     backstep=True),
       LivePlot('det', 'motor', markersize=10, marker='o'))

.. plot::

    from bluesky import RunEngine
    from bluesky.plans import adaptive_scan
    from bluesky.callbacks import LivePlot
    from bluesky.examples import motor, det

    RE = RunEngine({})

    RE(adaptive_scan([det], 'det', motor,
                     start=-15,
                     stop=10,
                     min_step=0.01,
                     max_step=5,
                     target_delta=.05,
                     backstep=True),
       LivePlot('det', 'motor', markersize=10, marker='o'))

From left to right, the scan lengthens its stride through the flat region. At
first, it steps past the peak. The large jump causes it to double back and then
sample more densely through the peak. As the peak flattens, it lengthens its
stride again.

.. autosummary::
   :toctree:
   :nosignatures:

   adaptive_scan
   rel_adaptive_scan

Misc.
~~~~~

.. autosummary::
   :toctree:
   :nosignatures:

   tweak
   fly

.. _stub_plans:

Stub Plans
----------
.. currentmodule:: bluesky.plan_stubs

These are the aforementioned "ingredients" for remixing, the pieces from which
the pre-assembled plans above were made. The next section provides many
examples.

Plans for interacting with hardware:

.. autosummary::
   :nosignatures:
   :toctree:

    abs_set
    rel_set
    mv
    mvr
    trigger
    read
    stage
    unstage
    configure
    stop

Plans for asynchronous acquisition:

.. autosummary::
   :nosignatures:
   :toctree:

    monitor
    unmonitor
    kickoff
    complete
    collect

Plans that control the RunEngine:

.. autosummary::
   :nosignatures:
   :toctree:

    open_run
    close_run
    create
    save
    pause
    deferred_pause
    checkpoint
    clear_checkpoint
    sleep
    input_plan
    subscribe
    unsubscribe
    wait
    wait_for
    null

Combinations of the above that are often convenient:

.. autosummary::
    trigger_and_read
    one_1d_step
    one_nd_step

We also provide :ref:`wrapper and decorator functions <preprocessors>` and
:ref:`utility functions <plan_utils>`, documented below, that make building
these easier.


.. _plan_examples1:

Examples
========

Changing a Parameter Between Runs
---------------------------------

Produce several runs, changing a parameter each time.

.. code-block:: python

    from bluesky.plans import scan
    from bluesky.examples import det, motor

    def scan_varying_density():
        "Run a scan several times, changing the step size each time."
        for num in range(5, 10):
            # Scan motor from -1 to 1, sampling more densely in each
            # iteration.
            yield from scan([det], motor, -1, 1, num)

Setting Devices to a Set Point
------------------------------

Next, we introduce :func:`abs_set`, which sets a motor to a position (or a
temperature controller to a temperature, etc.). See also :func:`rel_set`, which
sets *relative* to the current value.

.. code-block:: python

    from bluesky.plans import count, abs_set
    from bluesky.examples import det, motor

    def move_and_count():
        "Move a motor into place, then take a reading from detectors."
        yield from abs_set(motor, 3, wait=True)
        yield from count([det])

The argument ``wait=True`` blocks progress until the device reports that it is
ready (e.g., done moving or done triggering). Alternatively, use a :func:`wait`
plan, which is more flexible. Here, we move two motors at once and wait for
them both to finish.

.. code-block:: python

    from bluesky.plans import abs_set, wait
    from bluesky.examples import motor1, motor2

    def set_two_motors():
        "Set, trigger, read"
        yield from abs_set(motor1, 5, group='A')  # Start moving motor1.
        yield from abs_set(motor2, 5, group='A')  # Start moving motor2.
        yield from wait('A')  # Now wait for both to finish.

The ``group`` is just temporary label that we can use to refer to groups of
devices that we want to move or trigger simulataneously and then wait for them
as a group.  This plan will continue once both motors have reported that they
have finished moving successfully.

We could have written this some logic with a loop:

.. code-block:: python

    def set_multiple_motors(motors):
        "Set all motors moving; then wait for all motors to finish."
        for motor in motors:
            yield from abs_set(motor, 5, group='A')
        yield from wait('A')

Two convenient shortcuts are available for common cases. As shown at the
beginning of this section, if you are setting one motor at a time, use the
``wait`` keyword argument.

.. code-block:: python

    def set_one_motor():
        yield from abs_set(motor1, wait=True)
        # `wait=True` implicitly adds a group and `wait` plan to match.

The same works for :func:`rel_set` and :func:`trigger`. Also, if you are only
dealing with one group at a time, you do not actually need to label the group:

.. code-block:: python

    def set_multiple_motors(motors):
        "Set all motors moving; then wait for all motors to finish."
        for motor in motors:
            yield from abs_set(motor, 5)
        yield from wait()

But by using labels you can express complex logic, waiting for different groups
at different points in the plan:

.. code-block:: python

    def staggered_wait(det, fast_motors, slow_motor):
        # Start all the motors, fast and slow, moving at once.
        # Put all the fast_motors in one group...
        for motor in fast_motors:
            yield from abs_set(motor, 5, group='A')
        # ...but put the slow motor is separate group.
        yield from abs_set(slow_motor, 5, group='B')

        # Wait for all the fast motors.
        yield from wait('A')

        # Do some stuff that doesn't require the slow motor to be finished.

        # Then wait for the slow motor.
        yield from wait('B')

Before writing a custom plan to coordinate the motion of multiple devices,
consider whether your use case could be addressed with one of the built-in
:ref:`multi-dimensional_scans`.

Sleeping
--------

A "sleep" is a timed delay.

.. code-block:: python

    from bluesky.plans import sleep, abs_set
    from bluesky.examples import motor

    def sleepy():
        "Set motor; sleep for a fixed time; set it to a new position."
        yield from abs_set(motor, 5)
        yield from sleep(2)  # units: seconds
        yield from abs_set(motor, 10)

The :func:`sleep` plan is not the same as Python's built-in sleep function,
``time.sleep(...)``. Never use ``time.sleep(...)`` in a plan; use ``yield from
sleep(...)`` instead. It allows other tasks --- such as watching for Ctrl+C,
updating plots --- to be executed while the clock runs.

.. _planned_pauses:

Planned Pauses
--------------

Pausing is typically done :ref:`interactively <pausing_interactively>` (Ctrl+C)
but it can also be incorporated into a plan. The plan can pause the RunEngine,
requiring the user to type ``RE.resume()`` to continue or ``RE.stop()`` to
clean up and stop.

Pauses can be interspersed using :func:`chain`. Demo:

.. ipython:: python

    from bluesky.plans import pchain, count, pause
    from bluesky.examples import det
    RE(pchain(count([det]), pause(), count([det])))
    RE.state  # optional -- just doing this to show that we are paused
    RE.resume()  # or, alternatively, RE.stop()

Or pauses can be incorporated in a plan like so:

.. code-block:: python

    from bluesky.plans import pause, checkpoint

    def pausing_plan():
        while True:
            yield from some_plan(...)
            print("Type RE.resume() to go again or RE.stop() to stop.")
            yield from checkpoint()  # marking where to resume from
            yield from pause()

.. _customizing_metadata:

Customizing metadata
====================

Metadata can be loaded from a persistent file, specified by the user
interactively at execution time, or incorporated in a plan.

All of the pre-assembled plans also accept an ``md`` ("metadata") argument,
which makes it easy for a user-defined plan to pass in extra metadata.

.. code-block:: python

    from bluesky.plans import count
    from bluesky.examples import det

    def master_plan():
        "Read a detector with the shutter closed and then open."
        # ... insert code here to close shutter ...
        yield from count([det], md={'is_dark_frame': True})
        # ... insert code here to open shutter ...
        yield from count([det], md={'is_dark_frame': False})

By default, the :func:`count` plan records ``{'plan_name': 'count'}``. To
customize the ``plan_name`` --- say, to differentiate separate *reasons* for
running a count --- you can override this behavior.

.. code-block:: python

    def calib_count(dets, num=3):
        "A count whose data will be designated 'calibration'."
        md = {'plan_name': 'calib_count'}
        yield from count(dets, num=num, md=md)

The above records the ``{'plan_name': 'calib_count'}``.  To enable users to
pass in metadata that combines with and potentially overrides the hard-coded
metadata, use the following pattern:

.. code-block:: python

    def calib_count(dets, num=3, *, md=None):
        "A count whose data will be designated 'calibration'."
        _md = {'plan_name': 'calib_count'}  # default
        _md.update(md or {})  # supplemented/overridden by argument passed in
        yield from count(dets, num=num, md=_md)

For example, if the plan is called with the arguments:

.. code-block:: python

    calib_count([det], md={'plan_name': 'watermelon'})

then ``'watermelon'`` will override ``'calib_count'`` as the recorded plan
name.

.. _preprocessors:

Plan Preprocessors
==================
.. currentmodule:: bluesky.preprocessors

.. _supplemental_data:

Supplemental Data
-----------------

Plan preprocessors modify a plans contents on the fly. One common use of a
preprocessor is to take "baseline" readings of a group of devices at the
beginning and end of each run. It is convenient to apply this to *all* plans
executed by a RunEngine using the :class:`SupplementalData`.

.. autoclass:: SupplementalData
    :members:

Preprocessor Wrappers and Decorators
------------------------------------

Preprocessors can make arbirary modifcations to a plan, and can get quite
devious. For example, the :func:`relative_set_wrapper` rewrites all positions
to be relative to the initial position.

.. code-block:: python

    def rel_scan(detectors, motor, start, stop, num):
        absolute = scan(detectors, motor, start, stop, num)
        relative = relative_set_wrapper(absolute, [motor])
        yield from relative

This is a subtle but remarkably powerful feature.

Wrappers like :func:`relative_set_wrapper` operate on a generator *instance*,
like ``scan(...)``. There are corresponding decorator functions like
``relative_set_decorator`` that operate on a generator
*function* itself, like :func:`scan`.

.. code-block:: python

    # Using a decorator to modify a generator function
    def rel_scan(detectors, motor, start, stop, num):

        @relative_set_decorator([motor])  # unfamiliar syntax? -- see box below
        def inner_relative_scan():
            yield from scan(detectors, motor, start, stop, num)

        yield from inner_relative_scan()

Incidentally, the name ``inner_relative_scan`` is just an internal variable,
so why did we choose such a verbose name? Why not just name it ``f``? That
would work, of course, but using a descriptive name can make debugging easier.
When navigating gnarly, deeply nested tracebacks, it helps if internal variables
have clear names.

.. note::

    The decorator syntax --- the ``@`` --- is a succinct way of passing a
    function to another function.

    This:

    .. code-block:: python

        @g
        def f(...):
            pass

        f(...)

    is equivalent to

    .. code-block:: python

        g(f)(...)

Built-in Preprocessors
----------------------
.. currentmodule:: bluesky.preprocessors

Each of the following functions named ``<something>_wrapper`` operates on
a generator instance. The corresponding functions named
``<something_decorator>`` operate on a generator function.

.. autosummary::
   :nosignatures:
   :toctree:

    baseline_decorator
    baseline_wrapper
    finalize_decorator
    finalize_wrapper
    fly_during_decorator
    fly_during_wrapper
    inject_md_decorator
    inject_md_wrapper
    lazily_stage_decorator
    lazily_stage_wrapper
    monitor_during_decorator
    monitor_during_wrapper
    relative_set_decorator
    relative_set_wrapper
    reset_positions_decorator
    reset_positions_wrapper
    run_decorator
    run_wrapper
    stage_decorator
    stage_wrapper
    subs_decorator
    subs_wrapper

Custom Preprocessors
--------------------

The preprocessors are implemented using :func:`msg_mutator` (for altering
messages in place) and :func:`plan_mutator` (for inserting
messages into the plan or removing messages).

It's easiest to learn this by example, studying the implementations of the built-in
processors (catalogued above) in the
`the source of the plans module <https://github.com/NSLS-II/bluesky/blob/master/bluesky/plans.py>`_.

.. _exception_handling:

How Plans Handle Exceptions
===========================

If an exception is raised, the RunEngine gives the plan the opportunity to
catch the exception and either handle it or merely yield some "clean up"
messsages before re-raising the exception and killing plan execution.

The exception in question may originate from the plan itself or from the
RunEngine when it attempts to execute a given command.

.. code-block:: python

    # This example is illustrative, but it is not completely correct.
    # Use `finalize_wrapper` instead (or read its source code).

    def plan_with_cleanup():
        try:
            yield from main_plan()
        except Exception:
            # Catch the exception long enough to clean up.
            yield from cleanup_plan()
            raise  # Re-raise the exception.

The :func:`finalize_wrapper` preprocessor provides a succinct and fully correct
way of applying this general pattern.

.. code-block:: python

    from bluesky.plans import finalize_wrapper

    def plan_with_cleanup():
        yield from finalize_wrapper(main_plan(), cleanup_plan())

Or, at your preference, the same logic is available as a decorator:

.. code-block:: python

    from bluesky.plans import finalize_decorator

    plan_with_cleanup = finalize_decorator(cleanup_plan)(main_plan)

    # or, equivalently:

    @finalize_decorator(cleanup_plan)
    def plan_with_cleanup():
        yield from main_plan()

Customize Step Scans with ``per_step``
======================================

The one-dimensional and multi-dimensional plans are composed (1) setup,
(2) a loop over a plan to perform at each position, (3) cleanup.

We provide a hook for customizing step (2). This enables you to write a
variation of an existing plan without starting from scratch.

For one-dimensional plans, the default inner loop is:

.. code-block:: python

    from bluesky.plans import checkpoint, abs_set, trigger_and_read

    def one_1d_step(detectors, motor, step):
        """
        Inner loop of a 1D step scan

        This is the default function for ``per_step`` param in 1D plans.
        """
        yield from checkpoint()
        yield from abs_set(motor, step, wait=True)
        return (yield from trigger_and_read(list(detectors) + [motor]))

Some user-defined function, ``custom_step``, with the same signature can be
used in its place:

.. code-block:: python

    scan([det], motor, 1, 5, 5, per_step=custom_step)

For convenience, this could be wrapped into the definition of a new plan:

.. code-block:: python

    def custom_scan(detectors, motor, start, stop, step, *, md=None):
        yield from scan([det], motor, start, stop, step, md=md
                        per_step=custom_step)

For multi-dimensional plans, the default inner loop is:

.. code-block:: python

    from bluesky.utils import short_uid
    from bluesky.plans import checkpoint, abs_set, wait, trigger_and_read

    def one_nd_step(detectors, step, pos_cache):
        """
        Inner loop of an N-dimensional step scan

        This is the default function for ``per_step`` param in ND plans.

        Parameters
        ----------
        detectors : iterable
            devices to read
        step : dict
            mapping motors to positions in this step
        pos_cache : dict
            mapping motors to their last-set positions
        """
        def move():
            yield from checkpoint()
            grp = short_uid('set')
            for motor, pos in step.items():
                if pos == pos_cache[motor]:
                    # This step does not move this motor.
                    continue
                yield from abs_set(motor, pos, group=grp)
                pos_cache[motor] = pos
            yield from wait(group=grp)

        motors = step.keys()
        yield from move()
        yield from trigger_and_read(list(detectors) + list(motors))

Likewise, a custom function with the same signature may be passed into the
``per_step`` argument of any of the multi-dimensional plans.

.. _reimplementing_count:

Controlling the Scope of a "Run"
================================

By default, the :func:`count` plan generates one "run" (i.e., dataset)
with one "event" (i.e., one bundle of readings from the detectors, one row in
a table of the data).

.. code-block:: python

    from bluesky.examples import det1, det2
    from bluesky.plans import count

    dets = [det1, det2]
    RE(count(dets))

The ``num`` argument enables multiple events (rows) in one run.

.. code-block:: python

    # one 'run' with three 'events'
    RE(count(dets, num=3))

If we didn't provide a num option, how could you make one yourself?

A tempting --- but wrong! --- possibility is to loop over calls to
``RE(count(dets))``.

.. code-block:: python

    # Don't do this!
    for _ in range(3):
        RE(count(dets))

As stated earlier, this ruins error-recovery and interruption recovery. It's
much better to do the loop inside a custom plan, which we'll dub
``multicount``.

.. code-block:: python

    def multicount(dets):
        for _ in range(3):
            yield from count(dets)

    RE(multicount(dets))

In fact, instead of hard-coding 3, we could make it an argument configurable
by the user. We can make the configuration optional by providing 3 as a
default.

.. code-block:: python

    def multicount(dets, num=3):
        for _ in range(num):
            yield from count(dets)

But this still creates three runs --- three datasets --- for what we'd rather
think of as three events (rows) in one run. To fix that, we'll have to dive
deeper, re-implementing :func:`count` from scratch.

.. code-block:: python

    from bluesky.plans import run_decorator, stage_decorator, trigger_and_read

    def multicount(dets, num=3, *, md=None):

        @stage_decorator(dets)
        @run_decorator(md=md)
        def inner_multicount():
            for _ in range(num):
                yield from trigger_and_read(dets)

        yield from inner_multicount()


Starting from the middle and explaining outward:

* The :func:`trigger_and_read` plan generates an "event" (a row of data) from
  reading ``dets``. This happens inside of a loop, ``num`` times.
* The :func:`run_decorator` preprocessor designates the scope of one run.
* The :func:`stage_decorator` preprocessor addresses some hardware details. It
  primes the hardware for data collection.  For some devices, this has no
  effect at all. But for others, it ensures that the device is put into a
  ready, triggerable state and then restored to standby at the end of the plan.

Plans with Adaptive Logic
=========================

Two-way communication is possible between the generator and the RunEngine.
For example, the 'read' command responds with its reading. We can use it to
make an on-the-fly decision about whether to continue or stop.

.. code-block:: python

    from bluesky.plans import abs_set, trigger, read
    from bluesky.examples import det, motor

    def conditional_break(threshold):
        """Set, trigger, read until the detector reads intensity < threshold"""
        i = 0
        while True:
            print("LOOP %d" % i)
            yield from abs_set(motor, i)
            yield from trigger(det, wait=True)
            reading = yield from read(det)
            if reading['det']['value'] < threshold:
                print('DONE')
                break
            i += 1

Demo:

.. code-block:: python

    In [5]: RE(conditional_break(0.2))
    LOOP 0
    LOOP 1
    LOOP 2
    DONE
    Out[5]: []

The important line in this example is

.. code-block:: python

    reading = yield from read(det)

The action proceeds like this:

1. The plan yields a 'read' message to the RunEngine.
2. The RunEngine reads the detector.
3. The RunEngine sends that reading *back to the plan*, and that response is
   assigned to the variable ``reading``.

The response, ``reading``, is formatted like:

.. code-block:: python

     {<name>: {'value': <value>, 'timestamp': <timestamp>}, ...}

For a detailed technical description of the messages and their responses,
see :ref:`msg`.

Asynchronous Plans: "Fly Scans" and "Monitoring"
================================================

See the section on :doc:`async` for some context on these terms and, near the
end of the section, some example plans.

.. _plan_utils:

Plan Utilities
==============

These are useful utilities for defining custom plans and plan preprocessors.

.. autosummary::
   :toctree:
   :nosignatures:

    pchain
    msg_mutator
    plan_mutator
    single_gen
    make_decorator

.. currentmodule:: bluesky.plan_stubs

.. autosummary::
   :toctree:
   :nosignatures:

    broadcast_msg
    repeater
    caching_repeater
