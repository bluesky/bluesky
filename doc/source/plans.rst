.. currentmodule:: bluesky.plans

=====
Plans
=====

A *plan* is bluesky's concept of an experimental procedure. A plan may be any
iterable object (list, tuple, custom iterable class, ...) but most commonly it
is implemented as a Python generator. For a more technical discussion we refer
you :doc:`msg`.

A variety of pre-assembled plans are provided. Like sandwiches on a deli menu,
you can use our pre-assembled plans or assemble your own from the same
ingredients, catalogued under the heading :ref:`stub_plans` below.

.. note:: 

    In the examples that follow, we will assume that you have a RunEngine
    instance named ``RE``. This may have already been configured for you if you
    are a user at a facility that runs bluesky. See
    :ref:`this section of the tutorial <tutorial_run_engine_setup>` to sort out
    if you already have a RunEngine and to quickly make one if needed.

.. _preassembled_plans:

Pre-assembled Plans
===================

Below this summary table, we break the down the plans by category and show
examples with figures.

Summary
-------

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
   grid_scan
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
   ramp_plan
   fly


Time series ("count")
---------------------

Examples:

.. code-block:: python

    from ophyd.sim import det
    from bluesky.plans import count

    # a single reading of the detector 'det'
    RE(count([det]))

    # five consecutive readings
    RE(count([det], num=5))

    # five sequential readings separated by a 1-second delay
    RE(count([det], num=5, delay=1))

    # a variable delay
    RE(count([det], num=5, delay=[1, 2, 3, 4]))

    # Take readings forever, until interrupted (e.g., with Ctrl+C)
    RE(count([det], num=None))

.. code-block:: python

    # We'll use the 'noisy_det' example detector for a more interesting plot.
    from ophyd.sim import noisy_det

    RE(count([noisy_det], num=5))


.. plot::

    from bluesky import RunEngine
    from bluesky.plans import count
    from ophyd.sim import noisy_det
    from bluesky.callbacks.best_effort import BestEffortCallback
    bec = BestEffortCallback()
    RE = RunEngine({})
    RE.subscribe(bec)
    RE(count([noisy_det], num=5))

.. autosummary::
   :toctree: generated
   :nosignatures:

   count

Scans over one dimension
------------------------

The "dimension" might be a physical motor position, a temperature, or a
pseudo-axis. It's all the same to the plans. Examples:

.. code-block:: python

    from ophyd.sim import det, motor
    from bluesky.plans import scan, rel_scan, list_scan

    # scan a motor from 1 to 5, taking 5 equally-spaced readings of 'det'
    RE(scan([det], motor, 1, 5, 5))

    # scan a motor from 1 to 5 *relative to its current position*
    RE(rel_scan([det], motor, 1, 5, 5))

    # scan a motor through a list of user-specified positions
    RE(list_scan([det], motor, [1, 1, 2, 3, 5, 8]))

.. code-block:: python

    RE(scan([det], motor, 1, 5, 5))

.. plot::

    from bluesky import RunEngine
    from bluesky.plans import scan
    from ophyd.sim import det, motor
    RE = RunEngine({})
    from bluesky.callbacks.best_effort import BestEffortCallback
    bec = BestEffortCallback()
    RE.subscribe(bec)
    RE(scan([det], motor, 1, 5, 5))

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
-----------------------

See :ref:`tutorial_multiple_motors` in the tutorial for an introduction to the
common cases of moving multiple motors in coordination (i.e. moving X and Y
along a diagonal) with :func:`~bluesky.plans.scan` or in a grid with
:func:`~bluesky.plans.grid_scan` with equal spacing.

Both :func:`~bluesky.plan.scan` and :func:`~bluesky.plans.grid_scan` are built
on a more general-purpose plan, :func:`~bluesky.plan.scan_nd`, which we can use
for more specialized cases, such as:

* grids or trajectories with unequally-spaced steps
* moving some motors together

Some jargon: we speak of :func:`~bluesky.plans.scan`-like joint movement as an
"inner product" of trajectories and :func:`~bluesky.plans.grid_scan`-like
movement as an "outer product" of trajectories. The general case, moving some
motors together in an "inner product" against another motor (or motors) in an
"outer product," can be addressed using a ``cycler``.  Notice what happens when
we add or multiply ``cycler`` objects.

.. ipython:: python

    from cycler import cycler
    from ophyd.sim import motor1, motor2, motor3

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
   :toctree: generated
   :nosignatures:

   grid_scan
   rel_grid_scan
   scan_nd

Spiral trajectories
-------------------

We provide two-dimensional scans that trace out spiral trajectories.

A simple spiral:

.. plot::
   :include-source:

    from bluesky.simulators import plot_raster_path
    from ophyd.sim import motor1, motor2, det
    from bluesky.plans import spiral

    plan = spiral([det], motor1, motor2, x_start=0.0, y_start=0.0, x_range=1.,
                  y_range=1.0, dr=0.1, nth=10)
    plot_raster_path(plan, 'motor1', 'motor2', probe_size=.01)


A fermat spiral:

.. plot::
   :include-source:

    from bluesky.simulators import plot_raster_path
    from ophyd.sim import motor1, motor2, det
    from bluesky.plans import spiral_fermat

    plan = spiral_fermat([det], motor1, motor2, x_start=0.0, y_start=0.0,
                         x_range=2.0, y_range=2.0, dr=0.1, factor=2.0, tilt=0.0)
    plot_raster_path(plan, 'motor1', 'motor2', probe_size=.01, lw=0.1)


.. autosummary::
   :toctree: generated
   :nosignatures:

   spiral
   spiral_fermat
   rel_spiral
   rel_spiral_fermat

Adaptive scans
--------------

These are one-dimension scans with an adaptive step size tuned to move quickly
over flat regions can concentrate readings in areas of high variation by
computing the local slope aiming for a target delta y between consecutive
points.

This is a basic example of the power of adaptive plan logic.

.. code-block:: python

    from bluesky.plans import adaptive_scan
    from ophyd.sim import motor, det

    RE(adaptive_scan([det], 'det', motor,
                     start=-15,
                     stop=10,
                     min_step=0.01,
                     max_step=5,
                     target_delta=.05,
                     backstep=True))

.. plot::

    from bluesky import RunEngine
    from bluesky.plans import adaptive_scan
    from bluesky.callbacks.best_effort import BestEffortCallback
    bec = BestEffortCallback()
    from ophyd.sim import motor, det

    RE = RunEngine({})
    RE.subscribe(bec)

    RE(adaptive_scan([det], 'det', motor,
                     start=-15,
                     stop=10,
                     min_step=0.01,
                     max_step=5,
                     target_delta=.05,
                     backstep=True))

From left to right, the scan lengthens its stride through the flat region. At
first, it steps past the peak. The large jump causes it to double back and then
sample more densely through the peak. As the peak flattens, it lengthens its
stride again.

.. autosummary::
   :toctree: generated
   :nosignatures:

   adaptive_scan
   rel_adaptive_scan

Misc.
-----

.. autosummary::
   :toctree: generated
   :nosignatures:

   tweak
   fly

.. _stub_plans:

Stub Plans
==========
.. currentmodule:: bluesky.plan_stubs

These are the aforementioned "ingredients" for remixing, the pieces from which
the pre-assembled plans above were made. See :ref:`tutorial_custom_plans` in
the tutorial for a practical introduction to these components.

Plans for interacting with hardware:

.. autosummary::
   :nosignatures:
   :toctree: generated

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
   :toctree: generated

    monitor
    unmonitor
    kickoff
    complete
    collect

Plans that control the RunEngine:

.. autosummary::
   :nosignatures:
   :toctree: generated

    open_run
    close_run
    create
    save
    drop
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

Special utilities:

.. autosummary::

   repeat
   repeater
   caching_repeater
   broadcast_msg

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

We have installed a "preprocessor" on the RunEngine. A preprocessor modifies
plans, supplementing or altering their instructions in some way. From now on,
every time we type ``RE(some_plan())``, the RunEngine will silently change
``some_plan()`` to ``sd(some_plan())``, where ``sd`` may insert some extra
instructions. Envision the instructions flow from ``some_plan`` to ``sd`` and
finally to ``RE``. The ``sd`` preprocessors has the opportunity to inspect
he
instructions as they go by and modify them as it sees fit before they get
processed by the RunEngine.

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
   :toctree: generated

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

.. _per_step_hook:

Customize Step Scans with ``per_step``
======================================

The one-dimensional and multi-dimensional plans are composed (1) setup,
(2) a loop over a plan to perform at each position, (3) cleanup.

We provide a hook for customizing step (2). This enables you to write a
variation of an existing plan without starting from scratch.

For one-dimensional plans, the default inner loop is:

.. code-block:: python

    from bluesky.plan_stubs import checkpoint, abs_set, trigger_and_read

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
    from bluesky.plan_stubs import checkpoint, abs_set, wait, trigger_and_read

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

Asynchronous Plans: "Fly Scans" and "Monitoring"
================================================

See the section on :doc:`async` for some context on these terms and, near the
end of the section, some example plans.

.. _plan_utils:

Plan Utilities
==============

These are useful utilities for defining custom plans and plan preprocessors.

.. autosummary::
   :toctree: generated
   :nosignatures:

    pchain
    msg_mutator
    plan_mutator
    single_gen
    make_decorator

.. currentmodule:: bluesky.plan_stubs

.. autosummary::
   :toctree: generated
   :nosignatures:

    broadcast_msg
    repeat
