.. currentmodule:: bluesky.simulators

Simulation and Error Checking
=============================

Bluesky provides three different approaches for simulating a plan without
actually executing it:

1. Introspect a plan by passing it to a "simulator" instead of a RunEngine.
2. Execute a plan with the real RunEngine, but use simulated hardware objects.
3. Redefine the RunEngine commands to change their meanings.

Approaches (1) and (2) are the most straightforward and most common.

Introspection
-------------

Recall that plans yield messages that *describe* what should be done; they
do not communicate with hardware directly. Therefore it's easy to use (or
write) a simple function that iterates through the plan and summarizes or
analyzes its actions.

.. autosummary::
   :toctree: generated
   :nosignatures:

   summarize_plan
   plot_raster_path
   check_limits

Summarize
^^^^^^^^^

The simulator :func:`summarize_plan` print a summary of what a plan would do if
executed by the RunEngine.

.. ipython:: python

    from bluesky.simulators import summarize_plan
    from ophyd.sim import det, motor
    from bluesky.plans import scan
    summarize_plan(scan([det], motor, 1, 3 ,3))

To see the unabridged contents of a plan, simply use the builtin Python
function :func:`list`. Note that it is not possible to summarize plans that
have adaptive logic because their contents are determined dynamically during
plan executation.

.. ipython:: python

    list(scan([det], motor, 1, 3 ,3))

Check Limits
^^^^^^^^^^^^

.. ipython:: python
    :suppress:

    motor.limits = (-1000, 1000)

Suppose that this motor is configured with limits on its range of motion at +/-
1000. The :func:`check_limits` simulator can verify whether or not a plan will
violate these limits, saving you from discovering this part way through a long
experiment.

.. ipython:: python
    :okexcept:

    from bluesky.simulators import check_limits

    check_limits(scan([det], motor, 1, 3 ,3))  # no problem here
    check_limits(scan([det], motor, 1, -3000, 3000))  # should raise an error

Simulated Hardware
------------------

.. warning::

    This feature has recently been changed, and it has yet to be documented.
    
Customizing RunEngine Methods
-----------------------------

The RunEngine allows you to customize the meaning of commands (like 'set' and
'read'). One could use this feature to create a dummy RunEngine that, instead
of actually reading and writing to hardware, merely reports what it *would*
have done.

.. automethod:: bluesky.run_engine.RunEngine.register_command
   :noindex:

.. automethod:: bluesky.run_engine.RunEngine.unregister_command
   :noindex:
