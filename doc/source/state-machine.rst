Interruptions
*************

The RunEngine can be cleanly paused and resumed. Plans are resumed at specified
checkpoints to ensure that the interruption does not corrupt the data or a miss
a step. When an interruption occurs, the RunEngine "rewinds" the plan and
carefully re-executes the steps between the interruption and last checkpoint.

Pausing and Suspending
======================

A pause can be initiated interactively:

* Ctrl+C once: Pause at the next checkpoint (e.g., finish the current step
  in a scan first).
* Ctrl+C twice: Pause immediately.

.. ipython:: python
    :verbatim:

    In [1]: RE(scan([det], motor, -10, 10, 15), subs)
    +-----------+------------+------------+------------+
    |   seq_num |       time |      motor |        det |
    +-----------+------------+------------+------------+
    |         1 | 07:21:29.2 |    -10.000 |      0.000 |
    |         2 | 07:21:29.3 |     -8.571 |      0.000 |
    |         3 | 07:21:29.4 |     -7.143 |      0.000 |
    ^C^C
    Pausing...
    In [2]:

or planned as part of the logic of an experiment:

.. ipython:: python
    :verbatim:

    # count; pause and wait for the user to resume; count again
    In [1]: RE(pchain(count([det]), pause(), count([det]))

When the RunEngine is paused, it returns the command prompt to the user.
During the pause, the user can do anything: check readings, move motors, etc.
Then, from a paused state, the user can choose to resume:

.. ipython:: python
    :verbatim:

    In [2]: RE.resume()
    Resuming from last checkpoint...
    |         4 | 07:21:29.5 |     -5.714 |      0.000 |
    |         5 | 07:21:29.5 |     -4.286 |      0.000 |
    |         6 | 07:21:29.6 |     -2.857 |      0.017 |
    |         7 | 07:21:29.7 |     -1.429 |      0.360 |
    (etc.)

or choose to stop/abort. (Read on for the distinction between these two.)

.. ipython:: python
    :verbatim:

    In [3]: RE.abort()
    Aborting...
    Out[3]: ['8ef9388c-75d3-498c-a800-3b0bd24b88ed']

It can also be useful to interrupt execution automatically in response some
condition (e.g., shutter closed, beam dumped, temperature exceed some limit).
We use the word *suspension* to mean an unplanned pause initialized by some
agent running the background. The agent (a "suspender") monitors some condition
and, if it detects a problem, it suspends execution. When it detects that
conditions have returned to normal, it gives the RunEngine permission to resume
after some interval. This can operate unattended.

.. ipython::
    :verbatim:

    In [6]: RE(my_scan)
    +------------+-------------------+----------------+----------------+
    |   seq_num  |             time  |         theta  |    sclr_chan4  |
    +------------+-------------------+----------------+----------------+
    |         1  |  16:46:08.953815  |          0.03  |        290.00  |
    Suspending....To get prompt hit Ctrl-C to pause the scan
    |         2  |  16:46:20.868445  |          0.09  |        279.00  |
    |         3  |  16:46:29.077690  |          0.16  |        284.00  |
    |         4  |  16:46:33.540643  |          0.23  |        278.00  |
    +------------+-------------------+----------------+----------------+

A *suspended* plan does not return the prompt to the user. Like a paused plan,
it stops executing new instructions and rewinds to the most recent checkpoint.
But unlike a paused plan, it resumes execution automatically when conditions
return to normal.

To take manual control of a suspended plan, pause it using Ctrl+C. This will
override its plan to automatically resume.

Read on for an example of installing a suspender.

Checkpoints
-----------

Plan are specified as a sequence of granualor instructions like 'read' and
'set'. The instructions can optionally include one or more 'checkpoint'
messages, indicating a place where it safe to resume after an interruption. For
example, checkpoints are placed before each step of a `bluesky.plans.scan`.

Some experiments are not resumable: for example, the sample may be melting or
aging. Incorporating `bluesky.plans.clear_checkpoint` in a plan makes it
un-resuming. If a pause or suspension are requested, the plan will abort
instead.

.. note::

    For developers, here some gritty details about checkpoints.

    It is not legal to create checkpoint in the middle of a data point (between
    'create' and 'save') Checkpoints are implicitly created after actions that
    it is not safe to replay: staging a device, adding a monitor, or adding a
    subscription.


Deferred Pause vs Hard Pause
----------------------------

When a *deferred pause* is requested (Ctrl+C once), the RunEngine continues
processing messages until the next checkpoint or the end of the plan, whichever
happens first. When (if) it reaches a checkpoint, it pauses. Then it can be
resumed from that checkpoint without repeating any work.

When a *hard pause* is requested (Ctrl+C twice), the RunEngine pauses as soon
as possible --- normally within less than second.

Stopping vs Aborting
--------------------

To stop a paused plan, use ``RE.stop()`` or ``RE.abort()``. In both cases, any
data that has been generated will be saved. The only difference is that
aborted runs are marked with ``exit_status: 'abort'`` instead of
``exit_status: 'success'``, which may be a useful distinction during analysis.

Suspenders
==========

Bluesky includes several "suspenders" that work with ophyd Signals to monitor
conditions and suspend execution. It's also possible to write suspenders
from scratch to monitor anything at all.

We'll start with an example.

Example: Suspend a plan if a shutter closes; resume when it opens
-----------------------------------------------------------------

We will use a built-in utility that watches an EPICS PV. It tells the
RunEngine to suspend when the PV's value goes high. When it goes low
again, the RunEngine resumes.

.. code-block:: python

    from ophyd import EpicsSignal
    from bluesky.suspenders SuspendBoolHigh

    shutter = EpicsSignal('XF:23ID1-PPS{PSh}Pos-Sts')  # main shutter PV

    sus = SuspendBoolHigh(shutter)
    RE.install_suspender(sus)

The above is all that is required. It will watch the PV indefinitely.
In the following example, the shuttle was closed in the middle of the
second data point.

.. ipython::
    :verbatim:

    In [6]: RE(my_scan)
    +------------+-------------------+----------------+----------------+
    |   seq_num  |             time  |         theta  |    sclr_chan4  |
    +------------+-------------------+----------------+----------------+
    |         1  |  16:46:08.953815  |          0.03  |        290.00  |
    Suspending....To get prompt hit Ctrl-C to pause the scan
    |         2  |  16:46:20.868445  |          0.09  |        279.00  |
    |         3  |  16:46:29.077690  |          0.16  |        284.00  |
    |         4  |  16:46:33.540643  |          0.23  |        278.00  |
    +------------+-------------------+----------------+----------------+

Notice that the plan was suspended and then resumed.  When it resumed, it went
back to the last checkpoint and re-took the second data point cleanly.

Built-in Suspenders
-------------------

The example above demonstrates ``SuspendBoolHigh``. Several other variants
are built in, and it is straightforward to write customized ones.

.. autosummary::
   :toctree:
   :nosignatures:

   bluesky.suspenders.SuspendBoolHigh
   bluesky.suspenders.SuspendBoolLow
   bluesky.suspenders.SuspendFloor
   bluesky.suspenders.SuspendCeil
   bluesky.suspenders.SuspendInBand
   bluesky.suspenders.SuspendOutBand

Deferred Pause
--------------

When a *deferred pause* is requested, the RunEngine continues processing
messages until the next checkpoint or the end of the plan, whichever happens
first. When (if) it reaches a checkpoint, it pauses. Then it can be resumed
from that checkpoint without repeating any work.

Associated RunEngine Interface
==============================

State
-----

The RunEngine has a state machine defining its phases of operation and the
allowed transitions between them. As illustrated above, it can be inspected via
the ``state`` property.

The states are:

* ``'idle'``: RunEngine is waiting for instructions.
* ``'running'``: RunEngine is executing instructions.
* ``'paused'``: RunEngine is waiting for user input. It can be 

Request Methods
---------------

This method is called when Ctrl+C is pressed or when a 'pause' Message is
processed. It can also be called by user-defined agents. See the next example.

.. automethod:: bluesky.run_engine.RunEngine.request_pause

This method is used by the ``PVSuspend*`` classes above. It can also be called
by user-defined agents.

.. automethod:: bluesky.run_engine.RunEngine.request_suspend


Example: Requesting a pause from the asyncio event loop
-------------------------------------------------------

Since the user does not control of the prompt, calls to ``RE.request_pause``
must be planned in advance. Here is a example that pauses the plan after 5
seconds.

.. code-block:: python

    from bluesky.plans import null

    def loop_forever():
        "a silly plan"
        while True:
            yield from null()

    import asyncio
    loop = asyncio.get_event_loop()
    # Request a pause 5 seconds from now.
    loop.call_later(5, RE.request_pause)

    # Execute the plan.
    RE(loop_forever())

    # Five seconds after ``call_later`` was run, the plan is paused.
    # Observe that the RunEngine is in a 'paused' state.
    RE.state

Above, we passed ``True`` to ``RE.request_pause`` to request a deferred pause.
