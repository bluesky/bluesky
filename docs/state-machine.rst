Interruptions
*************

The RunEngine can be safely interrupted and resumed. All plans get this
feature "for free."

.. _pausing_interactively:

Pausing Interactively
=====================

.. note::

    Looking for a quick refresher on pausing, resuming, or aborting
    interactively? Skip to the :ref:`interactive_pause_summary`.

While the RunEngine is executing a plan, it captures SIGINT (Ctrl+C).

Pause Now: Ctrl+C twice
-----------------------

.. code-block:: python

    In [14]: RE(scan([det], motor, 1, 10, 10))
    Transient Scan ID: 2     Time: 2018/02/12 12:43:12
    Persistent Unique Scan ID: '33a16823-e214-4952-abdd-032a78b8478f'
    New stream: 'primary'
    +-----------+------------+------------+------------+
    |   seq_num |       time |      motor |        det |
    +-----------+------------+------------+------------+
    |         1 | 12:43:13.3 |      1.000 |      0.607 |
    |         2 | 12:43:14.3 |      2.000 |      0.135 |
    |         3 | 12:43:15.3 |      3.000 |      0.011 |
    ^C
    A 'deferred pause' has been requested. The RunEngine will pause at the next checkpoint. To pause immediately, hit Ctrl+C again in the next 10 seconds.
    Deferred pause acknowledged. Continuing to checkpoint.
    ^C
    Pausing...
    ---------------------------------------------------------------------------
    RunEngineInterrupted                      Traceback (most recent call last)
    <ipython-input-14-826ee9dfb918> in <module>()
    ----> 1 RE(scan([det], motor, 1, 10, 10))

    ~/Documents/Repos/bluesky/bluesky/run_engine.py in __call__(self, *args, **metadata_kw)
        670
        671             if self._interrupted:
    --> 672                 raise RunEngineInterrupted(self.pause_msg) from None
        673
        674         return tuple(self._run_start_uids)

    RunEngineInterrupted:
    Your RunEngine is entering a paused state. These are your options for changing
    the state of the RunEngine:

    RE.resume()    Resume the plan.
    RE.abort()     Perform cleanup, then kill plan. Mark exit_stats='aborted'.
    RE.stop()      Perform cleanup, then kill plan. Mark exit_status='success'.
    RE.halt()      Emergency Stop: Do not perform cleanup --- just stop.

Before returning the prompt to the user, the RunEngine ensures that all motors
that it has touched are stopped. It also performs any device-specific cleanup
defined in the device's (optional) ``pause()`` method.

If execution is later resumed, the RunEngine will "rewind" through the plan to
the most recent :ref:`checkpoint <checkpoints>`, the last safe place to restart.

Pause Soon: Ctrl+C once
-----------------------

Pause at the next :ref:`checkpoint <checkpoints>`: typically, the next step in
a step scan. We call this "deferred pause." It avoids having to repeat any work
when the plan is resumed.

Notice that this time when Ctrl+C (^C) is hit, the current step (4) is allowed
to complete before execution is paused.

.. code-block:: python

    In [12]: RE(scan([det], motor, 1, 10, 10))
    Transient Scan ID: 1     Time: 2018/02/12 12:40:36
    Persistent Unique Scan ID: 'c5db9bb4-fb7f-49f4-948b-72fb716d1f67'
    New stream: 'primary'
    +-----------+------------+------------+------------+
    |   seq_num |       time |      motor |        det |
    +-----------+------------+------------+------------+
    |         1 | 12:40:37.6 |      1.000 |      0.607 |
    |         2 | 12:40:38.7 |      2.000 |      0.135 |
    |         3 | 12:40:39.7 |      3.000 |      0.011 |
    ^CA 'deferred pause' has been requested. The RunEngine will pause at the next checkpoint. To pause immediately, hit Ctrl+C again in the next 10 seconds.
    Deferred pause acknowledged. Continuing to checkpoint.
    |         4 | 12:40:40.7 |      4.000 |      0.000 |
    Pausing...
    ---------------------------------------------------------------------------
    RunEngineInterrupted                      Traceback (most recent call last)
    <ipython-input-12-826ee9dfb918> in <module>()
    ----> 1 RE(scan([det], motor, 1, 10, 10))

    ~/Documents/Repos/bluesky/bluesky/run_engine.py in __call__(self, *args, **metadata_kw)
        670
        671             if self._interrupted:
    --> 672                 raise RunEngineInterrupted(self.pause_msg) from None
        673
        674         return tuple(self._run_start_uids)

    RunEngineInterrupted:
    Your RunEngine is entering a paused state. These are your options for changing
    the state of the RunEngine:

    RE.resume()    Resume the plan.
    RE.abort()     Perform cleanup, then kill plan. Mark exit_stats='aborted'.
    RE.stop()      Perform cleanup, then kill plan. Mark exit_status='success'.
    RE.halt()      Emergency Stop: Do not perform cleanup --- just stop.

What to do after pausing
------------------------

After being paused, the RunEngine holds on to information that it might need in
order to resume later. It "knows" that it is in a paused state, and you can
check that at any time:

.. code-block:: python

    In [2]: RE.state
    Out[2]: 'paused'


During the pause, we can do anything: check readings, move motors, etc. It will
not allow you to execute a new plan until the current one is either resumed or
terminated. Your options are:

Resume
^^^^^^

.. code-block:: python

    In [3]: RE.resume()
    |         4 | 07:21:29.5 |     -5.714 |      0.000 |
    |         5 | 07:21:29.5 |     -4.286 |      0.000 |
    |         6 | 07:21:29.6 |     -2.857 |      0.017 |
    |         7 | 07:21:29.7 |     -1.429 |      0.360 |
    (etc.)

Depending on the plan, it may "rewind" to safely continue on and ensure all
data is collected correctly.

Abort
^^^^^

Allow the plan to perform any final cleanup. For example, some plans move
motors back to their starting positions. Mark the data as having been aborted,
so that this fact can be noted (if desired) in later analysis. All of the data
collected up this point will be saved regardless.

From a paused state:

.. code-block:: python

    In [3]: RE.abort()
    Aborting...
    Out[3]: ['8ef9388c-75d3-498c-a800-3b0bd24b88ed']

Stop
^^^^

``RE.stop()`` is functionally identical to ``RE.abort()``. The only
difference is that aborted runs are marked with ``exit_status: 'abort'``
instead of ``exit_status: 'success'``. This may be a useful distinction
during analysis.

Halt
^^^^

Aborting or stopping allows the plan to perform cleanup. We already mentioned
the example of a plan moving motors back to their starting positions at the
end.

In some situations, you may wish to prevent the plan from doing *anything*
--- you want to halt immediately, skipping cleanup. For this, use
``RE.halt()``.

.. _interactive_pause_summary:

Summary
-------

Interactively Interrupt Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

======================= ===========
Command                 Outcome
======================= ===========
Ctrl+C                  Pause soon.
Ctrl+C twice            Pause now.
======================= ===========
    
From a paused state
^^^^^^^^^^^^^^^^^^^

============== ===========
Command        Outcome
============== ===========
RE.resume()    Safely resume plan.
RE.abort()     Perform cleanup. Mark as aborted.
RE.stop()      Perform cleanup. Mark as success.
RE.halt()      Do not perform cleanup --- just stop.
RE.state       Check if 'paused' or 'idle'.
============== ===========

.. _suspenders:

Automated Suspension
====================

It can also be useful to interrupt execution automatically in response to some
condition (e.g., shutter closed, beam dumped, temperature exceeded some limit).
We use the word *suspension* to mean an unplanned pause initialized by some
agent running the background. The agent (a "suspender") monitors some condition
and, if it detects a problem, it suspends execution. When it detects that
conditions have returned to normal, it gives the RunEngine permission to resume
after some interval. This can operate unattended.

.. ipython::
    :verbatim:

    In [1]: RE(scan([det], motor, -10, 10, 15), LiveTable([motor, det]))
    +------------+-------------------+----------------+----------------+
    |   seq_num  |             time  |         motor  |           det  |
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

To take manual control of a suspended plan, pause it by hitting Ctrl+C twice.
You will be given the prompt. When conditions are good again, you may manually
resume using ``RE.resume()``.

.. _installing_suspenders:

Installing Suspenders
---------------------

Bluesky includes several "suspenders" that work with ophyd Signals to monitor
conditions and suspend execution. It's also possible to write suspenders
from scratch to monitor anything at all.

We'll start with an example.

Example: Suspend a plan if the beam current dips low
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This defines a suspender and installs it on the RunEngine. With this, plans
will be automatically suspended when the ``beam_current`` signal goes below 2
and resumed once it exceeds 3.

.. code-block:: python

    from ophyd import EpicsSignal
    from bluesky.suspenders import SuspendFloor

    beam_current = EpicsSignal('...PV string...')
    sus = SuspendFloor(beam_current, 2, resume_thresh=3)
    RE.install_suspender(sus)

In the following example, the beam current dipped below 2 in the middle of
taking the second data point. It later recovered.

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

Notice that the plan was suspended and then resumed. When it resumed, it went
back to the last checkpoint and re-took the second data point cleanly.

See the API documentation (follow the links in the table below) for other
suspender types and options, including a waiting period and cleanup
procedures to run pre-suspend and pre-resume.

Built-in Suspenders
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   bluesky.suspenders.SuspendBoolHigh
   bluesky.suspenders.SuspendBoolLow
   bluesky.suspenders.SuspendFloor
   bluesky.suspenders.SuspendCeil
   bluesky.suspenders.SuspendWhenOutsideBand
   bluesky.suspenders.SuspendWhenChanged

.. _checkpoints:

Checkpoints
===========

Plans are specified as a sequence of :ref:`messages <msg>`, granular
instructions like 'read' and 'set'. The messages can optionally include one
or more 'checkpoint' messages, indicating a place where it is safe to resume
after an interruption. For example, checkpoints are placed before each step of a
:func:`bluesky.plans.scan`.

Some experiments are not resumable: for example, the sample may be melting or
aging. Incorporating :func:`bluesky.plan_stubs.clear_checkpoint` in a plan
makes it un-resuming. If a pause or suspension are requested, the plan will
abort instead.

.. note::

    Some details about checkpoints and when they are allowed:

    It is not legal to create a checkpoint in the middle of a data point
    (between 'create' and 'save'). Checkpoints are implicitly created after
    actions that it is not safe to replay: staging a device, adding a
    monitor, or adding a subscription.

.. _planned_pauses:

Planned Pauses
==============

Pausing is typically done :ref:`interactively <pausing_interactively>` (Ctrl+C)
but it can also be incorporated into a plan. The plan can pause the RunEngine,
requiring the user to type ``RE.resume()`` to continue or ``RE.stop()``
(or similar) to clean up and stop.

.. code-block:: python

    import bluesky.plan_stubs as bps

    def pausing_plan():
        while True:
            yield from some_plan(...)
            print("Type RE.resume() to go again or RE.stop() to stop.")
            yield from bps.checkpoint()  # marking where to resume from
            yield from bps.pause()

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
* ``'paused'``: RunEngine is waiting for user input.

Suspender-related Methods
-------------------------

.. automethod:: bluesky.run_engine.RunEngine.install_suspender
    :noindex:

.. automethod:: bluesky.run_engine.RunEngine.remove_suspender
    :noindex:

.. automethod:: bluesky.run_engine.RunEngine.clear_suspenders
    :noindex:

The RunEngine also has a ``suspenders`` property, a collection of the
currently-installed suspenders.

Request Methods
---------------

This method is called when Ctrl+C is pressed or when a 'pause' Message is
processed. It can also be called by user-defined agents. See the next example.

.. automethod:: bluesky.run_engine.RunEngine.request_pause
    :noindex:

This method is used by the ``PVSuspend*`` classes above. It can also be called
by user-defined agents.

.. automethod:: bluesky.run_engine.RunEngine.request_suspend
    :noindex:


Example: Requesting a pause from the asyncio event loop
-------------------------------------------------------

Since the user does not have control of the prompt, calls to
``RE.request_pause`` must be planned in advance. Here is a example that pauses
the plan after 5 seconds.

.. code-block:: python

    from bluesky.plan_stubs import null

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

Experimental: Record Interruptions
==================================

In the analysis stage, it can be useful to know if and when a run was
interrupted.  This experimental feature creates a special event stream
recording the time and nature of any interruptions.

.. warning::

    This is an experimental feature. It is tested but not yet widely used. It
    might be changed or removed in the future.

Activate this feature by setting

.. code-block:: python

    RE.record_interruptions = True

In this mode, the RunEngine emits a special event descriptor after opening a
new run. This name field in the descriptor is 'interruptions'. It has a single
data key:

.. code-block:: python

    {'interruptions': {'dtype': 'string',
                       'shape': None,
                       'source': 'RunEngine'}}

Each time the RunEngine is paused, suspended, or resumed during the run, an
Event document for that descriptor is created. The data payload
``event['data']['interruptions']`` is ``'pause'``, ``'suspend'``, or
``'resume'``. The associated time notes when the interruptions/resume was
processed.

To see this in action, try this example:

.. code-block:: python

    from bluesky.plans import count
    from bluesky.preprocessors import pchain
    from bluesky.plan_stubs import pause
    from ophyd.sim import det

    RE.record_interruptions = True

    RE(pchain(count([det]), pause(), count([det])), print)
    # ... RunEngine pauses
    RE.resume()

In the text that ``print`` dumps to the screen, look for the special
'interruptions' event descriptor and associated events.
