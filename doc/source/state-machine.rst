.. currentmodule:: bluesky

.. ipython:: python
   :suppress:

    from bluesky import RunEngine
    RE = RunEngine()
    RE.md['owner'] = 'demo'
    RE.md['group'] = 'Grant No. 12345'
    RE.md['config'] = {'detector_model': 'XYZ', 'pxiel_size': 10}
    RE.md['beamline_id'] = 'demo'

Pausing and Resuming
====================

Pausing
-------

The RunEngine can be cleanly paused and resumed. Scans are resumed at specified
checkpoints to ensure that the interruption does not corrupt the data or a miss
a step.

There are three ways to pause:

1. Pressing Ctrl+C
2. Writing a scan with a planned pause step
3. Advanced: Requesting a pause from a thread or event loop

When the RunEngine is paused, it returns control to the user, who can then
choose to resume, stop or abort. It is also possible to resume automatically:
see :ref:`suspending`

Resuming
--------

Use ``RE.resume()`` to resume a paused scan. It will "rewind" to the last
checkpoint and continue from there. If a scan has no checkpoints or has not yet
reached a checkpoint, the RunEngine has no safe place to resume. If told to
pause, it will abort instead.

What's a Checkpoint?
++++++++++++++++++++

Scans are specified as a sequence of messages, simple instructions
like 'read' and 'set'. The instructions can optionally include one or more
'checkpoint' message, indicating a place where it safe to resume after an
interruption. For example, checkpoints are placed before each step of an
`AScan`.

If a scan does not include any 'checkpoint' messages, then it cannot be
resumed after an interruption. If a pause is requested, the scan is aborted
instead.

Stopping or Aborting
--------------------

To stop a paused scan, use ``RE.stop()`` or ``RE.abort()``. In both cases, any
data that has been generated will be saved. The only difference is that
aborted runs are marked with ``exit_status: 'abort'`` instead of
``exit_status: 'success'``, which may be a useful distinction during analysis.

Example: Pausing a scan with Ctrl+C
-----------------------------------

.. ipython::
    :verbatim:

    In [1]: RE(my_scan)
    ^C
    Pausing...
    In [2]:

We have a command prompt back. We can resume like so:

.. ipython::
    :verbatim:

    In [2]: RE.resume()
    Resuming from last checkpoint...
    Out[2]: ['bad06177-32af-47c8-a1b5-2c3e068ac30a']

As explained above, we could also have chosen to end the scan, using
``RE.stop()`` or ``RE.abort()``.

.. _planned-pause:

Example: Using a scan that has a planned pause
----------------------------------------------

As an example, we'll use a step scan that pauses after each step, letting the
user decide whether to continue. We'll revisit this example in a later section
to see the code of the scan itself. For now, we focus on how to use it.

.. ipython:: python

    from bluesky.examples import cautious_stepscan, motor1, det1
    RE(cautious_stepscan(det1, motor1))
    RE.resume()
    RE.resume()
    RE.stop()

.. _suspending:

Suspending
----------

A *suspended* scan does not return the prompt to the user. Like a paused scan,
it stops executing new instructions and rewinds to the most recent checkpoint.
But unlike a paused scan, it can resume execution automatically.

To take manual control of a suspended scan, pause it using Ctrl+C. This will
override its plan to automatically resume.

Example: Suspend a scan if a shutter closes; resume when it opens
-----------------------------------------------------------------

We will use a built-in utility that watches an EPICS PV. It tells the
RunEngine to suspend when the PV's value goes high. When it goes low
again, the RunEngine resumes.

.. ipython::
    :verbatim:

    In [3]: pv_name = 'XF:23ID1-PPS{PSh}Pos-Sts'  # main shutter PV

    In [4]: import bluesky.epics_callbacks

    In [5]: my_s = bluesky.epics_callbacks.PVBoolHigh(RE, pv_name)

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

Notice that the scan was suspended and then resumed.
When it resumed, it went back to the last checkpoint and re-took
the second data point cleanly. As with pausing, if a scan with no checkpoints
is supended, the scan is immediately aborted because it cannot be cleanly
resumed.

Built-in PV Monitors for Conditionally Suspending
-------------------------------------------------

The example above demonstrates ``PVSuspendBoolHigh``. Several other variants
are built in, and it is straightforward to write customized ones.

.. autoclass:: bluesky.epics_callbacks.PVSuspendBoolHigh
.. autoclass:: bluesky.epics_callbacks.PVSuspendBoolLow
.. autoclass:: bluesky.epics_callbacks.PVSuspendFloor
.. autoclass:: bluesky.epics_callbacks.PVSuspendCeil
.. autoclass:: bluesky.epics_callbacks.PVSuspendInBand
.. autoclass:: bluesky.epics_callbacks.PVSuspendOutBand

Deferred Pause
--------------

When a *deferred pause* is requested, the RunEngine continues processing
messages until the next checkpoint or the end of the scan, whichever happens
first. When (if) it reaches a checkpoint, it pauses. Then it can be resumed
from that checkpoint without repeating any work.

Advanced: Pause or Suspend Programmatically
-------------------------------------------

Request a Pause
+++++++++++++++

This method is called when Ctrl+C is pressed or when a 'pause' Message is
processed. It can also be called by user-defined agents. See the next example.

.. automethod:: bluesky.run_engine.RunEngine.request_pause

Request a Suspension
++++++++++++++++++++

This method is used by the ``PVSuspend*`` classes above. It can also be called
by user-defined agents.

.. automethod:: bluesky.run_engine.RunEngine.request_suspend


Advanced Example: Requesting a pause from the asyncio event loop
----------------------------------------------------------------

Since the user does not control of the prompt, calls to ``RE.request_pause``
must be planned in advance. Here is a example that pauses the scan after 5
seconds.

.. ipython:: python

    from bluesky.examples import do_nothing
    import asyncio
    loop = asyncio.get_event_loop()
    # Request a pause 5 seconds from now.
    loop.call_later(5, RE.request_pause, True)  # or False to pause immediately

.. ipython:: python

    RE(do_nothing())
    # Observe that the RunEngine is in a 'paused' state.
    RE.state
    RE.resume()

Above, we passed ``True`` to ``RE.request_pause`` to request a deferred pause.

State Machine
-------------

The RunEngine has a state machine defining its phases of operation and the
allowed transitions between them. As illustrated above, it can be inspected via
the ``state`` property.

.. ipython:: python

    RE.state

The states are:

* ``'idle'``: RunEngine is waiting for instructions.
* ``'running'``: RunEngine is executing instructions.
* ``'paused'``: RunEngine is waiting for user input. It can be 


"Panic": an Emergency Stop
--------------------------

.. warning::

   Bluesky can immediately stop data collection in the event of a emergency
   stop, but it should not be relied on to protect hardware in the event
   of a dangerous condition. It may not have the necessary repsonse time or
   dependability.

A panic is similar to a pause. It is different in the following ways:

* A panic can happen from any state.
* It is requested by calling ``RE.panic()``, a method which takes no
  arguments.
* Once the beamline is "panicked," it is not possible to resume or run a new
  scan until ``RE.all_is_well()`` has been called.
* If a panic happens while RunEngine is in the 'running' state, it always
  aborts the ongoing run without the option of resuming it.
* If a panic happens while the RunEngine is in the 'paused' state, it is
  possible to resume after ``RE.all_is_well()`` has been called.

.. automethod:: bluesky.run_engine.RunEngine.panic
