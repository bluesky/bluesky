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

1. Writing a scan with a planned pause step
2. Pressing Ctrl+C
3. Calling ``RE.request_pause()``

When the RunEngine is paused, it returns control to the user, who can then
choose to resume, stop or abort.

Resuming
--------

Scans are composed of granular instructions like 'read' and 'set'. These
instructions can optionally include one or more 'checkpoint' messages,
indicating a place where it is safe to resume after an interruption. For
example, the built-in step-scans include a checkpoint before each step. If the
scan is interrupted in the middle of a step, it will resume from the beginning
of that step. There is no possibility of data being recorded twice.

When the RunEngnine is paused, use ``RE.resume()`` to resume from the last
checkpoint. If a scan has no checkpoints or has not yet reached a
checkpoint, the RunEngine has no safe place to resume. If paused, it will
abort instead.

Stopping or Aborting
--------------------

To stop a paused scan, use ``RE.stop()`` or ``RE.abort()``. In both cases, any
data that has been generated will be saved. The only difference is that
aborted runs are marked with ``exit_status: 'abort'`` instead of
``exit_status: 'success'``, which may be a useful distinction during analysis.

Scans are specified as a sequence of messages, simple instructions
like 'read' and 'set'. The instructions can optionally include one or more
'checkpoint' message, indicating a place where it safe to resume after an
interruption. For example, checkpoints are placed before each step of an
`AScan`.

If a scan does not include any 'checkpoint' messages, then it cannot be
resumed after an interruption. If a pause is requested, the scan is aborted
instead.

Deferred Pause
--------------

When a *deferred pause* is requested, the RunEngine continues processing
messages until the next checkpoint or the end of the scan, whichever happens
first. When (if) it reaches a checkpoint, it pauses. Then it can be resumed
from that checkpoint without repeating any work.

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

Example: Requesting a pause after a delay
-----------------------------------------

Suppose we want to pause a scan if it hasn't finished in a certian amount
of time. Then, we can decide whether to continue or abort.

The user cannot call ``RE.request_pause()`` directly while a scan is running.
It can only be a done from a separate thread. Here is a example code for a
pausing a scan if it goes on for more than 2 seconds.

.. ipython:: python

    import threading
    from time import sleep
    class SimplePausingAgent:
        def __init__(self, RE):
            self.RE = RE
        def issue_request(self, defer, delay=0):
            def requester():
                sleep(delay)
                self.RE.request_pause(defer)
            thread = threading.Thread(target=requester)
            thread.start()

We'll try it on a dummy scan that just loops through checkpoints for several
seconds.

.. ipython:: python

    from bluesky.examples import do_nothing
    agent = SimplePausingAgent(RE)
    # Pause the scan two seconds from when this is executed:
    agent.issue_request(hard=False, delay=2)
    RE(do_nothing())
    # Observe that the RunEngine is in a 'paused' state.
    RE.state
    RE.resume()

In some cases you may wish to prohibit resuming until whatever condition
that required the pause has passed. For example, an agent might request a
pause if the beam current becomes too low. It should sustain the pause request
utnil the beam current has recovered. To accomplish this, include a name and
a function with the pause request.

.. ipython:: python

    class PausingAgent:
        def __init__(self, RE, name):
            self.RE = RE
            self.name = name
        def issue_request(self, hard, delay=0):
            def callback():
                return self.permission
            def requester():
                sleep(delay)
                self.permission = False
                self.RE.request_pause(hard, self.name, callback)
            thread = threading.Thread(target=requester)
            thread.start()

When ``resume`` is called, the RunEngine will call ``callback()`` to check
whether is has permission to lift the pause.

.. ipython:: python

    agent = PausingAgent(RE, 'Timed Pausing Agent')
    agent.issue_request(hard=False, delay=2)
    RE(do_nothing())
    RE.state

This time, if we attempted to resume, the RunEngine will return a list of
names of pause requests that must be revoked before the pause can be lifted.

.. ipython:: python

    RE.resume()  # not allowed!
    agent.permission = True
    RE.resume()

When ``resume`` is called, the RunEngine will call ``callback()`` to check
whether is has permission to lift the pause.

Example: Use a file to trigger a pause
--------------------------------------

The example above can be adapted to do something a little different.
Instead of pausing based on a
timed delay, we can monitor a PV or a file. The example below regularly
checks a filepath. If it find a file there, it requests a pause.

.. ipython:: python

    class FileBasedPausingAgent:
        def __init__(self, RE, name, filepath, hard=False):
            self.RE = RE
            self.name = name
            self.filepath = filepath
            self.hard = hard
            def callback():
                return self.permission
            def requester():
                while True:
                    sleep(1)  # Check every second.
                    if self.permission:
                        if os.path.isfile(self.filepath):
                            # The file has been created! Request a pause.
                            self.permission = False
                            self.RE.request_pause(self.hard, self.name,
                                                  callback)
                    else:
                        if not os.path.isfile(self.filepath):
                            # The file is gone. Give permission to resume.
                            self.permission = True
            thread = threading.Thread(target=requester)
            thread.start()


State Machine
-------------

The RunEngine has a state machine defining its phases of operation and the
allowed transitions between them. As illustrated above, it can be inspected via
the ``state`` property.

.. ipython:: pytohn

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
