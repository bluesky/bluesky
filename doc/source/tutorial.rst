********
Tutorial
********

Before You Begin
================

* You will need Python 3.5 or newer. From a shell ("Terminal" on OSX,
  "Command Prompt" on Windows), check your current Python version.

  .. code-block:: bash

    python --version

  If that version is less than 3.5, you can easily install a new version of
  Python into "virtual environment". It will not interfere with any existing
  Python software:

  .. code-block:: bash

    python -m venv ~/bluesky-tutorial
    source ~/bluesky-tutorial/bin/activate

  Alternatively, if you are a
  `conda <https://conda.io/docs/user-guide/install/download.html>`_ user,
  you can create a conda environment:

  .. code-block:: bash

    conda create -n bluesky-tutorial "python>=3.5"
    conda activate bluesky-tutorial

* Install the latest versions of bluesky and ophyd. And, optionally, ipython (a
  Python interpreter designed by scientists for scientists).

  .. code-block:: bash

     python -m pip install --upgrade bluesky ophyd ipython

* Start ``ipython``. Can you ``import bluesky``? If so, you are ready to go.

If you get lost or confused...
==============================

...then we want to know! We have a friendly
`chat channel <https://gitter.im/NSLS-II/DAMA>`_, or you can
`file a bug <https://github.com/NSLS-II/Bug-Reports/issues>`_ to let us know
where our documentation could be made more clear.

Devices
=======

The notion of a "Device" serves two goals:

* Provide a standard interface to all hardware for the sake of generality
  and code reuse.
* Logically group individual signals into composite "Devices" that can be read
  together, as a unit, and configured in a coordinated way.

In bluesky's view of the world, there are only three different kinds of devices
used in data acquisition.

* Some devices can be read. This includes simple points detectors that produce
  a single number and large CCD detectors that produce big arrays.
* Some devices can be both read and set. Setting a motor physically moves it to
  a new position. Setting a temperature controller impels it to gradually
  change its temperature. Setting the exposure time on some detector promptly
  updates its configuration.
* Some devices produce data at a rate too high to be read out in real time, and
  instead buffer their data temporarily on a device or separate software.

Bluesky interacts with all devices via a :doc:`specified interface <hardware>`.
Each device is represented by a Python object with certain methods and
attributes (with names like ``read`` and ``set``). Some of these methods are
asynchronous, such as ``set``, which allows for the concurrent movement of
multiple devices.

`Ophyd <https://nsls-ii.github.io/ophyd>`_, a Python library that was
developed in tandem with bluesky, implements this interface for devices that
speak `EPICS <http://www.aps.anl.gov/epics/>`_. But bluesky is not tied to
ophyd specifically: any Python object may be used, so long as it provides the
specified methods and attributes that bluesky expects. For example, a
separately-developed library has experimentally implemented the bluesky
interface for LabView.

For example, to get a flavor for what it looks like to configure hardware in
ophyd, connecting to an EPICS motor looks like this:

.. code-block:: python

    from ophyd import EpicsMotor

    nano_top_x = EpicsMotor('XF:23ID1-ES{Dif:Nano-Ax:TopX}Mtr', name='nano_top_x')

The ``EpicsMotor`` device is a logical grouping of many signals. The most
important are the readback (actual position) and setpoint (target position).
All of the signals are summarized thus. The details here aren't important: the
take-away message is, "There is a lot of stuff to keep track of about a motor,
and a Device helpfully groups that stuff for us."

.. code-block:: none

    In [3]: nano_top_x.summary()
    data keys (* hints)
    -------------------
    *nano_top_x
    nano_top_x_user_setpoint

    read attrs
    ----------
    user_readback        EpicsSignalRO       ('nano_top_x')
    user_setpoint        EpicsSignal         ('nano_top_x_user_setpoint')

    config keys
    -----------
    nano_top_x_acceleration
    nano_top_x_motor_egu
    nano_top_x_user_offset
    nano_top_x_user_offset_dir
    nano_top_x_velocity

    configuration attrs
    ----------
    motor_egu            EpicsSignal         ('nano_top_x_motor_egu')
    velocity             EpicsSignal         ('nano_top_x_velocity')
    acceleration         EpicsSignal         ('nano_top_x_acceleration')
    user_offset          EpicsSignal         ('nano_top_x_user_offset')
    user_offset_dir      EpicsSignal         ('nano_top_x_user_offset_dir')

    Unused attrs
    ------------
    offset_freeze_switch EpicsSignal         ('nano_top_x_offset_freeze_switch')
    set_use_switch       EpicsSignal         ('nano_top_x_set_use_switch')
    motor_is_moving      EpicsSignalRO       ('nano_top_x_motor_is_moving')
    motor_done_move      EpicsSignalRO       ('nano_top_x_motor_done_move')
    high_limit_switch    EpicsSignal         ('nano_top_x_high_limit_switch')
    low_limit_switch     EpicsSignal         ('nano_top_x_low_limit_switch')
    direction_of_travel  EpicsSignal         ('nano_top_x_direction_of_travel')
    motor_stop           EpicsSignal         ('nano_top_x_motor_stop')
    home_forward         EpicsSignal         ('nano_top_x_home_forward')
    home_reverse         EpicsSignal         ('nano_top_x_home_reverse')

For this tutorial, we will not assume that you have access to real detectors or
motors. In the examples that follow, we will use simulated hardware from
ophyd's module ``ophyd.sim``, such as:

.. code-block:: python

    from ophyd.sim import det, motor

The RunEngine
=============

The RunEngine is the heart of bluesky, and we'll understand it better through
the examples that follow.

.. code-block:: python

    from bluesky import RunEngine

    RE = RunEngine({})

This RunEngine is ready to use --- but if you care about visualizing or saving
your data, there is more to do first....

Prepare Live Visualization
--------------------------

The RunEngine dispatches a live stream of metadata and data to one or more
consumers ("callbacks") for in-line data processing and visualization and
long-term storage. Example consumers include a live-updating plot, a curve-fitting
algorithm, a database, a message queue, or a file in your preferred format.

To start, let's use the all-purpose "Best-Effort Callback".

.. code-block:: python

    from bluesky.callbacks.best_effort import BestEffortCallback
    bec = BestEffortCallback()

    # Send all metadata/data captured to the BestEffortCallback.
    RE.subscribe(bec)

    # Make plots update live while scans run.
    from bluesky.utils import install_kicker
    install_kicker()

The Best-Effort Callback will receive the metadata/data in real time and
produce plots and text, doing its best to provide live feedback that strikes
the right balance between "comprehensive" and "overwhelming." For more tailored
feeback, taking account of the details of the experiment, you may configure
custom callbacks.

Prepare Data Saving/Export
--------------------------

The `databroker <https://nsls-ii.github.io>`_, also co-developed with bluesky,
is an interface to long-term, searchable storage for metadata and data
generated by bluesky. Additionally (or *alternatively*, if you are not
interested in the databroker) the metadata and data can be written into the
file format of your choice. See :ref:`examples <export>`.

For this tutorial, we will spin up a databroker backed by a temporary database.

.. code-block:: python

    from databroker import Broker
    db = Broker.named('temp')

    # Insert all metadata/data captured into db.
    RE.subscribe(db.insert)

.. warning::

    **This example makes a temporary database. Do not use it for important
    data.** The data will become difficult to access once Python exits or the
    variable ``db`` is deleted. Running ``Broker.named('temp')`` a second time
    creates a fresh, separate temporary database.

The RunEngine can do a lot more than this, but let's hold that thought for
later in the tutorial (:ref:`things_the_run_engine_can_do_for_free`). Let's
take some data!

Common Experiments ("Plans")
============================

Read Some Detectors
-------------------

Begin with a very simple experiment: trigger and read some detectors.
 
Bluesky calls this "counting" detectors---a term of art inherited from the
spectroscopy community. Before we begin, we'll need some simulated detectors
from ophyd's module of simulated hardware.

.. code-block:: python

    from ophyd.sim import det1, det2

Using the RunEngine, configured in the previous section, "count" the detectors:

.. code-block:: python

    from bluesky.plans import count
    dets = [det1, det2]   # a list of any number of detectors
 
    RE(count(dets))

A key feature of bluesky is that these detectors could simple photodiodes or
complex CCDs. All of those details are captured in the implementation of the
Device. From the point of view of bluesky, detectors are just Python objects
with certain methods.

See :func:`~bluesky.plans.count` for more options. You can also view this
documentation in IPython by typing ``count?``.

Try the following variations:

.. code-block:: python

    # five consecutive readings
    RE(count(dets, num=5))

    # five sequential readings separated by a 1-second delay
    RE(count(dets, num=5, delay=1))

    # a variable delay
    RE(count(dets, num=5, delay=[1, 2, 3, 4]))

    # Take readings forever, until interrupted (e.g., with Ctrl+C)
    RE(count(dets, num=None))
    # RunEngine is paused by Ctrl+C. It now needs to be 'stopped'.
    # See later section of tutorial for more on this....
    RE.stop()

Scan
----

Use :func:`~bluesky.plans.scan` to scan ``motor`` from ``-1`` to ``1`` in ten
equally-spaced steps, wait for it to arrive at each step, and then trigger and
read some detector, ``det``.

.. code-block:: python

    from ophyd.sim import det, motor
    from bluesky.plans import scan
    dets = [det]   # just one in this case, but it could be more than one

    RE(scan(dets, motor, -1, 1, 10))

A key feature of bluesky is that ``motor`` may be any "movable" devices,
including a temperature controller, a sample changer, or some pseudo-axis. From
the point of view of bluesky and the RunEngine, all of these are just objects
in Python with certain methods.

Use :func:`~bluesky.plans.rel_scan` to scan from ``-1`` to ``-1`` *relative to
the current position*.

.. code-block:: python

    RE(rel_scan(dets, motor, -1, 1, 10))

Use :func:`~bluesky.plans.list_scan` to scan points with some arbitrary
spacing.

.. code-block:: python

    points = [1, 1, 2, 3, 5, 8, 13]

    RE(list_scan(dets, motor, points))

For a complete list of scan variations see :doc:`plans`.

Scan Multiple Motors Together
-----------------------------

Again, we emphasize that these "motors" could be anything that can be set
(temperature controller, pseudo-axis, sample changer).

Scan Multiple Motors in a Grid
------------------------------

What is a "Plan" Really?
========================

Compose a Series of Plans
=========================

condensed aside on yield from

some plan stubs

.. _things_the_run_engine_can_do_for_free:

Things the RunEngine Can Do For Free
====================================

Interactive Pause & Resume
--------------------------

Sometimes it is convenient to pause data collection, check on some things, and
then either resume from where you left off or quit. The RunEngine makes it
possible to do this cleanly and safely on *every* plan, including user-defined
ones, with no special effort by the user.

(Of course, experiments on systems that evolve with time can't be arbitrarily
paused and resumed. It's up to the user to know that and use this feature only
when applicable.)

Take this example, a step scan over ten points.

.. code-block:: python

    from ophyd.sim import det, motor
    from bluesky.plans import scan

    motor.delay = 1  # simulate slow motor movement
    RE(scan([det], motor, 1, 10, 10))

Demo:

.. ipython::
    :verbatim:

    In [1]: RE(scan([det], motor, 1, 10, 10))
    Transient Scan ID: 1     Time: 2018/02/12 12:40:36
    Persistent Unique Scan ID: 'c5db9bb4-fb7f-49f4-948b-72fb716d1f67'
    New stream: 'primary'
    +-----------+------------+------------+------------+
    |   seq_num |       time |      motor |        det |
    +-----------+------------+------------+------------+
    |         1 | 12:40:37.6 |      1.000 |      0.607 |
    |         2 | 12:40:38.7 |      2.000 |      0.135 |
    |         3 | 12:40:39.7 |      3.000 |      0.011 |

At this point we decide to hit **Ctrl+C** (SIGINT). The RunEngine will catch
this signal and react like so.

.. code-block:: none

    ^C
    A 'deferred pause' has been requested.The RunEngine will pause at the next
    checkpoint. To pause immediately, hit Ctrl+C again in the next 10 seconds.
    Deferred pause acknowledged. Continuing to checkpoint.
    <...a few seconds later...>
    |         4 | 12:40:40.7 |      4.000 |      0.000 |
    Pausing...

    ---------------------------------------------------------------------------
    RunEngineInterrupted                      Traceback (most recent call last)
    <ipython-input-14-826ee9dfb918> in <module>()
    ----> 1 RE(scan([det], motor, 1, 10, 10))
    <...snipped details...>

    RunEngineInterrupted:
    Your RunEngine is entering a paused state. These are your options for changing
    the state of the RunEngine:
    RE.resume()    Resume the plan.
    RE.abort()     Perform cleanup, then kill plan. Mark exit_stats='aborted'.
    RE.stop()      Perform cleanup, then kill plan. Mark exit_status='success'.
    RE.halt()      Emergency Stop: Do not perform cleanup --- just stop.

When it pauses, the RunEngine immediately tells all Devices that is has touched
to "stop". (Devices define what that means to them in their ``stop()`` method.)
Now, all the hardware should be safe. At our leisure, we may:

* pause to think
* investigate the state of our hardware, such as the detector's exposure time
* turn on more verbose logging  (see :doc:`debugging`)
* decide whether to stop here or resume

Suppose we decide to resume.

.. ipython::
    :verbatim:

    In [13]: RE.resume()
    |         5 | 12:40:50.1 |      5.000 |      0.000 |
    |         6 | 12:40:51.1 |      6.000 |      0.000 |
    |         7 | 12:40:52.1 |      7.000 |      0.000 |
    |         8 | 12:40:53.1 |      8.000 |      0.000 |
    |         9 | 12:40:54.1 |      9.000 |      0.000 |
    |        10 | 12:40:55.1 |     10.000 |      0.000 |
    +-----------+------------+------------+------------+
    generator scan ['c5db9bb4'] (scan num: 1)

If you read the demo above closely, you will see that the RunEngine didn't
pause immediately: it finished the current step of the scan first. Quoting an
excerpt from the demo above:


.. code-block:: none

    ^C
    A 'deferred pause' has been requested.The RunEngine will pause at the next
    checkpoint. To pause immediately, hit Ctrl+C again in the next 10 seconds.
    Deferred pause acknowledged. Continuing to checkpoint.
    <...a few seconds later...>
    |         4 | 12:40:40.7 |      4.000 |      0.000 |
    Pausing...

To pause immediately without waiting for the next "checkpoint" (e.g. the
beginning of the next step) hit Ctrl+C *twice*.

Quoting again from the demo, notice that ``RE.resume()`` was only one of our
options. If we decide not to continue we can quit in three different ways:

.. code-block:: none

    Your RunEngine is entering a paused state. These are your options for changing
    the state of the RunEngine:
    RE.resume()    Resume the plan.
    RE.abort()     Perform cleanup, then kill plan. Mark exit_stats='aborted'.
    RE.stop()      Perform cleanup, then kill plan. Mark exit_status='success'.
    RE.halt()      Emergency Stop: Do not perform cleanup --- just stop.

"Aborting" and "stopping" are almost the same thing: they just record different
metadata about why the experiment was ended. Both signal to the plan that it
should end early, but they still let it specify more instructions so that it
can "clean up." For example, a :func:`~bluesky.plans.rel_scan` moves the motor
back to its starting position before quitting.

In rare cases, if we are worried that the plan's cleanup procedure might be
dangerous, we can "halt". Halting circumvents the cleanup instructions.

Automated Suspend & Resume
--------------------------

The RunEngine can be configured in advance to *automatically* pause and resume
in response to external signals. To distinguish automatic pause/resume for
interactive, user-initiated pause and resume, we call this behavior
"suspending."

Safe Error Handling
-------------------

Progress Bar
------------

Optional but nice to have. Add one like so:

.. code-block:: python

    from bluesky.utils import ProgressBarManager
    
    RE.waiting_hook = ProgressBarManager()

For example, two motors ``phi`` and ``theat`` moving simultaneously make a
display like this:

.. code-block:: none

    phi    9%|███▊                                       | 0.09/1.0 [00:00<00:01,  1.36s/deg]
    theta100%|████████████████████████████████████████████| 1.0/1.0 [00:01<00:00,  1.12s/deg]

The display includes the name of the device(s) being waited on and, if
available:

* distance (or degrees, etc.) traveled so far
* total distance to be covered
* time elapsed
* estimated time remaining and the of progress (determined empirically)

See :doc:`progress-bar` for more details and configuration.

Supplemental Data
-----------------

Persistent Metadata
-------------------

Write Custom Plans
==================

Write Custom Callbacks
======================

Export
------

Visualization
-------------

Fitting
-------
===========================   ======================================
interactive (blocking)        re-write for BlueSky plan()
===========================   ======================================
some.device.put("config")     yield from mv(some.device, "config")
motor.move(52)                yield from mv(motor, 52)
motor.velocity.put(5)         yield from mv(motor.velocity, 5)
===========================   ======================================

