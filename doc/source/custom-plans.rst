Messages
========

The built-in plans are heavily customizable and can satisfy many applications. Most users can find everything they need in :doc:`plans`.

This section explores Messages, the granular instructions that make up a plan,
in depth.


A message is comprised of:

* a command string, such as 'read', 'set', or 'pause'
* a target object, such as ``motor``, if applicable
* positional arguments
* keyword arguments


Examples:

.. code-block:: python

   from bluesky import Msg
   from bluesky.examples import motor

   Msg('read', motor)
   Msg('set', motor, 5)

The ``Msg`` object itself is a namedtuple.

Below, we build up a collection of example plans demonstrating a variety of
different commands and use cases.

Simplest Scan
-------------

Messages are passed to the RunEngine through a Python *generator* (more on
these below). Here is a very simple scan that sets a motor's position to 5
and reads the position back.

.. code-block:: python

    def simple_scan(det, motor):
        yield Msg('set', motor, 5)
        yield Msg('read', motor)

The RunEngine processes these messages like so:

.. code-block:: python

    motor.set(5)
    motor.read()

To read from a detector, we also need the 'trigger' command.

.. code-block:: python

    def simple_scan(det, motor):
        yield Msg('set', motor, 5)
        yield Msg('read', motor)
        yield Msg('trigger', det)
        yield Msg('read', det)

Making Scans Responsive
-----------------------

Two-way communication is possible between the generator and the RunEngine.
For example, the 'read' command responds with its reading. We can use it to
make an on-the-fly decision about whether to continue or stop.

.. code-block:: python

    def conditional_break(det, motor, threshold):
        """Set, trigger, read until the detector reads intensity < threshold"""
        i = 0
        while True:
            print("LOOP %d" % i)
            yield Msg('set', motor, i)
            yield Msg('trigger', det)
            reading = yield Msg('read', det)
            if reading['det']['value'] < threshold:
                print('DONE')
                break
            i += 1

The response from 'read' -- ``reading``, above -- is formatted like:

.. code-block:: python

     {<name>: {'value': <value>, 'timestamp': <timestamp>}, ...}

For a detailed technical description of the messages and their responses,
see :doc:`msg`.

Sleeping
--------

Sleeping is as simple as it sounds. It might be used, for example, to add
extra delay to allow a sample to equilibrate to the temperature set by a
temperature controller.

.. code-block:: python

    def sleepy(det, motor):
        "Set, trigger motor, sleep for a fixed time, trigger detector, read"
        yield Msg('set', motor, 5)
        yield Msg('sleep', None, 2)  # units: seconds
        yield Msg('trigger', det)
        yield Msg('read', det)

Notice that unlike 'set', 'read', and 'trigger', the 'sleep' command does
not have a target object. We use ``None`` as a placeholder.

Waiting
-------

Use the 'wait' command to block progress until an object report that it is
ready. For example, wait for a motor to finish moving.

First, give the 'set' command a ``block_group``
keyword argument. This is just a label that we can use to refer to it later.
Then, use 'wait' to tell the RunEngine to block progress until everything in
that ``block_group`` reports that it is ready.

.. code-block:: python

    def wait_one(det, motor):
        "Set, trigger, read"
        yield Msg('set', motor, 5, block_group='A')  # Add to group 'A'.
        yield Msg('wait', None, 'A')  # Wait for everything in group 'A'.
        yield Msg('trigger', det)
        yield Msg('read', det)

By assigning multiple objects to the same ``block_group``, you can wait until
the last one reports it is ready.

.. code-block:: python

    def wait_multiple(det, motors):
        "Set motors, trigger all motors, wait for all motors to move."
        for motor in motors:
            yield Msg('set', motor, 5, block_group='A')
        # Wait for everything in group 'A' to report done.
        yield Msg('wait', None, 'A')
        yield Msg('trigger', det)
        yield Msg('read', det)

If the above seems unnecessarily complex, here is the payoff. By using
different ``block_group`` labels, you can wait for different groups at
different points in the scan.

.. code-block:: python

    def wait_complex(det, motors):
        "Set motors, trigger motors, wait for all motors to move in groups."
        # Same as above...
        for motor in motors[:-1]:
            yield Msg('set', motor, 5, block_group='A')
        # ...but put the last motor is separate group.
        yield Msg('set', motors[-1], 5, block_group='B')
        # Wait for everything in group 'A' to report done.
        yield Msg('wait', None, 'A')
        yield Msg('trigger', det)
        yield Msg('read', det)
        # Wait for everything in group 'B' to report done.
        yield Msg('wait', None, 'B')
        yield Msg('trigger', det)
        yield Msg('read', det)

Pauseable Scans
---------------

The 'pause' command pauses the RunEngine. Details of pausing and resuming were
addressed :doc:`previously <state-machine>`.

The 'checkpoint' command defines where a scan can be safely resumed after an
interruption.

.. code-block:: python

    def conditional_pause(det, motor, defer):
        for i in range(5):
            yield Msg('checkpoint')
            yield Msg('set', motor, i)
            yield Msg('trigger', det)
            reading = yield Msg('read', det)
            if reading['det']['value'] < 0.2:
                yield Msg('pause', defer=defer)
            yield Msg('set', motor, i + 0.5)

If detector reading dips below 0.2, the scan is paused.

The next example is a step scan that pauses after each data point is collected.
(This is the function we used in the
:ref:`first pausing example <planned-pause>`.)


.. code-block:: python

    def cautious_stepscan(det, motor):
        for i in range(-5, 5):
            yield Msg('checkpoint')
            yield Msg('create')
            yield Msg('set', motor, i)
            yield Msg('trigger', det)
            ret_m = yield Msg('read', motor)
            ret_d = yield Msg('read', det)
            yield Msg('save')
            print("Value at {m} is {d}. Pausing.".format(
                m=ret_m[motor._name]['value'], d=ret_d[det1._name]['value']))
            yield Msg('pause', None, hard=False)

The 'create' and 'save' commands bundle and save the readings between them, as
described just below. Notice that 'checkpoint' occurs before 'create'. It is
illegal to place checkpoints inside a create--save pair.

Creating Documents (Saving the Data)
------------------------------------

Data is bundled into *Events*, logical groupings of measurements that can be
considered "simultaneous" for practical purposes. (Individual measurement
times are recorded, but they are usually ignored.) When readings are
bundled as an Event, an Event Document is created and made available to
:doc:`subscriptions <callbacks>`.

To bundle data into an Event, use the 'create' and 'save' commands. Any
'read' commands that occur between the two will be bundled into an Event.

.. code-block:: python

    def simple_scan_saving(motor, det):
        "Set, trigger, read"
        yield Msg('open_run')
        yield Msg('create')
        yield Msg('set', motor, 5)
        yield Msg('read', motor)
        yield Msg('trigger', det)
        yield Msg('read', det)
        yield Msg('save')
        yield Msg('close_run')

The above generates one Event. By looping through several create--save pairs,
we can generate many Events.

.. code-block:: python

    def stepscan(motor, det):
        yield Msg('open_run')
        for i in range(-5, 5):
            yield Msg('create')
            yield Msg('set', motor, i)
            yield Msg('trigger', det)
            yield Msg('read', motor)
            yield Msg('read', det)
            yield Msg('save')
        yield Msg('close_run')

Fly Scans
---------

From the point of view of bluesky, a "fly scan" is any object that needs to
be told to start and then, some time later, to return its data in bulk with
no supervision in between. These two steps are called "kickoff" and "collect"
respectively.

.. code-block:: python

    def flyscan(flyer):
        Msg('kickoff', flyer)
        # some time later...
        Msg('collect', flyer)

Obviously, all of the interesting action is up to ``flyer`` -- but that is
the point.


Registering Custom Commands
---------------------------

The RunEngine can be made to undersand any new commands. They can
be registered using the following methods.

.. automethod:: bluesky.run_engine.RunEngine.register_command
.. automethod:: bluesky.run_engine.RunEngine.unregister_command
