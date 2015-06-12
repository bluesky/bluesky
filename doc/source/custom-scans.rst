Writing Custom Scans
====================

.. ipython:: python
   :suppress:

    from bluesky import Msg
    from bluesky.examples import motor

Messages
--------

The RunEngine processes *messages*. A message is comprised of:

* a command, such as 'read', 'set', or 'pause'
* a target object, such as ``motor``, if applicable
* positional arguments
* keyword arguments

Examples:

.. ipython:: python

   Msg('read', motor)
   Msg('set', motor, 5)

The full list of built-in commands is covered systematically
:ref:`elsewhere <commands>`.
Below, we build up a collection of example scans demonstrating a variety of
use cases.

Simplest Scan
-------------

Messages are passed to the RunEngine through a Python *generator* (more on
these below). Here is a very simple scan that sets a motor's position to 5
and reads the position back.

.. ipython:: python

    def simple_scan(motor):
        yield Msg('set', motor, 5)
        yield Msg('read', motor)

The RunEngine processes these messages like so:

.. code-block:: python

    motor.set(5)
    motor.read()

To read from a detector, we also need the 'trigger' command.

.. ipython:: python

    def simple_scan(motor, det):
        yield Msg('set', motor, 5)
        yield Msg('read', motor)
        yield Msg('trigger', det)
        yield Msg('read', det)

Setting Objects with Multiple Degrees of Freedom
------------------------------------------------

It's possible to 'set' an object with multiple degress of freedom.

.. ipython:: python

    def multi_set(multi_motor):
        yield Msg('set', multi_motor, (), {'h': 1, 'k':2, 'l': 3})

The dictionary in this message is passed to ``multi_motor.set`` as keyword
arguments:

.. code-block:: python

    multi_motor.set(h=1, k=2, l=3)

Making Scans Responsive
-----------------------

Two-way communication is possible between the generator and the RunEngine.
For example, the 'read' command responds with its reading. We can use it to
make an on-the-fly decision about whether to continue or stop.

.. ipython:: python

    def conditional_break(motor, det, threshold):
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

.. ipython:: python

    def sleepy(motor, det):
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

.. ipython:: python

    def wait_one(motor, det):
        "Set, trigger, read"
        yield Msg('set', motor, 5, block_group='A')  # Add to group 'A'.
        yield Msg('wait', None, 'A')  # Wait for everything in group 'A'.
        yield Msg('trigger', det)
        yield Msg('read', det)

By assigning multiple objects to the same ``block_group``, you can wait until
the last one reports it is ready.

.. ipython:: python

    def wait_multiple(motors, det):
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

.. ipython:: python

    def wait_complex(motors, det):
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

.. ipython:: python

    def conditional_pause(motor, det, hard):
        for i in range(5):
            yield Msg('checkpoint')
            yield Msg('set', motor, i)
            yield Msg('trigger', det)
            reading = yield Msg('read', det)
            if reading['det']['value'] < 0.2:
                yield Msg('pause', hard=hard)
            yield Msg('set', motor, i + 0.5)

If detector reading dips below 0.2, the scan is paused.

The next example is a step scan that pauses after each data point is collected.
(This is the function we used in the
:ref:`first pausing example <planned-pause>`.)


.. ipython:: python

    def cautious_stepscan(motor, det):
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

.. ipython:: python

    def simple_scan_saving(motor, det):
        "Set, trigger, read"
        yield Msg('create')
        yield Msg('set', motor, 5)
        yield Msg('read', motor)
        yield Msg('trigger', det)
        yield Msg('read', det)
        yield Msg('save')

The above generates one Event. By looping through several create--save pairs,
we can generate many Events.

.. ipython:: python

    def stepscan(motor, det):
        for i in range(-5, 5):
            yield Msg('create')
            yield Msg('set', motor, i)
            yield Msg('trigger', det)
            yield Msg('read', motor)
            yield Msg('read', det)
            yield Msg('save')

Fly Scans
---------

TODO

Registering Custom Commands
---------------------------

TODO

Making Scans Reusable
---------------------

Generators
++++++++++

Python generators are iterable, like lists, but you can only iterate through
them once. Observe:

.. ipython:: python

    from bluesky.examples import motor, det
    s = stepscan(motor, det)
    def count_messages(s):
        return len(list(s))

    count_messages(s)
    count_messages(s)  # not reusable -- no messages left

``stepscan`` is a function that returns a generator. ``s`` is a generator.

Why not just use a list? Generators support two-way commuication through a
sophisticated language feature called coroutines, which makes it possible
to write adaptive and responsive scans.

Reusable Scans
++++++++++++++

By contrast, bluesky's built-in scans are reusable.

.. ipython:: python

    from bluesky.scans import Ascan
    s = Ascan(motor, [det], [1, 2, 3])
    count_messages(s)
    count_messages(s)  # reusable!

How does that work? ``Ascan`` is not function that returns a generator; it is
an iterable class that returns a fresh generator upon each new iteration.

You can use that same pattern to make our ``stepscan`` example---or any custom
scan---reusable. Follow this pattern:

.. ipython:: python

    class ReusableStepscan:
        def __init__(self, motor, det):
            self.motor = motor
            self.det = det
        def __iter__(self):
            return self._gen()
        def _gen(self):
            yield from stepscan(self.motor, self.det)

    # Check that it works.
    s = ReusableStepscan(motor, det)
    count_messages(s)
    count_messages(s)  # reusable!

Extra Credit: Less Typing, More Magic
+++++++++++++++++++++++++++++++++++++

If the ``__init__`` and ``__iter__`` blocks above seems tedious and reptitive,
subclass bluesky's ``Scan`` class.

.. ipython:: python

    from bluesky.scans import Scan
    class ReusableStepscan(Scan):
        _fields = ['motor', 'det']  # These magically become the args.
        def _gen(self):
            yield from stepscan(self.motor, self.det)

    # Check that it works.
    s = ReusableStepscan(motor, det)
    count_messages(s)
    count_messages(s)  # reusable!

Additional Examples
-------------------

Temperature Sweep
+++++++++++++++++

.. ipython:: python

    import numpy as np
    def temperature_sweep(temp_controller, det):
        # scan a temperature controller from 100 to 150 in 50 steps
        for temp in np.linspace(100, 150, 50):
            yield Msg('create')
            # set the temperature controller
            yield Msg('set', temp_controller, temp)
            # wait one second for temperature to stabilize
            yield Msg('sleep', None, 1)
            # trigger acquisition of the detector
            yield Msg('trigger', det)
            yield Msg('save')
