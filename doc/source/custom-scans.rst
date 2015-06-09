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


Creating Documents (Saving the Data)
------------------------------------

Data is bundled into *Events*, logical groupings of measurements that can be
considered "simultaneous" for the purposes of analysis. When readings are
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

Registering Custom Commands
---------------------------



###Message: ``sleep``
Sleep the scan thread. One use case for this is waiting for a temperature to 
stabilize before collecting an image after instructing its temperature 
controller to change temperatures

```python
def temperature_scan(temp_controller, det):
    # scan a temperature controller from 100 to 150 in 50 steps
    for temp in np.linspace(100, 150, 50):
        # set the temperature controller
        yield Msg('set', temp_controller, temp)
        # wait one second for temperature to stabilize
        yield Msg('sleep', None, 1)
        # trigger acquisition of the detector
        yield Msg('trigger', det)
```

###Message: ``read``
``read`` takes a positioner or detector as an argument and returns its 
current value as a dictionary formatted as:
```python
{data_key1: (data, timestamp), 
 data_key2: (data, timestamp),
 ...
```
Incorporate ``read`` into a scan
```python
def temperature_scan(temp_controller, det):
    # scan a temperature controller from 100 to 150 in 50 steps
    for temp in np.linspace(100, 150, 50):
        temp = []
        # set the temperature controller
        yield Msg('set', temp_controller, temp)
        # wait for the temperature to stabilize
        while temp_std > 0.1:
            cur_temp = yield Msg('read', temp_controller)
            temp.append(cur_temp)
            temp_std = np.std(temp)
            yield Msg('sleep', None, .05)
        # trigger acquisition of the detector
        yield Msg('trigger', det)
```

###Messages: ``create``, ``save``

``create`` and ``save`` are used to specify exactly which values 
should go into an event

```python
def temperature_scan(temp_controller, det):
    # scan a temperature controller from 100 to 150 in 50 steps
    for temp in np.linspace(100, 150, 50):
        # instruct the run engine to start watching for
        yield Msg('create')
        # set the temperature controller
        yield Msg('set', temp_controller, temp)
        # wait one second for temperature to stabilize
        yield Msg('sleep', None, 1)
        # trigger acquisition of the detector
        yield Msg('trigger', det)
        # read the temperature controller and the detector
        yield Msg('read', det)
        yield Msg('read', temp_controller)
        # bundle the two things that have been read via the ``read`` command 
        # into an event. If this is the first time an event is saved, it will
        # create a corresponding event descriptor
        yield Msg('save')
```

###Messages: ``checkpoint`` and ``pause``
``checkpoint`` and ``pause`` are used together.  ``checkpoint`` defines a point
 in the scan that is safe to resume operation from. ``pause`` will wait until
the ``RunEngine`` is no longer in its paused state and resume the scan from 
the last ``checkpoint``

```python
def conditional_hard_pause(motor, det):
    for i in range(5):
        yield Msg('checkpoint')
        yield Msg('set', motor, {'pos': i})
        yield Msg('trigger', det)
        reading = yield Msg('read', det)
        if reading['intensity']['value'] < 0.2:
            # this returns control to the main thread and the user can resume
            # scanning with RE.resume(). When the scan is resumed the 
            # ``RunEngine`` will re-evaluate all messages that were received 
            # since ``Msg('checkpoint')`` was received  
            yield Msg('pause')
```

###Messages: ``kickoff`` and ``collect``
``kickoff`` and ``collect`` are used together to perform fly scanning.  
``kickoff`` calls ``RunEngine._kickoff`` with the ``*args`` and ``kwargs`` of
 the message
```python
def step_fly(step_mtr, fly_mtr, fly_det, step_points, fly_start, fly_stop, 
             fly_velocity):
for step_point in step_points:
    yield Msg('create')
    yield Msg('set', step_mtr, step_point
    yield Msg('save')
    # start flying the motor and detector
    yield Msg('kickoff', fly_mtr, fly_start, fly_stop, fly_velocity)
    yield Msg('kickoff', fly_det)
    # collect and create the event streams for the flyscan motor and 
    # flyscan detector
    yield Msg('collect', fly_mtr)
    yield Msg('collect', fly_det)
```

###Messages: ``wait``
Wait for one motor to finish
```python
def wait_one(motor, det):
    "Set, trigger, read"
    yield Msg('set', motor, {'pos': 5}, block_group='A')  # Add to group 'A'.
    yield Msg('wait', None, 'A')  # Wait for everything in group 'A' to finish.
    yield Msg('trigger', det)
    yield Msg('read', det)
```
Wait for multiple motors to finish
```python
def wait_multiple(motors, det):
    "Set motors, trigger all motors, wait for all motors to move."
    for motor in motors:
        yield Msg('set', motor, {'pos': 5}, block_group='A')
    # Wait for everything in group 'A' to report done.
    yield Msg('wait', None, 'A')
    yield Msg('trigger', det)
    yield Msg('read', det)
```
