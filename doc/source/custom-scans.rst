Writing Custom Scans
====================

Messages
--------

The RunEngine processes Messages. Examples:

.. ipython:: python

   Msg('read', motor)
   Msg('set', motor, 5)

Simplest Scan
-------------

Messages are passed to the RunEngine through a Python *generator*. Here is a
very simple scan that moves a motor to ``5`` and reads its position.

.. ipython:: python

    def simple_scan(motor):
        yield Msg('set', motor, 5)
        yield Msg('read', motor)

Responsive Scans
----------------

Two-way communication is possible between the generator and the RunEngine.
The 'read' command responds with its data payload. We can inspect the data
point and use it to decide whether to continue.

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

The data is formatted like:

.. code-block:: python

     {<name>: {'value': <value>, 'timestamp': <timestamp>}, ...}

For a detailed technical description of the messages and their responses,
see :doc:`msg`.

Waiting and Sleeping
--------------------

Sleeping is as simple as it sounds.

.. ipython:: python

    def sleepy(motor, det):
        "Set, trigger motor, sleep for a fixed time, trigger detector, read"
        yield Msg('set', motor, 5)
        yield Msg('sleep', None, 2)  # units: seconds
        yield Msg('trigger', det)
        yield Msg('read', det)

To wait for a motor or any set-able thing to finish reponding to a 'set'
command, use 'wait'. First, give the 'set' command a ``block_group``
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

Fly Scans
---------




.. ipython:: python

   from bluesky import Msg
   Msg('trigger')


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
