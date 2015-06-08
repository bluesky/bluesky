Writing Custom Scans
====================

.. ipython:: python

   from bluesky import Msg
   Msg('trigger')


Trigger a detector to start acquiring
```python
def count(det):
    yield Msg('trigger', det)
```

###Message: ``set``
Set the position of a motor. By default, the motor is instructed to move
```python
def move(mtr, pos):
    # will move ``mtr`` to ``pos``
    yield Msg('set', mtr, pos)
```
is equivalent to
```python
def move(mtr, pos)
    yield Msg('set', mtr, pos, trigger=True)
    # the trigger command
```

If you want to set the motors position and then tell it to move at a later 
point in time, use the `trigger` kwarg
```python
def move(mtr, pos):
    # will instruct ``mtr`` to move to ``pos`` when triggered
    yield Msg('set', mtr, pos, trigger=False)
    # will instruct ``mtr`` to move to ``pos``
    yield Msg('trigger', mtr)
```

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
Create two different groups and wait for each to finish at a different point 
in the scan
```python
def wait_complex(motors, det):
    "Set motors, trigger motors, wait for all motors to move in groups."
    # Same as above...
    for motor in motors[:-1]:
        yield Msg('set', motor, {'pos': 5}, block_group='A')

    # ...but put the last motor is separate group.
    yield Msg('set', motors[-1], {'pos': 5}, block_group='B')
    # Wait for everything in group 'A' to report done.
    yield Msg('wait', None, 'A')
    yield Msg('trigger', det)
    yield Msg('read', det)

    # Wait for everything in group 'B' to report done.
    yield Msg('wait', None, 'B')
    yield Msg('trigger', det)
    yield Msg('read', det)
```

###Messages: ``null``
``null`` does ...?
