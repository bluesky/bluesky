#bluesky
duh!


#What is the problem that bluesky solves?
``bluesky`` is a Python implementation of a scanning framework that has been 
designed to accomodate the various and complex needs of the NSLS-II beamlines.
Its goal is to replace SPEC and add the full power of the Python language to
data acquisition.  

##``bluesky`` Features

- Beamline panic conditions 
  
    - will immediately terminate the run
  
- Programmatic pause conditions 

    - will pause the run until a resume condition has been met
    - e.g., Sample temperature is too high. Pause until the sample 
      temperature falls below some threshold

- Interactive pause conditions ``Ctrl + c``
  
    - will pause until the user decides to resume
    - A second ``Ctrl + c`` will terminate the scan

- Implement custom scans with messaging (``Msg``) framework

    - See ``bluesky.examples``
    - and ``bluesky.run_engine.Msg``
    - and ``bluesky.run_engine._command_registry``

- Main thread callback registry

  - Allows for "expensive" computations to take place on a different process 
    than the scan such as "live" plotting and data analysis


##Messaging framework
The ``bluesky`` run engine works via message passing.  As such, understanding
the messaging framework is critical.  

Mapping from message to `RunEngine` function
```python
self._command_registry = {
    'create': self._create,
    'save': self._save,
    'read': self._read,
    'null': self._null,
    'set': self._set,
    'trigger': self._trigger,
    'sleep': self._sleep,
    'wait': self._wait,
    'checkpoint': self._checkpoint,
    'pause': self._pause,
    'collect': self._collect,
    'kickoff': self._kickoff
}
```

###Message: ``trigger``
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
    
    # is equivalent to
    
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

##What is it not possible to do by design?

###Bundle fly scan data into a step scan event
In the ``step_fly`` example above, it would not be possible to bundle the 
data from `fly_mtr` into a filestore entry and `fly_det` into a filestore 
entry and then insert those into the same event as `step_mtr`. While it not 
necessarily illogical to create such an event, we would argue that we already
provide tools to do this reconstruction at data analysis time.  Therefore, it
is not necessary to store the data in such a manner. Also it would add 
unnecessary complication to the bluesky RunEngine to do this.

What a flyscan+stepscan event would look like
```
Event
=====
data            :
  step_mtr        : [5.123029068072772, 1433188852.2006402] 
  fly_mtr         : 8ae46bff-b577-414a-b3a3-009548250d3c
  fly_det         : 9a715a80-007e-4591-b243-2535d66e3ee5
seq_num         : 5                                       
time            : 1433188852.2006402                      
time_as_datetime: 2015-06-01 16:00:52.200640              
uid             : 4ae29c60-0699-4492-82ec-ccc137811a68    
```
