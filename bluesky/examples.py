import asyncio
import time as ttime
from collections import deque, OrderedDict
import numpy as np
from .run_engine import Msg


class MockSignal:
    "hotfix for 2016 winter cycle -- build out more thoroughly later"
    def __init__(self, field):
        self._field = field

    def read(self):
        return {self._field: (0, 0)}

    def describe(self):
        return {self._field: (0, 0)}


class NullStatus:
    "a simple Status object that is always immediately done"
    def __init__(self):
        self._cb = None
        self.done = True
        self.success = True

    @property
    def finished_cb(self):
        return self._cb

    @finished_cb.setter
    def finished_cb(self, cb):
        cb()
        self._cb = cb


class Reader:
    """

    Parameters
    ----------
    name : string
    read_fields : dict
        Mapping field names to functions that return simulated data. The
        function will be passed no arguments.
    conf_fields : dict, optional
        Like `read_fields`, but providing slow-changing configuration data.
        If `None`, the configuration will simply be an empty dict.

    Example
    -------
    A detector that always returns 5.
    >>> det = Detector('det', {'intensity': lambda: 5})

    A detector that is coupled to a motor, such that measured insensity
    varies with motor position.
    >>> motor = Motor('motor')
    >>> det = Detector('det',
    ...                {'intensity': lambda: 2 * motor.read()['value']})
    """
    def __init__(self, name, read_fields, conf_fields=None):
        self.name = name
        self.parent = None
        self._read_fields = read_fields
        if conf_fields is None:
            conf_fields = {}
        self._conf_fields = conf_fields

    def trigger(self):
        "No-op: returns a status object immediately marked 'done'."
        return NullStatus()

    def read(self):
        """
        Simulate readings by calling functions.

        The readings are collated with timestamps.
        """
        return {field: {'value': func(), 'timestamp': ttime.time()}
                        for field, func in self._read_fields.items()}

    def describe(self):
        """
        Provide metadata for each of the fields returned by `read`.

        In this simple example, the metadata is hard-coded: we assume all
        readings are numeric and scalar.
        """
        return {field: {'source': 'simulated using bluesky.examples',
                        'dtype': 'number',
                        'shape': [],
                        'precision': 2}
                for field in self._read_fields}

    def read_configuration(self):
        """
        Like `read`, but providing slow-changing configuration readings.
        """
        return {field: {'value': func(), 'timestamp': ttime.time()}
                        for field, func in self._conf_fields.items()}

    def describe_configuration(self):
        return {field: {'source': 'simulated using bluesky.examples',
                        'dtype': 'number',
                        'shape': [],
                        'precision': 2}
                for field in self._conf_fields}

    def configure(self, *args, **kwargs):
        old_conf = self.read_configuration()
        # Update configuration here.
        new_conf = self.read_configuration()
        return old_conf, new_conf


class Mover(Reader):
    """

    Parameters
    ----------
    name : string
    read_fields : dict
        Mapping field names to functions that return simulated data. The
        function will be passed the last set of argument given to ``set()``.
    conf_fields : dict, optional
        Like `read_fields`, but providing slow-changing configuration data.
        If `None`, the configuration will simply be an empty dict.
    initial_set : dict
        passed to ``set`` as ``set(**initial_set)`` to initialize readings
    fake_sleep : float
        simulate moving time

    Example
    -------
    A motor that simply goes where it is set.
    >>> motor = Positioner('motor', {'readback': lambda x: x},
    ...                              'setpoint': lambda x: x})

    A motor that adds jitter.
    >>> import numpy as np
    >>> motor = Positioner('motor', {'readback': lambda x: x + np.random.randn()},
    ...                              'setpoint': lambda x: x})
    """
    def __init__(self, name, read_fields, initial_set, conf_fields=None, *,
                 fake_sleep=0):
        super().__init__(name, read_fields, conf_fields)
        # Do initial set without any fake sleep.
        self._fake_sleep = 0
        self.set(**initial_set)
        self._fake_sleep = fake_sleep

    def set(self, *args, **kwargs):
        """
        Pass the arguments to the functions to create the next reading.
        """
        self._state = {field: {'value': func(*args, **kwargs),
                               'timestamp': ttime.time()}
                       for field, func in self._read_fields.items()}
        if self._fake_sleep:
            ttime.sleep(self._fake_sleep)
        return NullStatus()

    def read(self):
        return self._state

    @property
    def position(self):
        "A heuristic that picks a single scalar out of the `read` dict."
        return self.read()[list(self._read_fields)[0]]['value']

    def stop(self):
        pass


class SynGauss(Reader):
    """
    Evaluate a point on a Gaussian based on the value of a motor.

    Parameters
    ----------
    noise : {'poisson', 'uniform', None}
        Add noise to the gaussian peak.
    noise_multiplier : float
        Only relevant for 'uniform' noise. Multiply the random amount of
        noise by 'noise_multiplier'

    Example
    -------
    motor = Mover('motor', {'motor': lambda x: x}, {'x': 0})
    det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)
    """
    def __init__(self, name, motor, motor_field, center, Imax, sigma=1,
                 noise=None, noise_multiplier=1, exposure_time=0):
        if noise not in ('poisson', 'uniform', None):
            raise ValueError("noise must be one of 'poisson', 'uniform', None")
        self.exposure_time = exposure_time

        def func():
            m = motor.read()[motor_field]['value']
            v = Imax * np.exp(-(m - center)**2 / (2 * sigma**2))
            if noise == 'poisson':
                v = int(np.random.poisson(np.round(v), 1))
            elif noise == 'uniform':
                v += np.random.uniform(-1, 1) * noise_multiplier
            return v

        super().__init__(name, {name: func})

    def trigger(self):
        if self.exposure_time:
            ttime.sleep(self.exposure_time)
        return super().trigger()


class Syn2DGauss(Reader):
    """
    Evaluate a point on a Gaussian based on the value of a motor.

    Parameters
    ----------
    name : str
        The name of the detector
    motor0 : `Mover`
        The 'x' coordinate of the 2-D gaussian blob
    motor_field0 : str
        The name field of the motor. Should be the key in motor0.describe()
    motor1 : `Mover`
        The 'y' coordinate of the 2-D gaussian blob
    motor_field1 : str
        The name field of the motor. Should be the key in motor1.describe()
    center : iterable, optional
        The center of the gaussian blob
        Defaults to (0,0)
    Imax : float, optional
        The intensity at `center`
        Defaults to 1
    sigma : float, optional
        Standard deviation for gaussian blob
        Defaults to 1
    noise : {'poisson', 'uniform', None}
        Add noise to the gaussian peak..
        Defaults to None
    noise_multiplier : float, optional
        Only relevant for 'uniform' noise. Multiply the random amount of
        noise by 'noise_multiplier'
        Defaults to 1

    Example
    -------
    motor = Mover('motor', ['motor'])
    det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)
    """
    def __init__(self, name, motor0, motor_field0, motor1, motor_field1,
                 center, Imax, sigma=1, noise=None, noise_multiplier=1,
                 exposure_time=0):

        if noise not in ('poisson', 'uniform', None):
            raise ValueError("noise must be one of 'poisson', 'uniform', None")
        self.exposure_time = exposure_time

        def func():
            x = motor0.read()[motor_field0]['value']
            y = motor1.read()[motor_field1]['value']
            m = np.array([x, y])
            v = Imax * np.exp(-np.sum((m - center)**2) / (2 * sigma**2))
            if noise == 'poisson':
                v = int(np.random.poisson(np.round(v), 1))
            elif noise == 'uniform':
                v += np.random.uniform(-1, 1) * noise_multiplier
            return v

        super().__init__(name, {name: func})

    def trigger(self):
        if self.exposure_time:
            ttime.sleep(self.exposure_time)
        return super().trigger()


class Flyer:
    """Flyer that complies to the API but returns empty data."""
    def kickoff(self):
        return NullStatus()

    def describe_collect(self):
        return {'stream_name': {}}

    def complete(self):
        return NullStatus()

    def collect(self):
        for i in range(100):
            yield {'data': {}, 'timestamps': {}, 'time': i, 'seq_num': i}

    def stop(self):
        pass


class MockFlyer:
    """
    Class for mocking a flyscan API implemented with stepper motors.

    warning::
    
        This is old and should be not used as a reference for current good
        practice. Specifically, it is its own status object, which is
        confusing.
    """
    def __init__(self, detector, motor, loop):
        self._mot = motor
        self._detector = detector
        self._steps = None
        self._future = None
        self._data = deque()
        self._cb = None
        self.ready = False
        self.success = True
        self.loop = loop

    @property
    def done(self):
        return self.ready

    def read_configuration(self):
        return OrderedDict()

    def describe_configuration(self):
        return OrderedDict()

    def describe_collect(self):
        dd = dict()
        dd.update(self._mot.describe())
        dd.update(self._detector.describe())
        return {'stream_name': dd}

    def complete(self):
        self.success = True
        self.ready = False
        return self

    def kickoff(self, start, stop, steps):
        self._steps = np.linspace(start, stop, steps)
        self._data = deque()

        # setup the status object (self) that will be returned by
        # self.complete(). Separately, make dummy status object
        # that is immediately done, and return that, indicated that
        # the 'kickoff' step is done.
        self._future = self.loop.run_in_executor(None, self._scan)
        self._future.add_done_callback(lambda x: self._finish())

        return NullStatus()

    def collect(self):
        if not self.ready:
            raise RuntimeError("No reading until done!")

        yield from self._data
        self._thread = None

    def _scan(self):
        # this is needed because these objects are very dumb
        ttime.sleep(.1)
        for p in self._steps:
            stat = self._mot.set(p)
            while True:
                if stat.done:
                    break
                ttime.sleep(0.01)
            stat = self._detector.trigger()
            while True:
                if stat.done:
                    break
                ttime.sleep(0.01)

            event = dict()
            event['time'] = ttime.time()
            event['data'] = dict()
            event['timestamps'] = dict()
            for r in [self._mot, self._detector]:
                d = r.read()
                for k, v in d.items():
                    event['data'][k] = v['value']
                    event['timestamps'][k] = v['timestamp']
            self._data.append(event)
        self._finish()

    @property
    def finished_cb(self):
        """
        Callback to be run when the status is marked as finished

        The call back has no arguments
        """
        return self._cb

    @finished_cb.setter
    def finished_cb(self, cb):
        if self._cb is not None:
            raise RuntimeError("Can not change the call back")
        if self.done:
            cb()
        else:
            self._cb = cb

    def _finish(self):
        print('finish fired')
        self.ready = True
        if self._cb is not None:
            self._cb()
            self._cb = None

    def stop(self):
        pass


class FlyMagic:
    """
    Another old flyer example

    warning::
    
        This is old and should be not used as a reference for current good
        practice. Specifically, it is its own status object, which is
        confusing.
    """
    def __init__(self, name, motor, det, det2, scan_points=15):
        self.name = name
        self._motor = motor
        self._det = det
        self._det2 = det2
        self._scan_points = scan_points
        self._time = None
        self._fly_count = 0
        self.ready = False

    def read_configuration(self):
        return OrderedDict()

    def describe_configuration(self):
        return OrderedDict()

    def reset(self):
        self._fly_count = 0

    def kickoff(self):
        self.ready = False
        self._time = ttime.time()
        self._fly_count += 1
        self.ready = True
        return self

    def describe_collect(self):
        return {'stream1': {k: {'source': self.name, 'dtype': 'number'}
                 for k in [self._motor, self._det]},
                'stream2':
                {self._det2: {'source': self.name, 'dtype': 'number'}}}

    def collect(self):
        if self._time is None or not self.ready:
            raise RuntimeError("Must kick off flyscan before you collect")

        dtheta = (np.pi / 10) * self._fly_count
        X = np.linspace(0, 2*np.pi, self._scan_points)
        Y = np.sin(X + dtheta)
        dt = (ttime.time() - self._time) / self._scan_points
        T = dt * np.arange(self._scan_points) + self._time

        for j, (t, x, y) in enumerate(zip(T, X, Y)):
            ev = {'time': t,
                  'data': {self._motor: x,
                           self._det: y},
                  'timestamps': {self._motor: t,
                                 self._det: t}
                  }

            yield ev
            ttime.sleep(0.01)
            ev = {'time': t + .1,
                  'data': {self._det2: -y},
                  'timestamps': {self._det2: t + 0.1}
                  }
            yield ev
            ttime.sleep(0.01)
        self._time = None

    def complete(self):
        return NullStatus()

    def stop(self):
        pass


motor = Mover('motor', {'motor': lambda x: x}, {'x': 0})
motor1 = Mover('motor1', {'motor1': lambda x: x}, {'x': 0})
motor2 = Mover('motor2', {'motor2': lambda x: x}, {'x': 0})
motor3 = Mover('motor3', {'motor3': lambda x: x}, {'x': 0})
jittery_motor1 = ('jittery_motor1',
                  {'jiterry_motor1': lambda x: x + np.random.randn()},
                  {'x': 0})
jittery_motor2 = ('jittery_motor2',
                  {'jiterry_motor2': lambda x: x + np.random.randn()},
                  {'x': 0})
noisy_det = SynGauss('noisy_det', motor, 'motor', center=0, Imax=1,
                     noise='uniform', sigma=1)
det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)
det1 = SynGauss('det1', motor1, 'motor1', center=0, Imax=5, sigma=0.5)
det2 = SynGauss('det2', motor2, 'motor2', center=1, Imax=2, sigma=2)
det3 = SynGauss('det3', motor3, 'motor3', center=-1, Imax=2, sigma=1)
det4 = Syn2DGauss('det4', motor1, 'motor1', motor2, 'motor2',
                  center=(0, 0), Imax=1)
det5 = Syn2DGauss('det4', jittery_motor1, 'jittery_motor1', jittery_motor2,
                  'jittery_motor2', center=(0, 0), Imax=1)


def simple_scan(motor):
    yield Msg('open_run')
    yield Msg('set', motor, 5)
    yield Msg('read', motor)
    yield Msg('close_run')


def conditional_break(det, motor, threshold):
    """Set, trigger, read until the detector reads intensity < threshold"""
    i = 0
    yield Msg('open_run')
    while True:
        print("LOOP %d" % i)
        yield Msg('set', motor, i)
        yield Msg('trigger', det)
        reading = yield Msg('read', det)
        if reading[det.name]['value'] < threshold:
            print('DONE')
            yield Msg('close_run')
            break
        i += 1


def sleepy(det, motor):
    "Set, trigger motor, sleep for a fixed time, trigger detector, read"
    yield Msg('open_run')
    yield Msg('set', motor, 5)
    yield Msg('sleep', None, 2)  # units: seconds
    yield Msg('trigger', det)
    yield Msg('read', det)
    yield Msg('close_run')


def do_nothing(timeout=5):
    "Generate 'checkpoint' messages until timeout."
    t = ttime.time()
    yield Msg('open_run')
    while True:
        if ttime.time() > t + timeout:
            break
        ttime.sleep(0.1)
        yield Msg('checkpoint')
    yield Msg('close_run')


def checkpoint_forever():
    # simplest pauseable scan
    yield Msg('open_run')
    while True:
        ttime.sleep(0.1)
        yield Msg('checkpoint')
    yield Msg('close_run')


def wait_one(det, motor):
    "Set, trigger, read"
    yield Msg('open_run')
    yield Msg('set', motor, 5, group='A')  # Add to group 'A'.
    yield Msg('wait', None, group='A')  # Wait for everything in group 'A' to finish.
    yield Msg('trigger', det)
    yield Msg('read', det)
    yield Msg('close_run')


def wait_multiple(det, motors):
    "Set motors, trigger all motors, wait for all motors to move."
    yield Msg('open_run')
    for motor in motors:
        yield Msg('set', motor, 5, group='A')
    # Wait for everything in group 'A' to report done.
    yield Msg('wait', None, group='A')
    yield Msg('trigger', det)
    yield Msg('read', det)
    yield Msg('close_run')


def wait_complex(det, motors):
    "Set motors, trigger motors, wait for all motors to move in groups."
    # Same as above...
    yield Msg('open_run')
    for motor in motors[:-1]:
        yield Msg('set', motor, 5, group='A')

    # ...but put the last motor is separate group.
    yield Msg('set', motors[-1], 5, group='B')
    # Wait for everything in group 'A' to report done.
    yield Msg('wait', None, group='A')
    yield Msg('trigger', det)
    yield Msg('read', det)

    # Wait for everything in group 'B' to report done.
    yield Msg('wait', None, group='B')
    yield Msg('trigger', det)
    yield Msg('read', det)
    yield Msg('close_run')


def conditional_pause(det, motor, defer, include_checkpoint):
    yield Msg('open_run')
    for i in range(5):
        if include_checkpoint:
            yield Msg('checkpoint')
        yield Msg('set', motor, i)
        yield Msg('trigger', det)
        reading = yield Msg('read', det)
        if reading['det']['value'] < 0.2:
            yield Msg('pause', defer=defer)
        print("I'm not pausing yet.")
    yield Msg('close_run')


def simple_scan_saving(det, motor):
    "Set, trigger, read"
    yield Msg('open_run')
    yield Msg('create')
    yield Msg('set', motor, 5)
    yield Msg('read', motor)
    yield Msg('trigger', det)
    yield Msg('read', det)
    yield Msg('save')
    yield Msg('close_run')


def stepscan(det, motor):
    yield Msg('open_run')
    for i in range(-5, 5):
        yield Msg('create')
        yield Msg('set', motor, i)
        yield Msg('trigger', det)
        yield Msg('read', motor)
        yield Msg('read', det)
        yield Msg('save')
    yield Msg('close_run')


def cautious_stepscan(det, motor):
    yield Msg('open_run')
    for i in range(-5, 5):
        yield Msg('checkpoint')
        yield Msg('create')
        yield Msg('set', motor, i)
        yield Msg('trigger', det)
        ret_m = yield Msg('read', motor)
        ret_d = yield Msg('read', det)
        yield Msg('save')
        print("Value at {m} is {d}. Pausing.".format(
            m=ret_m[motor.name]['value'], d=ret_d[det.name]['value']))
        yield Msg('pause', None, defer=True)
    yield Msg('close_run')


def fly_gen(flyer, start, stop, step):
    yield Msg('open_run')
    yield Msg('kickoff', flyer, start, stop, step, group='fly-kickoff')
    yield Msg('wait', None, group='fly-kickoff')
    yield Msg('complete', flyer, group='fly-complete')
    yield Msg('wait', None, group='fly-complete')
    yield Msg('collect', flyer)
    yield Msg('kickoff', flyer, start, stop, step, group='fly-kickoff2')
    yield Msg('wait', None, group='fly-kickoff2')
    yield Msg('complete', flyer, group='fly-complete2')
    yield Msg('wait', None, group='fly-complete2')
    yield Msg('collect', flyer)
    yield Msg('close_run')


def multi_sample_temperature_ramp(detector, sample_names, sample_positions,
                                  scan_motor, start, stop, step,
                                  temp_controller, tstart, tstop, tstep):
    def read_and_store_temp():
        yield Msg('create')
        yield Msg('read', temp_controller)
        yield Msg('save')

    peak_centers = [-1+3*n for n in range(len((sample_names)))]
    detector.noise = True

    for idx, temp in enumerate(np.arange(tstart, tstop, tstep)):
        # todo would be cute to have the temperature reduce peak noise
        yield Msg('set', temp_controller, temp)
        for sample, sample_position, peak_pos in zip(sample_names,
                                                     sample_positions,
                                                     peak_centers):
            yield Msg('open_run', sample_name=sample, target_temp=temp)
            detector.center = peak_pos
            detector.sigma = .5 + .25 * idx
            detector.noise_factor = .05 + idx * 0.1
            for scan_pos in np.arange(start, stop, step):
                yield Msg('set', scan_motor, scan_pos)
                # be super paranoid about the temperature. Grab it before and
                # after each trigger!
                # Capturing the temperature data before and after each
                # trigger is resulting in unintended behavior. Uncomment the
                # two `yield from` statements and run this example to see
                # what I'm talking about.
                # yield from read_and_store_temp()
                yield Msg('trigger', detector)
                # yield from read_and_store_temp()
                yield Msg('create')
                yield Msg('read', scan_motor)
                yield Msg('read', detector)
                yield Msg('read', temp_controller)
                # yield Msg('sleep', None, .1)
                yield Msg('save')
            # generate the end of the run document
            yield Msg('close_run')
