import asyncio
import time as ttime
from collections import deque
import numpy as np
from .run_engine import Msg

loop = asyncio.get_event_loop()


class MockSignal:
    "hotfix for 2016 winter cycle -- build out more thoroughly later"
    def __init__(self, field):
        self._field = field

    def read(self):
        return {self._field: (0, 0)}

    def describe(self):
        return {self._field: (0, 0)}


class Base:
    def __init__(self, name, fields):
        self.name = name
        self._fields = fields
        self._cb = None
        self._ready = False
        self.configuration_attrs = []
        for field in fields:
            if isinstance(field, str):
                # Flyers pass objects in as fields, not names.
                # hotfix 2016 -- revisit this!
                setattr(self, field, MockSignal(field))
        self.success = True
        self.root = self
        self.precision = 3

    def describe(self):
        return {k: {'source': self.name, 'dtype': 'number', 'shape': None,
                    'precision': self.precision}
                for k in self._fields}

    def __repr__(self):
        return '{}: {}'.format(self._klass, self.name)

    def read_configuration(self):
        return {}

    def describe_configuration(self):
        return {}

    def configure(self, d):
        return {}, {}

    def stage(self):
        pass

    def unstage(self):
        pass

    @property
    def done(self):
        return self.ready

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
        self.ready = True
        if self._cb is not None:
            self._cb()
            self._cb = None


class Reader(Base):
    _klass = 'reader'

    def __init__(self, *args, **kwargs):
        super(Reader, self).__init__(*args, **kwargs)
        self._cnt = 0

    def read(self):
        data = dict()
        for k in self._fields:
            data[k] = {'value': self._cnt, 'timestamp': ttime.time()}
            self._cnt += 1

        return data

    def trigger(self):
        return self


class Mover(Base):
    _klass = 'mover'

    def __init__(self, name, fields, *, sleep_time=0, **kwargs):
        super(Mover, self).__init__(name, fields, **kwargs)
        self._data = {f: {'value': 0, 'timestamp': ttime.time()}
                      for f in self._fields}
        self.ready = True
        self._fake_sleep = sleep_time

    def read(self):
        return self._data

    def set(self, val, *, trigger=True, block_group=None):
        # If trigger is False, wait for a separate 'trigger' command to move.
        if not trigger:
            raise NotImplementedError
        # block_group is handled by the RunEngine
        self.ready = False
        if self._fake_sleep:
            ttime.sleep(self._fake_sleep)  # simulate moving time
        if isinstance(val, dict):
            for k, v in val.items():
                self._data[k] = v
        else:
            self._data = {f: {'value': val, 'timestamp': ttime.time()}
                          for f in self._fields}
        self.ready = True
        return self

    def settle(self):
        pass

    def stop(self):
        pass

    def trigger(self):
        return None


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
    motor = Mover('motor', ['motor'])
    det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)
    """
    _klass = 'reader'

    def __init__(self, name, motor, motor_field, center, Imax, sigma=1,
                 noise=None, noise_multiplier=1):
        super(SynGauss, self).__init__(name, [name, ])
        self.ready = True
        self._motor = motor
        self._motor_field = motor_field
        self.center = center
        self.Imax = Imax
        self.sigma = sigma
        self.noise = noise
        self.noise_multiplier = noise_multiplier
        if noise not in ('poisson', 'uniform', None):
            raise ValueError("noise must be one of 'poisson', 'uniform', None")

    def trigger(self, *, block_group=True):
        self.ready = False
        m = self._motor.read()[self._motor_field]['value']
        v = self.Imax * np.exp(-(m - self.center)**2 / (2 * self.sigma**2))
        if self.noise == 'poisson':
            v = int(np.random.poisson(np.round(v), 1))
        elif self.noise == 'uniform':
            v += np.random.uniform(-1, 1) * self.noise_multiplier
        self._data = {self.name: {'value': v, 'timestamp': ttime.time()}}
        ttime.sleep(0.05)  # simulate exposure time
        self.ready = True
        return self

    def read(self):
        return self._data


class Syn2DGauss(Reader):
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
    motor = Mover('motor', ['motor'])
    det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)
    """
    _klass = 'reader'

    def __init__(self, name, motor0, motor_field0, motor1, motor_field1,
                 center, Imax, sigma=1,
                 noise=None, noise_multiplier=1):
        super().__init__(name, [name, ])
        self.ready = True
        self._motor0 = motor0
        self._motor_field0 = motor_field0
        self._motor1 = motor1
        self._motor_field1 = motor_field1
        self.center = np.asarray(center)
        self.Imax = Imax
        self.sigma = sigma
        self.noise = noise
        self.noise_multiplier = noise_multiplier
        if noise not in ('poisson', 'uniform', None):
            raise ValueError("noise must be one of 'poisson', 'uniform', None")

    def trigger(self, *, block_group=True):
        self.ready = False
        x = self._motor0.read()[self._motor_field0]['value']
        y = self._motor1.read()[self._motor_field1]['value']
        m = np.array([x, y])
        v = self.Imax * np.exp(
            -np.sum((m - self.center)**2) / (2 * self.sigma**2))
        if self.noise == 'poisson':
            v = int(np.random.poisson(np.round(v), 1))
        elif self.noise == 'uniform':
            v += np.random.uniform(-1, 1) * self.noise_multiplier
        self._data = {self.name: {'value': v, 'timestamp': ttime.time()}}
        ttime.sleep(0.05)  # simulate exposure time
        self.ready = True
        return self

    def read(self):
        return self._data


class MockFlyer:
    """
    Class for mocking a flyscan API implemented with stepper motors.

    """
    def __init__(self, detector, motor):
        self._mot = motor
        self._detector = detector
        self._steps = None
        self._future = None
        self._data = deque()
        self._cb = None
        self.ready = False
        self.success = True

    @property
    def done(self):
        return self.ready

    def describe(self):
        dd = dict()
        dd.update(self._mot.describe())
        dd.update(self._detector.describe())
        return [dd, ]

    def kickoff(self, start, stop, steps):
        self.success = True
        self.ready = False
        self._data = deque()
        self._steps = np.linspace(start, stop, steps)
        self._future = loop.run_in_executor(None, self._scan)
        self._future.add_done_callback(lambda x: self._finish())
        return self

    def collect(self):
        if not self.ready:
            raise RuntimeError("No reading until done!")

        yield from self._data
        self._thread = None

    def _scan(self):
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
        self.ready = True
        if self._cb is not None:
            self._cb()
            self._cb = None


class FlyMagic(Base):
    _klass = 'flyer'

    def __init__(self, name, motor, det, det2, scan_points=15):
        super(FlyMagic, self).__init__(name, [motor, det, det2])
        self._motor = motor
        self._det = det
        self._det2 = det2
        self._scan_points = scan_points
        self._time = None
        self._fly_count = 0

    def reset(self):
        self._fly_count = 0

    def kickoff(self):
        self._time = ttime.time()
        self._fly_count += 1
        return self

    def describe(self):
        return [{k: {'source': self.name, 'dtype': 'number'}
                 for k in [self._motor, self._det]},
                {self._det2: {'source': self.name, 'dtype': 'number'}}]

    def collect(self):
        if self._time is None:
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

    def stop(self):
        pass

motor = Mover('motor', ['motor'])
motor1 = Mover('motor1', ['motor1'], sleep_time=.1)
motor2 = Mover('motor2', ['motor2'], sleep_time=.2)
motor3 = Mover('motor3', ['motor3'], sleep_time=.5)
noisy_det = SynGauss('det', motor, 'motor', center=0, Imax=1,
                     noise='uniform', sigma=1)
det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)
det1 = SynGauss('det1', motor1, 'motor1', center=0, Imax=5, sigma=0.5)
det2 = SynGauss('det2', motor2, 'motor2', center=1, Imax=2, sigma=2)
det3 = SynGauss('det3', motor3, 'motor3', center=-1, Imax=2, sigma=1)


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
        if reading['det']['value'] < threshold:
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
    yield Msg('set', motor, 5, block_group='A')  # Add to group 'A'.
    yield Msg('wait', None, 'A')  # Wait for everything in group 'A' to finish.
    yield Msg('trigger', det)
    yield Msg('read', det)
    yield Msg('close_run')


def wait_multiple(det, motors):
    "Set motors, trigger all motors, wait for all motors to move."
    yield Msg('open_run')
    for motor in motors:
        yield Msg('set', motor, 5, block_group='A')
    # Wait for everything in group 'A' to report done.
    yield Msg('wait', None, 'A')
    yield Msg('trigger', det)
    yield Msg('read', det)
    yield Msg('close_run')


def wait_complex(det, motors):
    "Set motors, trigger motors, wait for all motors to move in groups."
    # Same as above...
    yield Msg('open_run')
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


def panic_timer(RE, delay):
    loop = asyncio.get_event_loop()
    loop.call_later(delay, RE.panic)


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
    yield Msg('kickoff', flyer, start, stop, step, block_group='fly')
    yield Msg('wait', None, 'fly')
    yield Msg('collect', flyer)
    yield Msg('kickoff', flyer, start, stop, step, block_group='fly')
    yield Msg('wait', None, 'fly')
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
