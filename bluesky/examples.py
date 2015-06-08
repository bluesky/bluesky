import threading
import time as ttime
from collections import deque
import numpy as np
from lmfit.models import GaussianModel, LinearModel
from .run_engine import Msg

from .callbacks import *


class Base:
    def __init__(self, name, fields):
        self._name = name
        self._fields = fields

    def describe(self):
        return {k: {'source': self._name, 'dtype': 'number'}
                for k in self._fields}

    def __repr__(self):
        return '{}: {}'.format(self._klass, self._name)


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
        pass


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

    def settle(self):
        pass


class SynGauss(Reader):
    """
    Evaluate a point on a Gaussian based on the value of a motor.

    Example
    -------
    motor = Mover('motor', ['motor'])
    det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)
    """
    _klass = 'reader'

    def __init__(self, name, motor, motor_field, center, Imax, sigma=1):
        super(SynGauss, self).__init__(name, [name, ])
        self.ready = True
        self._motor = motor
        self._motor_field = motor_field
        self.center = center
        self.Imax = Imax
        self.sigma = sigma

    def trigger(self, *, block_group=True):
        self.ready = False
        m = self._motor._data[self._motor_field]['value']
        v = self.Imax * np.exp(-(m - self.center)**2 / (2 * self.sigma**2))
        self._data = {self._name: {'value': v, 'timestamp': ttime.time()}}
        ttime.sleep(0.05)  # simulate exposure time
        self.ready = True

    def read(self):
        return self._data


class MockFlyer:
    """
    Class for mocking a flyscan API implemented with stepper motors.

    Currently this does the 'collection' in a blocking fashion in the
    collect step, would be better do to this with a thread starting from
    kickoff
    """
    def __init__(self, motor, detector):
        self._mot = motor
        self._detector = detector
        self._steps = None
        self._thread = None
        self._data = deque()

    @property
    def ready(self):
        return self._thread and not self._thread.is_alive()

    def describe(self):
        dd = dict()
        dd.update(self._mot.describe())
        dd.update(self._detector.describe())
        return [dd, ]

    def kickoff(self, start, stop, steps):
        self._data = deque()
        self._steps = np.linspace(start, stop, steps)
        self._thread = threading.Thread(target=self._scan,
                                                name='mock_fly_thread')
        self._thread.start()

    def collect(self):
        if not self.ready:
            raise RuntimeError("No reading until done!")

        yield from self._data
        self._thread = None

    def _scan(self):
        for p in self._steps:
            self._mot.set(p)
            while True:
                if self._mot.ready:
                    break
                ttime.sleep(0.01)
            self._detector.trigger()
            event = dict()
            event['time'] = ttime.time()
            event['data'] = dict()
            event['timestamp'] = dict()
            for r in [self._mot, self._detector]:
                d = r.read()
                for k, v in d.items():
                    event['data'][k] = v['value']
                    event['timestamp'][k] = v['timestamp']
            self._data.append(event)


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

    def describe(self):
        return [{k: {'source': self._name, 'dtype': 'number'}
                 for k in [self._motor, self._det]},
                {self._det2: {'source': self._name, 'dtype': 'number'}}]

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


motor = Mover('motor', ['motor'])
motor1 = Mover('motor1', ['motor1'], sleep_time=.1)
motor2 = Mover('motor2', ['motor2'], sleep_time=.2)
motor3 = Mover('motor3', ['motor3'], sleep_time=.5)
det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)
det1 = SynGauss('det1', motor1, 'motor1', center=0, Imax=5, sigma=0.5)
det2 = SynGauss('det2', motor2, 'motor2', center=1, Imax=2, sigma=2)
det3 = SynGauss('det3', motor3, 'motor3', center=-1, Imax=2, sigma=1)


def simple_scan(motor):
    yield Msg('set', motor, 5)
    yield Msg('read', motor)


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


def sleepy(motor, det):
    "Set, trigger motor, sleep for a fixed time, trigger detector, read"
    yield Msg('set', motor, 5)
    yield Msg('sleep', None, 2)  # units: seconds
    yield Msg('trigger', det)
    yield Msg('read', det)


def checkpoint_forever():
    # simplest pauseable scan
    while True:
        ttime.sleep(0.1)
        yield Msg('checkpoint')


def wait_one(motor, det):
    "Set, trigger, read"
    yield Msg('set', motor, 5, block_group='A')  # Add to group 'A'.
    yield Msg('wait', None, 'A')  # Wait for everything in group 'A' to finish.
    yield Msg('trigger', det)
    yield Msg('read', det)


def wait_multiple(motors, det):
    "Set motors, trigger all motors, wait for all motors to move."
    for motor in motors:
        yield Msg('set', motor, 5, block_group='A')
    # Wait for everything in group 'A' to report done.
    yield Msg('wait', None, 'A')
    yield Msg('trigger', det)
    yield Msg('read', det)


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


def conditional_pause(motor, det, hard, include_checkpoint):
    for i in range(5):
        if include_checkpoint:
            yield Msg('checkpoint')
        yield Msg('set', motor, i)
        yield Msg('trigger', det)
        reading = yield Msg('read', det)
        if reading['det']['value'] < 0.2:
            yield Msg('pause', hard=hard)
        yield Msg('set', motor, i + 0.5)


class PausingAgent:
    def __init__(self, RE, name):
        self.RE = RE
        self.name = name

    def issue_request(self, hard, delay=0):
        def callback():
            return self.permission

        def requester():
            ttime.sleep(delay)
            self.permission = False
            self.RE.request_pause(hard, self.name, callback)

        thread = threading.Thread(target=requester)
        thread.start()

    def revoke_request(self, delay=0):
        ttime.sleep(delay)
        self.permission = True


def panic_timer(RE, delay):
    def f():
        ttime.sleep(delay)
        RE.panic()

    thread = threading.Thread(target=f)
    thread.start()


def simple_scan_saving(motor, det):
    "Set, trigger, read"
    yield Msg('create')
    yield Msg('set', motor, 5)
    yield Msg('read', motor)
    yield Msg('trigger', det)
    yield Msg('read', det)
    yield Msg('save')


def stepscan(motor, det):
    for i in range(-5, 5):
        yield Msg('create')
        yield Msg('set', motor, i)
        yield Msg('trigger', det)
        yield Msg('read', motor)
        yield Msg('read', det)
        yield Msg('save')


def MoveRead_gen(motor, detector):
    try:
        for j in range(10):
            yield Msg('create')
            yield Msg('set', motor, {'x': j})
            yield Msg('trigger', detector)
            yield Msg('read', detector)
            yield Msg('read', motor)
            yield Msg('save')
    finally:
        print('Generator finished')


def SynGauss_gen(syngaus, motor_steps, motor_limit=None):
    try:
        for x in motor_steps:
            yield Msg('create')
            yield Msg('set', syngaus, {syngaus.motor_name: x})
            yield Msg('trigger', syngaus)
            yield Msg('sleep', None, .1)
            ret = yield Msg('read', syngaus)
            yield Msg('save')
            if motor_limit is not None:
                if ret[syngaus.motor_name] > motor_limit:
                    break
    finally:
        print('generator finished')


def find_center_gen(syngaus, initial_center, initial_width,
                    output_mutable):
    tol = .01
    seen_x = deque()
    seen_y = deque()

    for x in np.linspace(initial_center - initial_width,
                         initial_center + initial_center,
                         5, endpoint=True):
        yield Msg('set', syngaus, {syngaus.motor_name: x})
        yield Msg('trigger', syngaus)
        yield Msg('sleep', None, .1,)
        ret = yield Msg('read', syngaus)
        seen_x.append(ret[syngaus.motor_name])
        seen_y.append(ret[syngaus.det_name])
    model = GaussianModel() + LinearModel()
    guesses = {'amplitude': np.max(seen_y),
               'center': initial_center,
               'sigma': initial_width,
               'slope': 0, 'intercept': 0}
    while True:
        x = np.asarray(seen_x)
        y = np.asarray(seen_y)
        res = model.fit(y, x=x, **guesses)
        old_guess = guesses
        guesses = res.values

        if np.abs(old_guess['center'] - guesses['center']) < tol:
            break

        yield Msg('set', syngaus, {syngaus.motor_name: guesses['center']})
        yield Msg('trigger', syngaus)
        yield Msg('sleep', None, .1)
        ret = yield Msg('read', syngaus)
        seen_x.append(ret[syngaus.motor_name])
        seen_y.append(ret[syngaus.det_name])

    output_mutable.update(guesses)


def fly_gen(flyer, start, stop, step):
    yield Msg('kickoff', flyer, start, stop, step, block_group='fly')
    yield Msg('wait', None, 'fly')
    yield Msg('collect', flyer)
    yield Msg('kickoff', flyer, start, stop, step, block_group='fly')
    yield Msg('wait', None, 'fly')
    yield Msg('collect', flyer)
