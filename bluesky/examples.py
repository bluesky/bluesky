import threading
import time as ttime
from collections import deque
import numpy as np
from lmfit.models import GaussianModel, LinearModel
from .run_engine import Msg
from filestore.file_writers import save_ndarray
import tempfile
import random
from .callbacks import *


class Base:
    def __init__(self, name, fields):
        self._name = name
        self._fields = fields

    def describe(self):
        return {k: {'source': self._name, 'dtype': 'number', 'shape': None}
                for k in self._fields}

    def __repr__(self):
        return '{}: {}'.format(self._klass, self._name)

    def configure(self, *args, **kwargs):
        pass

    def deconfigure(self, *args, **kwargs):
        pass

    @property
    def done(self):
        return self.ready


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


class SynGauss(Reader):
    """
    Evaluate a point on a Gaussian based on the value of a motor.

    Example
    -------
    motor = Mover('motor', ['motor'])
    det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)
    """
    _klass = 'reader'

    def __init__(self, name, motor, motor_field, center, Imax, sigma=1,
                 noise=False):
        super(SynGauss, self).__init__(name, [name, ])
        self.ready = True
        self._motor = motor
        self._motor_field = motor_field
        self.center = center
        self.Imax = Imax
        self.sigma = sigma
        self.noise = noise

    def trigger(self, *, block_group=True):
        self.ready = False
        m = self._motor._data[self._motor_field]['value']
        v = self.Imax * np.exp(-(m - self.center)**2 / (2 * self.sigma**2))
        if self.noise:
            v = int(np.random.poisson(np.round(v), 1))
        self._data = {self._name: {'value': v, 'timestamp': ttime.time()}}
        ttime.sleep(0.05)  # simulate exposure time
        self.ready = True
        return self

    def read(self):
        return self._data


class NoisyGaussian(Reader):
    """
    Evaluate a point on a Gaussian based on the value of a motor.

    Example
    -------
    motor = Mover('motor', ['motor'])
    det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)
    """
    _klass = 'reader'

    def __init__(self, name, motor, motor_field, center, Imax, sigma=1,
                 noise=False, noise_factor=1):
        super().__init__(name, [name, ])
        self.ready = True
        self._motor = motor
        self._motor_field = motor_field
        self.center = center
        self.Imax = Imax
        self.sigma = sigma
        self.noise = noise
        self.noise_factor = 1

    def trigger(self, *, block_group=True):
        self.ready = False
        m = self._motor._data[self._motor_field]['value']
        v = self.Imax * np.exp(-(m - self.center)**2 / (2 * self.sigma**2))
        if self.noise:
            v += random.uniform(-1, 1) * self.noise_factor
        self._data = {self._name: {'value': v, 'timestamp': ttime.time()}}
        ttime.sleep(0.05)  # simulate exposure time
        self.ready = True
        return self

    def read(self):
        return self._data


class SynGauss2D(Reader):
    """
    Evaluate a point on a Gaussian based on the value of a motor.

    Example
    -------
    motor = Mover('motor', ['motor'])
    det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)
    """
    _klass = 'reader'

    def __init__(self, name, motor, motor_field, center, Imax=1000, sigma=1,
                 nx=250, ny=250, img_sigma=50):
        super(SynGauss2D, self).__init__(name, [name, ])
        self.ready = True
        self._motor = motor
        self._motor_field = motor_field
        self.center = center
        self.Imax = Imax
        self.sigma = sigma
        self.dims = (nx, ny)
        self.img_sigma = img_sigma
        # stash these things in a temp directory. This might cause an
        # exception to be raised if/when the file system cleans its temp files
        self.output_dir = tempfile.gettempdir()

    def trigger(self, *, block_group=True):
        self.ready = False
        m = self._motor._data[self._motor_field]['value']
        v = self.Imax * np.exp(-(m - self.center)**2 / (2 * self.sigma**2))
        arr = self.gauss(self.dims, self.img_sigma) * v + np.random.random(
            self.dims) * .01
        fs_uid = save_ndarray(arr, self.output_dir)
        self._data = {self._name: {'value': fs_uid, 'timestamp': ttime.time()}}
        ttime.sleep(0.05)  # simulate exposure time
        self.ready = True
        return self

    def read(self):
        return self._data

    def _dist(self, dims):
        """
        Create array with pixel value equals to the distance from array center.

        Parameters
        ----------
        dims : list or tuple
            shape of array to create

        Returns
        -------
        arr : np.ndarray
            ND array whose pixels are equal to the distance from the center
            of the array of shape `dims`
        """
        dist_sum = []
        shape = np.ones(len(dims))
        for idx, d in enumerate(dims):
            vec = (np.arange(d) - d // 2) ** 2
            shape[idx] = -1
            vec = vec.reshape(*shape)
            shape[idx] = 1
            dist_sum.append(vec)

        return np.sqrt(np.sum(dist_sum, axis=0))

    def gauss(self, dims, sigma):
        """
        Generate Gaussian function in 2D or 3D.

        Parameters
        ----------
        dims : list or tuple
            shape of the data
        sigma : float
            standard deviation of gaussian function

        Returns
        -------
        Array :
            ND gaussian
        """
        x = self._dist(dims)
        y = np.exp(-(x / sigma)**2 / 2)
        return y / np.sum(y)

    def describe(self):
        return {self._name: {'source': self._name,
                             'dtype': 'array',
                             'shape': list(self.dims),
                             'external': 'FILESTORE:'}}


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

    @property
    def done(self):
        return self.ready

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
noisy_det = NoisyGaussian('det', motor, 'motor', center=0, Imax=1, sigma=1)
det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)
det1 = SynGauss('det1', motor1, 'motor1', center=0, Imax=5, sigma=0.5)
det2 = SynGauss('det2', motor2, 'motor2', center=1, Imax=2, sigma=2)
det3 = SynGauss('det3', motor3, 'motor3', center=-1, Imax=2, sigma=1)
det_2d = SynGauss2D('det_2d', motor, 'motor', center=0, Imax=1000, sigma=1,
                    nx=300, ny=300)
det1_2d = SynGauss2D('det1_2d', motor1, 'motor1', center=0, Imax=10,
                     sigma=1, nx=100, ny=600)
det2_2d = SynGauss2D('det2_2d', motor2, 'motor2', center=1, Imax=10,
                     sigma=.5, nx=1000, ny=1000)
det3_2d = SynGauss2D('det3_2d', motor3, 'motor3', center=-1, Imax=10,
                     sigma=1.5, nx=500, ny=200)


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


def do_nothing(timeout=5):
    "Generate 'checkpoint' messages until timeout."
    t = ttime.time()
    while True:
        if ttime.time() > t + timeout:
            break
        ttime.sleep(0.1)
        yield Msg('checkpoint')


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


def find_center_gen(motor, det, motor_name, det_name,
                    initial_center, initial_width, output_mutable=None,
                    tolerance=.01):
    """
    Attempts to find the center of a peak by moving a motor.

    This will leave the motor at what it thinks is the center.

    The motion is clipped to initial center +/- 6 initial width

    Works by :

    - sampling 5 points around the initial center
    - fitting to Gaussian + line
    - moving to the center of the Gaussian
    - while |old center - new center| > tolerance
      - taking a measurement
      - re-run fit
      - move to new center

    Parameters
    ----------
    motor : Mover
    det : Reader
    motor_name : string
        The data key to read out of the motor to check the position
    det_name : string
        The data key to read out of the detector
    initial_center : number
        Initial guess at where the center is
    initial_width : number
        Initial guess at the width
    output_mutable : dict-like, optional
        Must have 'update' method.  Mutable object to provide a side-band to return
        fitting parameters + data points
    tolerance : number
        Tolerance to declare good enough on finding the center.
    """
    tol = tolerance
    seen_x = deque()
    seen_y = deque()
    min_cen = initial_center - 6 * initial_width
    max_cen = initial_center + 6 * initial_width
    for x in np.linspace(initial_center - initial_width,
                         initial_center + initial_width,
                         5, endpoint=True):
        yield Msg('set', motor, x)
        yield Msg('trigger', det)
        yield Msg('sleep', None, .1,)
        yield Msg('create')
        ret_mot = yield Msg('read', motor)
        ret_det = yield Msg('read', det)
        yield Msg('save')
        seen_y.append(ret_det[det_name]['value'])
        seen_x.append(ret_mot[motor_name]['value'])

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
        next_cen = np.clip(guesses['center'] +
                           np.random.randn(1) * guesses['sigma'],
                           min_cen, max_cen)
        yield Msg('set', motor, next_cen)
        yield Msg('trigger', det)
        yield Msg('sleep', None, .1,)
        yield Msg('create')
        ret_mot = yield Msg('read', motor)
        ret_det = yield Msg('read', det)
        yield Msg('save')
        seen_y.append(ret_det[det_name]['value'])
        seen_x.append(ret_mot[motor_name]['value'])

    yield Msg('set', motor, np.clip(guesses['center'], min_cen, max_cen))

    if output_mutable is not None:
        output_mutable.update(guesses)
        output_mutable['x'] = np.array(seen_x)
        output_mutable['y'] = np.array(seen_y)
        output_mutable['model'] = res


def fly_gen(flyer, start, stop, step):
    yield Msg('kickoff', flyer, start, stop, step, block_group='fly')
    yield Msg('wait', None, 'fly')
    yield Msg('collect', flyer)
    yield Msg('kickoff', flyer, start, stop, step, block_group='fly')
    yield Msg('wait', None, 'fly')
    yield Msg('collect', flyer)


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
            yield Msg('new_run', sample_name=sample, target_temp=temp)
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
