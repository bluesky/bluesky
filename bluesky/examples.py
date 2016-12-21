import asyncio
import time as ttime
from collections import deque, OrderedDict
from threading import RLock
import numpy as np
from bluesky.utils import Msg
from uuid import uuid4
import uuid
from tempfile import mkdtemp
import os
from bluesky.utils import new_uid, short_uid


class SimpleStatus:
    """
    This provides a single-slot callback for when the operation has finished.

    It is "simple" because it does not support a timeout or a settling time.
    """

    def __init__(self, *, done=False, success=False):
        super().__init__()
        self._lock = RLock()
        self._cb = None
        self.done = done
        self.success = success

    def _finished(self, success=True, **kwargs):
        if self.done:
            return

        with self._lock:
            self.success = success
            self.done = True

            if self._cb is not None:
                self._cb()
                self._cb = None

    @property
    def finished_cb(self):
        """
        Callback to be run when the status is marked as finished

        The call back has no arguments
        """
        return self._cb

    @finished_cb.setter
    def finished_cb(self, cb):
        with self._lock:
            if self._cb is not None:
                raise RuntimeError("Cannot change the call back")
            if self.done:
                cb()
            else:
                self._cb = cb

    def __str__(self):
        return ('{0}(done={1.done}, '
                'success={1.success})'
                ''.format(self.__class__.__name__, self)
                )

    __repr__ = __str__


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
    fields : dict
        Mapping field names to functions that return simulated data. The
        function will be passed no arguments.
    read_attrs : list, optional
        List of field names to include in ``read()`` . By default, all fields.
    conf_attrs : list, optional
        List of field names to include in ``read_configuration()``. By default,
        no fields. Any field nmaes specified here are then not included in
        ``read_attrs`` by default.
    monitor_intervals : list, optional
        iterable of numbers, specifying the spacing in time of updates from the
        device (this applies only if the ``subscribe`` method is used)
    loop : asyncio.EventLoop, optional
        used for ``subscribe`` updates; uses ``asyncio.get_event_loop()`` if
        unspecified
    exposure_time : float, optional
       Simulated exposure time is seconds.  Defaults to 0

    Examples
    --------
    A detector that always returns 5.
    >>> det = Reader('det', {'intensity': lambda: 5})

    A detector that is coupled to a motor, such that measured insensity
    varies with motor position.
    >>> motor = Mover('motor', {'motor': lambda x: x}, {'x': 0})
    >>> det = Reader('det',
    ...                {'intensity': lambda: 2 * motor.read()['value']})
    """

    def __init__(self, name, fields, *,
                 read_attrs=None, conf_attrs=None, monitor_intervals=None,
                 loop=None, exposure_time=0):
        self.exposure_time = exposure_time
        self.name = name
        self.parent = None
        self._fields = fields
        if conf_attrs is None:
            conf_attrs = []
        if read_attrs is None:
            read_attrs = list(set(fields) - set(conf_attrs))
        self.conf_attrs = conf_attrs
        self.read_attrs = read_attrs

        # All this is used only by monitoring (subscribe/unsubscribe).
        self._futures = {}
        if monitor_intervals is None:
            monitor_intervals = []
        self._monitor_intervals = monitor_intervals
        if loop is None:
            loop = asyncio.get_event_loop()
        self.loop = loop

    def __str__(self):
        # Show just name for readability, as in the cycler example in the docs.
        return ('{0}(name={1.name})'
                ''.format(self.__class__.__name__, self)
                )

    __repr__ = __str__

    def __setstate__(self, val):
        name, fields, read_attrs, conf_attrs, monitor_intervals = val
        self.name = name
        self._fields = fields
        self.read_attrs = read_attrs
        self.conf_attrs = conf_attrs
        self._futures = {}
        self._monitor_intervals = monitor_intervals
        self.loop = asyncio.get_event_loop()

    def __getstate__(self):
        return (self.name, self._fields, self.read_attrs, self.conf_attrs,
                self._monitor_intervals)

    def trigger(self):
        delay_time = self.exposure_time
        if delay_time:
            if self.loop.is_running():
                st = SimpleStatus()
                self.loop.call_later(delay_time, st._finished)
                return st
            else:
                ttime.sleep(delay_time)
        return NullStatus()

    def read(self):
        """
        Simulate readings by calling functions.

        The readings are collated with timestamps.
        """
        return {field: {'value': func(), 'timestamp': ttime.time()}
                        for field, func in self._fields.items()
                        if field in self.read_attrs}

    def describe(self):
        ret = {}
        d = {field: {'value': func(), 'timestamp': ttime.time()}
             for field, func in self._fields.items()
             if field in self.read_attrs}
        for k, v in d.items():
            v = v['value']
            try:
                shape = v.shape
            except AttributeError:
                shape = []
            dtype = 'array' if len(shape) else 'number'
            ret[k] = {'source': 'simulated using bluesky.examples',
                      'dtype': dtype,
                      'shape': shape,
                      'precision': 2}
        return ret

    def read_configuration(self):
        """
        Like `read`, but providing slow-changing configuration readings.
        """
        return {field: {'value': func(), 'timestamp': ttime.time()}
                        for field, func in self._fields.items()
                        if field in self.conf_attrs}

    def describe_configuration(self):
        return {field: {'source': 'simulated using bluesky.examples',
                        'dtype': 'number',
                        'shape': [],
                        'precision': 2}
                for field in self.conf_attrs}

    def configure(self, *args, **kwargs):
        old_conf = self.read_configuration()
        # Update configuration here.
        new_conf = self.read_configuration()
        return old_conf, new_conf

    def subscribe(self, function):
        "Simulate monitoring updates from a device."

        def sim_monitor():
            for interval in self._monitor_intervals:
                ttime.sleep(interval)
                function()

        self._futures[function] = self.loop.run_in_executor(None, sim_monitor)

    def clear_sub(self, function):
        self._futures.pop(function).cancel()


class Mover(Reader):
    """

    Parameters
    ----------
    name : string
    fields : dict
        Mapping field names to functions that return simulated data. The
        function will be passed no arguments.
    initial_set : dict
        passed to ``set`` as ``set(**initial_set)`` to initialize readings
    fake_sleep : float, optional
        simulate moving time
    read_attrs : list, optional
        List of field names to include in ``read()`` . By default, all fields.
    conf_attrs : list, optional
        List of field names to include in ``read_configuration()``. By default,
        no fields. Any field nmaes specified here are then not included in
        ``read_attrs`` by default.
    monitor_intervals : list, optional
        iterable of numbers, specifying the spacing in time of updates from the
        device (this applies only if the ``subscribe`` method is used)
    loop : asyncio.EventLoop, optional
        used for ``subscribe`` updates; uses ``asyncio.get_event_loop()`` if
        unspecified

    Examples
    --------
    A motor with one field.
    >>> motor = Mover('motor', {'motor': lambda x: x}, {'x': 0})

    A motor that simply goes where it is set.
    >>> motor = Mover('motor',
    ...               OrderedDict([('motor', lambda x: x),
    ...                            ('motor_setpoint', lambda x: x)]),
    ...               {'x': 0})

    A motor that adds jitter.
    >>> import numpy as np
    >>> motor = Mover('motor',
    ...               OrderedDict([('motor', lambda x: x + np.random.randn()),
    ...                            ('motor_setpoint', lambda x: x)]),
    ...               {'x': 0})
    """

    def __init__(self, name, fields, initial_set, *, read_attrs=None,
                 conf_attrs=None, fake_sleep=0, monitor_intervals=None,
                 loop=None):
        super().__init__(name, fields, read_attrs=read_attrs,
                         conf_attrs=conf_attrs,
                         monitor_intervals=monitor_intervals,
                         loop=loop)
        # Do initial set without any fake sleep.
        self._fake_sleep = 0
        self.set(**initial_set)
        self._fake_sleep = fake_sleep

    def __setstate__(self, val):
        (name, fields, read_attrs, conf_attrs,
         monitor_intervals, state, fk_slp) = val
        self.name = name
        self._fields = fields
        self.read_attrs = read_attrs
        self.conf_attrs = conf_attrs
        self._futures = {}
        self._monitor_intervals = monitor_intervals
        self.loop = asyncio.get_event_loop()
        self._state = state
        self._fake_sleep = fk_slp

    def __getstate__(self):
        return (self.name, self._fields, self.read_attrs, self.conf_attrs,
                self._monitor_intervals, self._state, self._fake_sleep)

    def set(self, *args, **kwargs):
        """
        Pass the arguments to the functions to create the next reading.
        """
        self._state = {field: {'value': func(*args, **kwargs),
                               'timestamp': ttime.time()}
                       for field, func in self._fields.items()}

        if self._fake_sleep:
            if self.loop.is_running():
                st = SimpleStatus()
                self.loop.call_later(self._fake_sleep, st._finished)
                return st
            else:
                ttime.sleep(self._fake_sleep)
        return NullStatus()

    def read(self):
        return self._state

    def describe(self):
        ret = {}
        for k, v in self._state.items():
            v = v['value']
            try:
                shape = v.shape
            except AttributeError:
                shape = []
            dtype = 'array' if len(shape) else 'number'
            ret[k] = {'source': 'simulated using bluesky.examples',
                      'dtype': dtype,
                      'shape': shape,
                      'precision': 2}
        return ret

    @property
    def position(self):
        "A heuristic that picks a single scalar out of the `read` dict."
        return self.read()[list(self._fields)[0]]['value']

    def stop(self, *, success=False):
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
                 noise=None, noise_multiplier=1, **kwargs):
        if noise not in ('poisson', 'uniform', None):
            raise ValueError("noise must be one of 'poisson', 'uniform', None")

        def func():
            m = motor.read()[motor_field]['value']
            v = Imax * np.exp(-(m - center) ** 2 / (2 * sigma ** 2))
            if noise == 'poisson':
                v = int(np.random.poisson(np.round(v), 1))
            elif noise == 'uniform':
                v += np.random.uniform(-1, 1) * noise_multiplier
            return v

        super().__init__(name, {name: func}, **kwargs)


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
                 center, Imax, sigma=1, noise=None, noise_multiplier=1):

        if noise not in ('poisson', 'uniform', None):
            raise ValueError("noise must be one of 'poisson', 'uniform', None")

        def func():
            x = motor0.read()[motor_field0]['value']
            y = motor1.read()[motor_field1]['value']
            m = np.array([x, y])
            v = Imax * np.exp(-np.sum((m - center) ** 2) / (2 * sigma ** 2))
            if noise == 'poisson':
                v = int(np.random.poisson(np.round(v), 1))
            elif noise == 'uniform':
                v += np.random.uniform(-1, 1) * noise_multiplier
            return v

        super().__init__(name, {name: func})


class ReaderWithFileStore(Reader):
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
    monitor_intervals : list, optional
        iterable of numbers, specifying the spacing in time of updates from the
        device (this applies only if the ``subscribe`` method is used)
    loop : asyncio.EventLoop, optional
        used for ``subscribe`` updates; uses ``asyncio.get_event_loop()`` if
        unspecified
    fs : FileStore
        FileStore object that supports inserting resource and datum documents
    save_path : str, optional
        Path to save files to, if None make a temp dir, defaults to None.

    """

    def __init__(self, *args, fs, save_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fs = fs
        self._resource_id = None
        if save_path is None:
            self.save_path = mkdtemp()
        else:
            self.save_path = save_path
        self.filestore_spec = 'RWFS_NPY'  # spec name stored in resource doc

        self._file_stem = None
        self._path_stem = None
        self._result = {}

    def stage(self):
        self._file_stem = short_uid()
        self._path_stem = os.path.join(self.save_path, self._file_stem)
        self._resource_id = self.fs.insert_resource(self.filestore_spec,
                                                    self._file_stem, {},
                                                    root=self.save_path)

    def trigger(self):
        # save file stash file name
        self._result.clear()
        for idx, (name, reading) in enumerate(super().read().items()):
            # Save the actual reading['value'] to disk and create a record
            # in FileStore.
            np.save('{}_{}.npy'.format(self._path_stem, idx), reading['value'])
            datum_id = new_uid()
            self.fs.insert_datum(self._resource_id, datum_id,
                                 dict(index=idx))
            # And now change the reading in place, replacing the value with
            # a reference to FileStore.
            reading['value'] = datum_id
            self._result[name] = reading

        delay_time = self.exposure_time
        if delay_time:
            if self.loop.is_running():
                st = SimpleStatus()
                self.loop.call_later(delay_time, st._finished)
                return st
            else:
                ttime.sleep(delay_time)

        return NullStatus()

    def read(self):
        return self._result

    def describe(self):
        res = super().describe()
        for key in res:
            res[key]['external'] = "FILESTORE"
        return res

    def unstage(self):
        self._resource_id = None
        self._file_stem = None
        self._path_stem = None
        self._result.clear()


class TrivialFlyer:
    """Trivial flyer that complies to the API but returns empty data."""

    def kickoff(self):
        return NullStatus()

    def describe_collect(self):
        return {'stream_name': {}}

    def read_configuration(self):
        return OrderedDict()

    def describe_configuration(self):
        return OrderedDict()

    def complete(self):
        return NullStatus()

    def collect(self):
        for i in range(100):
            yield {'data': {}, 'timestamps': {}, 'time': i, 'seq_num': i}

    def stop(self, *, success=False):
        pass


class MockFlyer:
    """
    Class for mocking a flyscan API implemented with stepper motors.
    """

    def __init__(self, name, detector, motor, start, stop, num, loop=None):
        self.name = name
        self.parent = None
        self._mot = motor
        self._detector = detector
        self._steps = np.linspace(start, stop, num)
        self._data = deque()
        self._completion_status = None
        if loop is None:
            loop = asyncio.get_event_loop()
        self.loop = loop

    def __setstate__(self, val):
        name, detector, motor, steps = val
        self.name = name
        self.parent = None
        self._mot = motor
        self._detector = detector
        self._steps = steps
        self._completion_status = None
        self.loop = asyncio.get_event_loop()

    def __getstate__(self):
        return (self.name, self._detector, self._mot, self._steps)

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
        return self._completion_status

    def kickoff(self):
        if self._completion_status is not None:
            raise RuntimeError("Already kicked off.")
        self._data = deque()

        # Setup a status object that will be returned by
        # self.complete(). Separately, make dummy status object
        # that is immediately done, and return that, indicated that
        # the 'kickoff' step is done.
        self._future = self.loop.run_in_executor(None, self._scan)
        st = SimpleStatus()
        self._completion_status = st
        self._future.add_done_callback(lambda x: st._finished())

        return NullStatus()

    def collect(self):
        if self._completion_status is not None:
            raise RuntimeError("No reading until done!")

        yield from self._data

    def _scan(self):
        "This will be run on a separate thread, started in self.kickoff()"
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
        self._completion_status._finished()
        self._completion_status = None

    def stop(self, *, success=False):
        pass


class GeneralReaderWithFileStore(Reader):
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
    monitor_intervals : list, optional
        iterable of numbers, specifying the spacing in time of updates from the
        device (this applies only if the ``subscribe`` method is used)
    loop : asyncio.EventLoop, optional
        used for ``subscribe`` updates; uses ``asyncio.get_event_loop()`` if
        unspecified
    fs : FileStore
        FileStore object that supports inserting resource and datum documents
    save_path : str, optional
        Path to save files to, if None make a temp dir, defaults to None.
    save_func : function, optional
        The function to save the data, function signature must be:
        `func(file_path, array)`, defaults to np.save
    save_spec : str, optional
        The filestore spec for the save function, defaults to 'RWFS_NPY'
    save_ext : str, optional
        The extention to add to the file name, defaults to '.npy'

    """

    def __init__(self, *args, fs, save_path=None, save_func=np.save,
                 save_spec='RWFS_NPY', save_ext='.npy',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.fs = fs
        self.save_func = save_func
        self.save_ext = save_ext
        self._resource_id = None
        if save_path is None:
            self.save_path = mkdtemp()
        else:
            self.save_path = save_path
        self.filestore_spec = save_spec  # spec name stored in resource doc

        self._file_stem = None
        self._path_stem = None
        self._result = {}

    def stage(self):
        self._file_stem = short_uid()
        self._path_stem = os.path.join(self.save_path, self._file_stem)
        self._resource_id = self.fs.insert_resource(self.filestore_spec,
                                                    self._file_stem, {},
                                                    root=self.save_path)

    def trigger(self):
        # save file stash file name
        self._result.clear()
        for idx, (name, reading) in enumerate(super().read().items()):
            # Save the actual reading['value'] to disk and create a record
            # in FileStore.
            self.save_func('{}_{}.{}'.format(self._path_stem, idx,
                                             self.save_ext), reading['value'])
            datum_id = new_uid()
            self.fs.insert_datum(self._resource_id, datum_id,
                                 dict(index=idx))
            # And now change the reading in place, replacing the value with
            # a reference to FileStore.
            reading['value'] = datum_id
            self._result[name] = reading
        return NullStatus()

    def read(self):
        return self._result

    def describe(self):
        res = super().describe()
        for key in res:
            res[key]['external'] = "FILESTORE"
        return res

    def unstage(self):
        self._resource_id = None
        self._file_stem = None
        self._path_stem = None
        self._result.clear()


class ReaderWithFSHandler:
    specs = {'RWFS_NPY'}

    def __init__(self, filename, root=''):
        self._name = os.path.join(root, filename)

    def __call__(self, index):
        return np.load('{}_{}.npy'.format(self._name, index))

    def get_file_list(self, datum_kwarg_gen):
        "This method is optional. It is not needed for access, but for export."
        return ['{name}_{index}.npy'.format(name=self._name, **kwargs)
                for kwargs in datum_kwarg_gen]


motor = Mover('motor', OrderedDict([('motor', lambda x: x),
                                    ('motor_setpoint', lambda x: x)]),
              {'x': 0})
motor1 = Mover('motor1', OrderedDict([('motor1', lambda x: x),
                                      ('motor1_setpoint', lambda x: x)]),
               {'x': 0})
motor2 = Mover('motor2', OrderedDict([('motor2', lambda x: x),
                                      ('motor2_setpoint', lambda x: x)]),
               {'x': 0})
motor3 = Mover('motor3', OrderedDict([('motor3', lambda x: x),
                                      ('motor3_setpoint', lambda x: x)]),
               {'x': 0})
jittery_motor1 = Mover('jittery_motor1',
                       OrderedDict([('jittery_motor1',
                                     lambda x: x + np.random.randn()),
                                    ('jittery_motor1_setpoint', lambda x: x)]),
                       {'x': 0})
jittery_motor2 = Mover('jittery_motor2',
                       OrderedDict([('jittery_motor2',
                                     lambda x: x + np.random.randn()),
                                    ('jittery_motor2_setpoint', lambda x: x)]),
                       {'x': 0})
noisy_det = SynGauss('noisy_det', motor, 'motor', center=0, Imax=1,
                     noise='uniform', sigma=1, noise_multiplier=0.1)
det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)
det1 = SynGauss('det1', motor1, 'motor1', center=0, Imax=5, sigma=0.5)
det2 = SynGauss('det2', motor2, 'motor2', center=1, Imax=2, sigma=2)
det3 = SynGauss('det3', motor3, 'motor3', center=-1, Imax=2, sigma=1)
det4 = Syn2DGauss('det4', motor1, 'motor1', motor2, 'motor2',
                  center=(0, 0), Imax=1)
det5 = Syn2DGauss('det5', jittery_motor1, 'jittery_motor1', jittery_motor2,
                  'jittery_motor2', center=(0, 0), Imax=1)

flyer1 = MockFlyer('flyer1', det, motor, 1, 5, 20)
flyer2 = MockFlyer('flyer2', det, motor, 1, 5, 10)


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
    yield Msg('wait', None,
              group='A')  # Wait for everything in group 'A' to finish.
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


def fly_gen(flyer):
    yield Msg('open_run')
    yield Msg('kickoff', flyer, group='fly-kickoff')
    yield Msg('wait', None, group='fly-kickoff')
    yield Msg('complete', flyer, group='fly-complete')
    yield Msg('wait', None, group='fly-complete')
    yield Msg('collect', flyer)
    yield Msg('kickoff', flyer, group='fly-kickoff2')
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

    peak_centers = [-1 + 3 * n for n in range(len((sample_names)))]
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
