from collections import deque, defaultdict
import itertools
from boltons.iterutils import chunked
from cycler import cycler
from lmfit.models import GaussianModel, LinearModel
import numpy as np
from .run_engine import Msg
from .utils import Struct, snake_cyclers


class ScanBase(Struct):
    """
    This is a base class for writing reusable scans.

    It provides a default entry in the logbook at the start of the scan and a
    __iter__ method.

    To create a new sub-class you need to over-ride two things:

    - a ``_gen`` method which yields the instructions of the scan.
    - optionally, a class level ``_fields`` attribute which is used to
      construct the init signature via meta-class magic.


    If you do not use the class-level ``_fields`` and write a custom
    ``__init__`` (which you need to do if you want to have optional kwargs)
    you should provide an instance level ``_fields`` so that the logbook
    related messages will work.
    """
    def __iter__(self):
        yield Msg('open_run', **getattr(self, 'md', {}))
        yield from self._pre_scan()
        yield from self._gen()
        yield from self._post_scan()
        yield Msg('close_run')

    def _pre_scan(self):
        yield Msg('logbook', None, self.logmsg(), **self.logdict())

    def _post_scan(self):
        yield from []

    def _call_str(self):
        args = []
        for k in self._fields:
            args.append("{k}={{{k}!r}}".format(k=k))

        return ["RE({{scn_cls}}({args}))".format(args=', '.join(args)), ]

    def _logmsg(self):

        call_str = self._call_str()

        msgs = ['Scan Class: {scn_cls}', '']
        for k in self._fields:
            msgs.append('{k}: {{{k}!r}}'.format(k=k))
        msgs.append('')
        msgs.append('To call:')
        msgs.extend(call_str)
        return msgs

    def logmsg(self):
        msgs = self._logmsg()
        return '\n'.join(msgs)

    def logdict(self):
        out_dict = {k: getattr(self, k) for k in self._fields}
        out_dict['scn_cls'] = self.__class__.__name__
        return out_dict

    def _gen(self):
        raise NotImplementedError("ScanBase is a base class, you must "
                                  "sub-class it and override this method "
                                  "(_gen)")


class Count(ScanBase):
    """
    Take one or more readings from the detectors. Do not move anything.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    num : integer, optional
        number of readings to take; default is 1
    delay : float
        time delay between successive readings; default is 0

    Examples
    --------
    Count three detectors.

    >>> c = Count([det1, det2, det3])
    >>> RE(c)

    Count them five times with a one-second delay between readings.

    >>> c = Count([det1, det2, det3], 5, 1)
    >>> RE(c)
    """
    # We define _fields not for Struct, but for ScanBase.log* methods.
    _fields = ['detectors', 'num', 'delay']

    def __init__(self, detectors, num=1, delay=0):
        self.detectors = detectors
        self.num = num
        self.delay = delay

    def _gen(self):
        dets = self.detectors
        delay = self.delay
        for d in dets:
            yield Msg('configure', d)
        for i in range(self.num):
            yield Msg('checkpoint')
            yield Msg('create')
            for det in dets:
                yield Msg('trigger', det, block_group='A')
            for det in dets:
                yield Msg('wait', None, 'A')
            for det in dets:
                yield Msg('read', det)
            yield Msg('save')
            yield Msg('sleep', None, delay)
        for d in dets:
            yield Msg('deconfigure', d)


class Scan1D(ScanBase):
    _fields = ['detectors', 'motor', 'steps']

    def _gen(self):
        dets = self.detectors
        for d in dets:
            yield Msg('configure', d)
        for step in self._steps:
            yield Msg('checkpoint')
            yield Msg('set', self.motor, step, block_group='A')
            yield Msg('wait', None, 'A')
            yield Msg('create')
            yield Msg('read', self.motor)
            for det in dets:
                yield Msg('trigger', det, block_group='B')
            for det in dets:
                yield Msg('wait', None, 'B')
            for det in dets:
                yield Msg('read', det)
            yield Msg('save')
        for d in dets:
            yield Msg('deconfigure', d)


class AbsListScan(Scan1D):
    """
    Absolute scan over one variable in user-specified steps

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    steps : list
        list of positions
    """
    def _gen(self):
        self._steps = self.steps
        yield from super()._gen()


class DeltaListScan(Scan1D):
    """
    Delta (relative) scan over one variable in user-specified steps

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    steps : list
        list of positions relative to current position
    """
    def _pre_scan(self):
        ret = yield Msg('read', self.motor)
        if len(ret.keys()) > 1:
            raise NotImplementedError("Can't DScan this motor")
        key, = ret.keys()
        self._init_pos = ret[key]['value']
        yield from super()._pre_scan()

    def logdict(self):
        logdict = super().logdict()
        try:
            init_pos = self._init_pos
        except AttributeError:
            raise RuntimeError("Trying to create an olog entry for a DScan "
                               "without running the _pre_scan code to get "
                               "the baseline position.")
        logdict['init_pos'] = init_pos
        return logdict

    def _call_str(self):

        call_str = ["{motor!r}.set({init_pos})", ]
        call_str.extend(super()._call_str())
        return call_str

    def _gen(self):
        self._steps = self.steps + self._init_pos
        yield from super()._gen()

    def _post_scan(self):
        yield from super()._post_scan()
        try:
            init_pos = self._init_pos
            delattr(self, '_init_pos')
        except AttributeError:
            raise RuntimeError("Trying to run _post_scan code for a DScan "
                               "without running the _pre_scan code to get "
                               "the baseline position.")
        # Return the motor to its original position.
        yield Msg('set', self.motor, init_pos, block_group='A')
        yield Msg('wait', None, 'A')


class AbsScan(Scan1D):
    """
    Absolute scan over one variable in equally spaced steps

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    start : float
        starting position of motor
    stop : float
        ending position of motor
    num : int
        number of steps

    Examples
    --------
    Scan motor1 from 0 to 1 in ten steps.

    >>> my_scan = AbsScan(motor1, [det1, det2], 0, 1, 10)
    >>> RE(my_scan)
    # Adjust a Parameter and run again.
    >>> my_scan.num = 100
    >>> RE(my_scan)
    """
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']

    def _gen(self):
        self._steps = np.linspace(self.start, self.stop, self.num)
        yield from super()._gen()


class LogAbsScan(Scan1D):
    """
    Absolute scan over one variable in log-spaced steps

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    start : float
        starting position of motor
    stop : float
        ending position of motor
    num : int
        number of steps

    Examples
    --------
    Scan motor1 from 0 to 10 in ten log-spaced steps.

    >>> my_scan = LogAbsScan(motor1, [det1, det2], 0, 1, 10)
    >>> RE(my_scan)
    # Adjust a Parameter and run again.
    >>> my_scan.num = 100
    >>> RE(my_scan)
    """
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']

    def _gen(self):
        self._steps = np.logspace(self.start, self.stop, self.num)
        yield from super()._gen()


class DeltaScan(DeltaListScan):
    """
    Delta (relative) scan over one variable in equally spaced steps

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    start : float
        starting position of motor
    stop : float
        ending position of motor
    num : int
        number of steps

    Examples
    --------
    Scan motor1 from 0 to 1 in ten steps.

    >>> my_scan = DeltaScan(motor1, [det1, det2], 0, 1, 10)
    >>> RE(my_scan)
    # Adjust a Parameter and run again.
    >>> my_scan.num = 100
    >>> RE(my_scan)
    """
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']

    def _gen(self):
        self.steps = np.linspace(self.start, self.stop, self.num)
        yield from super()._gen()


class LogDeltaScan(DeltaListScan):
    """
    Delta (relative) scan over one variable in log-spaced steps

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    start : float
        starting position of motor
    stop : float
        ending position of motor
    num : int
        number of steps

    Examples
    --------
    Scan motor1 from 0 to 10 in ten log-spaced steps.

    >>> my_scan = LogDeltaScan(motor1, [det1, det2], 0, 1, 10)
    >>> RE(my_scan)
    # Adjust a Parameter and run again.
    >>> my_scan.num = 100
    >>> RE(my_scan)
    """
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']

    def _gen(self):
        self.steps = np.logspace(self.start, self.stop, self.num)
        yield from super()._gen()


class _AdaptiveScanBase(Scan1D):
    _fields = ['detectors', 'target_field', 'motor', 'start', 'stop',
               'min_step', 'max_step', 'target_delta', 'backstep']
    THRESHOLD = 0.8  # threshold for going backward and rescanning a region.

    def _gen(self):
        start = self.start + self._offset
        stop = self.stop + self._offset
        next_pos = start
        step = (self.max_step - self.min_step) / 2

        past_I = None
        cur_I = None
        cur_det = {}
        motor = self.motor
        dets = self.detectors
        target_field = self.target_field
        for d in dets:
            yield Msg('configure', d)
        while next_pos < stop:
            yield Msg('checkpoint')
            yield Msg('set', motor, next_pos)
            yield Msg('wait', None, 'A')
            yield Msg('create')
            yield Msg('read', motor)
            for det in dets:
                yield Msg('trigger', det, block_group='B')
            for det in dets:
                yield Msg('wait', None, 'B')
            for det in dets:
                cur_det = yield Msg('read', det)
                if target_field in cur_det:
                    cur_I = cur_det[target_field]['value']
            yield Msg('save')

            # special case first first loop
            if past_I is None:
                past_I = cur_I
                next_pos += step
                continue

            dI = np.abs(cur_I - past_I)

            slope = dI / step
            if slope:
                new_step = np.clip(self.target_delta / slope, self.min_step,
                                   self.max_step)
            else:
                new_step = np.min([step * 1.1, self.max_step])

            # if we over stepped, go back and try again
            if self.backstep and (new_step < step * self.THRESHOLD):
                next_pos -= step
                step = new_step
            else:
                past_I = cur_I
                step = 0.2 * new_step + 0.8 * step
            next_pos += step
        for d in dets:
            yield Msg('deconfigure', d)


class AdaptiveAbsScan(_AdaptiveScanBase):
    """
    Absolute scan over one variable with adaptively tuned step size

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    target_field : string
        data field whose output is the focus of the adaptive tuning
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    start : float
        starting position of motor
    stop : float
        ending position of motor
    min_step : float
        smallest step for fast-changing regions
    max_step : float
        largest step for slow-chaning regions
    target_delta : float
        desired fractional change in detector signal between steps
    backstep : bool
        whether backward steps are allowed -- this is concern with some motors
    """
    def _gen(self):
        self._offset = 0
        yield from super()._gen()


class AdaptiveDeltaScan(_AdaptiveScanBase):
    """
    Delta (relative) scan over one variable with adaptively tuned step size

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    target_field : string
        data field whose output is the focus of the adaptive tuning
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    start : float
        starting position of motor
    stop : float
        ending position of motor
    min_step : float
        smallest step for fast-changing regions
    max_step : float
        largest step for slow-chaning regions
    target_delta : float
        desired fractional change in detector signal between steps
    backstep : bool
        whether backward steps are allowed -- this is concern with some motors
    """
    def _gen(self):
        ret = yield Msg('read', self.motor)
        if len(ret.keys()) > 1:
            raise NotImplementedError("Can't DScan this motor")
        key, = ret.keys()
        current_value = ret[key]['value']
        self._offset = current_value
        yield from super()._gen()
        # Return the motor to its original position.
        yield Msg('set', self.motor, current_value, block_group='A')
        yield Msg('wait', None, 'A')


class Center(ScanBase):
    RANGE = 2  # in sigma, first sample this range around the guess
    RANGE_LIMIT = 6  # in sigma, never sample more than this far from the guess
    NUM_SAMPLES = 10
    # We define _fields not for Struct, but for ScanBase.log* methods.
    _fields = ['detectors', 'target_field', 'motor', 'initial_center',
               'initial_width', 'tolerance', 'output_mutable']

    def __init__(self, detectors, target_field, motor, initial_center,
                 initial_width, tolerance=0.1, output_mutable=None):
        """
        Attempts to find the center of a peak by moving a motor.

        This will leave the motor at what it thinks is the center.

        The motion is clipped to initial center +/- 6 initial width

        Works by :

        - sampling 10 points around the initial center
        - fitting to Gaussian + line
        - moving to the center of the Gaussian
        - while |old center - new center| > tolerance
        - taking a measurement
        - re-run fit
        - move to new center

        Parameters
        ----------
        detetectors : Reader
        target_field : string
            data field whose output is the focus of the adaptive tuning
        motor : Mover
        initial_center : number
            Initial guess at where the center is
        initial_width : number
            Initial guess at the width
        tolerance : number, optional
            Tolerance to declare good enough on finding the center. Default 0.01.
        output_mutable : dict-like, optional
            Must have 'update' method.  Mutable object to provide a side-band to
            return fitting parameters + data points
        """
        self.detectors = detectors
        self.target_field = target_field
        self.motor = motor
        self.initial_center = initial_center
        self.initial_width = initial_width
        self.output_mutable = output_mutable
        self.tolerance = tolerance

    @property
    def min_cen(self):
        return self.initial_center - self.RANGE_LIMIT * self.initial_width

    @property
    def max_cen(self):
        return self.initial_center + self.RANGE_LIMIT * self.initial_width

    def _gen(self):
        # For thread safety (paranoia) make copies of stuff
        dets = self.detectors
        target_field = self.target_field
        motor = self.motor
        initial_center = self.initial_center
        initial_width = self.initial_width
        tol = self.tolerance
        min_cen = self.min_cen
        max_cen = self.max_cen
        seen_x = deque()
        seen_y = deque()
        for x in np.linspace(initial_center - self.RANGE * initial_width,
                             initial_center + self.RANGE * initial_width,
                             self.NUM_SAMPLES, endpoint=True):
            yield Msg('set', motor, x)
            yield Msg('create')
            ret_mot = yield Msg('read', motor)
            key, = ret_mot.keys()
            seen_x.append(ret_mot[key]['value'])
            for det in dets:
                yield Msg('trigger', det, block_group='B')
            for det in dets:
                yield Msg('wait', None, 'B')
            for det in dets:
                ret_det = yield Msg('read', det)
                if target_field in ret_det:
                    seen_y.append(ret_det[target_field]['value'])
            yield Msg('save')

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
            yield Msg('create')
            ret_mot = yield Msg('read', motor)
            key, = ret_mot.keys()
            seen_x.append(ret_mot[key]['value'])
            for det in dets:
                yield Msg('trigger', det, block_group='B')
            for det in dets:
                yield Msg('wait', None, 'B')
            for det in dets:
                ret_det = yield Msg('read', det)
                if target_field in ret_det:
                    seen_y.append(ret_det[target_field]['value'])
            yield Msg('save')

        yield Msg('set', motor, np.clip(guesses['center'], min_cen, max_cen))

        if self.output_mutable is not None:
            self.output_mutable.update(guesses)
            self.output_mutable['x'] = np.array(seen_x)
            self.output_mutable['y'] = np.array(seen_y)
            self.output_mutable['model'] = res


class ScanND(ScanBase):
    _fields = ['detectors', 'cycler']

    def _gen(self):
        self.motors = self.cycler.keys
        self._last_set_point = {m: None for m in self.motors}
        dets = self.detectors
        for d in dets:
            yield Msg('configure', d)
        for step in list(self.cycler):
            yield Msg('checkpoint')
            for motor, pos in step.items():
                if pos == self._last_set_point[motor]:
                    # This step does not move this motor.
                    continue
                yield Msg('set', motor, pos, block_group='A')
                yield Msg('wait', None, 'A')
                self._last_set_point[motor] = pos
                yield Msg('create')
            for motor in self.motors:
                yield Msg('read', motor)
            for det in dets:
                yield Msg('trigger', det, block_group='B')
            for det in dets:
                yield Msg('wait', None, 'B')
            for det in dets:
                yield Msg('read', det)
            yield Msg('save')
        for d in dets:
            yield Msg('deconfigure', d)


class _OuterProductScanBase(ScanND):
    # We define _fields not for Struct, but for ScanBase.log* methods.
    _fields = ['detectors', 'args']

    def __init__(self, detectors, *args):
        args = list(args)
        # The first (slowest) axis is never "snaked." Insert False to
        # make it easy to iterate over the chunks or args..
        args.insert(4, False)
        if len(args) % 5 != 0:
            raise ValueError("wrong number of positional arguments")
        self.detectors = detectors
        self._args = args
        self.motors = []
        for motor, start, stop, num, snake in chunked(self.args, 5):
            self.motors.append(motor)

    def _pre_scan(self):
        # Build a Cycler for ScanND.
        cyclers = []
        snake_booleans = []
        for motor, start, stop, num, snake in chunked(self.args, 5):
            offset = self._offsets[motor]
            steps = offset + np.linspace(start, stop, num=num, endpoint=True)
            c = cycler(motor, steps)
            cyclers.append(c)
            snake_booleans.append(snake)
        self.cycler = snake_cyclers(cyclers, snake_booleans)
        yield from super()._pre_scan()

    @property
    def args(self):
        # Do this so that args is not settable. Too complex to allow updates.
        return self._args


class _InnerProductScanBase(ScanND):
    # We define _fields not for Struct, but for ScanBase.log* methods.
    _fields = ['detectors', 'num', 'args']

    def __init__(self, detectors, num, *args):
        if len(args) % 3 != 0:
            raise ValueError("wrong number of positional arguments")
        self.detectors = detectors
        self.num = num
        self._args = args
        self.motors = []
        for motor, start, stop, in chunked(self.args, 3):
            self.motors.append(motor)

    @property
    def args(self):
        # Do this so that args is not settable. Too complex to allow updates.
        return self._args

    def _pre_scan(self):
        # Build a Cycler for ScanND.
        num = self.num
        self.cycler = None
        for motor, start, stop, in chunked(self.args, 3):
            offset = self._offsets[motor]
            steps = offset + np.linspace(start, stop, num=num, endpoint=True)
            c = cycler(motor, steps)
            # Special case first pass because their is no
            # mutliplicative identity for cyclers.
            if self.cycler is None:
                self.cycler = c
            else:
                self.cycler += c
        yield from super()._pre_scan()


class InnerProductAbsScan(_InnerProductScanBase):
    """
    Absolute scan over one multi-motor trajectory

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    num : integer
        number of steps
    motor1, start1, stop1, ..., motorN, startN, stopN : list
        motors can be any 'setable' object (motor, temp controller, etc.)
    """
    def _pre_scan(self):
        self._offsets = defaultdict(lambda: 0)
        yield from super()._pre_scan()


class InnerProductDeltaScan(_InnerProductScanBase):
    """
    Delta (relative) scan over one multi-motor trajectory

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    num : integer
        number of steps
    motor1, start1, stop1, ..., motorN, startN, stopN : list
        motors can be any 'setable' object (motor, temp controller, etc.)
    """
    def _pre_scan(self):
        self._offsets = {}
        for motor, start, stop, in chunked(self.args, 3):
            ret = yield Msg('read', motor)
            if len(ret.keys()) > 1:
                raise NotImplementedError("Can't DScan this motor")
            key, = ret.keys()
            current_value = ret[key]['value']
            self._offsets[motor] = current_value
        yield from super()._pre_scan()

    def _post_scan(self):
        # Return the motor to its original position.
        yield from super()._post_scan()
        for motor, start, stop, in chunked(self.args, 3):
            yield Msg('set', motor, self._offsets[motor], block_group='A')
        yield Msg('wait', None, 'A')


class OuterProductAbsScan(_OuterProductScanBase):
    """
    Absolute scan over a mesh; each motor is on an independent trajectory

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    motor1, start1, stop1, num1, \
    motor2, start2, stop2, num2, snake2, \
    ..., motorN, startN, stopN, numN, snakeN : list
        motors can be any 'setable' object (motor, temp controller, etc.)
        Notice that the first motor is followed by start, stop, num.
        All other motors are followed by start, stop, num, snake where snake
        is a boolean indicating whether to following snake-like, winding
        trajectory or a simple left-to-right trajectory.
    """
    def _pre_scan(self):
        self._offsets = defaultdict(lambda: 0)
        yield from super()._pre_scan()


class OuterProductDeltaScan(_OuterProductScanBase):
    """
    Delta scan over a mesh; each motor is on an independent trajectory

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    motor1, start1, stop1, num1, \
    motor2, start2, stop2, num2, snake2, \
    ..., motorN, startN, stopN, numN, snakeN : list
        motors can be any 'setable' object (motor, temp controller, etc.)
        Notice that the first motor is followed by start, stop, num.
        All other motors are followed by start, stop, num, snake where snake
        is a boolean indicating whether to following snake-like, winding
        trajectory or a simple left-to-right trajectory.
    """
    def _pre_scan(self):
        self._offsets = {}
        for motor, start, stop, num, snake in chunked(self.args, 5):
            ret = yield Msg('read', motor)
            if len(ret.keys()) > 1:
                raise NotImplementedError("Can't DScan this motor")
            key, = ret.keys()
            current_value = ret[key]['value']
            self._offsets[motor] = current_value
        yield from super()._pre_scan()

    def _post_scan(self):
        # Return the motor to its original position.
        yield from super()._post_scan()
        for motor, start, stop, num, snake in chunked(self.args, 5):
            yield Msg('set', motor, self._offsets[motor], block_group='A')
        yield Msg('wait', None, 'A')


class Tweak(ScanBase):
    """
    Move and motor and read a detector with an interactive prompt.

    Parameters
    ----------
    detetector : Reader
    target_field : string
        data field whose output is the focus of the adaptive tuning
    motor : Mover
    """
    _fields = ['detector', 'target_field', 'motor', 'step']
    prompt_str = '{0}, {1:.3}, {2}, ({3}) '

    def _gen(self):
        d = self.detector
        target_field = self.target_field
        motor = self.motor
        step = self.step
        yield Msg('configure', d)
        while True:
            yield Msg('create')
            ret_mot = yield Msg('read', motor)
            key, = ret_mot.keys()
            pos = ret_mot[key]['value']
            yield Msg('trigger', d, block_group='A')
            yield Msg('wait', None, 'A')
            reading = Msg('read', d)[target_field]['value']
            yield Msg('save')
            prompt = prompt_str.format(motor._name, pos, reading, step)
            new_step = input(prompt)
            if new_step:
                try:
                    step = float(new_step)
                except ValueError:
                    break
            yield Msg('set', motor, pos + step, block_group='A')
            print('Motor moving...')
            sys.stdout.flush()
            yield Msg('wait', None, 'A')
            clear_output(wait=True)
            # stackoverflow.com/a/12586667/380231
            print('\x1b[1A\x1b[2K\x1b[1A')
        yield Msg('deconfigure', d)
