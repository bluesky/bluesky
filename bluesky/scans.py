from .run_engine import Msg
from .utils import Struct
import numpy as np


class Scan(Struct):
    def __iter__(self):
        return self._gen()

class Count(Scan):
    _fields = ['detectors']

    def _gen(self):
        dets = self.detectors
        yield Msg('checkpoint')
        yield Msg('create')
        for det in dets:
            yield Msg('trigger', det, block_group='A')
        for det in dets:
            yield Msg('wait', None, 'A')
        for det in dets:
            yield Msg('read', det)
        yield Msg('save')


class Scan1D(Scan):
    _fields = ['motor', 'detectors', 'steps']

    def _gen(self):
        dets = self.detectors
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


class Ascan(Scan1D):
    """
    Absolute scan over one variable in user-specified steps

    Parameters
    ----------
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    detectors : list
        list of 'readable' objects
    steps : list
        list of positions
    """
    def _gen(self):
        self._steps = self.steps
        yield from super()._gen()


class Dscan(Scan1D):
    """
    Delta (relative) scan over one variable in user-specified steps

    Parameters
    ----------
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    detectors : list
        list of 'readable' objects
    steps : list
        list of positions relative to current position
    """
    def _gen(self):
        ret = yield Msg('read', self.motor)
        if len(ret.keys()) > 1:
            raise NotImplementedError("Can't DScan this motor")
        key, = ret.keys()
        current_value = ret[key]['value']
        self._steps = self.steps + current_value
        yield from super()._gen()


class LinAscan(Scan1D):
    """
    Absolute scan over one variable in equally spaced steps

    Parameters
    ----------
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    detectors : list
        list of 'readable' objects
    start : float
        starting position of motor
    stop : float
        ending position of motor
    num : int
        number of steps

    Examples
    --------
    # Scan motor1 from 0 to 1 in ten steps.
    >>> my_scan = LinAscan(motor1, [det1, det2], 0, 1, 10)
    >>> RE(my_scan)
    # Adjust a Parameter and run again.
    >>> my_scan.num = 100
    >>> RE(my_scan)
    """
    _fields = ['motor', 'detectors', 'start', 'stop', 'num']

    def _gen(self):
        self._steps = np.linspace(self.start, self.stop, self.num)
        yield from super()._gen()


class LogAscan(Scan1D):
    """
    Absolute scan over one variable in log-spaced steps

    Parameters
    ----------
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    detectors : list
        list of 'readable' objects
    start : float
        starting position of motor
    stop : float
        ending position of motor
    num : int
        number of steps

    Examples
    --------
    # Scan motor1 from 0 to 10 in ten log-spaced steps.
    >>> my_scan = LogAscan(motor1, [det1, det2], 0, 1, 10)
    >>> RE(my_scan)
    # Adjust a Parameter and run again.
    >>> my_scan.num = 100
    >>> RE(my_scan)
    """
    _fields = ['motor', 'detectors', 'start', 'stop', 'num']

    def _gen(self):
        self._steps = np.logspace(self.start, self.stop, self.num)
        yield from super()._gen()


class LinDscan(Dscan):
    """
    Delta (relative) scan over one variable in equally spaced steps

    Parameters
    ----------
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    detectors : list
        list of 'readable' objects
    start : float
        starting position of motor
    stop : float
        ending position of motor
    num : int
        number of steps

    Examples
    --------
    # Scan motor1 from 0 to 1 in ten steps.
    >>> my_scan = LinDscan(motor1, [det1, det2], 0, 1, 10)
    >>> RE(my_scan)
    # Adjust a Parameter and run again.
    >>> my_scan.num = 100
    >>> RE(my_scan)
    """
    _fields = ['motor', 'detectors', 'start', 'stop', 'num']

    def _gen(self):
        self.steps = np.linspace(self.start, self.stop, self.num)
        yield from super()._gen()


class LogDscan(Dscan):
    """
    Delta (relative) scan over one variable in log-spaced steps

    Parameters
    ----------
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    detectors : list
        list of 'readable' objects
    start : float
        starting position of motor
    stop : float
        ending position of motor
    num : int
        number of steps

    Examples
    --------
    # Scan motor1 from 0 to 10 in ten log-spaced steps.
    >>> my_scan = LogDscan(motor1, [det1, det2], 0, 1, 10)
    >>> RE(my_scan)
    # Adjust a Parameter and run again.
    >>> my_scan.num = 100
    >>> RE(my_scan)
    """
    _fields = ['motor', 'detectors', 'start', 'stop', 'num']

    def _gen(self):
        self.steps = np.logspace(self.start, self.stop, self.num)
        yield from super()._gen()


class _AdaptiveScan(Scan1D):
    _fields = ['motor', 'detectors', 'target_field', 'start', 'stop',
               'min_step', 'max_step', 'target_delta']
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

            new_step = np.clip(self.target_delta / slope, self.min_step,
                               self.max_step)
            # if we over stepped, go back and try again
            if new_step < step * self.THRESHOLD:
                next_pos -= step
            else:
                past_I = cur_I
            step = new_step
            next_pos += step


class AdaptiveAscan(_AdaptiveScan):
    """
    Absolute scan over one variable with adaptively tuned step size

    Parameters
    ----------
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    detectors : list
        list of 'readable' objects
    target_field : string
        data field whose output is the focus of the adaptive tuning
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
    """
    def _gen(self):
        self._offset = 0
        yield from super()._gen()


class AdaptiveDscan(_AdaptiveScan):
    """
    Delta (relative) scan over one variable with adaptively tuned step size

    Parameters
    ----------
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    detectors : list
        list of 'readable' objects
    target_detector : obj
        detector whose output is the focus of the adaptive tuning
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
    """
    def _gen(self):
        ret = yield Msg('read', self.motor)
        if len(ret.keys()) > 1:
            raise NotImplementedError("Can't DScan this motor")
        key, = ret.keys()
        current_value = ret[key]['value']
        self._offset = current_value
        yield from super()._gen()
