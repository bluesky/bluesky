from .run_engine import Msg
from .utils import Struct
import numpy as np


class Scan(Struct):
    def __iter__(self):
        return self._gen()


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
