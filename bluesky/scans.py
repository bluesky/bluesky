from .run_engine import Msg
from .utils import Struct
import numpy as np


class LinAscan(Struct):
    """
    Linear absolute scan over one variable in equally spaced steps

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
    >>> my_scan = LinearAScan(motor1, [det1, det2], 0, 1, 10)
    >>> RE(my_scan)
    # Adjust a Parameter and run again.
    >>> my_scan.num = 100
    >>> RE(my_scan)
    """
    _fields = ['motor', 'detectors', 'start', 'stop', 'num']

    def _gen(self):
        steps = np.linspace(self.start, self.stop, self.num)
        for step in steps:
            yield Msg('checkpoint')
            yield Msg('set', self.motor, step, block_group='A')
            yield Msg('wait', None, 'A')
            yield Msg('create')
            for det in self.detectors:
                yield Msg('trigger', det)
                yield Msg('read', det)
            yield Msg('save')

    def __iter__(self):
        return self._gen()
