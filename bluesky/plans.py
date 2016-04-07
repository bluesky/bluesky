import sys
from collections import deque, defaultdict

import functools
import operator

from cycler import cycler
import numpy as np
from .run_engine import Msg
from .utils import (Struct, snake_cyclers, Subs, normalize_subs_input,
                    scalar_heuristic, separate_devices, apply_sub_factories,
                    update_sub_lists)
from .plan_tools import (fly_during, count, abs_scan, delta_scan,
                         abs_list_scan, delta_list_scan, log_abs_scan,
                         log_delta_scan, subscription_wrapper,
                         adaptive_abs_scan, adaptive_delta_scan, plan_nd,
                         outer_product_scan, inner_product_scan,
                         delta_outer_product_scan, delta_inner_product_scan,
                         tweak)


class Plan(Struct):
    """
    This is a base class for writing reusable plans.

    It provides a default entry in the logbook at the start of the scan and a
    __iter__ method.

    To create a new sub-class you need to over-ride two things:

    - a class level ``_fields`` attribute which is used to construct the init
      signature via meta-class magic

    If you do not use the class-level ``_fields`` and write a custom
    ``__init__`` (which you need to do if you want to have optional kwargs)
    you should provide an instance level ``_fields`` so that the metadata
    inspection will work.
    """
    subs = Subs({})
    sub_factories = Subs({})

    def __iter__(self):
        """
        Return an iterable of messages.
        """
        return self()

    def __call__(self, **kwargs):
        """
        Return an iterable of messages.

        Any keyword arguments override present settings.
        """
        subs = defaultdict(list)
        update_sub_lists(subs, self.subs)
        update_sub_lists(subs, apply_sub_factories(self.sub_factories, self))

        current_settings = {}
        for key, val in kwargs.items():
            current_settings[key] = getattr(self, key)
            setattr(self, key, val)
        try:
            plan = self._gen()
            plan = fly_during(plan, getattr(self, 'flyers', []))
            plan = subscription_wrapper(plan, subs)
            ret = yield from plan
            yield Msg('checkpoint')
        finally:
            for key, val in current_settings.items():
                setattr(self, key, val)
        return ret

    def _gen(self):
        "Subclasses override this to provide the main plan content."
        yield from []


PlanBase = Plan  # back-compat


class Count(Plan):
    """
    Take one or more readings from the detectors. Do not move anything.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    num : integer, optional
        number of readings to take; default is 1
    delay : iterable or scalar
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
    _fields = ['detectors', 'num', 'delay']

    def __init__(self, detectors, num=1, delay=0):
        self.detectors = detectors
        self.num = num
        self.delay = delay
        self.flyers = []

    def _gen(self):
        return count(self.detectors, self.num, self.delay)


class AbsListScanPlan(Plan):
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
    _fields = ['detectors', 'motor', 'steps']

    def _gen(self):
        return abs_list_scan(self.detectors, self.motor, self.steps)


class DeltaListScanPlan(Plan):
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
    _fields = ['detectors', 'motor', 'steps']

    def _gen(self):
        return delta_list_scan(self.detectors, self.motor, self.steps)



class AbsScanPlan(Plan):
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

    >>> my_plan = AbsScanPlan([det1, det2], motor, 0, 1, 10)
    >>> RE(my_plan)
    # Adjust a Parameter and run again.
    >>> my_plan.num = 100
    >>> RE(my_plan)
    """
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']

    def _gen(self):
        return abs_scan(self.detectors, self.motor, self.start, self.stop,
                        self.num)


class LogAbsScanPlan(Plan):
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

    >>> my_plan = LogAbsScanPlan([det1, det2], motor, 0, 1, 10)
    >>> RE(my_plan)
    # Adjust a Parameter and run again.
    >>> my_plan.num = 100
    >>> RE(my_plan)
    """
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']  # override super

    def _gen(self):
        return log_abs_scan(self.detectors, self.motor, self.start, self.stop,
                            self.num)


class DeltaScanPlan(Plan):
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

    >>> my_plan = DeltaScanPlan([det1, det2], motor, 0, 1, 10)
    >>> RE(my_plan)
    # Adjust a Parameter and run again.
    >>> my_plan.num = 100
    >>> RE(my_plan)
    """
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']

    def _gen(self):
        return delta_scan(self.detectors, self.motor, self.start, self.stop,
                          self.num)

class LogDeltaScanPlan(Plan):
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

    >>> my_plan = LogDeltaScanPlan([det1, det2], motor, 0, 1, 10)
    >>> RE(my_plan)
    # Adjust a Parameter and run again.
    >>> my_plan.num = 100
    >>> RE(my_plan)
    """
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']

    def _gen(self):
        return log_delta_scan(self.detectors, self.motor, self.start,
                              self.stop, self.num)


class AdaptiveAbsScanPlan(Plan):
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
    _fields = ['detectors', 'target_field', 'motor', 'start', 'stop',
               'min_step', 'max_step', 'target_delta', 'backstep',
               'threshold']

    def __init__(self, detectors, target_field, motor, start, stop,
                 min_step, max_step, target_delta, backstep,
                 threshold=0.8):
        self.detectors = detectors
        self.target_field = target_field
        self.motor = motor
        self.start = start
        self.stop = stop
        self.min_step = min_step
        self.max_step = max_step
        self.target_delta = target_delta
        self.backstep = backstep
        self.threshold = threshold
        self.flyers = []

    def _gen(self):
        return adaptive_abs_scan(self.detectors, self.target_field, self.motor,
                                 self.start, self.stop, self.min_step,
                                 self.max_step, self.target_delta,
                                 self.backstep, self.threshold)


class AdaptiveDeltaScanPlan(AdaptiveAbsScanPlan):
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
        return adaptive_delta_scan(self.detectors, self.target_field,
                                   self.motor, self.start, self.stop,
                                   self.min_step, self.max_step,
                                   self.target_delta, self.backstep,
                                   self.threshold)


class PlanND(PlanBase):
    _fields = ['detectors', 'cycler']

    def _gen(self):
        return plan_nd(self.detectors, self.cycler)


class InnerProductAbsScanPlan(Plan):
    """
    Absolute scan over one multi-motor trajectory

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    num : integer
        number of steps
    *args
        patterned like (``motor1, start1, stop1,`` ...,
                        ``motorN, startN, stopN``)
        Motors can be any 'setable' object (motor, temp controller, etc.)
    """
    def __init__(self, detectors, num, *args):
        self.detectors = detectors
        self.num = num
        self.args = args
        self.flyers = []

    def _gen(self):
        return inner_product_scan(self.detectors, self.num, *self.args)


class InnerProductDeltaScanPlan(InnerProductAbsScanPlan):
    """
    Delta (relative) scan over one multi-motor trajectory

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    num : integer
        number of steps
    *args
        patterned like (``motor1, start1, stop1,`` ...,
                        ``motorN, startN, stopN``)
        Motors can be any 'setable' object (motor, temp controller, etc.)
    """
    def _gen(self):
        return delta_inner_product_scan(self.detectors, self.num, *self.args)


class OuterProductAbsScanPlan(Plan):
    """
    Absolute scan over a mesh; each motor is on an independent trajectory

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    *args
        patterned like ``motor1, start1, stop1, num1, motor2, start2, stop2,
        num2, snake2,`` ..., ``motorN, startN, stopN, numN, snakeN``
        Motors can be any 'setable' object (motor, temp controller, etc.)
        Notice that the first motor is followed by start, stop, num.
        All other motors are followed by start, stop, num, snake where snake
        is a boolean indicating whether to following snake-like, winding
        trajectory or a simple left-to-right trajectory.
    """
    def __init__(self, detectors, *args):
        self.detectors = detectors
        self.args = args
        self.flyers = []

    def _gen(self):
        return outer_product_scan(self.detectors, *self.args)


class OuterProductDeltaScanPlan(OuterProductAbsScanPlan):
    """
    Delta scan over a mesh; each motor is on an independent trajectory

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    *args
        patterned like ``motor1, start1, stop1, num1, motor2, start2, stop2,
        num2, snake2,`` ..., ``motorN, startN, stopN, numN, snakeN``
        Motors can be any 'setable' object (motor, temp controller, etc.)
        Notice that the first motor is followed by start, stop, num.
        All other motors are followed by start, stop, num, snake where snake
        is a boolean indicating whether to following snake-like, winding
        trajectory or a simple left-to-right trajectory.
    """
    def _gen(self):
        return delta_outer_product_scan(self.detectors, *self.args)


class Tweak(Plan):
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
    
    def _gen(self):
        return tweak(self.detector, self.target_field, self.motor, self.step)
