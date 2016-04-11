import sys
from collections import deque, defaultdict

import functools
import operator

from cycler import cycler
import numpy as np
from .run_engine import Msg
from .utils import (Struct, snake_cyclers, Subs, normalize_subs_input,
                    separate_devices, apply_sub_factories, update_sub_lists)
from .plan_tools import (fly_during, count, scan, relative_scan,
                         list_scan, relative_list_scan, log_scan,
                         relative_log_scan, subscription_wrapper,
                         adaptive_scan, relative_adaptive_scan, plan_nd,
                         outer_product_scan, inner_product_scan,
                         relative_outer_product_scan,
                         relative_inner_product_scan, tweak)


class Plan(Struct):
    """
    This is a base class for wrapping plan generators in a stateful class.

    To create a new sub-class you need to over-ride two things:

    - an ``__init__`` method *or* a class level ``_fields`` attribute which is
      used to construct the init signature via meta-class magic
    - a ``_gen`` method, which should return a generator of Msg objects

    The class provides:

    - state stored in attributes that are used to re-generate a plan generator
      with the same parameters
    - a hook for adding "flyable" objects to a plan
    - attributes for adding subscriptions and subscription factory functions
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
    _fields = ['detectors', 'num', 'delay']
    __doc__ = count.__doc__

    def __init__(self, detectors, num=1, delay=0):
        self.detectors = detectors
        self.num = num
        self.delay = delay
        self.flyers = []

    def _gen(self):
        return count(self.detectors, self.num, self.delay)


class ListScan(Plan):
    _fields = ['detectors', 'motor', 'steps']
    __doc__ = list_scan.__doc__

    def _gen(self):
        return list_scan(self.detectors, self.motor, self.steps)

AbsListScanPlan = ListScan  # back-compat


class RelativeListScan(Plan):
    _fields = ['detectors', 'motor', 'steps']
    __doc__ = relative_list_scan.__doc__

    def _gen(self):
        return relative_list_scan(self.detectors, self.motor, self.steps)

DeltaListScanPlan = RelativeListScan  # back-compat


class Scan(Plan):
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']
    __doc__ = scan.__doc__

    def _gen(self):
        return scan(self.detectors, self.motor, self.start, self.stop,
                        self.num)

AbsScanPlan = Scan  # back-compat


class LogScan(Plan):
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']
    __doc__ = log_scan.__doc__

    def _gen(self):
        return log_scan(self.detectors, self.motor, self.start, self.stop,
                            self.num)

LogAbsScanPlan = LogScan  # back-compat


class RelativeScan(Plan):
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']
    __doc__ = relative_scan.__doc__

    def _gen(self):
        return relative_scan(self.detectors, self.motor, self.start, self.stop,
                             self.num)

DeltaScanPlan = RelativeScan  # back-compat


class RelativeLogScan(Plan):
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']
    __doc__ = relative_log_scan.__doc__

    def _gen(self):
        return relative_log_scan(self.detectors, self.motor, self.start,
                              self.stop, self.num)

LogDeltaScanPlan = RelativeLogScan  # back-compat


class AdaptiveScan(Plan):
    _fields = ['detectors', 'target_field', 'motor', 'start', 'stop',
               'min_step', 'max_step', 'target_delta', 'backstep',
               'threshold']
    __doc__ = adaptive_scan.__doc__


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
        return adaptive_scan(self.detectors, self.target_field, self.motor,
                             self.start, self.stop, self.min_step,
                             self.max_step, self.target_delta,
                             self.backstep, self.threshold)

AdaptiveAbsScanPlan = AdaptiveScan  # back-compat


class RelativeAdaptiveScan(AdaptiveAbsScanPlan):
    __doc__ = relative_adaptive_scan.__doc__

    def _gen(self):
        return relative_adaptive_scan(self.detectors, self.target_field,
                                      self.motor, self.start, self.stop,
                                      self.min_step, self.max_step,
                                      self.target_delta, self.backstep,
                                      self.threshold)

AdaptiveDeltaScanPlan = RelativeAdaptiveScan  # back-compat


class ScanND(PlanBase):
    _fields = ['detectors', 'cycler']
    __doc__ = plan_nd.__doc__

    def _gen(self):
        return plan_nd(self.detectors, self.cycler)

PlanND = ScanND  # back-compat


class InnerProductScan(Plan):
    __doc__ = inner_product_scan.__doc__

    def __init__(self, detectors, num, *args):
        self.detectors = detectors
        self.num = num
        self.args = args
        self.flyers = []

    def _gen(self):
        return inner_product_scan(self.detectors, self.num, *self.args)

InnerProductAbsScanPlan = InnerProductScan  # back-compat


class RelativeInnerProductScan(InnerProductScan):
    __doc__ = relative_inner_product_scan.__doc__
    def _gen(self):
        return relative_inner_product_scan(self.detectors, self.num,
                                           *self.args)

InnerProductDeltaScanPlan = RelativeInnerProductScan  # back-compat


class OuterProductScan(Plan):
    __doc__ = outer_product_scan.__doc__

    def __init__(self, detectors, *args):
        self.detectors = detectors
        self.args = args
        self.flyers = []

    def _gen(self):
        return outer_product_scan(self.detectors, *self.args)

OuterProductAbsScanPlan = OuterProductScan  # back-compat


class RelativeOuterProductScan(OuterProductScan):
    __doc__ = relative_outer_product_scan.__doc__

    def _gen(self):
        return relative_outer_product_scan(self.detectors, *self.args)

OuterProductDeltaScanPlan = RelativeOuterProductScan  # back-compat


class Tweak(Plan):
    _fields = ['detector', 'target_field', 'motor', 'step']
    __doc__ = tweak.__doc__
    
    def _gen(self):
        return tweak(self.detector, self.target_field, self.motor, self.step)
