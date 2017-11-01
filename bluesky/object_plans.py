# DO NOT USE THIS MODULE.

# This module contians a legacy API, an early approach to plans that
# was deprecated in v0.10.0. It will be removed in a future release. It should
# not be used. To build 'reusable' plans we now recommend `functools.partial`.

from . import utils
import warnings
from collections import defaultdict

# The code below adds no new logic, but it wraps the generators above in
# classes for an alternative interface that is more stateful.

from . import preprocessors as bpp
from .plans import (count, list_scan, rel_list_scan, log_scan,
                    rel_scan, adaptive_scan, rel_adaptive_scan,
                    scan_nd, inner_product_scan, relative_inner_product_scan,
                    grid_scan, scan, tweak, spiral, spiral_fermat,
                    rel_spiral_fermat, rel_spiral, rel_log_scan,
                    rel_grid_scan)


class Plan(utils.Struct):
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
    subs = utils.Subs({})
    sub_factories = utils.Subs({})

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
        warnings.warn("This plan and all object-oriented plans have been "
                      "deprecated and will be removed in a future release "
                      "of bluesky. Instead of Count or Scan use count or "
                      "scan, etc.", stacklevel=2)
        subs = defaultdict(list)
        utils.update_sub_lists(subs, self.subs)
        utils.update_sub_lists(
            subs, utils.apply_sub_factories(self.sub_factories, self))
        flyers = getattr(self, 'flyers', [])

        def cls_plan():
            current_settings = {}
            for key, val in kwargs.items():
                current_settings[key] = getattr(self, key)
                setattr(self, key, val)
            try:
                plan = self._gen()
                plan = bpp.subs_wrapper(plan, subs)
                plan = bpp.stage_wrapper(plan, flyers)
                plan = bpp.fly_during_wrapper(plan, flyers)
                return (yield from plan)
            finally:
                for key, val in current_settings.items():
                    setattr(self, key, val)

        cls_plan.__name__ = self.__class__.__name__
        return cls_plan()

    def _gen(self):
        "Subclasses override this to provide the main plan content."
        yield from utils.censure_generator([])


PlanBase = Plan  # back-compat


class Count(Plan):
    _fields = ['detectors', 'num', 'delay']
    __doc__ = count.__doc__

    def __init__(self, detectors, num=1, delay=0, *, md=None):
        self.detectors = detectors
        self.num = num
        self.delay = delay
        self.flyers = []
        self.md = md

    def _gen(self):
        return count(self.detectors, self.num, self.delay, md=self.md)


class ListScan(Plan):
    _fields = ['detectors', 'motor', 'steps']
    __doc__ = list_scan.__doc__

    def _gen(self):
        return list_scan(self.detectors, self.motor, self.steps,
                         md=self.md)


AbsListScanPlan = ListScan  # back-compat


class RelativeListScan(Plan):
    _fields = ['detectors', 'motor', 'steps']
    __doc__ = rel_list_scan.__doc__

    def _gen(self):
        return rel_list_scan(self.detectors, self.motor, self.steps,
                             md=self.md)


DeltaListScanPlan = RelativeListScan  # back-compat


class Scan(Plan):
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']
    __doc__ = scan.__doc__

    def _gen(self):
        return scan(self.detectors, self.motor, self.start, self.stop,
                    self.num, md=self.md)


AbsScanPlan = Scan  # back-compat


class LogScan(Plan):
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']
    __doc__ = log_scan.__doc__

    def _gen(self):
        return log_scan(self.detectors, self.motor, self.start, self.stop,
                        self.num, md=self.md)


LogAbsScanPlan = LogScan  # back-compat


class RelativeScan(Plan):
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']
    __doc__ = rel_scan.__doc__

    def _gen(self):
        return rel_scan(self.detectors, self.motor, self.start, self.stop,
                        self.num, md=self.md)


DeltaScanPlan = RelativeScan  # back-compat


class RelativeLogScan(Plan):
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']
    __doc__ = rel_log_scan.__doc__

    def _gen(self):
        return rel_log_scan(self.detectors, self.motor, self.start,
                            self.stop, self.num, md=self.md)


LogDeltaScanPlan = RelativeLogScan  # back-compat


class AdaptiveScan(Plan):
    _fields = ['detectors', 'target_field', 'motor', 'start', 'stop',
               'min_step', 'max_step', 'target_delta', 'backstep',
               'threshold']
    __doc__ = adaptive_scan.__doc__

    def __init__(self, detectors, target_field, motor, start, stop,
                 min_step, max_step, target_delta, backstep,
                 threshold=0.8, *, md=None):
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
        self.md = md

    def _gen(self):
        return adaptive_scan(self.detectors, self.target_field, self.motor,
                             self.start, self.stop, self.min_step,
                             self.max_step, self.target_delta,
                             self.backstep, self.threshold, md=self.md)


AdaptiveAbsScanPlan = AdaptiveScan  # back-compat


class RelativeAdaptiveScan(AdaptiveAbsScanPlan):
    __doc__ = rel_adaptive_scan.__doc__

    def _gen(self):
        return rel_adaptive_scan(self.detectors, self.target_field,
                                 self.motor, self.start, self.stop,
                                 self.min_step, self.max_step,
                                 self.target_delta, self.backstep,
                                 self.threshold, md=self.md)


AdaptiveDeltaScanPlan = RelativeAdaptiveScan  # back-compat


class ScanND(PlanBase):
    _fields = ['detectors', 'cycler']
    __doc__ = scan_nd.__doc__

    def _gen(self):
        return scan_nd(self.detectors, self.cycler, md=self.md)


PlanND = ScanND  # back-compat


class InnerProductScan(Plan):
    __doc__ = inner_product_scan.__doc__

    def __init__(self, detectors, num, *args, md=None):
        self.detectors = detectors
        self.num = num
        self.args = args
        self.flyers = []
        self.md = md

    def _gen(self):
        return inner_product_scan(self.detectors, self.num, *self.args,
                                  md=self.md)


InnerProductAbsScanPlan = InnerProductScan  # back-compat


class RelativeInnerProductScan(InnerProductScan):
    __doc__ = relative_inner_product_scan.__doc__

    def _gen(self):
        return relative_inner_product_scan(self.detectors, self.num,
                                           *self.args, md=self.md)


InnerProductDeltaScanPlan = RelativeInnerProductScan  # back-compat


class OuterProductScan(Plan):
    __doc__ = grid_scan.__doc__

    def __init__(self, detectors, *args, md=None):
        self.detectors = detectors
        self.args = args
        self.flyers = []
        self.md = md

    def _gen(self):
        return grid_scan(self.detectors, *self.args, md=self.md)


OuterProductAbsScanPlan = OuterProductScan  # back-compat


class RelativeOuterProductScan(OuterProductScan):
    __doc__ = rel_grid_scan.__doc__

    def _gen(self):
        return rel_grid_scan(self.detectors, *self.args,
                             md=self.md)


OuterProductDeltaScanPlan = RelativeOuterProductScan  # back-compat


class Tweak(Plan):
    _fields = ['detector', 'target_field', 'motor', 'step']
    __doc__ = tweak.__doc__

    def _gen(self):
        return tweak(self.detector, self.target_field, self.motor, self.step,
                     md=self.md)


class SpiralScan(Plan):
    _fields = ['detectors', 'x_motor', 'y_motor', 'x_start', 'y_start',
               'x_range', 'y_range', 'dr', 'nth', 'tilt']
    __doc__ = spiral.__doc__

    def _gen(self):
        return spiral(self.detectors, self.x_motor, self.y_motor, self.x_start,
                      self.y_start, self.x_range, self.y_range, self.dr,
                      self.nth, tilt=self.tilt, md=self.md)


class SpiralFermatScan(Plan):
    _fields = ['detectors', 'x_motor', 'y_motor', 'x_start', 'y_start',
               'x_range', 'y_range', 'dr', 'factor', 'tilt']
    __doc__ = spiral_fermat.__doc__

    def _gen(self):
        return spiral_fermat(self.detectors, self.x_motor, self.y_motor,
                             self.x_start, self.y_start, self.x_range,
                             self.y_range, self.dr, self.factor,
                             tilt=self.tilt, md=self.md)


class RelativeSpiralScan(Plan):
    _fields = ['detectors', 'x_motor', 'y_motor', 'x_range', 'y_range', 'dr',
               'nth', 'tilt']
    __doc__ = rel_spiral.__doc__

    def _gen(self):
        return rel_spiral(self.detectors, self.x_motor, self.y_motor,
                          self.x_range, self.y_range, self.dr, self.nth,
                          tilt=self.tilt, md=self.md)


class RelativeSpiralFermatScan(Plan):
    _fields = ['detectors', 'x_motor', 'y_motor', 'x_range', 'y_range', 'dr',
               'factor', 'tilt']
    __doc__ = rel_spiral_fermat.__doc__

    def _gen(self):
        return rel_spiral_fermat(self.detectors, self.x_motor,
                                 self.y_motor, self.x_range, self.y_range,
                                 self.dr, self.factor, tilt=self.tilt,
                                 md=self.md)
