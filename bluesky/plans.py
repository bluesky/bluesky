from collections import deque, defaultdict, Iterable

import functools
import operator
from boltons.iterutils import chunked
from cycler import cycler
import numpy as np
from .run_engine import Msg
from .utils import (Struct, snake_cyclers, Subs, normalize_subs_input,
                    scalar_heuristic, separate_devices)
from .plan_tools import run_wrapper, subscription_wrapper, fly_during


class PlanBase(Struct):
    """
    This is a base class for writing reusable plans.

    It provides a default entry in the logbook at the start of the scan and a
    __iter__ method.

    To create a new sub-class you need to over-ride two things:

    - a ``_gen`` method which yields the instructions of the scan.
    - an ``_objects`` attribute, a list of arbitrary category names (such
      'detectors' or 'motors') to lists of objects -- used for metadata and for
      staging/unstagin all objects touched by a plan
    - a class level ``_fields`` attribute which is used to construct the init
      signature via meta-class magic

    If you do not use the class-level ``_fields`` and write a custom
    ``__init__`` (which you need to do if you want to have optional kwargs)
    you should provide an instance level ``_fields`` so that the metadata
    inspection will work.
    """
    subs = Subs({})

    @property
    def md(self):
        """
        a metadata dictionary, passed to the 'open_run' Message
        """
        self._md['plan_type'] = type(self).__name__
        self._md['plan_args'] = {field: repr(getattr(self, field))
                                 for field in self._fields}
        for category, objs in self._objects.items():
            # e.g., self.md['motors'] = ['motor1', 'motor2']
            if isinstance(objs, Iterable):
                self._md[category] = [obj.name for obj in objs]
            else:
                obj = objs
                self._md[category] = obj.name
        return self._md

    def __iter__(self):
        """
        yields messages
        """
        yield from self._pre()
        run = run_wrapper(fly_during(self._gen(), self.flyers), md=self.md)
        yield from subscription_wrapper(run, self.subs)
        yield from self._post()
        yield Msg('checkpoint')

    def _pre(self):
        """
        Subclasses use this to inject some messages to be processed
        before the run is opened -- for example, configuration or
        taking some preliminary readings needed for metadata.
        """
        # prerun is expected to be a callable that takes the Scan object
        # itself as its argument and returns a generator of messages
        if self.pre_run is not None:
            yield from self.pre_run(self)
        staged = []  # to track duplicates
        for objs in self._objects.values():
            for obj in objs:
                root = obj.root
                if root not in staged:
                    yield Msg('stage', root)
                    staged.append(root)

    def _post(self):
        """
        Subclasses use this to inject some messages to be processed
        after the run is closed -- for example, returning motors to
        their original positions.
        """
        # postrun is expected to be a callable that takes the Scan object
        # itself as its argument and returns a generator of messages
        unstaged = []  # to track duplicates
        for objs in self._objects.values():
            for obj in reversed(list(objs)):
                root = obj.root
                if root not in unstaged:
                    yield Msg('unstage', root)
                    unstaged.append(root)
        if self.post_run is not None:
            yield from self.post_run(self)

    def _gen(self):
        "Subclasses should override this method to add the main plan content."
        yield from []


class Count(PlanBase):
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
    _fields = ['detectors', 'num', 'delay']

    def __init__(self, detectors, num=1, delay=0,
                 pre_run=None, post_run=None):
        self.detectors = detectors
        self.num = num
        self.delay = delay
        self.pre_run = pre_run
        self.post_run = post_run
        self._md = {}
        self.configuration = {}
        self.flyers = []

    @property
    def _objects(self):
        return {'detectors': self.detectors}

    def _gen(self):
        if self.num is None:
            # Count(num=None) runs forever, awaiting user interruption.
            while True:
                yield from self._one_count()
        else:
            for _ in range(self.num):
                yield from self._one_count()

    def _one_count(self):
        # This is broken out in a separate method so we can loop
        # over it in different ways in _gen, above.
        dets = separate_devices(self.detectors)
        yield Msg('checkpoint')
        yield Msg('create', None, name='primary')
        for det in dets:
            yield Msg('trigger', det, block_group='A')
        yield Msg('wait', None, 'A')
        for det in dets:
            yield Msg('read', det)
        yield Msg('save')
        yield Msg('sleep', None, self.delay)


class Plan1D(PlanBase):
    "Use AbsListScanPlan or DeltaListScanPlan. Subclasses must define _abs_steps."
    _fields = ['detectors', 'motor', 'steps']

    @property
    def _objects(self):
        return {'detectors': self.detectors, 'motors': [self.motor]}

    def _gen(self):
        dets = separate_devices(self.detectors)
        for step in self._abs_steps:
            yield Msg('checkpoint')
            yield Msg('set', self.motor, step, block_group='A')
            yield Msg('wait', None, 'A')
            yield Msg('create', None, name='primary')
            for det in dets:
                yield Msg('trigger', det, block_group='B')
            yield Msg('wait', None, 'B')
            for det in separate_devices(dets + [self.motor]):
                yield Msg('read', det)
            yield Msg('save')


class AbsListScanPlan(Plan1D):
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
    @property
    def _abs_steps(self):
        return self.steps


class DeltaListScanPlan(Plan1D):
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

    @property
    def init_pos(self):
        "None unless a scan is running"
        return getattr(self, '_init_pos', None)

    def _pre(self):
        "Get current position for the motor."
        self._init_pos = scalar_heuristic(self.motor)
        self._md['init_pos'] = self.init_pos
        self._abs_steps = np.asarray(self.steps) + self._init_pos
        yield from super()._pre()

    def logdict(self):
        logdict = super().logdict()
        try:
            init_pos = self.init_pos
        except AttributeError:
            raise RuntimeError("Trying to create an olog entry for a DScan "
                               "without running the _pre code to get "
                               "the baseline position.")
        logdict['init_pos'] = init_pos
        return logdict

    def _call_str(self):

        call_str = ["{motor!r}.set({init_pos})", ]
        call_str.extend(super()._call_str())
        return call_str

    def _post(self):
        yield from super()._post()
        try:
            init_pos = self.init_pos
            delattr(self, '_init_pos')
            del self._md['init_pos']
        except AttributeError:
            raise RuntimeError("Trying to run _post code for a DScan "
                               "without running the _pre code to get "
                               "the baseline position.")
        # Return the motor to its original position.
        yield Msg('set', self.motor, init_pos, block_group='A')
        yield Msg('wait', None, 'A')


class AbsScanPlan(AbsListScanPlan):
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

    @property
    def steps(self):
        return np.linspace(self.start, self.stop, self.num)


class LogAbsScanPlan(AbsListScanPlan):
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

    @property
    def steps(self):
        return np.logspace(self.start, self.stop, self.num)


class DeltaScanPlan(DeltaListScanPlan):
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
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']  # override super

    @property
    def steps(self):
        return np.linspace(self.start, self.stop, self.num)


class LogDeltaScanPlan(DeltaListScanPlan):
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
    _fields = ['detectors', 'motor', 'start', 'stop', 'num']  # override super

    @property
    def steps(self):
        return np.logspace(self.start, self.stop, self.num)


class _AdaptivePlanBase(PlanBase):
    _fields = ['detectors', 'target_field', 'motor', 'start', 'stop',
               'min_step', 'max_step', 'target_delta', 'backstep']
    THRESHOLD = 0.8  # threshold for going backward and rescanning a region.

    @property
    def _objects(self):
        return {'detectors': self.detectors, 'motors': [self.motor]}

    def _gen(self):
        start = self.start + self._init_pos
        stop = self.stop + self._init_pos
        next_pos = start
        step = (self.max_step - self.min_step) / 2

        past_I = None
        cur_I = None
        cur_det = {}
        motor = self.motor
        dets = separate_devices(self.detectors)
        target_field = self.target_field
        while next_pos < stop:
            yield Msg('checkpoint')
            yield Msg('set', motor, next_pos)
            yield Msg('wait', None, 'A')
            yield Msg('create', None, name='primary')
            for det in dets:
                yield Msg('trigger', det, block_group='B')
            yield Msg('wait', None, 'B')
            for det in separate_devices(dets + [self.motor]):
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


class AdaptiveAbsScanPlan(_AdaptivePlanBase):
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
    @property
    def _init_pos(self):
        # facilitate code-sharing with AdaptiveDeltaScanPlan
        return 0


class AdaptiveDeltaScanPlan(_AdaptivePlanBase):
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
    @property
    def init_pos(self):
        "None unless a scan is running"
        return getattr(self, '_init_pos', None)

    def _pre(self):
        self._init_pos = scalar_heuristic(self.motor)
        self._md['init_pos'] = self.init_pos
        yield from super()._pre()

    def _post(self):
        yield from super()._post()
        try:
            init_pos = self.init_pos
            delattr(self, '_init_pos')
            del self._md['init_pos']
        except AttributeError:
            raise RuntimeError("Trying to run _post code for a DScan "
                               "without running the _pre code to get "
                               "the baseline position.")
        # Return the motor to its original position.
        yield Msg('set', self.motor, init_pos, block_group='A')
        yield Msg('wait', None, 'A')


class Center(PlanBase):
    RANGE = 2  # in sigma, first sample this range around the guess
    RANGE_LIMIT = 6  # in sigma, never sample more than this far from the guess
    NUM_SAMPLES = 10
    NUM_SAMPLES = 10
    # We define _fields not for Struct, but for metadata in PlanBase.md.
    _fields = ['detectors', 'target_field', 'motor', 'initial_center',
               'initial_width', 'tolerance', 'output_mutable']

    def __init__(self, detectors, target_field, motor, initial_center,
                 initial_width, tolerance=0.1, output_mutable=None,
                 pre_run=None, post_run=None):
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
        try:
            from lmfit.models import GaussianModel, LinearModel
        except ImportError:
            raise ImportError("This scan requires the package lmfit.")
        self.detectors = separate_devices(detectors)
        self.target_field = target_field
        self.motor = motor
        self.initial_center = initial_center
        self.initial_width = initial_width
        self.output_mutable = output_mutable
        self.tolerance = tolerance
        self.pre_run = pre_run
        self.post_run = post_run
        self._md = {}
        self.configuration = {}
        self.flyers = []

    @property
    def _objects(self):
        return {'detectors': self.detectors, 'motors': [self.motor]}

    @property
    def min_cen(self):
        return self.initial_center - self.RANGE_LIMIT * self.initial_width

    @property
    def max_cen(self):
        return self.initial_center + self.RANGE_LIMIT * self.initial_width

    def _gen(self):
        # We checked in the __init__ that this import works.
        from lmfit.models import GaussianModel, LinearModel
        # For thread safety (paranoia) make copies of stuff
        dets = separate_devices(self.detectors)
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
            ret_mot = yield Msg('read', motor)
            yield Msg('create', None, name='primary')
            key, = ret_mot.keys()
            seen_x.append(ret_mot[key]['value'])
            for det in dets:
                yield Msg('trigger', det, block_group='B')
            yield Msg('wait', None, 'B')
            for det in separate_devices(dets + [motor]):
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
            ret_mot = yield Msg('read', motor)
            yield Msg('create', None, name='primary')
            key, = ret_mot.keys()
            seen_x.append(ret_mot[key]['value'])
            for det in dets:
                yield Msg('trigger', det, block_group='B')
            yield Msg('wait', None, 'B')
            for det in separate_devices(dets + [motor]):
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


class PlanND(PlanBase):
    _fields = ['detectors', 'cycler']

    @property
    def motors(self):
        return self.cycler.keys

    @property
    def _objects(self):
        return {'detectors': self.detectors, 'motors': self.motors}

    def _gen(self):
        self._last_set_point = {m: None for m in self.motors}
        dets = separate_devices(self.detectors)
        for step in list(self.cycler):
            yield Msg('checkpoint')
            for motor, pos in step.items():
                if pos == self._last_set_point[motor]:
                    # This step does not move this motor.
                    continue
                yield Msg('set', motor, pos, block_group='A')
                self._last_set_point[motor] = pos

            yield Msg('wait', None, 'A')
            yield Msg('create', None, name='primary')

            for det in dets:
                yield Msg('trigger', det, block_group='B')
            yield Msg('wait', None, 'B')
            for det in separate_devices(dets + list(self.motors)):
                yield Msg('read', det)
            yield Msg('save')


class _OuterProductPlanBase(PlanND):
    # We define _fields not for Struct, but for metadata in PlanBase.md.
    _fields = ['detectors', 'args']
    _derived_fields = ['motors', 'shape', 'num', 'extents', 'snaking']

    # Overriding PlanND is the only way to do this; we cannot build the cycler
    # until we measure the initial positions at the beginning of the run.
    @property
    def motors(self):
        return self._motors

    def __init__(self, detectors, *args, pre_run=None, post_run=None):
        args = list(args)
        # The first (slowest) axis is never "snaked." Insert False to
        # make it easy to iterate over the chunks or args..
        args.insert(4, False)
        if len(args) % 5 != 0:
            raise ValueError("wrong number of positional arguments")
        self.detectors = detectors
        self._motors = []
        self._args = args
        shape = []
        extent = []
        snaking = []
        for motor, start, stop, num, snake in chunked(self.args, 5):
            self._motors.append(motor)
            shape.append(num)
            extent.append([start, stop])
            snaking.append(snake)
        self.shape = tuple(shape)
        self.extents = tuple(extent)
        self.snaking = tuple(snaking)
        self.pre_run = pre_run
        self.post_run = post_run
        self._md = {'shape': self.shape, 'extents': self.extents,
                    'snaking': self.snaking, 'num': self.num}
        self.configuration = {}
        self.flyers = []

    @property
    def cycler(self):
        # Build a Cycler for PlanND.
        cyclers = []
        snake_booleans = []
        for motor, start, stop, num, snake in chunked(self.args, 5):
            init_pos = self._init_pos[motor]
            steps = init_pos + np.linspace(start, stop, num=num, endpoint=True)
            c = cycler(motor, steps)
            cyclers.append(c)
            snake_booleans.append(snake)
        return snake_cyclers(cyclers, snake_booleans)

    @property
    def num(self):
        # We could do len(cycler) but we dont know init_pos yet,
        # so the cycler cannot be computed.
        nums = [num for motor, start, stop, num, snake
                in chunked(self.args, 5)]
        total_steps = functools.reduce(operator.mul, nums)
        return total_steps

    @property
    def args(self):
        # Do this so that args is not settable. Too complex to allow updates.
        return self._args


class _InnerProductPlanBase(PlanND):
    # We define _fields not for Struct, but for metadata in PlanBase.md.
    _fields = ['detectors', 'num', 'args']

    # Overriding PlanND is the only way to do this; we cannot build the cycler
    # until we measure the initial positions at the beginning of the run.
    @property
    def motors(self):
        return self._motors

    def __init__(self, detectors, num, *args, pre_run=None, post_run=None):
        if len(args) % 3 != 0:
            raise ValueError("wrong number of positional arguments")
        self.detectors = detectors
        self.num = num
        self._args = args
        self._motors = []
        extents = []
        for motor, start, stop, in chunked(self.args, 3):
            self._motors.append(motor)
            extents.append([start, stop])
        self.extents = tuple(extents)
        self.pre_run = pre_run
        self.post_run = post_run
        self._md = {'extents': self.extents}
        self.configuration = {}
        self.flyers = []

    @property
    def args(self):
        # Do this so that args is not settable. Too complex to allow updates.
        return self._args

    @property
    def cycler(self):
        # Build a Cycler for PlanND.
        cyclers = []
        for motor, start, stop, in chunked(self.args, 3):
            init_pos = self._init_pos[motor]
            steps = init_pos + np.linspace(start, stop, num=self.num,
                                           endpoint=True)
            c = cycler(motor, steps)
            cyclers.append(c)
        return functools.reduce(operator.add, cyclers)


class InnerProductAbsScanPlan(_InnerProductPlanBase):
    """
    Absolute scan over one multi-motor trajectory

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    num : integer
        number of steps
    *args
        patterned like ``motor1, start1, stop1,`` ..., ``motorN, startN, stopN``
        Motors can be any 'setable' object (motor, temp controller, etc.)
    """

    @property
    def _init_pos(self):
        "Makes code re-use between Delta and Abs varieties easier"
        return defaultdict(lambda: 0)


class InnerProductDeltaScanPlan(_InnerProductPlanBase):
    """
    Delta (relative) scan over one multi-motor trajectory

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    num : integer
        number of steps
    *args
        patterned like ``motor1, start1, stop1,`` ..., ``motorN, startN, stopN``
        Motors can be any 'setable' object (motor, temp controller, etc.)
    """

    @property
    def init_pos(self):
        "None unless a scan is running"
        return getattr(self, '_init_pos', None)

    def _pre(self):
        "Get current position for each motor."
        self._init_pos = {}
        self._md['init_pos'] = {}
        for motor, start, stop, in chunked(self.args, 3):
            self._init_pos[motor] = scalar_heuristic(motor)
            self._md['init_pos'][motor.name] = self.init_pos[motor]
        yield from super()._pre()

    def _post(self):
        yield from super()._post()
        try:
            init_pos = self.init_pos
            delattr(self, '_init_pos')
            del self._md['init_pos']
        except AttributeError:
            raise RuntimeError("Trying to run _post code for a DScan "
                               "without running the _pre code to get "
                               "the baseline position.")
        # Return the motors to their original positions.
        for motor, start, stop, in chunked(self.args, 3):
            yield Msg('set', motor, init_pos[motor], block_group='A')
        yield Msg('wait', None, 'A')


class OuterProductAbsScanPlan(_OuterProductPlanBase):
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

    @property
    def _init_pos(self):
        "Makes code re-use between Delta and Abs varieties easier"
        return defaultdict(lambda: 0)


class OuterProductDeltaScanPlan(_OuterProductPlanBase):
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

    @property
    def init_pos(self):
        "None unless a scan is running"
        return getattr(self, '_init_pos', None)

    def _pre(self):
        "Get current position for each motor."
        self._init_pos = {}
        self._md['init_pos'] = {}
        for motor, start, stop, num, snake in chunked(self.args, 5):
            self._init_pos[motor] = scalar_heuristic(motor)
            self._md['init_pos'][motor.name] = self.init_pos[motor]
        yield from super()._pre()

    def _post(self):
        # Return the motor to its original position.
        yield from super()._post()
        try:
            init_pos = self.init_pos
            delattr(self, '_init_pos')
            del self._md['init_pos']
        except AttributeError:
            raise RuntimeError("Trying to run _post code for a DScan "
                               "without running the _pre code to get "
                               "the baseline position.")
        for motor, start, stop, num, snake in chunked(self.args, 5):
            yield Msg('set', motor, init_pos[motor], block_group='A')
        yield Msg('wait', None, 'A')


class Tweak(PlanBase):
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

    @property
    def _objects(self):
        return {'detectors': [self.detector], 'motors': [self.motor]}

    def _gen(self):
        d = self.detector
        target_field = self.target_field
        motor = self.motor
        step = self.step
        while True:
            yield Msg('create', None, name='primary')
            ret_mot = yield Msg('read', motor)
            key, = ret_mot.keys()
            pos = ret_mot[key]['value']
            yield Msg('trigger', d, block_group='A')
            yield Msg('wait', None, 'A')
            reading = Msg('read', d)[target_field]['value']
            yield Msg('save')
            prompt = prompt_str.format(motor.name, pos, reading, step)
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
