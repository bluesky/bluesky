import sys
import inspect
from itertools import chain, zip_longest
from functools import partial
import collections
from collections import defaultdict
import time

import numpy as np
try:
    # cytools is a drop-in replacement for toolz, implemented in Cython
    from cytools import partition
except ImportError:
    from toolz import partition

from . import plan_patterns

from . import utils
from .utils import Msg

from . import preprocessors as bpp
from . import plan_stubs as bps


def count(detectors, num=1, delay=None, *, per_shot=None, md=None):
    """
    Take one or more readings from detectors.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    num : integer, optional
        number of readings to take; default is 1

        If None, capture data until canceled
    delay : iterable or scalar, optional
        Time delay in seconds between successive readings; default is 0.
    per_shot : callable, optional
        hook for customizing action of inner loop (messages per step)
        Expected signature ::

           def f(detectors: Iterable[OphydObj]) -> Generator[Msg]:
               ...

    md : dict, optional
        metadata

    Notes
    -----
    If ``delay`` is an iterable, it must have at least ``num - 1`` entries or
    the plan will raise a ``ValueError`` during iteration.
    """
    if num is None:
        num_intervals = None
    else:
        num_intervals = num - 1
    _md = {'detectors': [det.name for det in detectors],
           'num_points': num,
           'num_intervals': num_intervals,
           'plan_args': {'detectors': list(map(repr, detectors)), 'num': num},
           'plan_name': 'count',
           'hints': {}
           }
    _md.update(md or {})
    _md['hints'].setdefault('dimensions', [(('time',), 'primary')])

    if per_shot is None:
        per_shot = bps.one_shot

    @bpp.stage_decorator(detectors)
    @bpp.run_decorator(md=_md)
    def inner_count():
        return (yield from bps.repeat(partial(per_shot, detectors),
                                      num=num, delay=delay))

    return (yield from inner_count())


def list_scan(detectors, *args, per_step=None, md=None):
    """
    Scan over one or more variables in steps simultaneously (inner product).

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    *args :
        For one dimension, ``motor, [point1, point2, ....]``.
        In general:

        .. code-block:: python

            motor1, [point1, point2, ...],
            motor2, [point1, point2, ...],
            ...,
            motorN, [point1, point2, ...]

        Motors can be any 'settable' object (motor, temp controller, etc.)

    per_step : callable, optional
        hook for customizing action of inner loop (messages per step)
        Expected signature:
        ``f(detectors, motor, step) -> plan (a generator)``
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.rel_list_scan`
    :func:`bluesky.plans.list_grid_scan`
    :func:`bluesky.plans.rel_list_grid_scan`
    """
    if len(args) % 2 != 0:
        raise ValueError("The list of arguments must contain a list of "
                         "points for each defined motor")

    md = md or {}  # reset md if it is None.

    # set some variables and check that all lists are the same length
    lengths = {}
    motors = []
    pos_lists = []
    length = None
    for motor, pos_list in partition(2, args):
        pos_list = list(pos_list)  # Ensure list (accepts any finite iterable).
        lengths[motor.name] = len(pos_list)
        if not length:
            length = len(pos_list)
        motors.append(motor)
        pos_lists.append(pos_list)
    length_check = all(elem == list(lengths.values())[0] for elem in
                       list(lengths.values()))

    if not length_check:
        raise ValueError("The lengths of all lists in *args must be the same. "
                         "However the lengths in args are : "
                         "{}".format(lengths))

    md_args = list(chain(*((repr(motor), pos_list)
                           for motor, pos_list in partition(2, args))))
    motor_names = list(lengths.keys())

    _md = {'detectors': [det.name for det in detectors],
           'motors': motor_names,
           'num_points': length,
           'num_intervals': length - 1,
           'plan_args': {'detectors': list(map(repr, detectors)),
                         'args': md_args,
                         'per_step': repr(per_step)},
           'plan_name': 'list_scan',
           'plan_pattern': 'inner_list_product',
           'plan_pattern_module': plan_patterns.__name__,
           'plan_pattern_args': dict(args=md_args),
           'hints': {},
           }
    _md.update(md or {})

    x_fields = []
    for motor in motors:
        x_fields.extend(getattr(motor, 'hints', {}).get('fields', []))

    default_dimensions = [(x_fields, 'primary')]

    default_hints = {}
    if len(x_fields) > 0:
        default_hints.update(dimensions=default_dimensions)

    # now add default_hints and override any hints from the original md (if
    # exists)
    _md['hints'] = default_hints
    _md['hints'].update(md.get('hints', {}) or {})

    full_cycler = plan_patterns.inner_list_product(args)

    return (yield from scan_nd(detectors, full_cycler, per_step=per_step,
                               md=_md))


def rel_list_scan(detectors, *args, per_step=None, md=None):
    """
    Scan over one variable in steps relative to current position.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    *args :
        For one dimension, ``motor, [point1, point2, ....]``.
        In general:

        .. code-block:: python

            motor1, [point1, point2, ...],
            motor2, [point1, point2, ...],
            ...,
            motorN, [point1, point2, ...]

        Motors can be any 'settable' object (motor, temp controller, etc.)
        point1, point2 etc are relative to the current location.

    motor : object
        any 'settable' object (motor, temp controller, etc.)
    steps : list
        list of positions relative to current position
    per_step : callable, optional
        hook for customizing action of inner loop (messages per step)
        Expected signature: ``f(detectors, motor, step)``
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.list_scan`
    :func:`bluesky.plans.list_grid_scan`
    :func:`bluesky.plans.rel_list_grid_scan`
    """
    # TODO read initial positions (redundantly) so they can be put in md here
    _md = {'plan_name': 'rel_list_scan'}
    _md.update(md or {})

    motors = [motor for motor, pos_list in partition(2, args)]

    @bpp.reset_positions_decorator(motors)
    @bpp.relative_set_decorator(motors)
    def inner_relative_list_scan():
        return (yield from list_scan(detectors, *args, per_step=per_step,
                                     md=_md))
    return (yield from inner_relative_list_scan())


def list_grid_scan(detectors, *args, snake_axes=False, per_step=None, md=None):
    """
    Scan over a mesh; each motor is on an independent trajectory.

    Parameters
    ----------
    detectors: list
        list of 'readable' objects
    args: list
        patterned like (``motor1, position_list1,``
                        ``motor2, position_list2,``
                        ``motor3, position_list3,``
                        ``...,``
                        ``motorN, position_listN``)

        The first motor is the "slowest", the outer loop. ``position_list``'s
        are lists of positions, all lists must have the same length. Motors
        can be any 'settable' object (motor, temp controller, etc.).
    snake_axes: boolean or iterable, optional
        which axes should be snaked, either ``False`` (do not snake any axes),
        ``True`` (snake all axes) or a list of axes to snake. "Snaking" an axis
        is defined as following snake-like, winding trajectory instead of a
        simple left-to-right trajectory.The elements of the list are motors
        that are listed in `args`. The list must not contain the slowest
        (first) motor, since it can't be snaked.
    per_step: callable, optional
        hook for customizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
        for details.
    md: dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.rel_list_grid_scan`
    :func:`bluesky.plans.list_scan`
    :func:`bluesky.plans.rel_list_scan`
    """

    full_cycler = plan_patterns.outer_list_product(args, snake_axes)

    md_args = []
    motor_names = []
    motors = []
    for i, (motor, pos_list) in enumerate(partition(2, args)):
        md_args.extend([repr(motor), pos_list])
        motor_names.append(motor.name)
        motors.append(motor)
    _md = {'shape': tuple(len(pos_list)
                          for motor, pos_list in partition(2, args)),
           'extents': tuple([min(pos_list), max(pos_list)]
                            for motor, pos_list in partition(2, args)),
           'snake_axes': snake_axes,
           'plan_args': {'detectors': list(map(repr, detectors)),
                         'args': md_args,
                         'per_step': repr(per_step)},
           'plan_name': 'list_grid_scan',
           'plan_pattern': 'outer_list_product',
           'plan_pattern_args': dict(args=md_args, snake_axes=snake_axes),
           'plan_pattern_module': plan_patterns.__name__,
           'motors': tuple(motor_names),
           'hints': {},
           }
    _md.update(md or {})
    try:
        _md['hints'].setdefault('dimensions', [(m.hints['fields'], 'primary')
                                               for m in motors])
    except (AttributeError, KeyError):
        ...

    return (yield from scan_nd(detectors, full_cycler,
                               per_step=per_step, md=_md))


def rel_list_grid_scan(detectors, *args, snake_axes=False, per_step=None,
                       md=None):
    """
    Scan over a mesh; each motor is on an independent trajectory. Each point is
    relative to the current position.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects

    args
        patterned like (``motor1, position_list1,``
                        ``motor2, position_list2,``
                        ``motor3, position_list3,``
                        ``...,``
                        ``motorN, position_listN``)

        The first motor is the "slowest", the outer loop. ``position_list``'s
        are lists of positions, all lists must have the same length. Motors
        can be any 'settable' object (motor, temp controller, etc.).

    snake_axes : boolean or Iterable, optional
        which axes should be snaked, either ``False`` (do not snake any axes),
        ``True`` (snake all axes) or a list of axes to snake. "Snaking" an axis
        is defined as following snake-like, winding trajectory instead of a
        simple left-to-right trajectory.The elements of the list are motors
        that are listed in `args`. The list must not contain the slowest
        (first) motor, since it can't be snaked.

    per_step : callable, optional
        hook for customizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
        for details.
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.list_grid_scan`
    :func:`bluesky.plans.list_scan`
    :func:`bluesky.plans.rel_list_scan`
    """
    _md = {'plan_name': 'rel_list_grid_scan'}
    _md.update(md or {})

    motors = [motor for motor, pos_list in partition(2, args)]

    @bpp.reset_positions_decorator(motors)
    @bpp.relative_set_decorator(motors)
    def inner_relative_list_grid_scan():
        return (yield from list_grid_scan(detectors, *args,
                                          snake_axes=snake_axes,
                                          per_step=per_step, md=_md))
    return (yield from inner_relative_list_grid_scan())


def _scan_1d(detectors, motor, start, stop, num, *, per_step=None, md=None):
    """
    Scan over one variable in equally spaced steps.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    motor : object
        any 'settable' object (motor, temp controller, etc.)
    start : float
        starting position of motor
    stop : float
        ending position of motor
    num : int
        number of steps
    per_step : callable, optional
        hook for customizing action of inner loop (messages per step)
        Expected signature: ``f(detectors, motor, step)``
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.rel_scan`
    """
    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor.name],
           'num_points': num,
           'num_intervals': num - 1,
           'plan_args': {'detectors': list(map(repr, detectors)), 'num': num,
                         'motor': repr(motor),
                         'start': start, 'stop': stop,
                         'per_step': repr(per_step)},
           'plan_name': 'scan',
           'plan_pattern': 'linspace',
           'plan_pattern_module': 'numpy',
           'plan_pattern_args': dict(start=start, stop=stop, num=num),
           'hints': {},
           }
    _md.update(md or {})
    try:
        dimensions = [(motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].setdefault('dimensions', dimensions)

    if per_step is None:
        per_step = bps.one_1d_step

    steps = np.linspace(**_md['plan_pattern_args'])

    @bpp.stage_decorator(list(detectors) + [motor])
    @bpp.run_decorator(md=_md)
    def inner_scan():
        for step in steps:
            yield from per_step(detectors, motor, step)

    return (yield from inner_scan())


def _rel_scan_1d(detectors, motor, start, stop, num, *, per_step=None,
                 md=None):
    """
    Scan over one variable in equally spaced steps relative to current positon.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    motor : object
        any 'settable' object (motor, temp controller, etc.)
    start : float
        starting position of motor
    stop : float
        ending position of motor
    num : int
        number of steps
    per_step : callable, optional
        hook for customizing action of inner loop (messages per step)
        Expected signature: ``f(detectors, motor, step)``
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.scan`
    """
    _md = {'plan_name': 'rel_scan'}
    _md.update(md or {})
    # TODO read initial positions (redundantly) so they can be put in md here

    @bpp.reset_positions_decorator([motor])
    @bpp.relative_set_decorator([motor])
    def inner_relative_scan():
        return (yield from _scan_1d(detectors, motor, start, stop,
                                    num, per_step=per_step, md=_md))

    return (yield from inner_relative_scan())


def log_scan(detectors, motor, start, stop, num, *, per_step=None, md=None):
    """
    Scan over one variable in log-spaced steps.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    motor : object
        any 'settable' object (motor, temp controller, etc.)
    start : float
        starting position of motor
    stop : float
        ending position of motor
    num : int
        number of steps
    per_step : callable, optional
        hook for customizing action of inner loop (messages per step)
        Expected signature: ``f(detectors, motor, step)``
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.rel_log_scan`
    """
    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor.name],
           'num_points': num,
           'num_intervals': num - 1,
           'plan_args': {'detectors': list(map(repr, detectors)), 'num': num,
                         'start': start, 'stop': stop, 'motor': repr(motor),
                         'per_step': repr(per_step)},
           'plan_name': 'log_scan',
           'plan_pattern': 'logspace',
           'plan_pattern_module': 'numpy',
           'plan_pattern_args': dict(start=start, stop=stop, num=num),
           'hints': {},
           }
    _md.update(md or {})

    try:
        dimensions = [(motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].setdefault('dimensions', dimensions)

    if per_step is None:
        per_step = bps.one_1d_step

    steps = np.logspace(**_md['plan_pattern_args'])

    @bpp.stage_decorator(list(detectors) + [motor])
    @bpp.run_decorator(md=_md)
    def inner_log_scan():
        for step in steps:
            yield from per_step(detectors, motor, step)

    return (yield from inner_log_scan())


def rel_log_scan(detectors, motor, start, stop, num, *, per_step=None,
                 md=None):
    """
    Scan over one variable in log-spaced steps relative to current position.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    motor : object
        any 'settable' object (motor, temp controller, etc.)
    start : float
        starting position of motor
    stop : float
        ending position of motor
    num : int
        number of steps
    per_step : callable, optional
        hook for customizing action of inner loop (messages per step)
        Expected signature: ``f(detectors, motor, step)``
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.log_scan`
    """
    # TODO read initial positions (redundantly) so they can be put in md here
    _md = {'plan_name': 'rel_log_scan'}
    _md.update(md or {})

    @bpp.reset_positions_decorator([motor])
    @bpp.relative_set_decorator([motor])
    def inner_relative_log_scan():
        return (yield from log_scan(detectors, motor, start, stop, num,
                                    per_step=per_step, md=_md))

    return (yield from inner_relative_log_scan())


def adaptive_scan(detectors, target_field, motor, start, stop,
                  min_step, max_step, target_delta, backstep,
                  threshold=0.8, *, md=None):
    """
    Scan over one variable with adaptively tuned step size.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    target_field : string
        data field whose output is the focus of the adaptive tuning
    motor : object
        any 'settable' object (motor, temp controller, etc.)
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
    threshold : float, optional
        threshold for going backward and rescanning a region, default is 0.8
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.rel_adaptive_scan`
    """
    if not 0 < min_step < max_step:
        raise ValueError("min_step and max_step must meet condition of "
                         "max_step > min_step > 0")

    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor.name],
           'plan_args': {'detectors': list(map(repr, detectors)),
                         'motor': repr(motor),
                         'start': start,
                         'stop': stop,
                         'min_step': min_step,
                         'max_step': max_step,
                         'target_delta': target_delta,
                         'backstep': backstep,
                         'threshold': threshold},
           'plan_name': 'adaptive_scan',
           'hints': {},
           }
    _md.update(md or {})
    try:
        dimensions = [(motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].setdefault('dimensions', dimensions)

    @bpp.stage_decorator(list(detectors) + [motor])
    @bpp.run_decorator(md=_md)
    def adaptive_core():
        next_pos = start
        step = (max_step - min_step) / 2
        past_I = None
        cur_I = None
        cur_det = {}
        if stop >= start:
            direction_sign = 1
        else:
            direction_sign = -1
        while next_pos * direction_sign < stop * direction_sign:
            yield Msg('checkpoint')
            yield from bps.mv(motor, next_pos)
            yield Msg('create', None, name='primary')
            for det in detectors:
                yield Msg('trigger', det, group='B')
            yield Msg('wait', None, 'B')
            for det in utils.separate_devices(detectors + [motor]):
                cur_det = yield Msg('read', det)
                if target_field in cur_det:
                    cur_I = cur_det[target_field]['value']
            yield Msg('save')

            # special case first first loop
            if past_I is None:
                past_I = cur_I
                next_pos += step * direction_sign
                continue

            dI = np.abs(cur_I - past_I)

            slope = dI / step
            if slope:
                new_step = np.clip(target_delta / slope, min_step, max_step)
            else:
                new_step = np.min([step * 1.1, max_step])

            # if we over stepped, go back and try again
            if backstep and (new_step < step * threshold):
                next_pos -= step
                step = new_step
            else:
                past_I = cur_I
                step = 0.2 * new_step + 0.8 * step
            next_pos += step * direction_sign

    return (yield from adaptive_core())


def rel_adaptive_scan(detectors, target_field, motor, start, stop,
                      min_step, max_step, target_delta, backstep,
                      threshold=0.8, *, md=None):
    """
    Relative scan over one variable with adaptively tuned step size.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    target_field : string
        data field whose output is the focus of the adaptive tuning
    motor : object
        any 'settable' object (motor, temp controller, etc.)
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
    threshold : float, optional
        threshold for going backward and rescanning a region, default is 0.8
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.adaptive_scan`
    """
    _md = {'plan_name': 'rel_adaptive_scan'}
    _md.update(md or {})

    @bpp.reset_positions_decorator([motor])
    @bpp.relative_set_decorator([motor])
    def inner_relative_adaptive_scan():
        return (yield from adaptive_scan(detectors, target_field,
                                         motor, start, stop, min_step,
                                         max_step, target_delta,
                                         backstep, threshold, md=_md))

    return (yield from inner_relative_adaptive_scan())


def tune_centroid(
        detectors, signal, motor,
        start, stop, min_step,
        num=10,
        step_factor=3.0,
        snake=False,
        *, md=None):
    r"""
    plan: tune a motor to the centroid of signal(motor)

    Initially, traverse the range from start to stop with
    the number of points specified.  Repeat with progressively
    smaller step size until the minimum step size is reached.
    Rescans will be centered on the signal centroid
    (for $I(x)$, centroid$= \sum{I}/\sum{x*I}$)
    with original scan range reduced by ``step_factor``.

    Set ``snake=True`` if your positions are reproducible
    moving from either direction.  This will not necessarily
    decrease the number of traversals required to reach convergence.
    Snake motion reduces the total time spent on motion
    to reset the positioner.  For some positioners, such as
    those with hysteresis, snake scanning may not be appropriate.
    For such positioners, always approach the positions from the
    same direction.

    Note:  Ideally the signal has only one peak in the range to
    be scanned.  It is assumed the signal is not polymodal
    between ``start`` and ``stop``.

    Parameters
    ----------
    detectors : Signal
        list of 'readable' objects
    signal : string
        detector field whose output is to maximize
    motor : object
        any 'settable' object (motor, temp controller, etc.)
    start : float
        start of range
    stop : float
        end of range, note: start < stop
    min_step : float
        smallest step size to use.
    num : int, optional
        number of points with each traversal, default = 10
    step_factor : float, optional
        used in calculating new range after each pass

        note: step_factor > 1.0, default = 3
    snake : bool, optional
        if False (default), always scan from start to stop
    md : dict, optional
        metadata

    Examples
    --------
    Find the center of a peak using synthetic hardware.

    >>> from ophyd.sim import SynAxis, SynGauss
    >>> motor = SynAxis(name='motor')
    >>> det = SynGauss(name='det', motor, 'motor',
    ...                center=-1.3, Imax=1e5, sigma=0.05)
    >>> RE(tune_centroid([det], "det", motor, -1.5, -0.5, 0.01, 10))
    """
    if min_step <= 0:
        raise ValueError("min_step must be positive")
    if step_factor <= 1.0:
        raise ValueError("step_factor must be greater than 1.0")
    try:
        motor_name, = motor.hints['fields']
    except (AttributeError, ValueError):
        motor_name = motor.name
    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor.name],
           'plan_args': {'detectors': list(map(repr, detectors)),
                         'motor': repr(motor),
                         'start': start,
                         'stop': stop,
                         'num': num,
                         'min_step': min_step, },
           'plan_name': 'tune_centroid',
           'hints': {},
           }
    _md.update(md or {})
    try:
        dimensions = [(motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].setdefault('dimensions', dimensions)

    low_limit = min(start, stop)
    high_limit = max(start, stop)

    @bpp.stage_decorator(list(detectors) + [motor])
    @bpp.run_decorator(md=_md)
    def _tune_core(start, stop, num, signal):
        next_pos = start
        step = (stop - start) / (num - 1)
        peak_position = None
        cur_I = None
        sum_I = 0       # for peak centroid calculation, I(x)
        sum_xI = 0

        while abs(step) >= min_step and low_limit <= next_pos <= high_limit:
            yield Msg('checkpoint')
            yield from bps.mv(motor, next_pos)
            ret = (yield from bps.trigger_and_read(detectors + [motor]))
            cur_I = ret[signal]['value']
            sum_I += cur_I
            position = ret[motor_name]['value']
            sum_xI += position * cur_I

            next_pos += step
            in_range = min(start, stop) <= next_pos <= max(start, stop)

            if not in_range:
                if sum_I == 0:
                    return
                peak_position = sum_xI / sum_I  # centroid
                sum_I, sum_xI = 0, 0    # reset for next pass
                new_scan_range = (stop - start) / step_factor
                start = np.clip(peak_position - new_scan_range/2,
                                low_limit, high_limit)
                stop = np.clip(peak_position + new_scan_range/2,
                               low_limit, high_limit)
                if snake:
                    start, stop = stop, start
                step = (stop - start) / (num - 1)
                next_pos = start
                # print("peak position = {}".format(peak_position))
                # print("start = {}".format(start))
                # print("stop = {}".format(stop))

        # finally, move to peak position
        if peak_position is not None:
            # improvement: report final peak_position
            # print("final position = {}".format(peak_position))
            yield from bps.mv(motor, peak_position)

    return (yield from _tune_core(start, stop, num, signal))


def scan_nd(detectors, cycler, *, per_step=None, md=None):
    """
    Scan over an arbitrary N-dimensional trajectory.

    Parameters
    ----------
    detectors : list
    cycler : Cycler
        cycler.Cycler object mapping movable interfaces to positions
    per_step : callable, optional
        hook for customizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
        for details.
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.inner_product_scan`
    :func:`bluesky.plans.grid_scan`

    Examples
    --------
    >>> from cycler import cycler
    >>> cy = cycler(motor1, [1, 2, 3]) * cycler(motor2, [4, 5, 6])
    >>> scan_nd([sensor], cy)
    """
    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor.name for motor in cycler.keys],
           'num_points': len(cycler),
           'num_intervals': len(cycler) - 1,
           'plan_args': {'detectors': list(map(repr, detectors)),
                         'cycler': repr(cycler),
                         'per_step': repr(per_step)},
           'plan_name': 'scan_nd',
           'hints': {},
           }
    _md.update(md or {})
    try:
        dimensions = [(motor.hints['fields'], 'primary')
                      for motor in cycler.keys]
    except (AttributeError, KeyError):
        # Not all motors provide a 'fields' hint, so we have to skip it.
        pass
    else:
        # We know that hints exists. Either:
        #  - the user passed it in and we are extending it
        #  - the user did not pass it in and we got the default {}
        # If the user supplied hints includes a dimension entry, do not
        # change it, else set it to the one generated above
        _md['hints'].setdefault('dimensions', dimensions)

    if per_step is None:
        per_step = bps.one_nd_step
    else:
        # Ensure that the user-defined per-step has the expected signature.
        sig = inspect.signature(per_step)

        def _verify_1d_step(sig):
            if len(sig.parameters) < 3:
                return False
            for name, (p_name, p) in zip_longest(['detectors', 'motor', 'step'], sig.parameters.items()):
                # this is one of the first 3 positional arguements, check that the name matches
                if name is not None:
                    if name != p_name:
                        return False
                # if there are any extra arguments, check that they have a default
                else:
                    if p.kind is p.VAR_KEYWORD or p.kind is p.VAR_POSITIONAL:
                        continue
                    if p.default is p.empty:
                        return False

            return True

        def _verify_nd_step(sig):
            if len(sig.parameters) < 3:
                return False
            for name, (p_name, p) in zip_longest(['detectors', 'step', 'pos_cache'], sig.parameters.items()):
                # this is one of the first 3 positional arguements, check that the name matches
                if name is not None:
                    if name != p_name:
                        return False
                # if there are any extra arguments, check that they have a default
                else:
                    if p.kind is p.VAR_KEYWORD or p.kind is p.VAR_POSITIONAL:
                        continue
                    if p.default is p.empty:
                        return False

            return True

        if sig == inspect.signature(bps.one_nd_step):
            pass
        elif _verify_nd_step(sig):
            # check other signature for back-compatibility
            pass
        elif _verify_1d_step(sig):
            # Accept this signature for back-compat reasons (because
            # inner_product_scan was renamed scan).
            dims = len(list(cycler.keys))
            if dims != 1:
                raise TypeError("Signature of per_step assumes 1D trajectory "
                                "but {} motors are specified.".format(dims))
            motor, = cycler.keys
            user_per_step = per_step

            def adapter(detectors, step, pos_cache):
                # one_nd_step 'step' parameter is a dict; one_id_step 'step'
                # parameter is a value
                step, = step.values()
                return (yield from user_per_step(detectors, motor, step))
            per_step = adapter
        else:
            raise TypeError("per_step must be a callable with the signature \n "
                            "<Signature (detectors, step, pos_cache)> or "
                            "<Signature (detectors, motor, step)>. \n"
                            "per_step signature received: {}".format(sig))
    pos_cache = defaultdict(lambda: None)  # where last position is stashed
    cycler = utils.merge_cycler(cycler)
    motors = list(cycler.keys)

    @bpp.stage_decorator(list(detectors) + motors)
    @bpp.run_decorator(md=_md)
    def inner_scan_nd():
        for step in list(cycler):
            yield from per_step(detectors, step, pos_cache)

    return (yield from inner_scan_nd())


def inner_product_scan(detectors, num, *args, per_step=None, md=None):
    # For scan, num is the _last_ positional arg instead of the first one.
    # Notice the swapped order here.
    md = md or {}
    md.setdefault('plan_name', 'inner_product_scan')
    yield from scan(detectors, *args, num, per_step=None, md=md)


def scan(detectors, *args, num=None, per_step=None, md=None):
    """
    Scan over one multi-motor trajectory.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    *args :
        For one dimension, ``motor, start, stop``.
        In general:

        .. code-block:: python

            motor1, start1, stop1,
            motor2, start2, stop2,
            ...,
            motorN, startN, stopN

        Motors can be any 'settable' object (motor, temp controller, etc.)
    num : integer
        number of points
    per_step : callable, optional
        hook for customizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
        for details.
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.relative_inner_product_scan`
    :func:`bluesky.plans.grid_scan`
    :func:`bluesky.plans.scan_nd`
    """
    # For back-compat reasons, we accept 'num' as the last positional argument:
    # scan(detectors, motor, -1, 1, 3)
    # or by keyword:
    # scan(detectors, motor, -1, 1, num=3)
    # ... which requires some special processing.
    if num is None:
        if len(args) % 3 != 1:
            raise ValueError("The number of points to scan must be provided "
                             "as the last positional argument or as keyword "
                             "argument 'num'.")
        num = args[-1]
        args = args[:-1]

    if not (float(num).is_integer() and num > 0.0):
        raise ValueError(f"The parameter `num` is expected to be a number of "
                         f"steps (not step size!) It must therefore be a "
                         f"whole number. The given value was {num}.")
    num = int(num)

    md_args = list(chain(*((repr(motor), start, stop)
                           for motor, start, stop in partition(3, args))))
    motor_names = tuple(motor.name for motor, start, stop
                        in partition(3, args))
    md = md or {}
    _md = {'plan_args': {'detectors': list(map(repr, detectors)),
                         'num': num, 'args': md_args,
                         'per_step': repr(per_step)},
           'plan_name': 'scan',
           'plan_pattern': 'inner_product',
           'plan_pattern_module': plan_patterns.__name__,
           'plan_pattern_args': dict(num=num, args=md_args),
           'motors': motor_names
           }
    _md.update(md)

    # get hints for best effort callback
    motors = [motor for motor, start, stop in partition(3, args)]

    # Give a hint that the motors all lie along the same axis
    # [(['motor1', 'motor2', ...], 'primary'), ] is 1D (this case)
    # [ ('motor1', 'primary'), ('motor2', 'primary'), ... ] is 2D for example
    # call x_fields because these are meant to be the x (independent) axis
    x_fields = []
    for motor in motors:
        x_fields.extend(getattr(motor, 'hints', {}).get('fields', []))

    default_dimensions = [(x_fields, 'primary')]

    default_hints = {}
    if len(x_fields) > 0:
        default_hints.update(dimensions=default_dimensions)

    # now add default_hints and override any hints from the original md (if
    # exists)
    _md['hints'] = default_hints
    _md['hints'].update(md.get('hints', {}) or {})

    full_cycler = plan_patterns.inner_product(num=num, args=args)

    return (yield from scan_nd(detectors, full_cycler,
                               per_step=per_step, md=_md))


def grid_scan(detectors, *args, snake_axes=None, per_step=None, md=None):
    """
    Scan over a mesh; each motor is on an independent trajectory.

    Parameters
    ----------
    detectors: list
        list of 'readable' objects
    ``*args``
        patterned like (``motor1, start1, stop1, num1,``
                        ``motor2, start2, stop2, num2,``
                        ``motor3, start3, stop3, num3,`` ...
                        ``motorN, startN, stopN, numN``)

        The first motor is the "slowest", the outer loop. For all motors
        except the first motor, there is a "snake" argument: a boolean
        indicating whether to following snake-like, winding trajectory or a
        simple left-to-right trajectory.
    snake_axes: boolean or iterable, optional
        which axes should be snaked, either ``False`` (do not snake any axes),
        ``True`` (snake all axes) or a list of axes to snake. "Snaking" an axis
        is defined as following snake-like, winding trajectory instead of a
        simple left-to-right trajectory. The elements of the list are motors
        that are listed in `args`. The list must not contain the slowest
        (first) motor, since it can't be snaked.
    per_step: callable, optional
        hook for customizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
        for details.
    md: dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.rel_grid_scan`
    :func:`bluesky.plans.inner_product_scan`
    :func:`bluesky.plans.scan_nd`
    """
    # Notes: (not to be included in the documentation)
    #   The deprecated function call with no 'snake_axes' argument and 'args'
    #         patterned like (``motor1, start1, stop1, num1,``
    #                         ``motor2, start2, stop2, num2, snake2,``
    #                         ``motor3, start3, stop3, num3, snake3,`` ...
    #                         ``motorN, startN, stopN, numN, snakeN``)
    #         The first motor is the "slowest", the outer loop. For all motors
    #         except the first motor, there is a "snake" argument: a boolean
    #         indicating whether to following snake-like, winding trajectory or a
    #         simple left-to-right trajectory.
    #   Ideally, deprecated and new argument lists should not be mixed.
    #   The function will still accept `args` in the old format even if `snake_axes` is
    #   supplied, but if `snake_axes` is not `None` (the default value), it overrides
    #   any values of `snakeX` in `args`.

    args_pattern = plan_patterns.classify_outer_product_args_pattern(args)
    if (snake_axes is not None) and \
            (args_pattern == plan_patterns.OuterProductArgsPattern.PATTERN_2):
        raise ValueError("Mixing of deprecated and new API interface is not allowed: "
                         "the parameter 'snake_axes' can not be used if snaking is "
                         "set as part of 'args'")

    # For consistency, set 'snake_axes' to False if new API call is detected
    if (snake_axes is None) and \
            (args_pattern != plan_patterns.OuterProductArgsPattern.PATTERN_2):
        snake_axes = False

    chunk_args = list(plan_patterns.chunk_outer_product_args(args, args_pattern))
    # 'chunk_args' is a list of tuples of the form: (motor, start, stop, num, snake)
    # If the function is called using deprecated pattern for arguments, then
    # 'snake' may be set True for some motors, otherwise the 'snake' is always False.

    # The list of controlled motors
    motors = [_[0] for _ in chunk_args]

    # Check that the same motor is not listed multiple times. This indicates an error in the script.
    if len(set(motors)) != len(motors):
        raise ValueError(f"Some motors are listed multiple times in the argument list 'args': "
                         f"'{motors}'")

    if snake_axes is not None:

        def _set_snaking(chunk, value):
            """Returns the tuple `chunk` with modified 'snake' value"""
            _motor, _start, _stop, _num, _snake = chunk
            return _motor, _start, _stop, _num, value

        if isinstance(snake_axes, collections.abc.Iterable) and not isinstance(snake_axes, str):
            # Always convert to a tuple (in case a `snake_axes` is an iterator).
            snake_axes = tuple(snake_axes)

            # Check if the list of axes (motors) contains repeated entries.
            if len(set(snake_axes)) != len(snake_axes):
                raise ValueError(f"The list of axes 'snake_axes' contains repeated elements: "
                                 f"'{snake_axes}'")

            # Check if the snaking is enabled for the slowest motor.
            if len(motors) and (motors[0] in snake_axes):
                raise ValueError(f"The list of axes 'snake_axes' contains the slowest motor: "
                                 f"'{snake_axes}'")

            # Check that all motors in the chunk_args are controlled in the scan.
            #   It is very likely that the script running the plan has a bug.
            if any([_ not in motors for _ in snake_axes]):
                raise ValueError(f"The list of axes 'snake_axes' contains motors "
                                 f"that are not controlled during the scan: "
                                 f"'{snake_axes}'")

            # Enable snaking for the selected axes.
            #   If the argument `snake_axes` is specified (not None), then
            #   any `snakeX` values that could be specified in `args` are ignored.
            for n, chunk in enumerate(chunk_args):
                if n > 0:  # The slowest motor is never snaked
                    motor = chunk[0]
                    if motor in snake_axes:
                        chunk_args[n] = _set_snaking(chunk, True)
                    else:
                        chunk_args[n] = _set_snaking(chunk, False)

        elif snake_axes is True:  # 'snake_axes' has boolean value `True`
            # Set all 'snake' values except for the slowest motor
            chunk_args = [_set_snaking(_, True) if n > 0 else _
                          for n, _ in enumerate(chunk_args)]
        elif snake_axes is False:  # 'snake_axes' has boolean value `True`
            # Set all 'snake' values
            chunk_args = [_set_snaking(_, False) for _ in chunk_args]
        else:
            raise ValueError(f"Parameter 'snake_axes' is not iterable, boolean or None: "
                             f"'{snake_axes}', type: {type(snake_axes)}")

    # Prepare the argument list for the `outer_product` function
    args_modified = []
    for n, chunk in enumerate(chunk_args):
        if n == 0:
            args_modified.extend(chunk[:-1])
        else:
            args_modified.extend(chunk)
    full_cycler = plan_patterns.outer_product(args=args_modified)

    md_args = []
    motor_names = []
    motors = []
    for i, (motor, start, stop, num, snake) in enumerate(chunk_args):
        md_args.extend([repr(motor), start, stop, num])
        if i > 0:
            # snake argument only shows up after the first motor
            md_args.append(snake)
        motor_names.append(motor.name)
        motors.append(motor)
    _md = {'shape': tuple(num for motor, start, stop, num, snake
                          in chunk_args),
           'extents': tuple([start, stop] for motor, start, stop, num, snake
                            in chunk_args),
           'snaking': tuple(snake for motor, start, stop, num, snake
                            in chunk_args),
           # 'num_points': inserted by scan_nd
           'plan_args': {'detectors': list(map(repr, detectors)),
                         'args': md_args,
                         'per_step': repr(per_step)},
           'plan_name': 'grid_scan',
           'plan_pattern': 'outer_product',
           'plan_pattern_args': dict(args=md_args),
           'plan_pattern_module': plan_patterns.__name__,
           'motors': tuple(motor_names),
           'hints': {},
           }
    _md.update(md or {})
    _md['hints'].setdefault('gridding', 'rectilinear')
    try:
        _md['hints'].setdefault('dimensions', [(m.hints['fields'], 'primary')
                                               for m in motors])
    except (AttributeError, KeyError):
        ...

    return (yield from scan_nd(detectors, full_cycler,
                               per_step=per_step, md=_md))


def rel_grid_scan(detectors, *args, snake_axes=None, per_step=None, md=None):
    """
    Scan over a mesh relative to current position.

    Parameters
    ----------
    detectors: list
        list of 'readable' objects
    ``*args``
        patterned like (``motor1, start1, stop1, num1,``
                        ``motor2, start2, stop2, num2,``
                        ``motor3, start3, stop3, num3,`` ...
                        ``motorN, startN, stopN, numN``)

        The first motor is the "slowest", the outer loop. For all motors
        except the first motor, there is a "snake" argument: a boolean
        indicating whether to following snake-like, winding trajectory or a
        simple left-to-right trajectory.
    snake_axes: boolean or iterable, optional
        which axes should be snaked, either ``False`` (do not snake any axes),
        ``True`` (snake all axes) or a list of axes to snake. "Snaking" an axis
        is defined as following snake-like, winding trajectory instead of a
        simple left-to-right trajectory. The elements of the list are motors
        that are listed in `args`. The list must not contain the slowest
        (first) motor, since it can't be snaked.
    per_step: callable, optional
        hook for customizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
        for details.
    md: dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.relative_inner_product_scan`
    :func:`bluesky.plans.grid_scan`
    :func:`bluesky.plans.scan_nd`
    """
    # Notes: the deprecated function call is also supported. See the notes
    #   following the docstring for 'grid_scan' function

    _md = {'plan_name': 'rel_grid_scan'}
    _md.update(md or {})
    motors = [m[0] for m in
              plan_patterns.chunk_outer_product_args(args)]

    @bpp.reset_positions_decorator(motors)
    @bpp.relative_set_decorator(motors)
    def inner_rel_grid_scan():
        return (yield from grid_scan(detectors, *args,
                                     snake_axes=snake_axes,
                                     per_step=per_step, md=_md))

    return (yield from inner_rel_grid_scan())


def relative_inner_product_scan(detectors, num, *args, per_step=None, md=None):
    # For rel_scan, num is the _last_ positional arg instead of the first one.
    # Notice the swapped order here.
    md = md or {}
    md.setdefault('plan_name', 'relative_inner_product_scan')
    yield from rel_scan(detectors, *args, num, per_step=per_step, md=md)


def rel_scan(detectors, *args, num=None, per_step=None, md=None):
    """
    Scan over one multi-motor trajectory relative to current position.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    *args :
        For one dimension, ``motor, start, stop``.
        In general:

        .. code-block:: python

            motor1, start1, stop1,
            motor2, start2, start2,
            ...,
            motorN, startN, stopN,

        Motors can be any 'settable' object (motor, temp controller, etc.)
    num : integer
        number of points
    per_step : callable, optional
        hook for customizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
        for details.
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.rel_grid_scan`
    :func:`bluesky.plans.inner_product_scan`
    :func:`bluesky.plans.scan_nd`
    """
    _md = {'plan_name': 'rel_scan'}
    md = md or {}
    _md.update(md)
    motors = [motor for motor, start, stop in partition(3, args)]

    @bpp.reset_positions_decorator(motors)
    @bpp.relative_set_decorator(motors)
    def inner_rel_scan():
        return (yield from scan(detectors, *args, num=num,
                                per_step=per_step, md=_md))

    return (yield from inner_rel_scan())


def tweak(detector, target_field, motor, step, *, md=None):
    """
    Move and motor and read a detector with an interactive prompt.

    Parameters
    ----------
    detetector : Device
    target_field : string
        data field whose output is the focus of the adaptive tuning
    motor : Device
    step : float
        initial suggestion for step size
    md : dict, optional
        metadata
    """
    prompt_str = '{0}, {1:.3}, {2:.3}, ({3}) '

    _md = {'detectors': [detector.name],
           'motors': [motor.name],
           'plan_args': {'detector': repr(detector),
                         'target_field': target_field,
                         'motor': repr(motor),
                         'step': step},
           'plan_name': 'tweak',
           'hints': {},
           }
    try:
        dimensions = [(motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].update({'dimensions': dimensions})
    _md.update(md or {})
    d = detector
    try:
        from IPython.display import clear_output
    except ImportError:
        # Define a no-op for clear_output.
        def clear_output(wait=False):
            pass

    @bpp.stage_decorator([detector, motor])
    @bpp.run_decorator(md=_md)
    def tweak_core():
        nonlocal step

        while True:
            yield Msg('create', None, name='primary')
            ret_mot = yield Msg('read', motor)
            if ret_mot is None:
                return
            key = list(ret_mot.keys())[0]
            pos = ret_mot[key]['value']
            yield Msg('trigger', d, group='A')
            yield Msg('wait', None, 'A')
            reading = yield Msg('read', d)
            val = reading[target_field]['value']
            yield Msg('save')
            prompt = prompt_str.format(motor.name, float(pos),
                                       float(val), step)
            new_step = yield Msg('input', prompt=prompt)
            if new_step:
                try:
                    step = float(new_step)
                except ValueError:
                    break
            yield Msg('set', motor, pos + step, group='A')
            print('Motor moving...')
            sys.stdout.flush()
            yield Msg('wait', None, 'A')
            clear_output(wait=True)
            # stackoverflow.com/a/12586667/380231
            print('\x1b[1A\x1b[2K\x1b[1A')

    return (yield from tweak_core())


def spiral_fermat(detectors, x_motor, y_motor, x_start, y_start, x_range,
                  y_range, dr, factor, *, dr_y=None, tilt=0.0, per_step=None,
                  md=None):
    '''Absolute fermat spiral scan, centered around (x_start, y_start)

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    x_motor : object
        any 'settable' object (motor, temp controller, etc.)
    y_motor : object
        any 'settable' object (motor, temp controller, etc.)
    x_start : float
        x center
    y_start : float
        y center
    x_range : float
        x width of spiral
    y_range : float
        y width of spiral
    dr : float
        delta radius
    factor : float
        radius gets divided by this
    dr_y : float, optional
        Delta radius along the major axis of the ellipse, if not specifed
        defaults to dr.
    tilt : float, optional
        Tilt angle in radians, default 0.0
    per_step : callable, optional
        hook for customizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
        for details.
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.spiral`
    :func:`bluesky.plans.rel_spiral`
    :func:`bluesky.plans.rel_spiral_fermat`
    '''
    pattern_args = dict(x_motor=x_motor, y_motor=y_motor, x_start=x_start,
                        y_start=y_start, x_range=x_range, y_range=y_range,
                        dr=dr, factor=factor, dr_y=dr_y, tilt=tilt)
    cyc = plan_patterns.spiral_fermat(**pattern_args)

    # Before including pattern_args in metadata, replace objects with reprs.
    pattern_args['x_motor'] = repr(x_motor)
    pattern_args['y_motor'] = repr(y_motor)
    _md = {'plan_args': {'detectors': list(map(repr, detectors)),
                         'x_motor': repr(x_motor), 'y_motor': repr(y_motor),
                         'x_start': x_start, 'y_start': y_start,
                         'x_range': x_range, 'y_range': y_range,
                         'dr': dr, 'factor': factor, 'dr_y': dr_y,
                         'tilt': tilt, 'per_step': repr(per_step)},
           'extents': tuple([[x_start - x_range, x_start + x_range],
                             [y_start - y_range, y_start + y_range]]),
           'plan_name': 'spiral_fermat',
           'plan_pattern': 'spiral_fermat',
           'plan_pattern_module': plan_patterns.__name__,
           'plan_pattern_args': pattern_args,
           'hints': {},
           }
    try:
        dimensions = [(x_motor.hints['fields'], 'primary'),
                      (y_motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].update({'dimensions': dimensions})
    _md.update(md or {})

    return (yield from scan_nd(detectors, cyc, per_step=per_step, md=_md))


def rel_spiral_fermat(detectors, x_motor, y_motor, x_range, y_range, dr,
                      factor, *, dr_y=None, tilt=0.0, per_step=None, md=None):
    '''Relative fermat spiral scan

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    x_motor : object
        any 'settable' object (motor, temp controller, etc.)
    y_motor : object
        any 'settable' object (motor, temp controller, etc.)
    x_range : float
        x width of spiral
    y_range : float
        y width of spiral
    dr : float
        delta radius
    factor : float
        radius gets divided by this
    dr_y : float, optional
        Delta radius along the major axis of the ellipse, if not specifed
        defaults to dr
    tilt : float, optional
        Tilt angle in radians, default 0.0
    per_step : callable, optional
        hook for customizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
        for details.
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.spiral`
    :func:`bluesky.plans.rel_spiral`
    :func:`bluesky.plans.spiral_fermat`
    '''
    _md = {'plan_name': 'rel_spiral_fermat'}
    _md.update(md or {})

    @bpp.reset_positions_decorator([x_motor, y_motor])
    @bpp.relative_set_decorator([x_motor, y_motor])
    def inner_relative_spiral_fermat():
        return (yield from spiral_fermat(detectors, x_motor, y_motor,
                                         0, 0,
                                         x_range, y_range,
                                         dr, factor, dr_y=dr_y, tilt=tilt,
                                         per_step=per_step, md=_md))

    return (yield from inner_relative_spiral_fermat())


def spiral(detectors, x_motor, y_motor, x_start, y_start, x_range, y_range, dr,
           nth, *, dr_y=None, tilt=0.0, per_step=None, md=None):
    '''Spiral scan, centered around (x_start, y_start)

    Parameters
    ----------
    x_motor : object
        any 'settable' object (motor, temp controller, etc.)
    y_motor : object
        any 'settable' object (motor, temp controller, etc.)
    x_start : float
        x center
    y_start : float
        y center
    x_range : float
        x width of spiral
    y_range : float
        y width of spiral
    dr : float
        Delta radius along the minor axis of the ellipse.
    dr_y : float, optional
        Delta radius along the major axis of the ellipse. If None, defaults to
        dr.
    nth : float
        Number of theta steps
    tilt : float, optional
        Tilt angle in radians, default 0.0
    per_step : callable, optional
        hook for customizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
        for details.
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.rel_spiral`
    :func:`bluesky.plans.spiral_fermat`
    :func:`bluesky.plans.rel_spiral_fermat`
    '''
    pattern_args = dict(x_motor=x_motor, y_motor=y_motor, x_start=x_start,
                        y_start=y_start, x_range=x_range, y_range=y_range,
                        dr=dr, nth=nth, dr_y=dr_y, tilt=tilt)
    cyc = plan_patterns.spiral(**pattern_args)

    # Before including pattern_args in metadata, replace objects with reprs.
    pattern_args['x_motor'] = repr(x_motor)
    pattern_args['y_motor'] = repr(y_motor)
    _md = {'plan_args': {'detectors': list(map(repr, detectors)),
                         'x_motor': repr(x_motor), 'y_motor': repr(y_motor),
                         'x_start': x_start, 'y_start': y_start,
                         'x_range': x_range, 'y_range': y_range,
                         'dr': dr, 'dr_y': dr_y, 'nth': nth, 'tilt': tilt,
                         'per_step': repr(per_step)},
           'extents': tuple([[x_start - x_range, x_start + x_range],
                             [y_start - y_range, y_start + y_range]]),
           'plan_name': 'spiral',
           'plan_pattern': 'spiral',
           'plan_pattern_args': pattern_args,
           'plan_pattern_module': plan_patterns.__name__,
           'hints': {},
           }
    try:
        dimensions = [(x_motor.hints['fields'], 'primary'),
                      (y_motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].update({'dimensions': dimensions})
    _md.update(md or {})

    return (yield from scan_nd(detectors, cyc, per_step=per_step, md=_md))


def rel_spiral(detectors, x_motor, y_motor, x_range, y_range, dr, nth,
               *, dr_y=None, tilt=0.0, per_step=None, md=None):

    '''Relative spiral scan

    Parameters
    ----------
    x_motor : object
        any 'settable' object (motor, temp controller, etc.)
    y_motor : object
        any 'settable' object (motor, temp controller, etc.)
    x_range : float
        x width of spiral
    y_range : float
        y width of spiral
    dr : float
        Delta radius along the minor axis of the ellipse.
    dr_y : float, optional
        Delta radius along the major axis of the ellipse. If None, it
        defaults to dr.
    nth : float
        Number of theta steps
    tilt : float, optional
        Tilt angle in radians, default 0.0
    per_step : callable, optional
        hook for customizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
        for details.
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.spiral`
    :func:`bluesky.plans.spiral_fermat`
    '''
    _md = {'plan_name': 'rel_spiral'}
    _md.update(md or {})

    @bpp.reset_positions_decorator([x_motor, y_motor])
    @bpp.relative_set_decorator([x_motor, y_motor])
    def inner_relative_spiral():
        return (yield from spiral(detectors, x_motor, y_motor,
                                  0, 0,
                                  x_range, y_range, dr, nth,
                                  dr_y=dr_y, tilt=tilt,
                                  per_step=per_step, md=_md))

    return (yield from inner_relative_spiral())


def spiral_square(detectors, x_motor, y_motor, x_center, y_center, x_range,
                  y_range, x_num, y_num, *, per_step=None, md=None):
    '''Absolute square spiral scan, centered around (x_center, y_center)

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    x_motor : object
        any 'settable' object (motor, temp controller, etc.)
    y_motor : object
        any 'settable' object (motor, temp controller, etc.)
    x_center : float
        x center
    y_center : float
        y center
    x_range : float
        x width of spiral
    y_range : float
        y width of spiral
    x_num : float
        number of x axis points
    y_num : float
        Number of y axis points.
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plans.one_nd_step` (the default) for
        details.
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.relative_spiral_square`
    :func:`bluesky.plans.spiral`
    :func:`bluesky.plans.relative_spiral`
    :func:`bluesky.plans.spiral_fermat`
    :func:`bluesky.plans.relative_spiral_fermat`
    '''
    pattern_args = dict(x_motor=x_motor, y_motor=y_motor, x_center=x_center,
                        y_center=y_center, x_range=x_range, y_range=y_range,
                        x_num=x_num, y_num=y_num)
    cyc = plan_patterns.spiral_square_pattern(**pattern_args)

    # Before including pattern_args in metadata, replace objects with reprs.
    pattern_args['x_motor'] = repr(x_motor)
    pattern_args['y_motor'] = repr(y_motor)
    _md = {'plan_args': {'detectors': list(map(repr, detectors)),
                         'x_motor': repr(x_motor), 'y_motor': repr(y_motor),
                         'x_center': x_center, 'y_center': y_center,
                         'x_range': x_range, 'y_range': y_range,
                         'x_num': x_num, 'y_num': y_num,
                         'per_step': repr(per_step)},
           'plan_name': 'spiral_square',
           'plan_pattern': 'spiral_square',
           'shape': (y_num, x_num),
           'extents': ((y_center - y_range / 2, y_center + y_range / 2),
                       (x_center - x_range / 2, x_center + x_range / 2)),
           'hints': {},
           }
    _md.update(md or {})
    _md['hints'].setdefault('gridding', 'rectilinear_nonsequential')
    try:
        _md['hints'].setdefault('dimensions', [(m.hints['fields'], 'primary')
                                               for m in [y_motor, x_motor]])
    except (AttributeError, KeyError):
        ...

    return (yield from scan_nd(detectors, cyc, per_step=per_step, md=_md))


def rel_spiral_square(detectors, x_motor, y_motor, x_range, y_range,
                      x_num, y_num, *, per_step=None, md=None):
    '''Relative square spiral scan, centered around current (x, y) position.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    x_motor : object
        any 'settable' object (motor, temp controller, etc.)
    y_motor : object
        any 'settable' object (motor, temp controller, etc.)
    x_range : float
        x width of spiral
    y_range : float
        y width of spiral
    x_num : float
        number of x axis points
    y_num : float
        Number of y axis points.
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plans.one_nd_step` (the default) for
        details.
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.spiral_square`
    :func:`bluesky.plans.spiral`
    :func:`bluesky.plans.relative_spiral`
    :func:`bluesky.plans.spiral_fermat`
    :func:`bluesky.plans.relative_spiral_fermat`
    '''
    _md = {'plan_name': 'rel_spiral_square'}
    _md.update(md or {})

    @bpp.reset_positions_decorator([x_motor, y_motor])
    @bpp.relative_set_decorator([x_motor, y_motor])
    def inner_relative_spiral():
        return (yield from spiral_square(detectors, x_motor, y_motor,
                                         0, 0,
                                         x_range, y_range, x_num, y_num,
                                         per_step=per_step, md=_md))

    return (yield from inner_relative_spiral())


def ramp_plan(go_plan,
              monitor_sig,
              inner_plan_func,
              take_pre_data=True,
              timeout=None,
              period=None, md=None):
    '''Take data while ramping one or more positioners.

    The pseudo code for this plan is ::

       sts = (yield from go_plan)

       yield from open_run()
       yield from inner_plan_func()
       while not st.done:
           yield from inner_plan_func()
       yield from inner_plan_func()

       yield from close_run()

    Parameters
    ----------
    go_plan : generator
        plan to start the ramp.  This will be run inside of a open/close
        run.

        This plan must return a `ophyd.StatusBase` object.

    inner_plan_func : generator function
        generator which takes no input

        This will be called for every data point.  This should create
        one or more events.

    timeout : float, optional
        If not None, the maximum time the ramp can run.

        In seconds

    take_pre_data: Bool, optional
        If True, add a pre data at beginning

    period : float, optional
        If not None, take data no faster than this.  If None, take
        data as fast as possible

        If running the inner plan takes longer than `period` than take
        data with no dead time.

        In seconds.
    '''
    _md = {'plan_name': 'ramp_plan'}
    _md.update(md or {})

    @bpp.monitor_during_decorator((monitor_sig,))
    @bpp.run_decorator(md=_md)
    def polling_plan():
        fail_time = None
        if timeout is not None:
            # sort out if we should watch the clock
            fail_time = time.time() + timeout

        # take a 'pre' data point
        if take_pre_data:
            yield from inner_plan_func()
        # start the ramp
        status = (yield from go_plan)

        while not status.done:
            start_time = time.time()
            yield from inner_plan_func()
            if fail_time is not None:
                if time.time() > fail_time:
                    raise utils.RampFail()
            if period is not None:
                cur_time = time.time()
                wait_time = (start_time + period) - cur_time
                if wait_time > 0:
                    yield from bps.sleep(wait_time)
            # take a 'post' data point
        yield from inner_plan_func()

    return (yield from polling_plan())


def fly(flyers, *, md=None):
    """
    Perform a fly scan with one or more 'flyers'.

    Parameters
    ----------
    flyers : collection
        objects that support the flyer interface
    md : dict, optional
        metadata

    Yields
    ------
    msg : Msg
        'kickoff', 'wait', 'complete, 'wait', 'collect' messages

    See Also
    --------
    :func:`bluesky.preprocessors.fly_during_wrapper`
    :func:`bluesky.preprocessors.fly_during_decorator`
    """
    uid = yield from bps.open_run(md)
    for flyer in flyers:
        yield from bps.kickoff(flyer, wait=True)
    for flyer in flyers:
        yield from bps.complete(flyer, wait=True)
    for flyer in flyers:
        yield from bps.collect(flyer)
    yield from bps.close_run()
    return uid


def x2x_scan(detectors, motor1, motor2, start, stop, num, *,
             per_step=None, md=None):
    """
    Relatively scan over two motors in a 2:1 ratio

    This is a generalized version of a theta2theta scan

    Parameters
    ----------
    detectors : list
        list of 'readable' objects

    motor1, motor2 : Positioner
        The second motor will move half as much as the first

    start, stop : float
        The relative limits of the first motor.  The second motor
        will move between ``start / 2`` and ``stop / 2``

    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
        for details.

    md : dict, optional
        metadata


    """
    _md = {'plan_name': 'x2x_scan',
           'plan_args': {'detectors': list(map(repr, detectors)),
                         'motor1': motor1.name,
                         'motor2': motor2.name,
                         'start': start, 'stop': stop, 'num': num,
                         'per_step': repr(per_step)}
           }

    _md.update(md or {})
    return (yield from relative_inner_product_scan(
        detectors, num,
        motor1, start, stop,
        motor2, start / 2, stop / 2,
        per_step=per_step,
        md=_md))


relative_list_scan = rel_list_scan  # back-compat
relative_scan = rel_scan  # back-compat
relative_log_scan = rel_log_scan  # back-compat
relative_adaptive_scan = rel_adaptive_scan  # back-compat
outer_product_scan = grid_scan  # back-compat
relative_outer_product_scan = rel_grid_scan  # back-compat
relative_spiral_fermat = rel_spiral_fermat  # back-compat
relative_spiral = rel_spiral  # back-compat
