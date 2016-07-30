"""
This module implements plan generators that close over the "global state"
singleton, ``bluesky.global_state.gs``. Changing the attributes of ``gs``
changes the behavior of these plans.

    DETS  # list of detectors
    MASTER_DET  # detector to use for tw
    MASTER_DET_FIELD  # detector field to use for tw
    TH_MOTOR
    TTH_MOTOR

Page numbers in the code comments refer to the SPEC manual at
http://www.certif.com/downloads/css_docs/spec_man.pdf
"""
from collections import deque
import matplotlib.pyplot as plt
from bluesky import plans
from bluesky.callbacks import LiveTable, LivePlot, LiveRaster
from bluesky.callbacks.scientific import PeakStats
from boltons.iterutils import chunked
from bluesky.global_state import gs
from bluesky.utils import (first_key_heuristic)

from bluesky.plans import (subs_context, count, scan,
                           relative_scan, relative_inner_product_scan,
                           outer_product_scan, inner_product_scan,
                           tweak, configure_count_time_wrapper, planify,
                           baseline_wrapper)
import itertools
from itertools import chain
from collections import ChainMap
from inspect import signature, Parameter
import warnings

# ## Factory functions for generating callbacks


def _figure_name(base_name):
    """Helper to compute figure name

    This takes in a base name an return the name of the figure to use.

    If gs.OVERPLOT, then this is a no-op.  If not gs.OVERPLOT then append '(N)'
    to the end of the string until a non-existing figure is found

    """
    if not gs.OVERPLOT:
        if not plt.fignum_exists(base_name):
            pass
        else:
            for j in itertools.count(1):
                numbered_template = '{} ({})'.format(base_name, j)
                if not plt.fignum_exists(numbered_template):
                    base_name = numbered_template
                    break
    return base_name


def setup_plot(*, motors, dets, gs):
    """Setup a LivePlot by inspecting motors and gs.

    If motors is empty, use sequence number.
    """
    y_key = gs.PLOT_Y
    if motors:
        x_key = first_key_heuristic(list(motors)[0])
        fig_name = _figure_name('BlueSky {} v {}'.format(y_key, x_key))
        fig = plt.figure(fig_name)
        return LivePlot(y_key, x_key, fig=fig)
    else:
        fig_name = _figure_name('BlueSky: {} v sequence number'.format(y_key))
        fig = plt.figure(fig_name)
        return LivePlot(y_key, fig=fig)


def setup_peakstats(*, motors, dets, gs):
    "Set up peakstats"
    motor = list(motors)[0]
    key = first_key_heuristic(motor)
    ps = PeakStats(key, gs.MASTER_DET_FIELD, **gs.PS_CONFIG)
    gs.PS = ps
    ps.motor = motor
    return ps


def setup_livetable(*, motors, dets, gs):
    return LiveTable(motors + [gs.PLOT_Y] + gs.TABLE_COLS)


def setup_liveraster(*, motors, dets, gs, shape, extent):
    if len(motors) != 2:
        return None
    ylab, xlab = [first_key_heuristic(m) for m in motors]
    raster = LiveRaster(shape, gs.MASTER_DET_FIELD, xlabel=xlab,
                        ylabel=ylab, extent=extent, clim=[0, 1])
    return raster


def construct_subs(plan_name, motors, dets, **kwargs):
    factories = gs.SUB_FACTORIES.get('common', [])
    factories.extend(gs.SUB_FACTORIES.get(plan_name, []))
    subs = []
    for factory in factories:
        sig = signature(factory)
        fact_kwargs = {}
        missing_kwargs = set()
        for k, v in sig.parameters.items():
            if v.kind is Parameter.KEYWORD_ONLY:
                if k in kwargs:
                    fact_kwargs[k] = kwargs[k]
                else:
                    if v.default is Parameter.empty:
                        missing_kwargs.add(k)
        if missing_kwargs:
            warnings.warn('The factory {fn} could not be run due to missing '
                          'mandatory kwargs {missing!r}'.format(
                              fn=factory.__name__, missing=missing_kwargs))
        else:
            l_sub = factory(motors=motors, dets=dets, gs=gs, **fact_kwargs)
            if l_sub is None:
                continue
            elif callable(l_sub):
                subs.append(l_sub)
            else:
                subs.extend(l_sub)

    return {'all': subs}


# ## Counts (p. 140) ###

@planify
def ct(num=1, delay=None, time=None, *, md=None):
    """
    Take one or more readings from the global detectors.

    Parameters
    ----------
    num : integer, optional
        number of readings to take; default is 1.

        If None, capture data until canceled
    delay : iterable or scalar, optional
        time delay between successive readings; default is 0
    time : float, optional
        applied to any detectors that have a `count_time` setting
    md : dict, optional
        metadata
    """
    if md is None:
        md = {}
    md = ChainMap(md, {'plan_name': 'ct',
                       gs.MD_TIME_KEY: time})
    subs = construct_subs('ct', [], gs.DETS)
    if num is not None and num > 1:
        subs['all'].append(setup_plot(motors=[], dets=gs.DETS, gs=gs))

    plan_stack = deque()
    with subs_context(plan_stack, subs):
        plan = count(gs.DETS, num, delay, md=md)
        plan = baseline_wrapper(plan, gs.BASELINE_DEVICES)
        plan = configure_count_time_wrapper(plan, time)
        plan_stack.append(plan)
    return plan_stack
gs.SUB_FACTORIES['ct'] = [setup_livetable]


# ## Motor Scans (p. 146) ###

@planify
def ascan(motor, start, finish, intervals, time=None, *, md=None):
    """
    Scan over one variable in equally spaced steps.

    Parameters
    ----------
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    start : float
        starting position of motor
    finish : float
        ending position of motor
    intervals : int
        number of strides (number of points - 1)
    time : float, optional
        applied to any detectors that have a `count_time` setting
    md : dict, optional
        metadata
    """
    if md is None:
        md = {}
    md = ChainMap(md, {'plan_name': 'ascan',
                       gs.MD_TIME_KEY: time})
    subs = construct_subs('ascan', [motor], gs.DETS)

    plan_stack = deque()
    with subs_context(plan_stack, subs):
        plan = scan(gs.DETS, motor, start, finish, 1 + intervals, md=md)
        plan = baseline_wrapper(plan, [motor] + gs.BASELINE_DEVICES)
        plan = configure_count_time_wrapper(plan, time)
        plan_stack.append(plan)
        return plan_stack

gs.SUB_FACTORIES['ascan'] = [setup_livetable,
                             setup_plot,
                             setup_peakstats]


@planify
def dscan(motor, start, finish, intervals, time=None, *, md=None):
    """
    Scan over one variable in equally spaced steps relative to current pos.

    Parameters
    ----------
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    start : float
        starting position of motor
    finish : float
        ending position of motor
    intervals : int
        number of strides (number of points - 1)
    time : float, optional
        applied to any detectors that have a `count_time` setting
    md : dict, optional
        metadata
    """
    subs = construct_subs('dscan', [motor], gs.DETS)
    if md is None:
        md = {}
    md = ChainMap(md, {'plan_name': 'dscan',
                       gs.MD_TIME_KEY: time})
    plan_stack = deque()
    with subs_context(plan_stack, subs):
        plan = relative_scan(gs.DETS, motor, start, finish, 1 + intervals,
                             md=md)
        plan = baseline_wrapper(plan, [motor] + gs.BASELINE_DEVICES)
        plan = configure_count_time_wrapper(plan, time)
        plan_stack.append(plan)
    return plan_stack

gs.SUB_FACTORIES['dscan'] = [setup_livetable,
                             setup_plot,
                             setup_peakstats]


@planify
def mesh(*args, time=None, md=None):
    """
    Scan over a mesh; each motor is on an independent trajectory.

    Parameters
    ----------
    *args
        patterned like (``motor1, start1, stop1, num1,```
                        ``motor2, start2, stop2, num2,,``
                        ``motor3, start3, stop3, num3,,`` ...
                        ``motorN, startN, stopN, numN,``)

        The first motor is the "slowest", the outer loop.
    md : dict, optional
        metadata
    """
    if md is None:
        md = {}
    md = ChainMap(md, {'plan_name': 'mesh',
                       gs.MD_TIME_KEY: time})
    if len(args) % 4 != 0:
        raise ValueError("wrong number of positional arguments")
    motors = []
    shape = []
    extents = []
    for motor, start, stop, num, in chunked(args, 4):
        motors.append(motor)
        shape.append(num)
        extents.append([start, stop])

    # shape goes in (rr, cc)
    # extents go in (x, y)
    subs = construct_subs('mesh', motors, gs.DETS, shape=shape,
                           extent=list(chain(*extents[::-1])))

    # outer_product_scan expects a 'snake' param for all but fist motor
    chunked_args = iter(chunked(args, 4))
    new_args = list(next(chunked_args))
    for chunk in chunked_args:
        new_args.extend(list(chunk) + [False])

    plan_stack = deque()
    with subs_context(plan_stack, subs):
        plan = outer_product_scan(gs.DETS, *new_args, md=md)
        plan = baseline_wrapper(plan, motors + gs.BASELINE_DEVICES)
        plan = configure_count_time_wrapper(plan, time)
        plan_stack.append(plan)
    return plan_stack

gs.SUB_FACTORIES['mesh'] = [setup_livetable, setup_liveraster]


@planify
def a2scan(*args, time=None, md=None):
    """
    Scan over one multi-motor trajectory.

    Parameters
    ----------
    *args
        patterned like (``motor1, start1, stop1,`` ...,
                        ``motorN, startN, stopN, intervals``)
        where 'intervals' in the number of strides (number of points - 1)
        Motors can be any 'setable' object (motor, temp controller, etc.)
    time : float, optional
        applied to any detectors that have a `count_time` setting
    md : dict, optional
        metadata
    """
    if md is None:
        md = {}
    md = ChainMap(md, {'plan_name': 'a2scan',
                       gs.MD_TIME_KEY: time})
    if len(args) % 3 != 1:
        raise ValueError("wrong number of positional arguments")
    motors = []
    for motor, start, stop, in chunked(args[:-1], 3):
        motors.append(motor)

    subs = construct_subs('a2scan', motors, gs.DETS)

    intervals = list(args)[-1]
    num = 1 + intervals

    plan_stack = deque()
    with subs_context(plan_stack, subs):
        plan = inner_product_scan(gs.DETS, num, *args[:-1], md=md)
        plan = baseline_wrapper(plan, motors + gs.BASELINE_DEVICES)
        plan = configure_count_time_wrapper(plan, time)
        plan_stack.append(plan)
    return plan_stack
gs.SUB_FACTORIES['a2scan'] = [setup_livetable,
                              setup_plot,
                              setup_peakstats]


# This implementation works for *all* dimensions, but we follow SPEC naming.
def a3scan(*args, time=None, md=None):
    if md is None:
        md = {}
    md = ChainMap(md, {'plan_name': 'a3scan'})
    return (yield from a2scan(*args, time=time, md=md))
a3scan.__doc__ = a2scan.__doc__


@planify
def d2scan(*args, time=None, md=None):
    """
    Scan over one multi-motor trajectory relative to current positions.

    Parameters
    ----------
    *args
        patterned like (``motor1, start1, stop1,`` ...,
                        ``motorN, startN, stopN, intervals``)
        where 'intervals' in the number of strides (number of points - 1)
        Motors can be any 'setable' object (motor, temp controller, etc.)
    time : float, optional
        applied to any detectors that have a `count_time` setting
    md : dict, optional
        metadata
    """
    if md is None:
        md = {}
    md = ChainMap(md, {'plan_name': 'd2scan',
                       gs.MD_TIME_KEY: time})
    if len(args) % 3 != 1:
        raise ValueError("wrong number of positional arguments")
    motors = []
    for motor, start, stop, in chunked(args[:-1], 3):
        motors.append(motor)
    subs = construct_subs('d2scan', motors, gs.DETS)
    intervals = list(args)[-1]
    num = 1 + intervals

    plan_stack = deque()
    with subs_context(plan_stack, subs):
        plan = relative_inner_product_scan(gs.DETS, num, *(args[:-1]), md=md)
        plan = baseline_wrapper(plan, motors + gs.BASELINE_DEVICES)
        plan = configure_count_time_wrapper(plan, time)
        plan_stack.append(plan)
    return plan_stack
gs.SUB_FACTORIES['d2scan'] = [setup_livetable,
                              setup_plot,
                              setup_peakstats]


# This implementation works for *all* dimensions, but we follow SPEC naming.
def d3scan(*args, time=None, md=None):
    if md is None:
        md = {}
    md = ChainMap(md, {'plan_name': 'd3scan'})
    return (yield from d2scan(*args, time=time, md=md))
d3scan.__doc__ = d2scan.__doc__


@planify
def th2th(start, finish, intervals, time=None, *, md=None):
    """
    Scan the theta and two-theta motors together.

    gs.TTH_MOTOR scans from ``start`` to ``finish`` while gs.TH_MOTOR scans
    from ``start/2`` to ``finish/2``.

    Parameters
    ----------
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    start : float
        starting position of motor
    finish : float
        ending position of motor
    intervals : int
        number of strides (number of points - 1)
    time : float, optional
        applied to any detectors that have a `count_time` setting
    md : dict, optional
        metadata
    """
    if md is None:
        md = {}
    md = ChainMap(md, {'plan_name': 'th2th'})
    plan = d2scan(gs.TTH_MOTOR, start, finish,
                  gs.TH_MOTOR, start/2, finish/2,
                  intervals, time=time, md=md)
    return [plan]


@planify
def tw(motor, step, time=None, *, md=None):
    """
    Move and motor and read a detector with an interactive prompt.

    ``gs.MASTER_DET`` must be set to a detector, and ``gs.MASTER_DET_FIELD``
    must be set the name of the field to be watched.

    Parameters
    ----------
    motor : Device
    step : float
        initial suggestion for step size
    md : dict, optional
        metadata
    """
    if md is None:
        md = {}
    md = ChainMap(md, {'plan_name': 'tw',
                       gs.MD_TIME_KEY: time})
    plan = tweak(gs.MASTER_DET, gs.MASTER_DET_FIELD, motor, step, md=md)
    plan = baseline_wrapper(plan, [motor] + gs.BASELINE_DEVICES)
    plan = configure_count_time_wrapper(plan, time)
    return [plan]


@planify
def afermat(x_motor, y_motor, x_start, y_start, x_range, y_range, dr, factor,
            time=None, *, per_step=None, md=None):
    '''Absolute fermat spiral scan, centered around (0, 0)

    Parameters
    ----------
    x_motor : object
        any 'setable' object (motor, temp controller, etc.)
    y_motor : object
        any 'setable' object (motor, temp controller, etc.)
    x_start : float
        x center
    y_start : float
        y center
    x_range : float
        x range of spiral
    y_range : float
        y range of spiral
    dr : float
        delta radius
    factor : float
        radius gets divided by this
    time : float, optional
        applied to any detectors that have a `count_time` setting
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        See docstring of bluesky.plans.one_nd_step (the default) for
        details.
    md : dict, optional
        metadata

    See Also
    --------
    `bluesky.spec_api.fermat`
    `bluesky.spec_api.aspiral`
    `bluesky.spec_api.spiral`
    '''
    if md is None:
        md = {}
    md = ChainMap(md, {'plan_name': 'afermat',
                       gs.MD_TIME_KEY: time})
    subs = construct_subs('afermat', [x_motor, y_motor], gs.DETS)

    plan_stack = deque()
    with plans.subs_context(plan_stack, subs):
        plan = plans.spiral_fermat(gs.DETS, x_motor, y_motor, x_start, y_start,
                                   x_range, y_range, dr, factor,
                                   per_step=per_step, md=md)
        plan = baseline_wrapper(plan, [x_motor, y_motor] + gs.BASELINE_DEVICES)
        plan = configure_count_time_wrapper(plan, time)
        plan_stack.append(plan)
    return plan_stack
gs.SUB_FACTORIES['afermat'] = [setup_livetable]


def fermat(x_motor, y_motor, x_range, y_range, dr, factor, time=None, *,
           per_step=None, md=None):
    '''Relative fermat spiral scan

    Parameters
    ----------
    x_motor : object
        any 'setable' object (motor, temp controller, etc.)
    y_motor : object
        any 'setable' object (motor, temp controller, etc.)
    x_range : float
        x range of spiral
    y_range : float
        y range of spiral
    dr : float
        delta radius
    factor : float
        radius gets divided by this
    time : float, optional
        applied to any detectors that have a `count_time` setting
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        See docstring of bluesky.plans.one_nd_step (the default) for
        details.
    md : dict, optional
        metadata

    See Also
    --------
    `bluesky.spec_api.afermat`
    `bluesky.spec_api.aspiral`
    `bluesky.spec_api.spiral`
    '''
    if md is None:
        md = {}
    md = ChainMap(md, {'plan_name': 'fermat',
                       gs.MD_TIME_KEY: time})
    plan = afermat(x_motor, y_motor, x_motor.position, y_motor.position,
                   x_range, y_range, dr, factor, time=time, per_step=per_step,
                   md=md)
    plan = plans.reset_positions_wrapper(plan)  # return motors to starting pos
    yield from plan


@planify
def aspiral(x_motor, y_motor, x_start, y_start, x_range, y_range, dr, nth,
            time=None, *, per_step=None, md=None):
    '''Spiral scan, centered around (x_start, y_start)

    Parameters
    ----------
    x_motor : object
        any 'setable' object (motor, temp controller, etc.)
    y_motor : object
        any 'setable' object (motor, temp controller, etc.)
    x_range : float
        X range, in engineering units
    y_range : float
        Y range, in engineering units
    dr : float
        Delta radius, in engineering units
    nth : float
        Number of theta steps
    time : float, optional
        applied to any detectors that have a `count_time` setting
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        See docstring of bluesky.plans.one_nd_step (the default) for
        details.
    md : dict, optional
        metadata

    See Also
    --------
    `bluesky.spec_api.fermat`
    `bluesky.spec_api.afermat`
    `bluesky.spec_api.spiral`
    '''
    if md is None:
        md = {}
    md = ChainMap(md, {'plan_name': 'aspiral',
                       gs.MD_TIME_KEY: time})
    subs = construct_subs('aspiral', [x_motor, y_motor], gs.DETS)

    plan_stack = deque()
    with plans.subs_context(plan_stack, subs):
        plan = plans.spiral(gs.DETS, x_motor, y_motor, x_start, y_start,
                            x_range, y_range, dr, nth, per_step=per_step,
                            md=md)
        plan = baseline_wrapper(plan, [x_motor, y_motor] + gs.BASELINE_DEVICES)
        plan = configure_count_time_wrapper(plan, time)
        plan_stack.append(plan)
    return plan_stack
gs.SUB_FACTORIES['aspiral'] = [setup_livetable]


def spiral(x_motor, y_motor, x_range, y_range, dr, nth, time=None, *,
           per_step=None, md=None):
    '''Relative spiral scan

    Parameters
    ----------
    x_motor : object
        any 'setable' object (motor, temp controller, etc.)
    y_motor : object
        any 'setable' object (motor, temp controller, etc.)
    x_range : float
        X range, in engineering units
    y_range : float
        Y range, in engineering units
    dr : float
        Delta radius, in engineering units
    nth : float
        Number of theta steps
    time : float, optional
        applied to any detectors that have a `count_time` setting
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        See docstring of bluesky.plans.one_nd_step (the default) for
        details.
    md : dict, optional
        metadata

    See Also
    --------
    `bluesky.spec_api.fermat`
    `bluesky.spec_api.afermat`
    `bluesky.spec_api.aspiral`
    '''
    if md is None:
        md = {}
    md = ChainMap(md, {'plan_name': 'spiral',
                       gs.MD_TIME_KEY: time})
    plan = aspiral(x_motor, y_motor, x_motor.position, y_motor.position,
                   x_range, y_range, dr, nth, time=time, per_step=per_step,
                   md=md)
    plan = plans.reset_positions_wrapper(plan)  # return to starting pos
    yield from plan
