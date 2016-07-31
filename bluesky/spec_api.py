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
from collections import deque, OrderedDict

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
                           baseline_wrapper, subs_wrapper)
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


def setup_plot(*, motors, gs):
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


def setup_ct_plot(*, num, motors, gs):
    if num is not None and num > 1:
        return setup_plot(motors=motors, gs=gs)
    return None


def setup_peakstats(*, motors, gs):
    "Set up peakstats"
    motor = list(motors)[0]
    key = first_key_heuristic(motor)
    ps = PeakStats(key, gs.MASTER_DET_FIELD, **gs.PS_CONFIG)
    gs.PS = ps
    ps.motor = motor
    return ps


def setup_livetable(*, motors,  gs):
    return LiveTable(motors + [gs.PLOT_Y] + gs.TABLE_COLS)


def setup_liveraster(*, motors, gs, shape, extent):
    if len(motors) != 2:
        return None
    ylab, xlab = [first_key_heuristic(m) for m in motors]
    raster = LiveRaster(shape, gs.MASTER_DET_FIELD, xlabel=xlab,
                        ylabel=ylab, extent=extent, clim=[0, 1])
    return raster


def construct_subs(plan_name, **kwargs):
    '''Construct the subscriptions functions from factories.


    This is a small dependency injection tool to construct subscriptions
    based on `plan_name` by consulting `gs.SUB_FACTORIES` and combining
    `'common'` and `plan_name` lists.

    The values in `**kwargs` are injected into the factories as needed
    by consulting the signature.

    Parameters
    ----------
    plan_name : str
        The key used to get the subscription factories

    Returns
    -------
    subs : dict
       A dictionary suitable for handing to the RE or to the sub_wrapper
       plan.
    '''
    factories = gs.SUB_FACTORIES.get('common', [])
    factories.extend(gs.SUB_FACTORIES.get(plan_name, []))
    subs = []
    kwargs.setdefault('gs', gs)

    for factory in factories:
        # TODO, support positional injection
        sig = signature(factory)
        fact_kwargs = {}
        missing_kwargs = set()
        missing_args = set()
        for k, v in sig.parameters.items():
            if v.kind is Parameter.POSITIONAL_ONLY:
                if v.defaut is Parameter.empty:
                    missing_args.add(k)
            else:
                if k in kwargs:
                    fact_kwargs[k] = kwargs[k]
                else:
                    if v.default is Parameter.empty:
                        missing_kwargs.add(k)

        if missing_kwargs:
            warnings.warn('The factory {fn} could not be run due to missing '
                          'mandatory kwargs {missing!r}'.format(
                              fn=factory.__name__, missing=missing_kwargs))
        elif missing_args:
            warnings.warn('The factory {fn} could not be run due to having '
                          'positional only args {missing!r}'.format(
                              fn=factory.__name__, missing=missing_kwargs))
        else:
            l_sub = factory(**fact_kwargs)
            if l_sub is None:
                continue
            elif callable(l_sub):
                subs.append(l_sub)
            else:
                subs.extend(l_sub)

    return {'all': subs}


# TODO rename this
def plan_sub_factory_input(plan_name):
    '''Helper function for introspecting sub_factories

    Assume this will be re-written, do not program against.
    '''
    factories = gs.SUB_FACTORIES.get('common', [])
    factories.extend(gs.SUB_FACTORIES.get(plan_name, []))
    opt = set()
    req = set()
    by_fac = OrderedDict()
    for fac in factories:
        try:
            ret = get_sub_factory_input(fac)
        except InvalidFactory as e:
            by_fac[fac.__name__] = e
        else:
            opt.update(ret['opt'])
            req.update(ret['req'])
            by_fac[fac.__name__] = ret

    return {'opt': opt, 'req': req}, by_fac


class InvalidFactory(TypeError):
    ...


# TODO rename this
def get_sub_factory_input(factory):
    '''Helper function to re-organize signature information

    Assume this will be re-written, do not program against.
    '''
    sig = signature(factory)
    reqed = set()
    optional = set()
    for k, v in sig.parameters.items():
        if v.kind is Parameter.POSITIONAL_ONLY:
            # TODO make this nice?
            raise InvalidFactory()
        elif v.default is Parameter.empty:
            reqed.add(k)
        else:
            optional.add(k)
    return {'req': reqed, 'opt': optional}


def inner_spec_decorator(plan_name, time, motors, **subs_kwargs):
    subs = construct_subs(plan_name, motors=motors, **subs_kwargs)

    # create the paramterized decorator
    def outer(func):
        # create the decorated function to return
        def inner(*args, md=None, **kwargs):
            # inject the plan name + time into the metadata
            if md is None:
                md = {}
            md = ChainMap(md, {'plan_name': plan_name,
                               gs.MD_TIME_KEY: time})
            # get the 'base' plan
            plan = func(*args, md=md, **kwargs)
            # set up the baseline measurements
            plan = baseline_wrapper(plan, list(gs.BASELINE_DEVICES) + motors)
            # set up the 'count time' measurements'
            plan = configure_count_time_wrapper(plan, time)
            # set up the subscriptions
            plan = subs_wrapper(plan, subs)
            # yield from the whole thing
            return (yield from plan)
        return inner
    return outer

# ## Counts (p. 140) ###


def ct(num=1, delay=None, time=None, *, md=None, **kwargs):
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

    @inner_spec_decorator('ct', time, [], num=num, **kwargs)
    def inner(*args, **kwargs):
        return (yield from count(*args, **kwargs))

    return (yield from inner(gs.DETS, num, delay, md=md))
gs.SUB_FACTORIES['ct'] = [setup_livetable, setup_ct_plot]


# ## Motor Scans (p. 146) ###

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
    motors = [motor]

    @inner_spec_decorator('ascan', time, motors)
    def inner(*args, **kwargs):
        return (yield from scan(*args, **kwargs))

    return (yield from inner(gs.DETS, motor, start, finish,
                             1 + intervals, md=md))
gs.SUB_FACTORIES['ascan'] = [setup_livetable,
                             setup_plot,
                             setup_peakstats]


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
    motors = [motor]

    @inner_spec_decorator('dscan', time, motors)
    def inner(*args, **kwargs):
        return (yield from relative_scan(*args, **kwargs))

    return (yield from inner(gs.DETS, motor, start, finish, 1 + intervals,
                             md=md))
gs.SUB_FACTORIES['dscan'] = [setup_livetable,
                             setup_plot,
                             setup_peakstats]


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
    if len(args) % 4 != 0:
        raise ValueError("wrong number of positional arguments")
    motors = []
    shape = []
    extents = []
    for motor, start, stop, num, in chunked(args, 4):
        motors.append(motor)
        shape.append(num)
        extents.append([start, stop])

    # outer_product_scan expects a 'snake' param for all but fist motor
    chunked_args = iter(chunked(args, 4))
    new_args = list(next(chunked_args))
    for chunk in chunked_args:
        new_args.extend(list(chunk) + [False])

    # shape goes in (rr, cc)
    # extents go in (x, y)
    @inner_spec_decorator('mesh', time, motors=motors, shape=shape,
                          extent=list(chain(*extents[::-1])))
    def inner(*args, **kwargs):
        return (yield from outer_product_scan(*args, **kwargs))

    return (yield from inner(gs.DETS, *new_args, md=md))
gs.SUB_FACTORIES['mesh'] = [setup_livetable, setup_liveraster]


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
    if len(args) % 3 != 1:
        raise ValueError("wrong number of positional arguments")
    motors = []
    for motor, start, stop, in chunked(args[:-1], 3):
        motors.append(motor)

    intervals = list(args)[-1]
    num = 1 + intervals

    @inner_spec_decorator('a2scan', time, motors=motors)
    def inner(*args, **kwargs):
        return (yield from inner_product_scan(*args, **kwargs))

    return (yield from inner(gs.DETS, num, *args[:-1], md=md))

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
gs.SUB_FACTORIES['a3scan'] = gs.SUB_FACTORIES['a2scan']


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
    if len(args) % 3 != 1:
        raise ValueError("wrong number of positional arguments")
    motors = []
    for motor, start, stop, in chunked(args[:-1], 3):
        motors.append(motor)

    intervals = list(args)[-1]
    num = 1 + intervals

    @inner_spec_decorator('d2scan', time, motors=motors)
    def inner(*args, **kwargs):
        return (yield from relative_inner_product_scan(*args, **kwargs))

    return (yield from inner(gs.DETS, num, *(args[:-1]), md=md))

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
gs.SUB_FACTORIES['d3scan'] = gs.SUB_FACTORIES['d2scan']


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
    return (yield from plan)
gs.SUB_FACTORIES['th2th'] = gs.SUB_FACTORIES['d2scan']


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
    @inner_spec_decorator('tw', time, motors=[motor])
    def inner(*args, **kwargs):
        return (yield from tweak(*args, **kwargs))

    return (yield from inner(gs.MASTER_DET, gs.MASTER_DET_FIELD, motor,
                             step, md=md))


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
    motors = [x_motor, y_motor]

    @inner_spec_decorator('afermat', time, motors=motors)
    def inner(*args, **kwargs):
        return (yield from plans.spiral_fermat(*args, **kwargs))

    return (yield from inner(gs.DETS, x_motor, y_motor, x_start, y_start,
                             x_range, y_range, dr, factor,
                             per_step=per_step, md=md))
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
    md = ChainMap(md, {'plan_name': 'fermat'})
    plan = afermat(x_motor, y_motor, x_motor.position, y_motor.position,
                   x_range, y_range, dr, factor, time=time, per_step=per_step,
                   md=md)
    plan = plans.reset_positions_wrapper(plan)  # return motors to starting pos
    return (yield from plan)
gs.SUB_FACTORIES['fermat'] = gs.SUB_FACTORIES['afermat']


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
    motors = [x_motor, y_motor]

    @inner_spec_decorator('aspiral', time, motors=motors)
    def inner(*args, **kwargs):
        return (yield from plans.spiral(*args, **kwargs))

    return (yield from inner(gs.DETS, x_motor, y_motor, x_start, y_start,
                             x_range, y_range, dr, nth, per_step=per_step,
                             md=md))
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
    md = ChainMap(md, {'plan_name': 'spiral'})
    plan = aspiral(x_motor, y_motor, x_motor.position, y_motor.position,
                   x_range, y_range, dr, nth, time=time, per_step=per_step,
                   md=md)
    plan = plans.reset_positions_wrapper(plan)  # return to starting pos
    return (yield from plan)
gs.SUB_FACTORIES['spiral'] = gs.SUB_FACTORIES['aspiral']
