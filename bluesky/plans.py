import uuid
import sys
from functools import wraps
import itertools
import functools
import operator
from contextlib import contextmanager
from collections import OrderedDict, Iterable, defaultdict, deque

import numpy as np
from cycler import cycler
from boltons.iterutils import chunked
import matplotlib.pyplot as plt
from matplotlib import collections as mcollections
from matplotlib import patches as mpatches

from .run_engine import Msg
from .callbacks import LiveTable, LivePlot
from .utils import (Struct, snake_cyclers, Subs, normalize_subs_input,
                    separate_devices, apply_sub_factories, update_sub_lists)


def _short_uid(label, truncate=6):
    "Return a readable but unique id like 'label-fjfi5a'"
    return '-'.join([label, str(uuid.uuid4())[:truncate]])


def planify(func):
    """
    Turn a function that returns a list of generators into a coroutine.

    Parameters
    ----------
    func : callable
        expected to return a list of generators that yield Msg objects;
        the function may have an arbitrary signature
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        gen_stack = func(*args, **kwargs)
        for g in gen_stack:
            yield from g

    return wrapped


@context_manager
def subs_context(genstack, subs):
    """
    Subscribe to callbacks to the document stream; then unsubscribe on exit.

    Parameters
    ----------
    genstack : collection
        collection of generators that yield messages
    subs : callable, list of callables, or dict of lists of callables
         Documents of each type are routed to a list of functions.
         Input is normalized to a dict of lists of functions, like so:

         None -> {'all': [], 'start': [], 'stop': [], 'event': [],
                  'descriptor': []}

         func -> {'all': [func], 'start': [], 'stop': [], 'event': [],
                  'descriptor': []}

         [f1, f2] -> {'all': [f1, f2], 'start': [], 'stop': [], 'event': [],
                      'descriptor': []}

         {'event': [func]} ->  {'all': [], 'start': [], 'stop': [],
                                'event': [func], 'descriptor': []}

         Signature of functions must confirm to `f(name, doc)` where
         name is one of {'all', 'start', 'stop', 'event', 'descriptor'} and
         doc is a dictionary.
    """
    subs = normalize_subs_input(subs)
    tokens = set()

    def subscribe():
        for name, funcs in subs.items():
            for func in funcs:
                token = yield Msg('subscribe', None, name, func)
                tokens.add(token)

    def unsubscribe():
        for token in tokens:
            yield Msg('unsubscribe', None, token)

    genstack.append(subscribe())
    try:
        yield genstack
    finally:
        # The RunEngine might never process these if the execution fails,
        # but it keeps its own cache of tokens and will try to remove them
        # itself if this plan fails to do so.
        genstack.append(unsubscribe())


def single_gen(x):
    """Utility to wrapping a single object in a generator

    If ``lambda x: yield x`` were valid Python, this would be equivalent.
    """
    yield x


@contextmanager
def run_context(genstack, md=None):
    """Enclose in 'open_run' and 'close_run' messages.

    Parameters
    ----------
    genstack : collection
        collection of generators that yield messages
    md : dict, optional
        metadata to be passed into the 'open_run' message

    Yields
    ------
    Msg
    """
    if md is None:
        md = dict()
    md = dict(md)
    genstack.append(single_gen(Msg('open_run', None, **md)))
    try:
        yield genstack

    # This block is an example of how custom exception handling can be
    # inserted. Without any handling in the plan itself, the RunEngine will
    # close the run and mark it as errored.
    except Exception:
        genstack.append(single_gen(Msg('close_run', None,
                                       exit_status='error')))
        raise

    else:
        genstack.append(single_gen(Msg('close_run')))


@context_manager
def event_context(genstack, name='primary'):
    """Bundle readings into an 'event' (a datapoint).

    This encloses the contents in 'create' and 'save' messages.

    Parameters
    ----------
    genstack : collection
        collection of generators that yield messages
    name : string, optional
        name of event stream; default is 'primary'
    """
    genstack.append(single_gen(Msg('create', None, name=name)))
    yield genstack
    genstack.append(single_gen(Msg('save')))


def simple_preprocessor(func, cleanup_msgs=None):
    """
    Create generator wrapper than mutates or inserts messages.

    This utility makes simple cases easier to write, but it does not provide
    access to the *result* of processing the messages -- for, say, an
    adaptive plan. This utility only re-writes messsages on the way in.

    For examples, see ``relative_set``, ``put_back``, and ``fly_during``. For
    an example where this utility is *not* sufficiently general, see
    ``stage_wrapper``.

    Parameters
    ----------
    func : callable
        expected_signature: ``f(msg) -> msgs_before, original_msg, msgs_after``
    cleanup_msgs : list, optional
        list of cleanup-related messages to yield in finally block
    """
    if cleanup_msgs is None:
        cleanup_msgs = []
    def f(plan):
        try:
            ret = None
            while True:
                msg = plan.send(ret)
                before, main, after = func(msg)
                yield from before
                ret = yield main
                yield from after
        finally:
            for msg in cleanup_msgs:
                yield msg
        return ret
    return f


def fly_during(plan, flyers):
    """
    Kickoff and collect "flyer" (asynchronously collect) objects during runs.

    Parameters
    ----------
    plan : iterable
    flyers : iterable
        objects that support the flyer interface
    """
    grp = _short_uid('flyers')
    kickoff_msgs = [Msg('kickoff', flyer, block_group=grp) for flyer in flyers]
    collect_msgs = [Msg('collect', flyer) for flyer in flyers]
    if flyers:
        collect_msgs = [Msg('wait', None, grp)] + collect_msgs

    def insert_after_open(msg):
        if msg.command == 'open_run':
            return [], msg, kickoff_msgs
        else:
            return [], msg, []

    def insert_before_close(msg):
        if msg.command == 'close_run':
            return collect_msgs, msg, []
        else:
            return [], msg, []

    plan = simple_preprocessor(insert_after_open)(plan)
    plan = simple_preprocessor(insert_before_close)(plan)
    ret = yield from plan
    return ret


def stage_context(genstack, detectors):
    """
    This is a preprocessor that inserts 'stage' Messages.

    The first time an object is read, set, triggered (or, for flyers,
    the first time is it told to "kickoff") a 'stage' Msg is inserted first.
    It stages the the object's ultimate parent, pointed to be its `root`
    property.

    At the end, an 'unstage' Message issued for every 'stage' Message.
    """
    COMMANDS = set(['read', 'set', 'trigger', 'kickoff'])
    # Cache devices in the order they are staged; then unstage in reverse.
    devices_staged = []
    genstack.append(stage())

    def unstage():
        yield from broadcast_msg('unstage', reversed(devices_staged))

    def stage():
        # ???
        pass

    ####
    ret = None
    try:
        while True:
            msg = plan.send(ret)
            if msg.command in COMMANDS and msg.obj not in devices_staged:
                root = msg.obj.root
                ret = yield Msg('stage', root)
                if ret is None:
                    # The generator may be being list-ified.
                    # This is a hack to make that possible. Is it a good idea?
                    ret = [root]
                devices_staged.extend(ret)
            ret = yield msg
    ###
        yield genstack
    finally:
        genstack.append(unstage())


def relative_set(plan, devices=None):
    """
    Interpret 'set' messages on devices as relative to initial position.

    Parameters
    ----------
    plan : iterable
    devices : iterable or None, optional
        if default (None), apply to all devices that are moved by the plan
    """
    initial_positions = {}
    def f(msg):
        if msg.command == 'set' and (devices is None or msg.obj in devices):
            if msg.obj not in initial_positions:
                pos = msg.obj.position
                initial_positions[msg.obj] = pos
            rel_pos, = msg.args
            abs_pos = initial_positions[msg.obj] + rel_pos
            new_msg = msg._replace(args=(abs_pos,))
            return [], new_msg, []
        else:
            return [], msg, []
    plan = simple_preprocessor(f)(plan)
    ret = yield from plan
    return ret


def put_back(plan, devices=None):
    """
    Return movable devices to the original positions at the end.

    Parameters
    ----------
    plan : iterable
    devices : iterable or None, optional
        if default (None), apply to all devices that are moved by the plan
    """
    initial_positions = OrderedDict()  # local var for debugging purposes
    grp = _short_uid('put_back')
    cleanup_msgs = [Msg('wait', None, grp)]
    def f(msg):
        obj = msg.obj
        if msg.command == 'set' and (devices is None or obj in devices):
            if obj not in initial_positions:
                pos = obj.position
                initial_positions[obj] = pos
                cleanup_msgs.insert(0, Msg('set', obj, pos, block_group=grp))
        return [], msg, []
    plan = simple_preprocessor(f, cleanup_msgs)(plan)
    ret = yield from plan
    return ret


def wrap_with_decorator(wrapper, *outer_args, **outer_kwargs):
    """Paramaterized decorator for wrapping generators with wrappers

    The wrapped function must be a generator and wrapper wrap an
    iterable.
    """
    def outer(func):
        @wraps(func)
        def inner(*args, **kwargs):
            ret = yield from wrapper(func(*args, **kwargs),
                                     *outer_args, **outer_kwargs)
            return ret
        return inner
    return outer


def trigger_and_read(devices):
    """Trigger and read a list of detectors.

    Parameters
    ----------
    devices : iterable
        devices to trigger (if they have a trigger method) and then read
    """
    devices = separate_devices(devices)  # remove redundant entries
    grp = _short_uid('trigger')
    for obj in separate_devices(devices):
        if hasattr(obj, 'trigger'):
            yield Msg('trigger', obj, block_group=grp)
    yield Msg('wait', None, grp)
    for obj in devices:
        yield Msg('read', obj)


def broadcast_msg(command, objs, *args, **kwargs):
    """
    Generate many copies of a mesasge, applying it to a list of devices.

    Parameters
    ----------
    command : string
    devices : iterable
    *args
        args for message
    **kwargs
        kwargs for message
    """
    return_vals = []
    for o in objs:
        ret = yield Msg(command, o, *args, **kwargs)
        return_vals.append(ret)

    return return_vals


def repeater(n, gen_func, *args, **kwargs):
    """
    Generate n chained copies of the messages from gen_func
    
    Parameters
    ----------
    n : int
        total number of repetitions
    gen_func : callable
        returns generator instance
    *args
        args for gen_func
    **kwargs
        kwargs for gen_func
    """
    it = range
    if n is None:
        n = 0
        it = itertools.count

    for j in it(n):
        yield from gen_func(*args, **kwargs)


def caching_repeater(n, plan):
    """
    Generate n chained copies of the messages in a plan.

    This is different from ``repeater`` above because it takes in a
    generator or iterator, not a function that returns one.

    Parameters
    ----------
    n : int
        total number of repetitions
    plan : iterable
    """
    lst_plan = list(plan)
    for j in range(n):
        yield from (m for m in lst_plan)


@planify
def count(detectors, num=1, delay=None, *, md=None):
    """
    Take one or more readings from detectors.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    num : integer, optional
        number of readings to take; default is 1
    delay : iterable or scalar, optional
        time delay between successive readings; default is 0
    md : dict, optional
        metadata

    Note
    ----
    If ``delay`` is an iterable, it must have at least ``num`` entries or the
    plan will raise a ``StopIteration`` error.
    """
    if md is None:
        md = {}
    md.update({'detectors': [det.name for det in detectors]})
    md['plan_args'] = {'detectors': repr(detectors), 'num': num}

    if num is None:
        counter = itertools.count()  # run forever, until interrupted
    else:
        counter = range(num)

    # If delay is a scalar, repeat it forever. If it is an iterable, leave it.
    if not isinstance(delay, Iterable):
        delay = itertools.repeat(delay)
    else:
        delay = iter(delay)

    genstack = deque()
    with stage_context(genstack, detectors):
        with run_context(genstack, md):
            for _ in counter:
                genstack.append(single_gen(Msg('checkpoint')))
                with event_context(genstack):
                    genstack.append(trigger_and_read(detectors))
                d = next(delay)
                if d is not None:
                    genstack.append(single_gen(Msg('sleep', None, d)))
    return genstack


def _one_step(detectors, motor, step):
    """
    Inner loop of a 1D step scan
    
    This is the default function for ``per_step`` param`` in 1D plans.
    """
    grp = _short_uid('set')
    yield Msg('checkpoint')
    yield Msg('set', motor, step, block_group=grp)
    yield Msg('wait', None, grp)
    ret = yield from trigger_and_read(list(detectors) + [motor])
    return ret


def _step_scan_core(detectors, motor, steps, *, per_step=None):
    "Just the steps. This should be wrapped in stage_wrapper, run_context."
    if per_step is None:
        per_step = _one_step
    for step in steps:
        ret = yield from per_step(detectors, motor, step)
    return ret


def list_scan(detectors, motor, steps, *, per_step=None, md=None):
    """
    Scan over one variable in steps.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    steps : list
        list of positions
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        Expected signature: ``f(detectors, motor, step)`` 
    md : dict, optional
        metadata
    """
    if md is None:
        md = {}
    md.update({'detectors': [det.name for det in detectors],
               'motors': [motor.name]})
    md['plan_args'] = {'detectors': repr(detectors), 'steps': steps}

    plan = _step_scan_core(detectors, motor, steps, per_step=per_step)
    plan = stage_wrapper(plan)
    plan = run_context(plan, md)
    ret = yield from plan
    return ret


def relative_list_scan(detectors, motor, steps, *, per_step=None, md=None):
    """
    Scan over one variable in steps relative to current position.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    motor : object
        any 'setable' object (motor, temp controller, etc.)
    steps : list
        list of positions relative to current position
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        Expected signature: ``f(detectors, motor, step)`` 
    md : dict, optional
        metadata
    """
    # TODO read initial positions (redundantly) so they can be put in md here
    plan = list_scan(detectors, motor, steps, per_step=per_step, md=md)
    plan = relative_set(plan, [motor])  # re-write trajectory as relative
    plan = put_back(plan, [motor])  # return motors to starting pos
    ret = yield from plan
    return ret


def scan(detectors, motor, start, stop, num, *, per_step=None, md=None):
    """
    Scan over one variable in equally spaced steps.

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
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        Expected signature: ``f(detectors, motor, step)`` 
    md : dict, optional
        metadata
    """
    if md is None:
        md = {}
    md.update({'detectors': [det.name for det in detectors],
               'motors': [motor.name]})
    md['plan_args'] = {'detectors': repr(detectors), 'num': num,
                       'start': start, 'stop': stop}

    steps = np.linspace(start, stop, num)
    plan = _step_scan_core(detectors, motor, steps, per_step=per_step)
    plan = stage_wrapper(plan)
    plan = run_context(plan, md)
    ret = yield from plan
    return ret


def relative_scan(detectors, motor, start, stop, num, *, per_step=None,
                  md=None):
    """
    Scan over one variable in equally spaced steps relative to current positon.

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
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        Expected signature: ``f(detectors, motor, step)`` 
    md : dict, optional
        metadata
    """
    # TODO read initial positions (redundantly) so they can be put in md here
    plan = scan(detectors, motor, start, stop, num, per_step=per_step, md=md)
    plan = relative_set(plan, [motor])  # re-write trajectory as relative
    plan = put_back(plan, [motor])  # return motors to starting pos
    ret = yield from plan


def log_scan(detectors, motor, start, stop, num, *, per_step=None, md=None):
    """
    Scan over one variable in log-spaced steps.

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
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        Expected signature: ``f(detectors, motor, step)`` 
    md : dict, optional
        metadata
    """
    if md is None:
        md = {}
    md.update({'detectors': [det.name for det in detectors],
               'motors': [motor.name]})
    md['plan_args'] = {'detectors': repr(detectors), 'num': num,
                       'start': start, 'stop': stop}

    steps = np.logspace(start, stop, num)
    plan = _step_scan_core(detectors, motor, steps, per_step=per_step)
    plan = stage_wrapper(plan)
    plan = run_context(plan, md)
    ret = yield from plan
    return ret


def relative_log_scan(detectors, motor, start, stop, num, *, per_step=None,
                      md=None):
    """
    Scan over one variable in log-spaced steps relative to current position.

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
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        Expected signature: ``f(detectors, motor, step)`` 
    md : dict, optional
        metadata
    """
    # TODO read initial positions (redundantly) so they can be put in md here
    plan = log_scan(detectors, motor, start, stop, num, per_step=per_step,
                    md=md)
    plan = relative_set(plan, [motor])  # re-write trajectory as relative
    plan = put_back(plan, [motor])  # return motors to starting pos
    ret = yield from plan
    return ret


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
    threshold : float, optional
        threshold for going backward and rescanning a region, default is 0.8
    md : dict, optional
        metadata
    """
    def core():
        next_pos = start
        step = (max_step - min_step) / 2
        past_I = None
        cur_I = None
        cur_det = {}
        while next_pos < stop:
            yield Msg('checkpoint')
            yield Msg('set', motor, next_pos)
            yield Msg('wait', None, 'A')
            yield Msg('create', None, name='primary')
            for det in detectors:
                yield Msg('trigger', det, block_group='B')
            yield Msg('wait', None, 'B')
            for det in separate_devices(detectors + [motor]):
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
            next_pos += step
    plan = core()
    plan = stage_wrapper(plan)
    plan = run_context(plan, md)
    ret = yield from plan
    return ret


def relative_adaptive_scan(detectors, target_field, motor, start, stop,
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
    threshold : float, optional
        threshold for going backward and rescanning a region, default is 0.8
    md : dict, optional
        metadata
    """
    plan = adaptive_scan(detectors, target_field, motor, start, stop,
                         min_step, max_step, target_delta, backstep,
                         threshold, md=md)
    plan = relative_set(plan, [motor])  # re-write trajectory as relative
    plan = put_back(plan, [motor])  # return motors to starting pos
    ret = yield from plan
    return ret


def _one_nd_step(detectors, step, pos_cache):
    """
    Inner loop of an N-dimensional step scan
    
    This is the default function for ``per_step`` param`` in ND plans.

    Parameters
    ----------
    detectors : iterable
        devices to read
    step : dict
        mapping motors to positions in this step
    pos_cache : dict
        mapping motors to their last-set positions
    """
    yield Msg('checkpoint')
    for motor, pos in step.items():
        grp = _short_uid('set')
        if pos == pos_cache[motor]:
            # This step does not move this motor.
            continue
        yield Msg('set', motor, pos, block_group=grp)
        pos_cache[motor] = pos
        yield Msg('wait', None, grp)
    motors = step.keys()
    ret = yield from trigger_and_read(list(detectors) + list(motors))


def _nd_step_scan_core(detectors, cycler, per_step=None):
    """
    Just the steps. This should be wrapped in stage_wrapper, run_context.

    See ``plan_nd`` below.
    """
    if per_step is None:
        per_step = _one_nd_step
    motors = cycler.keys
    last_pos = defaultdict(lambda: None)
    for step in list(cycler):
        ret = yield from per_step(detectors, step, last_pos)
    return ret


def plan_nd(detectors, cycler, *, per_step=None, md=None):
    """
    Scan over an arbitrary N-dimensional trajectory.

    Parameters
    ----------
    detectors : list
    cycler : Cycler
        list of dictionaries mapping motors to positions
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        See docstring of bluesky.plans._one_nd_step (the default) for
        details.
    md : dict, optional
        metadata
    """
    if md is None:
        md = {}
    md.update({'detectors': [det.name for det in detectors],
               'motors': [motor.name for motor in cycler.keys]})
    md['plan_args'] = {'detectors': repr(detectors), 'cycler': repr(cycler)}

    plan = _nd_step_scan_core(detectors, cycler, per_step)
    plan = stage_wrapper(plan)
    plan = run_context(plan, md)
    ret = yield from plan
    return ret


def inner_product_scan(detectors, num, *args, per_step=None, md=None):
    """
    Scan over one multi-motor trajectory.

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
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        See docstring of bluesky.plans._one_nd_step (the default) for
        details.
    md : dict, optional
        metadata
    """
    if len(args) % 3 != 0:
        raise ValueError("wrong number of positional arguments")
    cyclers = []
    for motor, start, stop, in chunked(args, 3):
        steps = np.linspace(start, stop, num=num, endpoint=True)
        c = cycler(motor, steps)
        cyclers.append(c)
    full_cycler = functools.reduce(operator.add, cyclers)

    ret = yield from plan_nd(detectors, full_cycler, per_step=per_step, md=md)
    return ret


def outer_product_scan(detectors, *args, per_step=None, md=None):
    """
    Scan over a mesh; each motor is on an independent trajectory.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    *args
        patterned like (``motor1, start1, stop1, num1,```
                        ``motor2, start2, stop2, num2, snake2,``
                        ``motor3, start3, stop3, num3, snake3,`` ...
                        ``motorN, startN, stopN, numN, snakeN``)

        The first motor is the "slowest", the outer loop. For all motors
        except the first motor, there is a "snake" argument: a boolean
        indicating whether to following snake-like, winding trajectory or a
        simple left-to-right trajectory.
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        See docstring of bluesky.plans._one_nd_step (the default) for
        details.
    md : dict, optional
        metadata
    """
    args = list(args)
    # The first (slowest) axis is never "snaked." Insert False to
    # make it easy to iterate over the chunks or args..
    args.insert(4, False)
    if len(args) % 5 != 0:
        raise ValueError("wrong number of positional arguments")
    shape = []
    extents = []
    snaking = []
    cyclers = []
    for motor, start, stop, num, snake in chunked(args, 5):
        shape.append(num)
        extents.append([start, stop])
        snaking.append(snake)
        steps = np.linspace(start, stop, num=num, endpoint=True)
        c = cycler(motor, steps)
        cyclers.append(c)
    full_cycler = snake_cyclers(cyclers, snaking)

    if md is None:
        md = {}
    md.update({'shape': tuple(shape), 'extents': tuple(extents),
               'snaking': tuple(snaking), 'num': len(full_cycler)})

    ret = yield from plan_nd(detectors, full_cycler, per_step=per_step, md=md)
    return ret


def relative_outer_product_scan(detectors, *args, per_step=None, md=None):
    """
    Scan over a mesh relative to current position.

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
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        See docstring of bluesky.plans._one_nd_step (the default) for
        details.
    md : dict, optional
        metadata
    """
    # There is some duplicate effort here to obtain the list of motors.
    _args = list(args)
    # The first (slowest) axis is never "snaked." Insert False to
    # make it easy to iterate over the chunks or args..
    _args.insert(4, False)
    if len(_args) % 5 != 0:
        raise ValueError("wrong number of positional arguments")
    motors = []
    for motor, start, stop, num, snake in chunked(_args, 5):
        motors.append(motor)

    plan = outer_product_scan(detectors, *args, per_step=per_step, md=md)
    plan = relative_set(plan, motors)  # re-write trajectory as relative
    plan = put_back(plan, motors)  # return motors to starting pos
    ret = yield from plan
    return ret


def relative_inner_product_scan(detectors, num, *args, per_step=None, md=None):
    """
    Scan over one multi-motor trajectory relative to current position.

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
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        See docstring of bluesky.plans._one_nd_step (the default) for
        details.
    md : dict, optional
        metadata
    """
    # There is some duplicate effort here to obtain the list of motors.
    _args = list(args)
    if len(_args) % 3 != 0:
        raise ValueError("wrong number of positional arguments")
    motors = []
    for motor, start, stop, in chunked(_args, 3):
        motors.append(motor)

    plan = inner_product_scan(detectors, num, *args, per_step=per_step, md=md)
    plan = relative_set(plan, motors)  # re-write trajectory as relative
    plan = put_back(plan, motors)  # return motors to starting pos
    ret = yield from plan
    return ret


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
    prompt_str = '{0}, {1:.3}, {2}, ({3}) '

    if md is None:
        md = {}
    md.update({'detectors': [detector.name],
               'motors': [motor.name]})

    d = detector
    try:
        from IPython.display import clear_output
    except ImportError:
        # Define a no-op for clear_output.
        def clear_output(wait=False):
            pass

    def core():
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

    plan = core()
    plan = stage_wrapper(plan)
    plan = run_context(plan, md)
    ret = yield from plan
    return ret


# The code below adds no new logic, but it wraps the generators above in
# classes for an alternative interface that is more stateful.


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
