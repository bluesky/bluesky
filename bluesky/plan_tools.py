import numpy as np
from cycler import cycler
import uuid
from functools import wraps
import itertools
import functools
import operator
from boltons.iterutils import chunked
from collections import OrderedDict, Iterable

import matplotlib.pyplot as plt
from matplotlib import collections as mcollections
from matplotlib import patches as mpatches

from bluesky import Msg
from bluesky.utils import (normalize_subs_input, scalar_heuristic,
                           snake_cyclers, separate_devices)
from bluesky.callbacks import LiveTable, LivePlot


def plot_raster_path(plan, x_motor, y_motor, ax=None, probe_size=None):
    """Plot the raster path for this plan

    Parameters
    ----------
    plan : iterable
       Must yield `Msg` objects and not be a co-routine

    x_motor, y_motor : str
       Names of the x and y motors

    ax : matplotlib.axes.Axes
       The axes to plot to, if none, make new figure + axes

    probe_size : float, optional
       If not None, use as radius of probe (in same units as motor positions)
    """
    if ax is None:
        ax = plt.subplots()[1]
    ax.set_aspect('equal')

    cur_x = cur_y = None
    traj = []
    for msg in plan:
        cmd = msg.command
        if cmd == 'set':
            if msg.obj.name == x_motor:
                cur_x = msg.args[0]
            if msg.obj.name == y_motor:
                cur_y = msg.args[0]
        elif cmd == 'save':
            traj.append((cur_x, cur_y))

    x, y = zip(*traj)
    path, = ax.plot(x, y, marker='', linestyle='-', lw=2)
    if probe_size is None:
        read_points = ax.scatter(x, y, marker='o', lw=2)
    else:
        circles = [mpatches.Circle((_x, _y), probe_size,
                                   facecolor='black', alpha=0.5)
                   for _x, _y in traj]

        read_points = mcollections.PatchCollection(circles,
                                                   match_original=True)
        ax.add_collection(read_points)
    return {'path': path, 'events': read_points}


def print_summary(plan):
    """Print summary of plan

    Prints a minimal version of the plan, showing only moves and
    where events are created.

    Parameters
    ----------
    plan : iterable
        Must yield `Msg` objects
    """

    read_cache = []
    for msg in plan:
        cmd = msg.command
        if cmd == 'open_run':
            print('{:=^80}'.format(' Open Run '))
        elif cmd == 'close_run':
            print('{:=^80}'.format(' Close Run '))
        elif cmd == 'set':
            print('{motor.name} -> {args[0]}'.format(motor=msg.obj,
                                                     args=msg.args))
        elif cmd == 'create':
            pass
        elif cmd == 'read':
            read_cache.append(msg.obj.name)
        elif cmd == 'save':
            print('  Read {}'.format(read_cache))
            read_cache = []


def subscription_wrapper(plan, subs):
    """
    Subscribe to callbacks, yield from plan, then unsubscribe.

    Parameters
    ----------
    plan : iterable
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
    tokens = set()
    subs = normalize_subs_input(subs)
    for name, funcs in subs.items():
        for func in funcs:
            token = yield Msg('subscribe', None, name, func)
            tokens.add(token)
    try:
        ret = yield from plan
    finally:
        # The RunEngine might never process these if the execution fails,
        # but it keeps its own cache of tokens and will try to remove them
        # itself if this plan fails to do so.
        for token in tokens:
            yield Msg('unsubscribe', None, token)
    return ret


def fly_during(plan, flyers):
    if flyers:
        for flyer in flyers:
            yield Msg('kickoff', flyer, block_group='_flyers')
        yield Msg('wait', None, '_flyers')
    ret = yield from plan
    if flyers:
        for flyer in flyers:
            yield Msg('collect', flyer, block_group='_flyers')
        yield Msg('wait', None, '_flyers')
    return ret


def run_wrapper(plan, md=None, **kwargs):
    """Automatically adds RunStart and RunStop Msgs around a plan

    Yields
    ------
    Msg
    """
    if md is None:
        md = dict()
    md = dict(md)
    md.update(kwargs)
    yield Msg('open_run', None, **md)
    try:
        ret = yield from plan
    except:
        yield Msg('close_run', None, exit_status='error')
        raise
    else:
        yield Msg('close_run', None, exit_status='success')
    return ret


def event_wrapper(plan, name=None):
    """Wrap an iterator with a create and save messages

    Yields
    ------
    Msg
    """
    yield Msg('create', name=name)
    ret = yield from plan
    yield Msg('save')

    return ret


def stage_wrapper(plan):
    """
    This is a preprocessor that inserts 'stage' Messages.

    The first time an object is read, set, triggered (or, for flyers,
    the first time is it told to "kickoff") a 'stage' Msg is inserted first.
    It stages the the object's ultimate parent, pointed to be its `root`
    property.

    At the end, an 'unstage' Message issued for every 'stage' Message.
    """
    COMMANDS = ['read', 'set', 'trigger', 'kickoff']
    devices_staged = []
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
    finally:
        yield from broadcast_msg('unstage', reversed(devices_staged))
    return ret


def relative_set(plan, objects):
    """
    Interpret set messages on objects as relative to current position.
    """
    initial_positions = {}
    ret = None
    while True:
        msg = plan.send(ret)
        if msg.command == 'set' and msg.obj in objects:
            if msg.obj not in initial_positions:
                pos = scalar_heuristic(msg.obj)
                initial_positions[msg.obj] = pos
            rel_pos, = msg.args
            abs_pos = initial_positions[msg.obj] + rel_pos
            new_msg = msg._replace(args=(abs_pos,))
            ret = yield new_msg
        else:
            ret = yield msg


def put_back(plan, objects):
    initial_positions = OrderedDict()
    ret = None
    try:
        while True:
            msg = plan.send(ret)
            if msg.command == 'set' and msg.obj in objects:
                if msg.obj not in initial_positions:
                    pos = scalar_heuristic(msg.obj)
                    initial_positions[msg.obj] = pos
            ret = yield msg
    finally:
        for obj, pos in reversed(list(initial_positions.items())):
            yield Msg('set', obj, pos, block_group='_restore')
            yield Msg('wait', None, '_restore')


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


@wrap_with_decorator(event_wrapper)
def trigger_and_read(triggerable, not_triggerable, *, name=None):
    """Trigger and read a list of detectors bundled into a single event.

    triggerable : iterable
        devices to trigger and then read (e.g., detectors)
    not_triggerable : iterable
        dervices to just read (e.g., motors)
    """
    grp = str(uuid.uuid4())
    for det in separate_devices(triggerable):
        yield Msg('trigger', det, block_group=grp)
    yield Msg('wait', None, grp)

    for det in separate_devices(list(triggerable) + list(not_triggerable)):
        yield Msg('read', det)


def broadcast_msg(command, objs, *args, **kwargs):
    return_vals = []
    for o in objs:
        ret = yield Msg(command, o, *args, **kwargs)
        return_vals.append(ret)

    return return_vals


def repeater(n, gen_func, *args, **kwargs):
    it = range
    if n is None:
        n = 0
        it = itertools.count

    for j in it(n):
        yield from gen_func(*args, **kwargs)


def caching_repeater(n, plan):
    lst_plan = list(plan)
    for j in range(n):
        yield from (m for m in lst_plan)


def count(detectors, num=1, delay=None, **md):
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
    **md
        Additional keyword arguments are interpreted as metadata.
    """
    md.update({'detectors': [det.name for det in detectors]})
    md['plan_args'] = {'detectors': repr(detectors), 'num': num}

    # If delay is a scalar, repeat it forever. If it is an iterable, leave it.
    if not isinstance(delay, Iterable):
        delay = itertools.repeat(delay)
    else:
        delay = iter(delay)

    def single_point():
        yield Msg('checkpoint')
        ret = yield from trigger_and_read(detectors, [], name='primary')
        d = next(delay)
        if d is not None:
            yield Msg('sleep', None, d)
        return ret

    plan = repeater(num, single_point)
    plan = stage_wrapper(plan)
    plan = run_wrapper(plan, md=md)
    ret = yield from plan
    return ret

def _step_scan_core(detectors, motor, steps):
    "Just the steps. This should be wrapped in stage_wrapper, run_wrapper."
    def one_step(step):
        grp = str(uuid.uuid4())
        yield Msg('checkpoint')
        yield Msg('set', motor, step, block_group=grp)
        yield Msg('wait', None, grp)
        ret = yield from trigger_and_read(detectors, [motor])
        return ret

    for step in steps:
        ret = yield from one_step(step)
    return ret


def list_scan(detectors, motor, steps, **md):
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
    **md
        Additional keyword arguments are interpreted as metadata.
    """
    md.update({'detectors': [det.name for det in detectors],
               'motors': [motor.name]})
    md['plan_args'] = {'detectors': repr(detectors), 'steps': steps}

    plan = stage_wrapper(_step_scan_core(detectors, motor, steps))
    plan = run_wrapper(plan, md)
    ret = yield from plan
    return ret


def relative_list_scan(detectors, motor, steps, **md):
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
    **md
        Additional keyword arguments are interpreted as metadata.
    """
    # TODO read initial positions (redundantly) so they can be put in md here
    plan = list_scan(detectors, motor, steps, **md)
    plan = relative_set(plan, [motor])  # re-write trajectory as relative
    plan = put_back(plan, [motor])  # return motors to starting pos
    ret = yield from plan
    return ret


def scan(detectors, motor, start, stop, num, **md):
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
    **md
        Additional keyword arguments are interpreted as metadata.
    """
    md.update({'detectors': [det.name for det in detectors],
               'motors': [motor.name]})
    md['plan_args'] = {'detectors': repr(detectors), 'num': num,
                       'start': start, 'stop': stop}

    steps = np.linspace(start, stop, num)
    plan = stage_wrapper(_step_scan_core(detectors, motor, steps))
    plan = run_wrapper(plan, md=md)
    ret = yield from plan
    return ret


def relative_scan(detectors, motor, start, stop, num, **md):
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
    **md
        Additional keyword arguments are interpreted as metadata.
    """
    # TODO read initial positions (redundantly) so they can be put in md here
    plan = scan(detectors, motor, start, stop, num, **md)
    plan = relative_set(plan, [motor])  # re-write trajectory as relative
    plan = put_back(plan, [motor])  # return motors to starting pos
    ret = yield from plan


def log_scan(detectors, motor, start, stop, num, **md):
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
    **md
        Additional keyword arguments are interpreted as metadata.
    """
    md.update({'detectors': [det.name for det in detectors],
               'motors': [motor.name]})
    md['plan_args'] = {'detectors': repr(detectors), 'num': num,
                       'start': start, 'stop': stop}

    steps = np.logspace(start, stop, num)
    plan = stage_wrapper(_step_scan_core(detectors, motor, steps))
    plan = run_wrapper(plan, md=md)
    ret = yield from plan
    return ret


def relative_log_scan(detectors, motor, start, stop, num, **md):
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
    **md
        Additional keyword arguments are interpreted as metadata.
    """
    # TODO read initial positions (redundantly) so they can be put in md here
    plan = log_scan(detectors, motor, start, stop, num, **md)
    plan = relative_set(plan, [motor])  # re-write trajectory as relative
    plan = put_back(plan, [motor])  # return motors to starting pos
    ret = yield from plan
    return ret


def bind_to_run_engine(RE, gen_func, name):

    def inner(*args, md=None, **kwargs):
        if md is None:
            md = {}
        plan = gen_func(*args, **kwargs)
        return RE(plan, **md)

    return inner


def adaptive_scan(detectors, target_field, motor, start, stop,
                  min_step, max_step, target_delta, backstep,
                  threshold=0.8, **md):
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
    **md
        Additional keyword arguments are interpreted as metadata.
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
    plan = run_wrapper(plan, **md)
    ret = yield from plan
    return ret


def relative_adaptive_scan(detectors, target_field, motor, start, stop,
                           min_step, max_step, target_delta, backstep,
                           threshold=0.8, **md):
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
    **md
        Additional keyword arguments are interpreted as metadata.
    """
    plan = adaptive_scan(detectors, target_field, motor, start, stop,
                         min_step, max_step, target_delta, backstep,
                         threshold, **md)
    plan = relative_set(plan, [motor])  # re-write trajectory as relative
    plan = put_back(plan, [motor])  # return motors to starting pos
    ret = yield from plan
    return ret

def _nd_step_scan_core(detectors, cycler):
    motors = cycler.keys
    last_set_point = {m: None for m in motors}
    for step in list(cycler):
        yield Msg('checkpoint')
        for motor, pos in step.items():
            grp = str(uuid.uuid4())
            if pos == last_set_point[motor]:
                # This step does not move this motor.
                continue
            yield Msg('set', motor, pos, block_group=grp)
            last_set_point[motor] = pos
            yield Msg('wait', None, grp)
        ret = yield from trigger_and_read(detectors, motors)
    return ret


def plan_nd(detectors, cycler, **md):
    """
    Scan over an arbitrary N-dimensional trajectory.

    Parameters
    ----------
    detectors : list
    cycler : Cycler
        list of dictionaries mapping motors to positions
    **md
        Additional keyword arguments are interpreted as metadata.
    """
    md.update({'detectors': [det.name for det in detectors],
               'motors': [motor.name for motor in cycler.keys]})
    md['plan_args'] = {'detectors': repr(detectors), 'cycler': repr(cycler)}

    plan = _nd_step_scan_core(detectors, cycler)
    plan = stage_wrapper(plan)
    plan = run_wrapper(plan, md)
    ret = yield from plan
    return ret


def inner_product_scan(detectors, num, *args, **md):
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
    **md
        Additional keyword arguments are interpreted as metadata.
    """
    if len(args) % 3 != 0:
        raise ValueError("wrong number of positional arguments")
    cyclers = []
    for motor, start, stop, in chunked(args, 3):
        steps = np.linspace(start, stop, num=num, endpoint=True)
        c = cycler(motor, steps)
        cyclers.append(c)
    full_cycler = functools.reduce(operator.add, cyclers)

    ret = yield from plan_nd(detectors, full_cycler, **md)
    return ret


def outer_product_scan(detectors, *args, **md):
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
    **md
        Additional keyword arguments are interpreted as metadata.
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

    md.update({'shape': tuple(shape), 'extents': tuple(extents),
               'snaking': tuple(snaking), 'num': len(full_cycler)})

    ret = yield from plan_nd(detectors, full_cycler, **md)
    return ret


def relative_outer_product_scan(detectors, *args, **md):
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
    **md
        Additional keyword arguments are interpreted as metadata.
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

    plan = outer_product_scan(detectors, *args, **md)
    plan = relative_set(plan, motors)  # re-write trajectory as relative
    plan = put_back(plan, motors)  # return motors to starting pos
    ret = yield from plan
    return ret


def relative_inner_product_scan(detectors, num, *args, **md):
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
    **md
        Additional keyword arguments are interpreted as metadata.
    """
    # There is some duplicate effort here to obtain the list of motors.
    _args = list(args)
    if len(_args) % 3 != 0:
        raise ValueError("wrong number of positional arguments")
    motors = []
    for motor, start, stop, in chunked(_args, 3):
        motors.append(motor)

    plan = inner_product_scan(detectors, num, *args, **md)
    plan = relative_set(plan, motors)  # re-write trajectory as relative
    plan = put_back(plan, motors)  # return motors to starting pos
    ret = yield from plan
    return ret


def tweak(detector, target_field, motor, step, **md):
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
    **md
        Additional keyword arguments are interpreted as metadata.
    """
    prompt_str = '{0}, {1:.3}, {2}, ({3}) '

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
    plan = run_wrapper(plan, **md)
    ret = yield from plan
    return ret
