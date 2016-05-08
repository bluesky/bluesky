import uuid
import sys
from functools import wraps
import itertools
import functools
import operator
from contextlib import contextmanager
from collections import OrderedDict, Iterable, defaultdict, deque
import types

import numpy as np
from cycler import cycler
from boltons.iterutils import chunked

from . import Msg

from .plan_tools import ensure_generator
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
        expected to return a list of generators that yield messages (`Msg`
        objects) the function may have an arbitrary signature

    Returns
    -------
    gen : generator
        a single generator that yields messages
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        gen_stack = func(*args, **kwargs)
        for g in gen_stack:
            yield from g

    return wrapped


def plan_mutator(plan, msg_proc):
    """
    Alter the contents of a plan on the fly by changing or inserting messages.

    Parameters
    ----------
    plan : generator
        a generator that yields messages (`Msg` objects)
    msg_proc : callable
        functions that takes in a message and returns replacement messages

        fucntion signatures:

        msg -> None, None (no op)
        msg -> gen, None (mutate and/or insert before current message;
                        last message in gen must invoke a response compatible
                        with original msg)
        msg -> gen, tail (same as above, but insert some messages *after*)
        msg -> None, tail (illegal -- raises RuntimeError)

    Yields
    ------
    msg : Msg
        messages from `plan`, altered by `msg_proc`
    """
    # internal stacks
    msgs_seen = dict()
    plan_stack = deque()
    result_stack = deque()
    tail_cache = dict()
    tail_result_cache = dict()

    # seed initial conditions
    plan_stack.append(plan)
    result_stack.append(None)

    while True:
        # get last result
        ret = result_stack.pop()
        try:
            # send last result to the top most generator in the
            # stack this may raise StopIteration
            msg = plan_stack[-1].send(ret)

        except StopIteration:
            # discard the exhausted generator
            # TODO capture gen.close()?
            exhausted_gen = plan_stack.pop()
            # if we just came out of a 'tail' generator, discard its
            # return value and replace it with the cached one (from the last
            # message in its paired 'new_gen')
            if id(exhausted_gen) in tail_result_cache:
                ret = tail_result_cache.pop(id(exhausted_gen))

            result_stack.append(ret)

            if id(exhausted_gen) in tail_cache:
                gen = tail_cache.pop(id(exhausted_gen))
                if gen is not None:
                    plan_stack.append(gen)
                    saved_result = result_stack.pop()
                    tail_result_cache[id(gen)] = saved_result
                    # must use None to prime generator
                    result_stack.append(None)

            if plan_stack:
                continue
            else:
                return plan.close()
        except Exception as ex:
            # we are here because an exception came out of the send
            # this may be due to
            # a) the plan really raising or
            # b) an exception that came out of the run engine via ophyd
            # If the plan is raising from its side, then the next plan
            # gets to see the exception on its way out to the user.
            # If this is a plan that came in through the RE, the top plan has
            # had its shot do deal with it and now the next plan gets it.
            plan_stack.pop()
            if plan_stack:
                msg = plan_stack[-1].throw(ex)
                plan_stack.append(single_gen(msg))
                result_stack.append(None)
                continue
            else:
                raise ex

        # if inserting / mutating, put new generator on the stack
        # and replace the current msg with the first element from the
        # new generator
        if id(msg) not in msgs_seen:
            # Use the id as a hash, and hold a reference to the msg so that
            # it cannot be garbage collected until the plan is complete.
            msgs_seen[id(msg)] = msg

            new_gen, tail_gen = msg_proc(msg)
            # mild correctness check
            if tail_gen is not None and new_gen is None:
                raise RuntimeError("This makes no sense")
            if new_gen is not None:
                # stash the new generator
                plan_stack.append(new_gen)
                # put in a result value to prime it
                result_stack.append(None)
                # stash the tail generator
                tail_cache[id(new_gen)] = tail_gen
                # go to the top of the loop
                continue

        try:
            # yield out the 'current message' and collect the return
            inner_ret = yield msg
        except Exception as ex:
            msg = plan.throw(ex)
            plan_stack.append(single_gen(msg))
            result_stack.append(None)
            continue

        result_stack.append(inner_ret)


def msg_mutator(plan, msg_proc):
    """
    A simple preprocessor that mutates or deletes single messages in a plan

    To *insert* messages, use ``plan_mutator`` instead.

    Parameters
    ----------
    plan : generator
        a generator that yields messages (`Msg` objects)
    msg_proc : callable
        Expected signature `f(msg) -> new_msg or None`

    Yields
    ------
    msg : Msg
        messages from `plan`, altered by `msg_proc`
    """
    ret = None
    while True:
        try:
            msg = plan.send(ret)
            msg = msg_proc(msg)
            # if None, just skip message
            # feed 'None' back down into the base plan,
            # this may break some plans
            if msg is None:
                ret = None
                continue
            ret = yield msg
        except StopIteration:
            break
    return plan.close()


def pchain(*args):
    '''Like `itertools.chain` but using `yield from`

    This ensures than `.send` works as expected and the underlying
    plans get the return values

    Parameters
    ----------
    args :
        generators (plans)

    Yields
    ------
    msg : Msg
        The messages from each plan in turn
    '''
    rets = deque()
    for p in args:
        rets.append((yield from p))
    return tuple(rets)


def single_gen(msg):
    '''Turn a single message into a plan

    If ``lambda x: yield x`` were valid Python, this would be equivalent.
    In Python 3.6 or 3.7 we might get lambda generators.

    Parameters
    ----------
    msg : Msg
        a single message

    Yields
    ------
    msg : Msg
        the input message
    '''
    yield msg


def create(name='primary'):
    """
    Bundle future readings into a new Event document.

    Parameters
    ----------
    name : string, optional
        name given to event stream, used to convenient identification
        default is 'primary'

    Yields
    ------
    msg : Msg
        Msg('create', name=name)
    """
    return (yield from single_gen(Msg('create', name=name)))


def save():
    """
    Close a bundle of readings and emit a completed Event document.

    Yields
    -------
    msg : Msg
        Msg('save')
    """
    return (yield from single_gen(Msg('save')))


def read(obj):
    """
    Take a reading and add it to the current bundle of readings.

    Parameters
    ----------
    obj : Device or Signal

    Yields
    ------
    msg : Msg
        Msg('read', obj)
    """
    return (yield from single_gen(Msg('read', obj)))


def monitor(obj, *args, name=None, **kwargs):
    """
    Asynchronously monitor for new values and emit Event documents.

    Parameters
    ----------
    obj : Signal
    args :
        passed through to ``obj.subscribe()``
    name : string, optional
        name of event stream; default is None
    kwargs :
        passed through to ``obj.subscribe()``

    Yields
    ------
    msg : Msg
        Msg('monitor', obj, *args, **kwargs)
    """
    return (yield from single_gen(Msg('monitor', obj, *args, name=name,
                                      **kwargs)))


def unmonitor(obj):
    """
    Stop monitoring.

    Parameters
    ----------
    obj : Signal

    Yields
    ------
    msg : Msg
        Msg('unmonitor', obj)
    """
    return (yield from single_gen(Msg('unmonitor', obj)))


def null():
    """
    Yield a no-op Message. (Primarily for debugging and testing.)

    Yields
    ------
    msg : Msg
        Msg('null')
    """
    return (yield from single_gen(Msg('null')))


def abs_set(obj, *args, group=None, wait=False, **kwargs):
    """
    Set a value. Optionally, wait for it to complete before continuing.

    Parameters
    ----------
    obj : Device
    group : string (or any hashable object), optional
        identifier used by 'wait'
    wait : boolean, optional
        If True, wait for completion before processing any more messages.
        False by default.
    args :
        passed to obj.set()
    kwargs :
        passed to obj.set()

    Yields
    ------
    msg : Msg
    """
    ret = yield from single_gen(Msg('set', obj, *args, group=group, **kwargs))
    if wait:
        yield from single_gen(Msg('wait', None, group=group))
    return ret


def rel_set(obj, *args, group=None, wait=False, **kwargs):
    """
    Set a value relative to current value. Optionally, wait before continuing.

    Parameters
    ----------
    obj : Device
    group : string (or any hashable object), optional
        identifier used by 'wait'; None by default
    wait : boolean, optional
        If True, wait for completion before processing any more messages.
        False by default.
    args :
        passed to obj.set()
    kwargs :
        passed to obj.set()

    Yields
    ------
    msg : Msg
    """
    ret = yield from relative_set(abs_set(obj, *args, group=group, **kwargs))
    if wait:
        yield from single_gen(Msg('wait', None, group=group))
    return ret


def trigger(obj, *, group=None, wait=False):
    """
    Trigger and acquisition. Optionally, wait for it to complete.

    Parameters
    ----------
    obj : Device
    group : string (or any hashable object), optional
        identifier used by 'wait'; None by default
    wait : boolean, optional
        If True, wait for completion before processing any more messages.
        False by default.

    Yields
    ------
    msg : Msg
    """
    ret = yield from single_gen(Msg('trigger', obj, group=group))
    if wait:
        yield from single_gen(Msg('wait', None, group=group))
    return ret


def sleep(time):
    """
    Tell the RunEngine to sleep, while asynchronously doing other processing.

    This is not the same as ``import time; time.sleep()`` because it allows
    other actions, like interruptions, to be processed during the sleep.

    Parameters
    ----------
    time : float
        seconds

    Yields
    ------
    msg : Msg
        Msg('sleep', None, time)
    """
    return (yield from single_gen(Msg('sleep', None, time)))


def wait(group=None):
    """
    Wait for all statuses in a group to report being finished.

    Parameters
    ----------
    group : string (or any hashable object), optional
        idenified given to `abs_set`, `rel_set`, `trigger`; None by default

    Yields
    ------
    msg : Msg
        Msg('wait', None, group=group)
    """
    return (yield from single_gen(Msg('wait', None, group=group)))


_wait = wait  # for internal references to avoid collision with 'wait' kwarg


def checkpoint():
    """
    If interrupted, rewind to this point.

    Yields
    ------
    msg : Msg
        Msg('checkpoint')
    """
    return (yield from single_gen(Msg('checkpoint')))


def clear_checkpoint():
    """
    Designate that it is not safe to resume. If interrupted or paused, abort.

    Yields
    ------
    msg : Msg
        Msg('clear_checkpoint')
    """
    return (yield from single_gen(Msg('clear_checkpoint')))


def pause():
    """
    Pause and wait for the user to resume.

    Yields
    ------
    msg : Msg
        Msg('pause')
    """
    return (yield from single_gen(Msg('pause', None, defer=False)))


def deferred_pause():
    """
    Pause at the next checkpoint.

    Yields
    ------
    msg : Msg
        Msg('pause', defer=True)
    """
    return (yield from single_gen(Msg('pause', None, defer=True)))


def input(prompt=''):
    """
    Prompt the user for text input.

    Parameters
    ----------
    prompt : str
        prompt string, e.g., 'enter user name' or 'enter next position'

    Yields
    ------
    msg : Msg
        Msg('input', prompt=prompt)
    """
    return (yield from single_gen(Msg('input', prompt=prompt)))


def kickoff(obj, *, group=None, wait=False, **kwargs):
    """
    Kickoff a fly-scanning device.

    Parameters
    ----------
    obj : fly-able
        Device with 'kickoff', 'complete', and 'collect' methods
    group : string (or any hashable object), optional
        identifier used by 'wait'
    wait : boolean, optional
        If True, wait for completion before processing any more messages.
        False by default.
    kwargs
        passed through to ``obj.kickoff()``

    Yields
    ------
    msg : Msg
        Msg('kickoff', obj)
    """
    ret = (yield from single_gen(
         Msg('kickoff', obj, group=group, **kwargs)))
    if wait:
        yield from _wait(group=group)
    return ret


def complete(obj, *, group=None, wait=True, **kwargs):
    """
    Tell a flyer, 'stop collecting, whenver you are ready'.

    The flyer returns a status object. Some flyers respond to this
    command by stopping collection and returning a finished status
    object immedately. Other flyers finish their given course and
    finish whenever they finish, irrespective of when this command is
    issued.

    Parameters
    ----------
    obj : fly-able
        Device with 'kickoff', 'complete', and 'collect' methods
    group : string (or any hashable object), optional
        identifier used by 'wait'
    wait : boolean, optional
        If True, wait for completion before processing any more messages.
        False by default.
    kwargs
        passed through to ``obj.complete()``

    Yields
    ------
    msg : Msg
        a 'complete' Msg and maybe a 'wait' message
    """
    ret = (yield from single_gen(
         Msg('complete', obj, group=group, **kwargs)))
    if wait:
        yield from _wait(group=group)
    return ret



def collect(obj, *, stream=False):
    """
    Collect data cached by a fly-scanning device and emit documents.

    Parameters
    ----------
    obj : fly-able
        Device with 'kickoff', 'complete', and 'collect' methods
    stream : boolean
        If False (default), emit events documents in one bulk dump. If True,
        emit events one at time.

    Yields
    ------
    msg : Msg
        Msg('collect', obj)
    """
    return (yield from single_gen(Msg('collect', obj, stream=stream)))


def configure(obj, *args, **kwargs):
    """
    Change Device configuration and emit an updated Event Descriptor document.

    Parameters
    ----------
    obj : Device
    args
        passed through to ``obj.configure()``
    kwargs
        passed through to ``obj.configure()``

    Yields
    ------
    msg : Msg
        Msg('configure', obj, *args, **kwargs)
    """
    return (yield from single_gen(Msg('configure', obj, *args, **kwargs)))


def stage(obj):
    """
    'Stage' a device (i.e., prepare it for use, 'arm' it).

    Parameters
    ----------
    obj : Device

    Yields
    ------
    msg : Msg
        Msg('stage', obj)
    """
    return (yield from single_gen(Msg('stage', obj)))


def unstage(obj):
    """
    'Unstage' a device (i.e., put it in standby, 'disarm' it).

    Parameters
    ----------
    obj : Device

    Yields
    ------
    msg : Msg
        Msg('unstage', obj)
    """
    return (yield from single_gen(Msg('unstage', obj)))


def subscribe(name, func):
    """
    Subscribe the stream of emitted documents.

    Parameters
    ----------
    name : {'all', 'start', 'descriptor', 'event', 'stop'}
    func : callable
        Expected signature: ``f(name, doc)`` where ``name`` is one of the
        strings above ('all, 'start', ...) and ``doc`` is a dict

    Yields
    ------
    msg : Msg
        Msg('subscribe', None, name, func)
    """
    return (yield from single_gen(Msg('subscribe', None, name, func)))


def unsubscribe(token):
    """
    Remove a subscription.

    Parameters
    ----------
    token : int
        token returned by processing a 'subscribe' message

    Yields
    ------
    msg : Msg
        Msg('unsubscribe', token=token)
    """
    return (yield from single_gen(Msg('unsubscribe', token=token)))


def open_run(md=None):
    """
    Mark the beginning of a new 'run'. Emit a RunStart document.

    Parameters
    ----------
    md : dict, optional
        metadata

    Yields
    ------
    msg : Msg
        Msg('open_run', **md)
    """
    if md is None:
        md = {}
    return (yield from single_gen(Msg('open_run', **md)))


def close_run():
    """
    Mark the end of the current 'run'. Emit a RunStop document.

    Yields
    ------
    msg : Msg
        Msg('close_run')
    """
    return (yield from single_gen(Msg('close_run')))


def wait_for(futures, **kwargs):
    """
    Low-level: wait for a list of ``asyncio.Future`` objects to set (complete).

    Parameters
    ----------
    futures : collection
        collection of asyncio.Future objects
    kwargs
        passed through to ``asyncio.wait()``

    Yields
    ------
    msg : Msg
        Msg('wait_for', None, futures, **kwargs)
    """
    return (yield from single_gen(Msg('wait_for', None, futures, **kwargs)))


def finalize(plan, final_plan):
    '''try...finally helper

    Run the first plan and then the second.  If any of the messages
    raise an error in the RunEngine (or otherwise), the second plan
    will attempted to be run anyway.

    Parameters
    ----------
    plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects
    final_plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects; attempted to be
        run no matter what happens in the first plan

    Yields
    ------
    msg : Msg
        messages from `plan` until it terminates or an error is raised, then
        messages from `final_plan`
    '''
    try:
        ret = yield from plan
    finally:
        yield from ensure_generator(final_plan)
    return ret


@contextmanager
def subs_context(plan_stack, subs):
    """
    Subscribe to callbacks to the document stream; then unsubscribe on exit.

    Parameters
    ----------
    plan_stack : list-like
        appendable collection of generators that yield messages (`Msg` objects)
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
            yield Msg('unsubscribe', None, token=token)

    plan_stack.append(subscribe())
    try:
        yield plan_stack
    finally:
        # The RunEngine might never process these if the execution fails,
        # but it keeps its own cache of tokens and will try to remove them
        # itself if this plan fails to do so.
        plan_stack.append(unsubscribe())


@contextmanager
def run_context(plan_stack, md=None):
    """Enclose in 'open_run' and 'close_run' messages.

    Parameters
    ----------
    plan_stack : list-like
        appendable collection of generators that yield messages (`Msg` objects)
    md : dict, optional
        metadata to be passed into the 'open_run' message
    """
    if md is None:
        md = dict()
    md = dict(md)
    plan_stack.append(single_gen(Msg('open_run', None, **md)))
    yield plan_stack
    plan_stack.append(single_gen(Msg('close_run')))


@contextmanager
def event_context(plan_stack, name='primary'):
    """Bundle readings into an 'event' (a datapoint).

    This encloses the contents in 'create' and 'save' messages.

    Parameters
    ----------
    plan_stack : list-like
        appendable collection of generators that yield messages (`Msg` objects)
    name : string, optional
        name of event stream; default is 'primary'
    """
    plan_stack.append(single_gen(Msg('create', None, name=name)))
    yield plan_stack
    plan_stack.append(single_gen(Msg('save')))


def fly(flyers, *, md=None):
    """
    Perform a fly scan with one or more 'flyers'.

    Parameters
    ----------
    flyers : collection
        objects that support the flyer interface

    Yields
    ------
    msg : Msg
        'kickoff', 'wait', 'complete, 'wait', 'collect' messages
    md : dict, optional
        metadata

    See Also
    --------
    `bluesky.plans.fly_during`
    """
    yield from open_run(md)
    for flyer in flyers:
        yield from kickoff(flyer, wait=True)
    for flyer in flyers:
        yield from complete(flyer, wait=True)
    for flyer in flyers:
        yield from collect(flyer)
    yield from close_run()


def fly_during(plan, flyers):
    """
    Kickoff and collect "flyer" (asynchronously collect) objects during runs.

    This is a preprocessor that insert messages immediately after a run is
    opened and before it is closed.

    Parameters
    ----------
    plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects
    flyers : collection
        objects that support the flyer interface

    Yields
    ------
    msg : Msg
        messages from plan with 'kickoff', 'wait' and 'collect' messages
        inserted
    """
    grp1 = _short_uid('flyers-kickoff')
    grp2 = _short_uid('flyers-complete')
    kickoff_msgs = [Msg('kickoff', flyer, group=grp1) for flyer in flyers]
    complete_msgs = [Msg('complete', flyer, group=grp1) for flyer in flyers]
    collect_msgs = [Msg('collect', flyer) for flyer in flyers]
    if flyers:
        # If there are any flyers, insert a 'wait' Msg after kickoff, complete
        kickoff_msgs += [Msg('wait', None, group=grp1)]
        complete_msgs += [Msg('wait', None, group=grp2)]

    def insert_after_open(msg):
        if msg.command == 'open_run':
            def new_gen():
                yield from ensure_generator(kickoff_msgs)
            return single_gen(msg), new_gen()
        else:
            return None, None

    def insert_before_close(msg):
        if msg.command == 'close_run':
            def new_gen():
                yield from ensure_generator(complete_msgs)
                yield from ensure_generator(collect_msgs)
                yield msg
            return new_gen(), None
        else:
            return None, None

    # Apply nested mutations.
    plan1 = plan_mutator(plan, insert_after_open)
    plan2 = plan_mutator(plan1, insert_before_close)
    return (yield from plan2)


def lazily_stage(plan):
    """
    This is a preprocessor that inserts 'stage' messages and appends 'unstage'.

    The first time an object is seen in `plan`, it is staged. To avoid
    redundant staging we actually stage the object's ultimate parent, pointed
    to be its `root` property.

    At the end, in a `finally` block, an 'unstage' Message issued for every
    'stage' Message.

    Parameters
    ----------
    plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects

    Yields
    ------
    msg : Msg
        messages from plan with 'stage' messages inserted and 'unstage'
        messages appended
    """
    COMMANDS = set(['read', 'set', 'trigger', 'kickoff'])
    # Cache devices in the order they are staged; then unstage in reverse.
    devices_staged = []

    def inner(msg):
        if msg.command in COMMANDS and msg.obj not in devices_staged:
            root = msg.obj.root

            def new_gen():
                # Here we insert a 'stage' message
                ret = yield Msg('stage', root)
                # and cache the result
                if ret is None:
                    # The generator may be being list-ified.
                    # This is a hack to make that possible.
                    ret = [root]
                devices_staged.extend(ret)
                # and then proceed with our regularly scheduled programming
                yield msg
            return new_gen(), None
        else:
            return None, None

    def unstage_all():
        for device in reversed(devices_staged):
            yield Msg('unstage', device)

    return (yield from finalize(plan_mutator(plan, inner), unstage_all()))


@contextmanager
def stage_context(plan_stack, devices):
    """
    Stage devices upon entering context and unstage upon exiting.

    Parameters
    ----------
    plan_stack : list-like
        appendable collection of generators that yield messages (`Msg` objects)
    devices : collection
        list of devices to stage immediately on entrance and unstage on exit
    """
    # Resolve unique devices, avoiding redundant staging.
    devices = [device.root for device in devices]

    def stage():
        # stage devices explicitly passed to 'devices' argument
        yield from broadcast_msg('stage', devices)

    def unstage():
        # unstage devices explicitly passed to 'devices' argument
        yield from broadcast_msg('unstage', reversed(devices))

    plan_stack.append(stage())
    yield plan_stack
    plan_stack.append(unstage())


def relative_set(plan, devices=None):
    """
    Interpret 'set' messages on devices as relative to initial position.

    Parameters
    ----------
    plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects
    devices : collection or None, optional
        if default (None), apply to all devices that are moved by the plan

    Yields
    ------
    msg : Msg
        messages from plan, with 'read' messages inserted and 'set' messages
        mutated
    """
    initial_positions = {}

    def read_and_stash_a_motor(obj):
        # obj should have a `position` attribution
        try:
            cur_pos = obj.position
        except AttributeError:
            # ... but as a fallback we can read obj and grab the value of the
            # first key
            reading = yield Msg('read', obj)
            if reading is None:
                # this plan may be being list-ified
                cur_pos = 0
            else:
                k = list(reading.keys())[0]
                cur_pos = reading[k]['value']
        initial_positions[obj] = cur_pos

    def rewrite_pos(msg):
        if (msg.command == 'set') and (msg.obj in initial_positions):
            rel_pos, = msg.args
            abs_pos = initial_positions[msg.obj] + rel_pos
            new_msg = msg._replace(args=(abs_pos,))
            return new_msg
        else:
            return msg

    def insert_reads(msg):
        eligible = (devices is None) or (msg.obj in devices)
        seen = msg.obj in initial_positions
        if (msg.command == 'set') and eligible and not seen:
                return pchain(read_and_stash_a_motor(msg.obj),
                              single_gen(msg)), None
        else:
            return None, None

    plan = plan_mutator(plan, insert_reads)
    plan = msg_mutator(plan, rewrite_pos)
    return (yield from plan)


def reset_positions(plan, devices=None):
    """
    Return movable devices to their initial positions at the end.

    Parameters
    ----------
    plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects
    devices : collection or None, optional
        If default (None), apply to all devices that are moved by the plan.

    Yields
    ------
    msg : Msg
        messages from plan with 'read' and finally 'set' messages inserted
    """
    initial_positions = OrderedDict()

    def read_and_stash_a_motor(obj):
        try:
            cur_pos = obj.position
        except AttributeError:
            reading = yield Msg('read', obj)
            if reading is None:
                # this plan may be being list-ified
                cur_pos = 0
            else:
                k = list(reading.keys())[0]
                cur_pos = reading[k]['value']
        initial_positions[obj] = cur_pos

    def insert_reads(msg):
        eligible = devices is None or msg.obj in devices
        seen = msg.obj in initial_positions
        if (msg.command == 'set') and eligible and not seen:
            return pchain(read_and_stash_a_motor(msg.obj),
                          single_gen(msg)), None
        else:
            return None, None

    def reset():
        blk_grp = 'reset-{}'.format(str(uuid.uuid4())[:6])
        for k, v in initial_positions.items():
            yield Msg('set', k, v, group=blk_grp)
        yield Msg('wait', None, group=blk_grp)

    return (yield from finalize(plan_mutator(plan, insert_reads), reset()))


def configure_count_time(plan, time):
    """
    Preprocessor that sets all devices with a `count_time` to the same time.

    The original setting is stashed and restored at the end.

    Parameters
    ----------
    plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects
    time : float or None
        If None, the plan passes through unchanged.

    Yields
    ------
    msg : Msg
        messages from plan, with 'set' messages inserted
    """
    devices_seen = set()
    original_times = {}

    def insert_set(msg):
        obj = msg.obj
        if obj is not None and obj not in devices_seen:
            devices_seen.add(obj)
            if hasattr(obj, 'count_time'):
                # TODO Do this with a 'read' Msg once reads can be
                # marked as belonging to a different event stream (or no
                # event stream.
                original_times[obj] = obj.count_time.get()
                grp = _short_uid('set-count-time')
                return pchain(single_gen(Msg('set', obj.count_time, time,
                                             group=grp)),
                              single_gen(Msg('wait', None, group=grp)),
                              single_gen(msg)), None
        return None, None

    def reset():
        for obj, time in original_times.items():
            grp = _short_uid('reset-count-time')
            yield Msg('set', obj.count_time, time, group=grp)
            yield Msg('wait', None, group=grp)

    if time is None:
        # no-op
        return (yield from plan)
    else:
        return (yield from finalize(plan_mutator(plan, insert_set), reset()))


@contextmanager
def baseline_context(plan_stack, devices, name='baseline'):
    """
    Read every device once upon entering and exiting the context.

    The readings are designated for a separate event stream named 'baseline'
    by default.

    Parameters
    ----------
    plan_stack : list-like
        appendable collection of generators that yield messages (`Msg` objects)
    devices : collection
        collection of Devices to read
    name : string, optional
        name for event stream; by default, 'baseline'
    """
    plan_stack.append(trigger_and_read(devices, name=name))
    yield
    plan_stack.append(trigger_and_read(devices, name=name))


@contextmanager
def monitor_context(plan_stack, signals):
    """
    Asynchronously monitor signals, generating separate event streams.

    Upon exiting the context, stop monitoring.

    Parameters
    ----------
    plan_stack : list-like
        appendable collection of generators that yield messages (`Msg` objects)
    signals : dict or list
        either a dict mapping Signals to event stream names or simply a list
        of Signals, in which case the event stream names default to None
    name : string, optional
        name for event stream; by default, None

    Examples
    --------
    >>> plan_stack = deque()

    With custom event stream names
    >>> with monitor_context(plan_stack, {sig1: 'sig1', sig2: 'sig2'}):
            ...

    With no event stream names
    >>> with monitor_context(plan_stack, [sig1, sig2]):
            ...
    """
    if hasattr(signals, 'items'):
        # interpret input as dict of signals mapped to event stream names
        pass
    else:
        # interpet input as list of signals
        signals = {sig: None for sig in signals}

    for sig, name in signals.items():
        plan_stack.append(single_gen(Msg('monitor', sig, name=name)))
    yield
    for sig, name in signals.items():
        plan_stack.append(single_gen(Msg('unmonitor', sig)))


@planify
def trigger_and_read(devices, name='primary'):
    """
    Trigger and read a list of detectors and bundle readings into one Event.

    Parameters
    ----------
    devices : iterable
        devices to trigger (if they have a trigger method) and then read
    name : string, optional
        event stream name, a convenient human-friendly identifier; default
        name is 'primary'

    Yields
    ------
    msg : Msg
        messages to 'trigger', 'wait' and 'read'
    """
    devices = separate_devices(devices)  # remove redundant entries
    grp = _short_uid('trigger')
    plan_stack = deque()
    for obj in devices:
        if hasattr(obj, 'trigger'):
            plan_stack.append(single_gen(Msg('trigger', obj, group=grp)))
    if plan_stack:
        plan_stack.append(single_gen(Msg('wait', None, group=grp)))
    with event_context(plan_stack, name=name):
        for obj in devices:
            plan_stack.append(single_gen(Msg('read', obj)))
    return plan_stack


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

    Yields
    ------
    msg : Msg
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
    n : int or None
        total number of repetitions; if None, infinite
    gen_func : callable
        returns generator instance
    *args
        args for gen_func
    **kwargs
        kwargs for gen_func

    Yields
    ------
    msg : Msg
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
    n : int or None
        total number of repetitions; if None, infinite
    plan : iterable

    Yields
    ------
    msg : Msg
    """
    it = range
    if n is None:
        n = 0
        it = itertools.count

    lst_plan = list(plan)
    for j in it(n):
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
    md['plan_args'] = {'detectors': list(map(repr, detectors)), 'num': num}

    if num is None:
        counter = itertools.count()  # run forever, until interrupted
    else:
        counter = range(num)

    # If delay is a scalar, repeat it forever. If it is an iterable, leave it.
    if not isinstance(delay, Iterable):
        delay = itertools.repeat(delay)
    else:
        delay = iter(delay)

    plan_stack = deque()
    with stage_context(plan_stack, detectors):
        with run_context(plan_stack, md):
            for _ in counter:
                plan_stack.append(single_gen(Msg('checkpoint')))
                plan_stack.append(trigger_and_read(detectors))
                d = next(delay)
                if d is not None:
                    plan_stack.append(single_gen(Msg('sleep', None, d)))
    return plan_stack


@planify
def one_1d_step(detectors, motor, step):
    """
    Inner loop of a 1D step scan

    This is the default function for ``per_step`` param in 1D plans.
    """
    def move():
        grp = _short_uid('set')
        yield Msg('checkpoint')
        yield Msg('set', motor, step, group=grp)
        yield Msg('wait', None, group=grp)

    plan_stack = deque()
    plan_stack.append(move())
    plan_stack.append(trigger_and_read(list(detectors) + [motor]))
    return plan_stack


@planify
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
        Expected signature:
        ``f(detectors, motor, step) -> plan (a generator)
    md : dict, optional
        metadata
    """
    if md is None:
        md = {}
    md.update({'detectors': [det.name for det in detectors],
               'motors': [motor.name]})
    md['plan_args'] = {'detectors': list(map(repr, detectors)),
                       'motor': repr(motor), 'steps': steps,
                       'per_step': repr(per_step)}
    if per_step is None:
        per_step = one_1d_step

    plan_stack = deque()
    with stage_context(plan_stack, list(detectors) + [motor]):
        with run_context(plan_stack, md):
            for step in steps:
                plan_stack.append(per_step(detectors, motor, step))
    return plan_stack


@planify
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
    plan = relative_set(plan)  # re-write trajectory as relative
    plan = reset_positions(plan)  # return motors to starting pos
    return [plan]


@planify
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
    md['plan_args'] = {'detectors': list(map(repr, detectors)), 'num': num,
                       'start': start, 'stop': stop,
                       'per_step': repr(per_step)}

    if per_step is None:
        per_step = one_1d_step

    steps = np.linspace(start, stop, num)

    plan_stack = deque()
    with stage_context(plan_stack, list(detectors) + [motor]):
        with run_context(plan_stack, md):
            for step in steps:
                plan_stack.append(per_step(detectors, motor, step))
    return plan_stack


@planify
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
    plan = reset_positions(plan, [motor])  # return motors to starting pos
    return [plan]


@planify
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
    md['plan_args'] = {'detectors': list(map(repr, detectors)), 'num': num,
                       'start': start, 'stop': stop,
                       'per_step': repr(per_step)}

    if per_step is None:
        per_step = one_1d_step

    steps = np.logspace(start, stop, num)

    plan_stack = deque()
    with stage_context(plan_stack, list(detectors) + [motor]):
        with run_context(plan_stack, md):
            for step in steps:
                plan_stack.append(per_step(detectors, motor, step))
    return plan_stack


@planify
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
    plan = reset_positions(plan, [motor])  # return motors to starting pos
    return [plan]


@planify
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
                yield Msg('trigger', det, group='B')
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

    plan_stack = deque()
    with stage_context(plan_stack, list(detectors) + [motor]):
        with run_context(plan_stack, md):
            plan_stack.append(core())
    return plan_stack


@planify
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
    plan = reset_positions(plan, [motor])  # return motors to starting pos
    return [plan]


@planify
def one_nd_step(detectors, step, pos_cache):
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
    def move():
        yield Msg('checkpoint')
        grp = _short_uid('set')
        for motor, pos in step.items():
            if pos == pos_cache[motor]:
                # This step does not move this motor.
                continue
            yield Msg('set', motor, pos, group=grp)
            pos_cache[motor] = pos
        yield Msg('wait', None, group=grp)

    motors = step.keys()
    plan_stack = deque()
    plan_stack.append(move())
    plan_stack.append(trigger_and_read(list(detectors) + list(motors)))
    return plan_stack


@planify
def scan_nd(detectors, cycler, *, per_step=None, md=None):
    """
    Scan over an arbitrary N-dimensional trajectory.

    Parameters
    ----------
    detectors : list
    cycler : Cycler
        list of dictionaries mapping motors to positions
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        See docstring of bluesky.plans.one_nd_step (the default) for
        details.
    md : dict, optional
        metadata
    """
    if md is None:
        md = {}
    md.update({'detectors': [det.name for det in detectors],
               'motors': [motor.name for motor in cycler.keys]})
    md['plan_args'] = {'detectors': list(map(repr, detectors)),
                       'cycler': repr(cycler)}

    if per_step is None:
        per_step = one_nd_step
    pos_cache = defaultdict(lambda: None)  # where last position is stashed
    motors = list(cycler.keys)

    plan_stack = deque()
    with stage_context(plan_stack, list(detectors) + motors):
        with run_context(plan_stack, md):
            for step in list(cycler):
                plan_stack.append(per_step(detectors, step, pos_cache))
    return plan_stack


@planify
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
        See docstring of bluesky.plans.one_nd_step (the default) for
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

    plan = scan_nd(detectors, full_cycler, per_step=per_step, md=md)
    return [plan]


@planify
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
        See docstring of bluesky.plans.one_nd_step (the default) for
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

    plan = scan_nd(detectors, full_cycler, per_step=per_step, md=md)
    return [plan]


@planify
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
        See docstring of bluesky.plans.one_nd_step (the default) for
        details.
    md : dict, optional
        metadata
    """
    plan = outer_product_scan(detectors, *args, per_step=per_step, md=md)
    plan = relative_set(plan)  # re-write trajectory as relative
    plan = reset_positions(plan)  # return motors to starting pos
    return [plan]


@planify
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
        See docstring of bluesky.plans.one_nd_step (the default) for
        details.
    md : dict, optional
        metadata
    """
    plan = inner_product_scan(detectors, num, *args, per_step=per_step, md=md)
    plan = relative_set(plan)  # re-write trajectory as relative
    plan = reset_positions(plan)  # return motors to starting pos
    return [plan]


@planify
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
        nonlocal step

        while True:
            yield Msg('create', None, name='primary')
            try:
                pos = motor.position
            except AttributeError:
                ret_mot = yield Msg('read', motor)
                key = list(ret_mot.keys())[0]
                pos = ret_mot[key]['value']
            yield Msg('trigger', d, group='A')
            yield Msg('wait', None, 'A')
            reading = yield Msg('read', d)
            val = reading[target_field]['value']
            yield Msg('save')
            prompt = prompt_str.format(motor.name, pos, val, step)
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

    plan_stack = deque()
    with stage_context(plan_stack, [detector, motor]):
        with run_context(plan_stack, md):
            plan_stack.append(core())
    return plan_stack


def spiral_fermat(detectors, x_motor, y_motor, x_start, y_start, x_range,
                  y_range, dr, factor, *, per_step=None, md=None):
    '''Absolute fermat spiral scan, centered around (x_start, y_start)

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    x_motor : object
        any 'setable' object (motor, temp controller, etc.)
    y_motor : object
        any 'setable' object (motor, temp controller, etc.)
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
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        See docstring of bluesky.plans.one_nd_step (the default) for
        details.
    md : dict, optional
        metadata

    See Also
    --------
    `bluesky.plans.spiral`
    `bluesky.plans.relative_spiral`
    `bluesky.plans.relative_spiral_fermat`
    '''
    phi = 137.508 * np.pi / 180.

    half_x = x_range / 2
    half_y = y_range / 2

    x_points, y_points = [], []

    diag = np.sqrt(half_x ** 2 + half_y ** 2)
    num_rings = int((1.5 * diag / (dr / factor)) ** 2)
    for i_ring in range(1, num_rings):
        radius = np.sqrt(i_ring) * dr / factor
        angle = phi * i_ring
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        if abs(x) <= half_x and abs(y) <= half_y:
            x_points.append(x_start + x)
            y_points.append(y_start + y)

    cyc = cycler(x_motor, x_points)
    cyc += cycler(y_motor, y_points)
    yield from scan_nd(detectors, cyc, per_step=per_step, md=md)


def relative_spiral_fermat(detectors, x_motor, y_motor, x_range, y_range, dr,
                           factor, *, per_step=None, md=None):
    '''Relative fermat spiral scan

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    x_motor : object
        any 'setable' object (motor, temp controller, etc.)
    y_motor : object
        any 'setable' object (motor, temp controller, etc.)
    x_range : float
        x width of spiral
    y_range : float
        y width of spiral
    dr : float
        delta radius
    factor : float
        radius gets divided by this
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        See docstring of bluesky.plans.one_nd_step (the default) for
        details.
    md : dict, optional
        metadata

    See Also
    --------
    `bluesky.plans.spiral`
    `bluesky.plans.relative_spiral`
    `bluesky.plans.spiral_fermat`
    '''
    yield from spiral_fermat(detectors, x_motor, y_motor, x_motor.position,
                             y_motor.position, x_range, y_range, dr, factor,
                             per_step=per_step, md=md)


def spiral(detectors, x_motor, y_motor, x_start, y_start, x_range, y_range, dr,
           nth, *, per_step=None, md=None):
    '''Spiral scan, centered around (x_start, y_start)

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
        x width of spiral
    y_range : float
        y width of spiral
    dr : float
        Delta radius
    nth : float
        Number of theta steps
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        See docstring of bluesky.plans.one_nd_step (the default) for
        details.
    md : dict, optional
        metadata

    See Also
    --------
    `bluesky.plans.relative_spiral`
    `bluesky.plans.spiral_fermat`
    `bluesky.plans.relative_spiral_fermat`
    '''
    half_x = x_range / 2
    half_y = y_range / 2

    r_max = np.sqrt(half_x ** 2 + half_y ** 2)
    num_ring = 1 + int(r_max / dr)

    x_points = []
    y_points = []
    for i_ring in range(1, num_ring + 2):
        radius = i_ring * dr
        angle_step = 2. * np.pi / (i_ring * nth)

        for i_angle in range(int(i_ring * nth)):
            angle = i_angle * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            if abs(x) <= half_x and abs(y) <= half_y:
                x_points.append(x_start + x)
                y_points.append(y_start + y)

    cyc = cycler(x_motor, x_points)
    cyc += cycler(y_motor, y_points)
    yield from scan_nd(detectors, cyc, per_step=per_step, md=md)


def relative_spiral(detectors, x_motor, y_motor, x_range, y_range, dr, nth,
                    *, per_step=None, md=None):
    '''Relative spiral scan

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
        x width of spiral
    y_range : float
        y width of spiral
    dr : float
        Delta radius
    nth : float
        Number of theta steps
    per_step : callable, optional
        hook for cutomizing action of inner loop (messages per step)
        See docstring of bluesky.plans.one_nd_step (the default) for
        details.
    md : dict, optional
        metadata

    See Also
    --------
    `bluesky.plans.spiral`
    `bluesky.plans.spiral_fermat`
    '''
    yield from spiral(detectors, x_motor, y_motor, x_motor.position,
                      y_motor.position, x_range, y_range, dr, nth,
                      per_step=per_step, md=md)


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
        flyers = getattr(self, 'flyers', [])

        current_settings = {}
        for key, val in kwargs.items():
            current_settings[key] = getattr(self, key)
            setattr(self, key, val)
        try:
            plan_stack = deque()
            with stage_context(plan_stack, flyers):
                with subs_context(plan_stack, subs):
                    plan = self._gen()
                    plan_stack.append(fly_during(plan, flyers))
                    plan_stack.append(single_gen(Msg('checkpoint')))
        finally:
            for key, val in current_settings.items():
                setattr(self, key, val)

        for gen in plan_stack:
            yield from gen

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
    __doc__ = scan_nd.__doc__

    def _gen(self):
        return scan_nd(self.detectors, self.cycler)

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


class SpiralScan(Plan):
    _fields = ['detectors', 'x_motor', 'y_motor', 'x_start', 'y_start',
               'x_range', 'y_range', 'dr', 'nth']
    __doc__ = spiral.__doc__

    def _gen(self):
        return spiral(self.detectors, self.x_motor, self.y_motor, self.x_start,
                      self.y_start, self.x_range, self.y_range, self.dr,
                      self.nth)


class SpiralFermatScan(Plan):
    _fields = ['detectors', 'x_motor', 'y_motor', 'x_start', 'y_start',
               'x_range', 'y_range', 'dr', 'factor']
    __doc__ = spiral_fermat.__doc__

    def _gen(self):
        return spiral_fermat(self.detectors, self.x_motor, self.y_motor,
                             self.x_start, self.y_start, self.x_range,
                             self.y_range, self.dr, self.factor)


class RelativeSpiralScan(Plan):
    _fields = ['detectors', 'x_motor', 'y_motor', 'x_range', 'y_range', 'dr',
               'nth']
    __doc__ = relative_spiral.__doc__

    def _gen(self):
        return relative_spiral(self.detectors, self.x_motor, self.y_motor,
                               self.x_range, self.y_range, self.dr, self.nth)


class RelativeSpiralFermatScan(Plan):
    _fields = ['detectors', 'x_motor', 'y_motor', 'x_range', 'y_range', 'dr',
               'factor']
    __doc__ = relative_spiral_fermat.__doc__

    def _gen(self):
        return relative_spiral_fermat(self.detectors, self.x_motor,
                                      self.y_motor, self.x_range, self.y_range,
                                      self.dr, self.factor)
