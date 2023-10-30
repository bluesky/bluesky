from __future__ import generator_stop

from collections import OrderedDict, deque, ChainMap
from collections.abc import Iterable
import uuid

from bluesky.protocols import Locatable
from .utils import (get_hinted_fields, normalize_subs_input, root_ancestor,
                    separate_devices,
                    Msg, ensure_generator, single_gen,
                    short_uid as _short_uid, make_decorator,
                    RunEngineControlException, merge_axis)
from functools import wraps
from .plan_stubs import (open_run, close_run, mv, pause, trigger_and_read,
                         declare_stream, stage_all, unstage_all)


def plan_mutator(plan, msg_proc):
    """
    Alter the contents of a plan on the fly by changing or inserting messages.

    Parameters
    ----------
    plan : generator

        a generator that yields messages (`Msg` objects)
    msg_proc : callable
        This function takes in a message and specifies messages(s) to replace
        it with. The function must account for what type of response the
        message would prompt. For example, an 'open_run' message causes the
        RunEngine to send a uid string back to the plan, while a 'set' message
        causes the RunEngine to send a status object back to the plan. The
        function should return a pair of generators ``(head, tail)`` that yield
        messages. The last message out of the ``head`` generator is the one
        whose response will be sent back to the host plan. Therefore, that
        message should prompt a response compatible with the message that it is
        replacing. Any responses to all other messages will be swallowed. As
        shorthand, either ``head`` or ``tail`` can be replaced by ``None``.
        This means:

        * ``(None, None)`` No-op. Let the original message pass through.
        * ``(head, None)`` Mutate and/or insert messages before the original
          message.
        * ``(head, tail)`` As above, and additionally insert messages after.
        * ``(None, tail)`` Let the original message pass through and then
          insert messages after.

        The reason for returning a pair of generators instead of just one is to
        provide a way to specify which message's response should be sent out to
        the host plan. Again, it's the last message yielded by the first
        generator (``head``).

    Yields
    ------
    msg : Msg
        messages from `plan`, altered by `msg_proc`

    See Also
    --------
    :func:`bluesky.plans.msg_mutator`
    """
    # internal stacks
    msgs_seen = dict()
    plan_stack = deque()
    result_stack = deque()
    tail_cache = dict()
    tail_result_cache = dict()
    exception = None

    parent_plan = plan
    ret_value = None
    # seed initial conditions
    plan_stack.append(plan)
    result_stack.append(None)

    while True:
        # get last result
        if exception is not None:
            # if we have a stashed exception, pass it along
            try:
                msg = plan_stack[-1].throw(exception)
            except StopIteration as e:
                # discard the exhausted generator
                exhausted_gen = plan_stack.pop()
                # if this is the parent plan, capture it's return value
                if exhausted_gen is parent_plan:
                    ret_value = e.value

                # if we just came out of a 'tail' generator,
                # discard its return value and replace it with the
                # cached one (from the last message in its paired
                # 'new_gen')
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
                    return ret_value
            except Exception as e:
                # if we catch an exception,
                # the current top plan is dead so pop it
                plan_stack.pop()
                if plan_stack:
                    # stash the exception and go to the top
                    exception = e
                    continue
                else:
                    raise
            else:
                exception = None
        else:
            ret = result_stack.pop()
            try:
                msg = plan_stack[-1].send(ret)
            except StopIteration as e:
                # discard the exhausted generator
                exhausted_gen = plan_stack.pop()
                # if this is the parent plan, capture it's return value
                if exhausted_gen is parent_plan:
                    ret_value = e.value

                # if we just came out of a 'tail' generator,
                # discard its return value and replace it with the
                # cached one (from the last message in its paired
                # 'new_gen')
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
                    return ret_value
            except Exception as ex:
                # we are here because an exception came out of the send
                # this may be due to
                # a) the plan really raising or
                # b) an exception that came out of the run engine via ophyd

                # in either case the current plan is dead so pop it
                failed_gen = plan_stack.pop()
                if id(failed_gen) in tail_cache:
                    gen = tail_cache.pop(id(failed_gen))
                    if gen is not None:
                        plan_stack.append(gen)
                # if there is at least
                if plan_stack:
                    exception = ex
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
                new_gen = single_gen(msg)
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
        except GeneratorExit:
            # special case GeneratorExit.  We must clean up all of our plans
            # and exit with out yielding anything else.
            for p in plan_stack:
                p.close()
            raise
        except Exception as ex:
            if plan_stack:
                exception = ex
                continue
            else:
                raise
        else:
            result_stack.append(inner_ret)


def msg_mutator(plan, msg_proc):
    """
    A simple preprocessor that mutates or deletes single messages in a plan.

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

    See Also
    --------
    :func:`bluesky.plans.plan_mutator`
    """
    try:
        msg = plan.send(None)
    except StopIteration as _e:
        ret = _e.value
    else:
        while 1:
            try:
                msg = msg_proc(msg)
                # if None, just skip message
                # feed 'None' back down into the base plan,
                # this may break some plans
                if msg is None:
                    _s = None
                else:
                    _s = yield msg
            except GeneratorExit:
                plan.close()
                raise
            except BaseException as _e:
                try:
                    msg = plan.throw(_e)
                except StopIteration as _e:
                    ret = _e.value
                    break
            else:
                try:
                    msg = plan.send(_s)
                except StopIteration as _e:
                    ret = _e.value
                    break

    return ret


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


def print_summary_wrapper(plan):
    """Print summary of plan as it goes by

    Prints a minimal version of the plan, showing only moves and
    where events are created.  Yields the `Msg` unchanged.

    Parameters
    ----------
    plan : iterable
        Must yield `Msg` objects

    Yields
    ------
    msg : `Msg`
    """

    read_cache = []

    def spy(msg):
        nonlocal read_cache

        cmd = msg.command
        if cmd == 'open_run':
            print('{:=^80}'.format(' Open Run '))
        elif cmd == 'close_run':
            print('{:=^80}'.format(' Close Run '))
        elif cmd == 'set':
            print('{motor.name} -> {args[0]}'.format(motor=msg.obj,
                                                     args=msg.args))
        elif cmd == 'create':
            read_cache = []
        elif cmd == 'read':
            read_cache.append(msg.obj.name)
        elif cmd == 'save':
            print('  Read {}'.format(read_cache))
        return msg

    return (yield from msg_mutator(plan, spy))


def run_wrapper(plan, *, md=None):
    """Enclose in 'open_run' and 'close_run' messages.

    Parameters
    ----------
    plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects
    md : dict, optional
        metadata to be passed into the 'open_run' message
    """
    rs_uid = yield from open_run(md)

    def except_plan(e):
        if isinstance(e, RunEngineControlException):
            yield from close_run(exit_status=e.exit_status)
        else:
            yield from close_run(exit_status='fail', reason=str(e))

    yield from contingency_wrapper(plan,
                                   except_plan=except_plan,
                                   else_plan=close_run)
    return rs_uid


def subs_wrapper(plan, subs):
    """
    Subscribe callbacks to the document stream; finally, unsubscribe.

    Parameters
    ----------
    plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects
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

         Signature of functions must conform to `f(name, doc)` where
         name is one of {'all', 'start', 'stop', 'event', 'descriptor'} and
         doc is a dictionary.

    Yields
    ------
    msg : Msg
        messages from plan, with 'subscribe' and 'unsubscribe' messages
        inserted and appended
    """
    subs = normalize_subs_input(subs)
    tokens = set()

    def _subscribe():
        for name, funcs in subs.items():
            for func in funcs:
                token = yield Msg('subscribe', None, func, name)
                tokens.add(token)

    def _unsubscribe():
        for token in tokens:
            yield Msg('unsubscribe', None, token=token)

    def _inner_plan():
        yield from _subscribe()
        return (yield from plan)

    return (yield from finalize_wrapper(_inner_plan(),
                                        _unsubscribe()))


def suspend_wrapper(plan, suspenders):
    """
    Install suspenders to the RunEngine, and remove them at the end.

    Parameters
    ----------
    plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects
    suspenders : suspender or list of suspenders
        Suspenders to use for the duration of the wrapper

    Yields
    ------
    msg : Msg
        messages from plan, with 'install_suspender' and 'remove_suspender'
        messages inserted and appended
    """
    if not isinstance(suspenders, Iterable):
        suspenders = [suspenders]

    def _install():
        for susp in suspenders:
            yield Msg('install_suspender', None, susp)

    def _remove():
        for susp in suspenders:
            yield Msg('remove_suspender', None, susp)

    def _inner_plan():
        yield from _install()
        return (yield from plan)

    return (yield from finalize_wrapper(_inner_plan(),
                                        _remove()))


def configure_count_time_wrapper(plan, time):
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
                # TODO do this with configure
                return pchain(mv(obj.count_time, time),
                              single_gen(msg)), None
        return None, None

    def reset():
        for obj, time in original_times.items():
            yield from mv(obj.count_time, time)

    if time is None:
        # no-op
        return (yield from plan)
    else:
        return (yield from finalize_wrapper(plan_mutator(plan, insert_set),
                                            reset()))


def finalize_wrapper(plan, final_plan, *, pause_for_debug=False):
    '''try...finally helper

    Run the first plan and then the second.  If any of the messages
    raise an error in the RunEngine (or otherwise), the second plan
    will attempted to be run anyway.

    See :func:`contingency_wrapper` for a more complex and more
    feature-complete error-handling preprocessor.

    Parameters
    ----------
    plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects
    final_plan : callable, iterable or iterator
        a generator, list, or similar containing `Msg` objects or a callable
        that reurns one; attempted to be run no matter what happens in the
        first plan
    pause_for_debug : bool, optional
        If the plan should pause before running the clean final_plan in
        the case of an Exception.  This is intended as a debugging tool only.

    Yields
    ------
    msg : Msg
        messages from `plan` until it terminates or an error is raised, then
        messages from `final_plan`

    See Also
    --------
    :func:`contingency_wrapper`
    '''
    # If final_plan is a generator *function* (as opposed to a generator
    # *instance*), call it.
    if callable(final_plan):
        final_plan_instance = final_plan()
    else:
        final_plan_instance = final_plan
    cleanup = True
    try:
        ret = yield from plan
    except GeneratorExit:
        cleanup = False
        raise
    except BaseException:
        if pause_for_debug:
            yield from pause()
        raise
    finally:
        # if the exception raised in `GeneratorExit` that means
        # someone called `gen.close()` on this generator.  In those
        # cases generators must either re-raise the GeneratorExit or
        # raise a different exception.  Trying to yield any values
        # results in a RuntimeError being raised where `close` is
        # called.  Thus, we catch, the GeneratorExit, disable cleanup
        # and then re-raise

        # https://docs.python.org/3/reference/expressions.html?#generator.close
        if cleanup:
            yield from ensure_generator(final_plan_instance)
    return ret


def contingency_wrapper(plan, *,
                        except_plan=None,
                        else_plan=None,
                        final_plan=None,
                        pause_for_debug=False,
                        auto_raise=True):
    '''try...except...else...finally helper

    See :func:`finalize_wrapper` for a simplified but less powerful
    error-handling preprocessor.

    Parameters
    ----------
    plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects
    except_plan : generator function, optional
        This will be called with the exception as the only input.  The
        plan does not need to re-raise, but may if you want to change the
        exception.

        Only subclasses of `Exception` will be passed in, will not see
        `GeneratorExit`, `SystemExit`, or `KeyboardInterrupt`
    else_plan : generator function, optional
        This will be called with no arguments if plan completes without raising
    final_plan : generator function, optional
        a generator, list, or similar containing `Msg` objects or a callable
        that reurns one; attempted to be run no matter what happens in the
        first plan
    pause_for_debug : bool, optional
        If the plan should pause before running the clean final_plan in
        the case of an Exception.  This is intended as a debugging tool only.
    auto_raise : bool, optional
        If the exception should be always be re-raised, reagardless of what
        except_plan does. Note this defaults to True for backwards compatibility,
        which is not the usual behaviour of an except statement

    Yields
    ------
    msg : Msg
        messages from `plan` until it terminates or an error is raised, then
        messages from `final_plan`

    See Also
    --------
    :func:`finalize_wrapper`
    '''
    cleanup = True
    try:
        ret = yield from plan
    except GeneratorExit:
        cleanup = False
        raise
    except Exception as e:
        if pause_for_debug:
            yield from pause()
        if except_plan:
            # it might be better to throw this in, but this is simpler
            # to implement for now
            ret = yield from except_plan(e)
            if auto_raise:
                raise
            else:
                return ret
        else:
            raise
    else:
        if else_plan:
            yield from else_plan()
    finally:
        # if the exception raised in `GeneratorExit` that means
        # someone called `gen.close()` on this generator.  In those
        # cases generators must either re-raise the GeneratorExit or
        # raise a different exception.  Trying to yield any values
        # results in a RuntimeError being raised where `close` is
        # called.  Thus, we catch, the GeneratorExit, disable cleanup
        # and then re-raise

        # https://docs.python.org/3/reference/expressions.html?#generator.close
        if cleanup and final_plan:
            yield from final_plan()
    return ret


def finalize_decorator(final_plan):
    '''try...finally helper

    Run the first plan and then the second.  If any of the messages
    raise an error in the RunEngine (or otherwise), the second plan
    will attempted to be run anyway.

    Notice that, this decorator requires a generator *function* so that it can
    be used multiple times, whereas :func:`bluesky.plans.finalize_wrapper`
    accepts either a generator function or a generator instance.

    Parameters
    ----------
    final_plan : callable
        a callable that returns a generator, list, or similar containing `Msg`
        objects; attempted to be run no matter what happens in the first plan

    Yields
    ------
    msg : Msg
        messages from `plan` until it terminates or an error is raised, then
        messages from `final_plan`
    '''
    def dec(gen_func):
        @wraps(gen_func)
        def dec_inner(*inner_args, **inner_kwargs):
            if not callable(final_plan):
                raise TypeError("final_plan must be a callable (e.g., a "
                                "generator function) not an iterable (e.g., a "
                                "generator instance).")
            final_plan_instance = final_plan()
            plan = gen_func(*inner_args, **inner_kwargs)
            cleanup = True
            try:
                ret = yield from plan
            except GeneratorExit:
                cleanup = False
                raise
            finally:
                # if the exception raised in `GeneratorExit` that means
                # someone called `gen.close()` on this generator.  In those
                # cases generators must either re-raise the GeneratorExit or
                # raise a different exception.  Trying to yield any values
                # results in a RuntimeError being raised where `close` is
                # called.  Thus, we catch, the GeneratorExit, disable cleanup
                # and then re-raise

                # https://docs.python.org/3/reference/expressions.html?#generator.close
                if cleanup:
                    yield from ensure_generator(final_plan_instance)
            return ret
        return dec_inner
    return dec


def rewindable_wrapper(plan, rewindable):
    '''Toggle the 'rewindable' state of the RE

    Allow or disallow rewinding during the processing of the wrapped messages.
    Then restore the initial state (rewindable or not rewindable).

    Parameters
    ----------
    plan : generator
        The plan to wrap in a 'rewindable' or 'not rewindable' context
    rewindable : bool

    '''
    initial_rewindable = True

    def capture_rewindable_state():
        nonlocal initial_rewindable
        initial_rewindable = yield Msg('rewindable', None, None)

    def set_rewindable(rewindable):
        if initial_rewindable != rewindable:
            return (yield Msg('rewindable', None, rewindable))

    def restore_rewindable():
        if initial_rewindable != rewindable:
            return (yield Msg('rewindable', None, initial_rewindable))

    if not rewindable:
        yield from capture_rewindable_state()
        yield from set_rewindable(rewindable)
        return (yield from finalize_wrapper(plan,
                                            restore_rewindable()))
    else:
        return (yield from plan)


def inject_md_wrapper(plan, md):
    """
    Inject additional metadata into a run.

    This takes precedences over the original metadata dict in the event of
    overlapping keys, but it does not mutate the original metadata dict.
    (It uses ChainMap.)

    Parameters
    ----------
    plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects
    md : dict
        metadata
    """
    def _inject_md(msg):
        if msg.command == 'open_run':
            msg = msg._replace(kwargs=ChainMap(md, msg.kwargs))
        return msg

    return (yield from msg_mutator(plan, _inject_md))


def stub_wrapper(plan):
    """
    Remove Msg object in order to use plan as a stub

    This will remove any `open_run`, `close_run`, `stage` and `unstage` `Msg`
    objects present in the plan in order for it to be run as part of a larger
    scan. Note, that any metadata from the provided plan will not be sent to
    the RunEngine automatically.

    Parameters
    ----------
    plan : iterable or iterator
        A generator list or similar containing `Msg` objects

    Returns
    -------
    md : dict
        Metadata discovered from `open_run` Msg
    """
    md = {}

    def _block_run_control(msg):
        """
        Block open and close run messages
        """
        # Capture the metadata from open_run
        if msg.command == 'open_run':
            md.update(msg.kwargs)
            return None
        elif msg.command in ('close_run', 'stage', 'unstage'):
            return None
        return msg

    yield from msg_mutator(plan, _block_run_control)
    return md


def monitor_during_wrapper(plan, signals):
    """
    Monitor (asynchronously read) devices during runs.

    This is a preprocessor that insert messages immediately after a run is
    opened and before it is closed.

    Parameters
    ----------
    plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects
    signals : collection
        objects that support the Signal interface

    Yields
    ------
    msg : Msg
        messages from plan with 'monitor', and 'unmontior' messages inserted

    See Also
    --------
    :func:`bluesky.plans.fly_during_wrapper`
    """
    monitor_msgs = [Msg('monitor', sig, name=sig.name + '_monitor')
                    for sig in signals]
    unmonitor_msgs = [Msg('unmonitor', sig) for sig in signals]

    def insert_after_open(msg):
        if msg.command == 'open_run':
            def new_gen():
                yield from ensure_generator(monitor_msgs)
            return single_gen(msg), new_gen()
        else:
            return None, None

    def insert_before_close(msg):
        if msg.command == 'close_run':
            def new_gen():
                yield from ensure_generator(unmonitor_msgs)
                yield msg
            return new_gen(), None
        else:
            return None, None

    # Apply nested mutations.
    plan1 = plan_mutator(plan, insert_after_open)
    plan2 = plan_mutator(plan1, insert_before_close)
    return (yield from plan2)


def fly_during_wrapper(plan, flyers):
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

    See Also
    --------
    :func:`bluesky.plans.fly`
    """
    grp1 = _short_uid('flyers-kickoff')
    grp2 = _short_uid('flyers-complete')
    kickoff_msgs = [Msg('kickoff', flyer, group=grp1) for flyer in flyers]
    complete_msgs = [Msg('complete', flyer, group=grp2) for flyer in flyers]
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


def lazily_stage_wrapper(plan):
    """
    This is a preprocessor that inserts 'stage' messages and appends 'unstage'.

    The first time an object is seen in `plan`, it is staged. To avoid
    redundant staging we actually stage the object's ultimate parent.

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
            root = root_ancestor(msg.obj)

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

    def inner_unstage_all():
        yield from unstage_all(*reversed(devices_staged))

    return (yield from finalize_wrapper(plan_mutator(plan, inner),
                                        inner_unstage_all()))


def stage_wrapper(plan, devices):
    """
    'Stage' devices (i.e., prepare them for use, 'arm' them) and then unstage.

    Parameters
    ----------
    plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects
    devices : collection
        list of devices to stage immediately on entrance and unstage on exit

    Yields
    ------
    msg : Msg
        messages from plan with 'stage' and finally 'unstage' messages inserted

    See Also
    --------
    :func:`bluesky.plans.lazily_stage_wrapper`
    :func:`bluesky.plans.stage`
    :func:`bluesky.plans.unstage`
    """
    devices = separate_devices(root_ancestor(device) for device in devices)

    def stage_devices():
        yield from stage_all(*devices)

    def unstage_devices():
        yield from unstage_all(*reversed(devices))

    def inner():
        yield from stage_devices()
        return (yield from plan)

    return (yield from finalize_wrapper(inner(), unstage_devices()))


def _normalize_devices(devices):
    coupled_parents = set()
    # if we have any pseudo devices then setting any part of it
    # needs to trigger the relative behavior.
    io, co, go = merge_axis(devices)
    devices = set(devices) | set(io) | set(co) | set(go)
    # if a device with coupled children is directly in the
    # list, include all the coupled children as well
    for obj in co:
        devices |= set(obj.pseudo_positioners)
        coupled_parents.add(obj)

    # if at least one child of a device with coupled children
    # only include the coupled children if at least of the children
    # directly included is one of the coupled ones.
    for obj, type_map in go.items():
        if len(type_map['pseudo']) > 0:
            devices |= set(obj.pseudo_positioners)
            coupled_parents.add(obj)
    return devices, coupled_parents


def __get_result_of_message(msg_type: str, obj):
    result = yield Msg(msg_type, obj)
    if result is None:
        # this plan may be being list-ified
        print(f"*** all positions for {obj.name} are relative to current position ***")
    return result


def __read_and_stash_a_motor(obj, initial_positions, coupled_parents):
    """Internal plan for relative set and reset wrappers


    .. warning ::

       Do not use this plan directly for any reason.

    """
    # First check if it is a locatable, and use its location if it is
    if isinstance(obj, Locatable):
        location = yield from __get_result_of_message("locate", obj)
        if location is None:
            setpoint = 0
        else:
            setpoint = location["setpoint"]
    # Otherwise it might have a `position` attribution
    elif hasattr(obj, "position"):
        setpoint = obj.position
    # Otherwise fallback to read obj and grab the value of the first key
    else:
        reading = yield from __get_result_of_message("read", obj)
        if reading is None:
            setpoint = 0
        else:
            fields = get_hinted_fields(obj)
            if len(fields) == 1:
                k, = fields
                setpoint = reading[k]['value']
            elif len(fields) == 0:
                k = list(reading.keys())[0]
                setpoint = reading[k]['value']
            else:
                raise Exception("do not yet know how to deal with "
                                "non pseudopositioner multi-axis.  Please "
                                "contact DAMA to justify why you need "
                                "this.")

    initial_positions[obj] = setpoint

    # if we move a pseudo positioner also stash it's children
    if obj in coupled_parents:
        for c, p in zip(obj.pseudo_positioners, setpoint):
            initial_positions[c] = p

    # if we move a pseudo single, also stash it's parent and siblings
    parent = obj.parent
    if parent in coupled_parents and obj in parent.pseudo_positioners:
        parent_pos = parent.position
        initial_positions[parent] = parent_pos
        for c, p in zip(parent.pseudo_positioners, parent_pos):
            initial_positions[c] = p

    # TODO forbid mixed pseudo / real motion


def relative_set_wrapper(plan, devices=None):
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
    if devices is not None:
        devices, coupled_parents = _normalize_devices(devices)
    else:
        coupled_parents = set()

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
            return (pchain(
                __read_and_stash_a_motor(
                    msg.obj, initial_positions, coupled_parents),
                single_gen(msg)), None)
        else:
            return None, None

    plan = plan_mutator(plan, insert_reads)
    plan = msg_mutator(plan, rewrite_pos)
    return (yield from plan)


def reset_positions_wrapper(plan, devices=None):
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
    if devices is not None:
        devices, coupled_parents = _normalize_devices(devices)
    else:
        coupled_parents = set()

    def insert_reads(msg):
        eligible = devices is None or msg.obj in devices
        seen = msg.obj in initial_positions
        if (msg.command == 'set') and eligible and not seen:
            return (pchain(
                    __read_and_stash_a_motor(
                        msg.obj, initial_positions, coupled_parents),
                    single_gen(msg)), None)
        else:
            return None, None

    def reset():
        blk_grp = 'reset-{}'.format(str(uuid.uuid4())[:6])
        for k, v in initial_positions.items():
            if k.parent in coupled_parents:
                continue
            yield Msg('set', k, v, group=blk_grp)
        yield Msg('wait', None, group=blk_grp)

    return (yield from finalize_wrapper(plan_mutator(plan, insert_reads),
                                        reset()))


def baseline_wrapper(plan, devices, name='baseline'):
    """
    Preprocessor that records a baseline of all `devices` after `open_run`

    The readings are designated for a separate event stream named 'baseline' by
    default.

    Parameters
    ----------
    plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects
    devices : collection
        collection of Devices to read
        If None, the plan passes through unchanged.
    name : string, optional
        name for event stream; by default, 'baseline'

    Yields
    ------
    msg : Msg
        messages from plan, with 'set' messages inserted
    """
    def head():
        yield from declare_stream(*devices, name=name)
        yield from trigger_and_read(devices, name=name)

    def insert_baseline(msg):
        if msg.command == 'open_run':
            return None, head()

        elif msg.command == 'close_run':
            def post_baseline():
                yield from trigger_and_read(devices, name=name)
                return (yield msg)

            return post_baseline(), None

        return None, None

    if not devices:
        # no-op
        return (yield from plan)
    else:
        return (yield from plan_mutator(plan, insert_baseline))


# Make generator function decorator for each generator instance wrapper.
baseline_decorator = make_decorator(baseline_wrapper)
subs_decorator = make_decorator(subs_wrapper)
suspend_decorator = make_decorator(suspend_wrapper)
relative_set_decorator = make_decorator(relative_set_wrapper)
reset_positions_decorator = make_decorator(reset_positions_wrapper)
# finalize_decorator is custom-made since it takes a plan as its
# argument. See its docstring for details why.
lazily_stage_decorator = make_decorator(lazily_stage_wrapper)
stage_decorator = make_decorator(stage_wrapper)
fly_during_decorator = make_decorator(fly_during_wrapper)
monitor_during_decorator = make_decorator(monitor_during_wrapper)
inject_md_decorator = make_decorator(inject_md_wrapper)
run_decorator = make_decorator(run_wrapper)
contingency_decorator = make_decorator(contingency_wrapper)
stub_decorator = make_decorator(stub_wrapper)
configure_count_time_decorator = make_decorator(configure_count_time_wrapper)


class SupplementalData:
    """
    A configurable preprocessor for supplemental measurements

    This is a plan preprocessor. It inserts messages into plans to:

    * take "baseline" readings at the beginning and end of each run for the
      devices listed in its ``baseline`` atrribute
    * monitor signals in its ``monitors`` attribute for asynchronous
      updates during each run.
    * kick off "flyable" devices listed in its ``flyers`` attribute at the
      beginning of each run and collect their data at the end

    Internally, it uses the plan preprocessors:

    * :func:`baseline_wrapper`
    * :func:`monitor_during_wrapper`
    * :func:`fly_during_wrapper`

    Parameters
    ----------
    baseline : list
        Devices to be read at the beginning and end of each run
    monitors : list
        Signals (not multi-signal Devices) to be monitored during each run,
        generating readings asynchronously
    flyers : list
        "Flyable" Devices to be kicked off before each run and collected
        at the end of each run

    Examples
    --------
    Create an instance of SupplementalData and apply it to a RunEngine.

    >>> sd = SupplementalData(baseline=[some_motor, some_detector]),
    ...                       monitors=[some_signal],
    ...                       flyers=[some_flyer])
    >>> RE = RunEngine({})
    >>> RE.preprocessors.append(sd)

    Now all plans executed by RE will be modified to add baseline readings
    (before and after each run), monitors (during each run), and flyers
    (kicked off before each run and collected afterward).

    Inspect or update the lists of devices interactively.

    >>> sd.baseline
    [some_motor, some_detector]

    >>> sd.baseline.remove(some_motor)

    >>> sd.baseline
    [some_detector]

    >>> sd.baseline.append(another_detector)

    >>> sd.baseline
    [some_detector, another_detector]

    Each attribute (``baseline``, ``monitors``, ``flyers``) is an ordinary
    Python list, support all the standard list methods, such as:

    >>> sd.baseline.clear()

    The arguments to SupplementalData are optional. All the lists
    will empty by default.  As shown above, they can be populated
    interactively.

    >>> sd = SupplementalData()
    >>> RE = RunEngine({})
    >>> RE.preprocessors.append(sd)
    >>> sd.baseline.append(some_detector)
    """
    def __init__(self, *, baseline=None, monitors=None, flyers=None):
        if baseline is None:
            baseline = []
        if monitors is None:
            monitors = []
        if flyers is None:
            flyers = []
        self.baseline = list(baseline)
        self.monitors = list(monitors)
        self.flyers = list(flyers)

    def __repr__(self):
        return ("{cls}(baseline={baseline}, monitors={monitors}, "
                "flyers={flyers})"
                "").format(cls=type(self).__name__, **vars(self))

    # I'm not sure why anyone would want to pickle this but it's good manners
    # to avoid breaking pickling.

    def __setstate__(self, state):
        baseline, monitors, flyers = state
        self.baseline = baseline
        self.monitors = monitors
        self.flyers = flyers

    def __getstate__(self):
        return (self.baseline, self.monitors, self.flyers)

    def __call__(self, plan):
        """
        Insert messages into a plan.

        Parameters
        ----------
        plan : iterable or iterator
            a generator, list, or similar containing `Msg` objects
        """
        # Read this as going from the inside out: first we wrap the plan in the
        # flying instructions, then monitoring, then baseline, so that the
        # order of operations is:
        # - Take baseline readings
        # - Start monitoring.
        # - Kick off flyers and wait for them to be kicked off.
        # - Do `plan`.
        # - Complete and collect flyers.
        # - Stop monitoring.
        # - Take baseline readings.
        plan = fly_during_wrapper(plan, self.flyers)
        plan = monitor_during_wrapper(plan, self.monitors)
        plan = baseline_wrapper(plan, self.baseline)
        return (yield from plan)


def set_run_key_wrapper(plan, run):
    """
    Add a run key to each message in wrapped plan

    Parameters
    ----------
    plan : iterable or iterator
        a generator, list, or similar containing `Msg` objects
    run : str or any other type except None
        The run key to set on each Msg. It is recommended that run key represents
        informative string for better readability of plans. But value of any other
        type can be used if needed.
    """
    if run is None:
        raise ValueError("run ID can not be None")

    def _set_run_key(msg):
        # Replace only the default value None
        if msg.run is None:
            msg = msg._replace(run=run)
        return msg

    return (yield from msg_mutator(plan, _set_run_key))


set_run_key_decorator = make_decorator(set_run_key_wrapper)
