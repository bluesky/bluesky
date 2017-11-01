# DO NOT USE THIS MODULE.

# This module contians a legacy API, an early approach to composing plans that
# was little used and finally deprecated in v0.10.0. It will be removed in a
# future release. It should not be used.

from functools import wraps
from contextlib import contextmanager
import warnings

from .utils import (normalize_subs_input, root_ancestor,
                    separate_devices,
                    Msg, single_gen)

from .plan_stubs import (broadcast_msg, trigger_and_read)


def planify(func):
    """Turn a function that returns a list of generators into a coroutine.

    Parameters
    ----------
    func : callable
        expected to return a list of generators that yield messages (`Msg`
        objects) the function may have an arbitrary signature

    Returns
    -------
    gen : generator
        a single generator that yields messages. The return value from
        the generator is the return of the last plan in the plan
        stack.

    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        gen_stack = func(*args, **kwargs)
        ret = None
        for g in gen_stack:
            ret = yield from g
        return ret

    return wrapped


@contextmanager
def subs_context(plan_stack, subs):
    """
    Subscribe callbacks to the document stream; then unsubscribe on exit.

    .. deprecated:: 0.10.0
        Use :func:`subs_wrapper` or :func:`subs_decorator` instead.

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
    warnings.warn("subs_context is deprecated. "
                  "Use subs_wrapper or subs_decorator.")
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

    plan_stack.append(_subscribe())
    try:
        yield plan_stack
    finally:
        # The RunEngine might never process these if the execution fails,
        # but it keeps its own cache of tokens and will try to remove them
        # itself if this plan fails to do so.
        plan_stack.append(_unsubscribe())


@contextmanager
def run_context(plan_stack, *, md=None):
    """Enclose in 'open_run' and 'close_run' messages.

    .. deprecated:: 0.10.0
        Use :func:`run_wrapper` or :func:`run_decorator` instead.

    Parameters
    ----------
    plan_stack : list-like
        appendable collection of generators that yield messages (`Msg` objects)
    md : dict, optional
        metadata to be passed into the 'open_run' message
    """
    warnings.warn(
        "run_context is deprecated. Use run_wrapper or run_decorator.")
    plan_stack.append(single_gen(Msg('open_run', None, **dict(md or {}))))
    yield plan_stack
    plan_stack.append(single_gen(Msg('close_run')))


@contextmanager
def event_context(plan_stack, name='primary'):
    """Bundle readings into an 'event' (a datapoint).

    This encloses the contents in 'create' and 'save' messages.

    .. deprecated:: 0.10.0
        Use the :func:`create` and :func:`save` plans directly. Also,
        :func:`trigger_and_read` addresses the common case of reading one or
        more devices into one Event.

    Parameters
    ----------
    plan_stack : list-like
        appendable collection of generators that yield messages (`Msg` objects)
    name : string, optional
        name of event stream; default is 'primary'
    """
    warnings.warn(
        "event_context is deprecated. Use create, save, or trigger_and_read.")
    plan_stack.append(single_gen(Msg('create', None, name=name)))
    yield plan_stack
    plan_stack.append(single_gen(Msg('save')))


@contextmanager
def stage_context(plan_stack, devices):
    """
    Stage devices upon entering context and unstage upon exiting.

    .. deprecated:: 0.10.0
        Use :func:`stage_wrapper` or :func:`stage_decorator`.

    Parameters
    ----------
    plan_stack : list-like
        appendable collection of generators that yield messages (`Msg` objects)
    devices : collection
        list of devices to stage immediately on entrance and unstage on exit

    See Also
    --------
    :func:`bluesky.plans.lazily_stage`
    """
    warnings.warn("stage_context is deprecated. "
                  "Use stage_wrapper or stage_decorator.")
    # Resolve unique devices, avoiding redundant staging.
    devices = separate_devices(root_ancestor(device) for device in devices)

    def stage():
        # stage devices explicitly passed to 'devices' argument
        yield from broadcast_msg('stage', devices)

    def unstage():
        # unstage devices explicitly passed to 'devices' argument
        yield from broadcast_msg('unstage', reversed(devices))

    plan_stack.append(stage())
    yield plan_stack
    plan_stack.append(unstage())


@contextmanager
def baseline_context(plan_stack, devices, name='baseline'):
    """
    Read every device once upon entering and exiting the context.

    .. deprecated:: 0.10.0
        Use :func:`baseline_wrapper` or :func:`baseline_decorator`.

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
    warnings.warn("baseline_context is deprecated. Use baseline_wrapper or "
                  "baseline_decorator.")
    plan_stack.append(trigger_and_read(devices, name=name))
    yield
    plan_stack.append(trigger_and_read(devices, name=name))


@contextmanager
def monitor_context(plan_stack, signals):
    """
    Asynchronously monitor signals, generating separate event streams.

    .. deprecated:: 0.10.0
        Use :func:`monitor_wrapper` or :func:`monitor_decorator`.

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
    warnings.warn("monitor_context is deprecated. Use monitor_wrapper or "
                  "monitor_decorator.")
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
