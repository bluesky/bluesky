import asyncio
import datetime
import inspect
import itertools
import operator
import os
import sys
import uuid
import warnings
from functools import reduce, wraps
from typing import Any, AsyncIterable, AsyncIterator, Dict, Iterable, List, Optional, Union

import numpy as np
from cycler import cycler

from bluesky.protocols import (
    Asset,
    HasHints,
    HasParent,
    Hints,
    Movable,
    Readable,
    StreamAsset,
    SyncOrAsync,
    SyncOrAsyncIterator,
    T,
    WritesExternalAssets,
    WritesStreamAssets,
    check_supports,
)

try:
    # cytools is a drop-in replacement for toolz, implemented in Cython
    from cytools import groupby
except ImportError:
    from toolz import groupby


SUBS_NAMES = ["all", "start", "stop", "event", "descriptor"]


def normalize_subs_input(subs):
    "Accept a callable, a list, or a dict. Normalize to a dict of lists."
    normalized = {name: [] for name in SUBS_NAMES}
    if subs is None:
        pass
    elif callable(subs):
        normalized["all"].append(subs)
    elif hasattr(subs, "items"):
        for key, funcs in list(subs.items()):
            if key not in SUBS_NAMES:
                raise KeyError(f"Keys must be one of {SUBS_NAMES!r:0}")
            if callable(funcs):
                normalized[key].append(funcs)
            else:
                normalized[key].extend(funcs)
    elif isinstance(subs, Iterable):
        normalized["all"].extend(subs)
    else:
        raise ValueError(
            "Subscriptions should be a callable, a list of "
            "callables, or a dictionary mapping subscription "
            "names to lists of callables."
        )
    # Validates that all entries are callables.
    for name, funcs in normalized.items():  # noqa: B007
        for func in funcs:
            if not callable(func):
                raise ValueError(
                    "subs values must be functions or lists of functions. The offending entry is\n " f"{func}"
                )
    return normalized


def snake_cyclers(cyclers, snake_booleans):
    """
    Combine cyclers with a 'snaking' back-and-forth order.

    Parameters
    ----------
    cyclers : cycler.Cycler
        or any iterable that yields dictionaries of lists
    snake_booleans : list
        a list of the same length as cyclers indicating whether each cycler
        should 'snake' (True) or not (False). Note that the first boolean
        does not make a difference because the first (slowest) dimension
        does not repeat.

    Returns
    -------
    result : cycler
    """
    if len(cyclers) != len(snake_booleans):
        raise ValueError("number of cyclers does not match number of booleans")
    if not any(snake_booleans[1:]):
        return reduce(operator.mul, cyclers)
    lengths = []
    new_cyclers = []
    for c in cyclers:
        lengths.append(len(c))
    total_length = np.prod(lengths)
    for i, (c, snake) in enumerate(zip(cyclers, snake_booleans)):
        num_tiles = np.product(lengths[:i])
        num_repeats = np.product(lengths[i + 1 :])
        for k, v in c._transpose().items():
            if snake:
                v = v + v[::-1]
            v2 = np.tile(np.repeat(v, num_repeats), num_tiles)
            expanded = v2[:total_length]
            new_cyclers.append(cycler(k, expanded))
    return reduce(operator.add, new_cyclers)


def first_key_heuristic(device):
    """
    Get the fully-qualified data key for the first entry in describe().

    This will raise is that entry's `describe()` method does not return a
    dictionary with exactly one key.
    """
    return next(iter(device.describe()))


def ancestry(obj):
    """
    List self, parent, grandparent, ... back to ultimate ancestor.

    Parameters
    ----------
    obj : object
        must have a `parent` attribute

    Returns
    -------
    ancestry : list
        list of objects, starting with obj and tracing parents recursively
    """
    ancestry = []
    ancestor = obj
    while True:
        ancestry.append(ancestor)
        if ancestor.parent is None:
            return ancestry
        ancestor = ancestor.parent


def root_ancestor(obj):
    """
    Traverse ancestry to obtain root ancestor.

    Parameters
    ----------
    obj : object
        must have a `parent` attribute

    Returns
    -------
    root : object
    """
    return ancestry(obj)[-1]


def share_ancestor(obj1, obj2):
    """
    Check whether obj1 and obj2 have a common ancestor.

    Parameters
    ----------
    obj1 : object
        must have a `parent` attribute
    obj2 : object
        must have a `parent` attribute

    Returns
    -------
    result : boolean
    """
    return ancestry(obj1)[-1] is ancestry(obj2)[-1]


def separate_devices(devices):
    """
    Filter out elements that have other elements as their ancestors.

    If A is an ancestor of B, [A, B, C] -> [A, C].

    Paremeters
    ----------
    devices : list
        All elements must have a `parent` attribute.

    Returns
    -------
    result : list
        subset of input, with order retained
    """
    result = []
    for det in devices:
        for existing_det in result[:]:
            if existing_det in ancestry(det):
                # known issue: here we assume that det is in the read_attrs
                # of existing_det -- to be addressed after plans.py refactor
                break
            elif det in ancestry(existing_det):
                # existing_det is redundant; use det in its place
                result.remove(existing_det)
        else:
            result.append(det)
    return result


def all_safe_rewind(devices):
    """If all devices can have their trigger method re-run on resume.

    Parameters
    ----------
    devices : list
        List of devices

    Returns
    -------
    safe_rewind : bool
       If all the device can safely re-triggered
    """
    for d in devices:
        if hasattr(d, "rewindable"):
            rewindable = d.rewindable.get()
            if not rewindable:
                return False
    return True


SEARCH_PATH = []
ENV_VAR = "BLUESKY_HISTORY_PATH"
if ENV_VAR in os.environ:
    SEARCH_PATH.append(os.environ[ENV_VAR])
SEARCH_PATH.extend(
    [
        os.path.expanduser("~/.config/bluesky/bluesky_history.db"),
        "/etc/bluesky/bluesky_history.db",
    ]
)


def get_history():
    """
    DEPRECATED: Return a dict-like object for stashing metadata.

    If historydict is not installed, return a dict.

    If historydict is installed, look for a sqlite file in:
      - $BLUESKY_HISTORY_PATH, if defined
      - ~/.config/bluesky/bluesky_history.db
      - /etc/bluesky/bluesky_history.db

    If no existing file is found, create a new sqlite file in:
      - $BLUESKY_HISTORY_PATH, if defined
      - ~/.config/bluesky/bluesky_history.db, otherwise
    """
    try:
        import historydict
    except ImportError:
        print(
            "You do not have historydict installed, your metadata "
            "will not be persistent or have any history of the "
            "values."
        )
        return dict()  # noqa: C408
    else:
        for path in SEARCH_PATH:
            if os.path.isfile(path):
                print("Loading metadata history from %s" % path)
                return historydict.HistoryDict(path)
        # No existing file was found. Try creating one.
        path = SEARCH_PATH[0]
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print("Storing metadata history in a new file at %s." % path)
            return historydict.HistoryDict(path)
        except OSError as exc:
            print(exc)
            print("Failed to create metadata history file at %s" % path)
            print("Storing HistoryDict in memory; it will not persist when session is ended.")
            return historydict.HistoryDict(":memory:")


_QT_KICKER_INSTALLED = {}
_NB_KICKER_INSTALLED = {}


def install_kicker(loop=None, update_rate=0.03):
    """
    Install a periodic callback to integrate drawing and asyncio event loops.

    This dispatches to :func:`install_qt_kicker` or :func:`install_nb_kicker`
    depending on the current matplotlib backend.

    Parameters
    ----------
    loop : event loop, optional
    update_rate : number
        Seconds between periodic updates. Default is 0.03.
    """
    import matplotlib

    backend = matplotlib.get_backend()
    if backend == "nbAgg":
        install_nb_kicker(loop=loop, update_rate=update_rate)
    elif backend in ("Qt4Agg", "Qt5Agg"):
        install_qt_kicker(loop=loop, update_rate=update_rate)
    else:
        raise NotImplementedError(f"The matplotlib backend {backend} is not yet supported.")


def install_qt_kicker(loop=None, update_rate=0.03):
    """Install a periodic callback to integrate Qt and asyncio event loops.

    DEPRECATED: This functionality is now handled automatically by default and
    is configurable via the RunEngine's new ``during_task`` parameter. Calling
    this function now has no effect. It will be removed in a future release of
    bluesky.

    Parameters
    ----------
    loop : event loop, optional
    update_rate : number
        Seconds between periodic updates. Default is 0.03.
    """
    warnings.warn(  # noqa: B028
        "bluesky.utils.install_qt_kicker is no longer necessary and "
        "has no effect. Please remove your use of it. It may be "
        "removed in a future release of bluesky."
    )


def install_remote_qt_kicker(loop=None, update_rate=0.03):
    """Install a periodic callback to integrate Qt and asyncio event loops.

    If a version of the Qt bindings are not already imported, this function
    will do nothing.

    It is safe to call this function multiple times.

    This is used when a Qt event loop is required and the process that needs
    it is different that the process that created the RunEngine,
    (see docstring for deprecated install_qt_kicker above)

    Parameters
    ----------
    loop : event loop, optional
    update_rate : number
        Seconds between periodic updates. Default is 0.03.
    """
    if loop is None:
        loop = asyncio.get_event_loop()
    global _QT_KICKER_INSTALLED
    if loop in _QT_KICKER_INSTALLED:
        return
    if not any(p in sys.modules for p in ["PyQt4", "pyside", "PyQt5"]):
        return

    import matplotlib.backends.backend_qt
    from matplotlib._pylab_helpers import Gcf
    from matplotlib.backends.backend_qt import _create_qApp

    _create_qApp()
    qApp = matplotlib.backends.backend_qt.qApp

    try:
        _draw_all = Gcf.draw_all  # mpl version >= 1.5
    except AttributeError:
        # slower, but backward-compatible
        def _draw_all():
            for f_mgr in Gcf.get_all_fig_managers():
                f_mgr.canvas.draw_idle()

    def _qt_kicker():
        # The RunEngine Event Loop interferes with the qt event loop. Here we
        # kick it to keep it going.
        _draw_all()

        qApp.processEvents()
        loop.call_later(update_rate, _qt_kicker)

    _QT_KICKER_INSTALLED[loop] = loop.call_soon(_qt_kicker)


def install_nb_kicker(loop=None, update_rate=0.03):
    """
    Install a periodic callback to integrate ipykernel and asyncio event loops.

    It is safe to call this function multiple times.

    Parameters
    ----------
    loop : event loop, optional
    update_rate : number
        Seconds between periodic updates. Default is 0.03.
    """
    import matplotlib

    if loop is None:
        loop = asyncio.get_event_loop()
    global _NB_KICKER_INSTALLED
    if loop in _NB_KICKER_INSTALLED:
        return

    def _nbagg_kicker():
        # This is more brute-force variant of the _qt_kicker function used
        # inside install_qt_kicker.
        for f_mgr in matplotlib._pylab_helpers.Gcf.get_all_fig_managers():
            if f_mgr.canvas.figure.stale:
                f_mgr.canvas.draw()

        loop.call_later(update_rate, _nbagg_kicker)

    _NB_KICKER_INSTALLED[loop] = loop.call_soon(_nbagg_kicker)


def apply_sub_factories(factories, plan):
    """Run sub factory functions for a plan

    Factory functions should return lists, which will be added onto the
    subscription key (e.g., 'all' or 'start') specified in the factory
    definition.

    If the factory function returns None, the list will not be modified.
    """
    factories = normalize_subs_input(factories)
    out = {
        k: list(itertools.filterfalse(lambda x: x is None, (sf(plan) for sf in v))) for k, v in factories.items()
    }
    return out


def update_sub_lists(out, inp):
    """Extends dictionary `out` lists with those in `inp`

    Assumes dictionaries where all values are lists
    """
    for k, v in inp.items():
        try:
            out[k].extend(v)
        except KeyError:
            out[k] = list(v)


def register_transform(RE, *, prefix="<", ip=None):
    """Register RunEngine IPython magic convenience transform
    Assuming the default parameters
    This maps `< stuff(*args, **kwargs)` -> `RE(stuff(*args, **kwargs))`
    RE is assumed to be available in the global namespace
    Parameters
    ----------
    RE : str
        The name of a valid RunEngine instance in the global IPython namespace
    prefix : str, optional
        The prefix to trigger this transform on.  If this collides with
        valid python syntax or an existing transform you are on your own.
    ip : IPython shell, optional
        If not passed, uses `IPython.get_ipython()` to get the current shell
    """
    import IPython

    if ip is None:
        ip = IPython.get_ipython()

    if IPython.__version__ >= "7":

        def tr_re(lines):
            if len(lines) != 1:
                return lines
            (line,) = lines
            head, split, tail = line.partition(prefix)
            if split == prefix and head.strip() == "":
                line = f"{RE}({tail.strip()})\n"

            return [line]

        ip.input_transformers_post.append(tr_re)

    else:
        from IPython.core.inputtransformer import StatelessInputTransformer

        @StatelessInputTransformer.wrap
        def tr_re(line):
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()
                return f"{RE}({line})"
            return line

        ip.input_splitter.logical_line_transforms.append(tr_re())
        ip.input_transformer_manager.logical_line_transforms.append(tr_re())


def new_uid():
    return str(uuid.uuid4())


def sanitize_np(val):
    "Convert any numpy objects into built-in Python types."
    if isinstance(val, (np.generic, np.ndarray)):
        if np.isscalar(val):
            return val.item()
        return val.tolist()
    return val


def expiring_function(func, loop, *args, **kwargs):
    """
    If timeout has not occurred, call func(*args, **kwargs).

    This is meant to used with the event loop's run_in_executor
    method. Outside that context, it doesn't make any sense.
    """

    def dummy(start_time, timeout):
        if loop.time() > start_time + timeout:
            return
        func(*args, **kwargs)
        return

    return dummy


def short_uid(label=None, truncate=6):
    "Return a readable but unique id like 'label-fjfi5a'"
    if label:
        return "-".join([label, new_uid()[:truncate]])
    else:
        return new_uid()[:truncate]


def ensure_uid(doc_or_uid):
    """
    Accept a uid or a dict with a 'uid' key. Return the uid.
    """
    try:
        return doc_or_uid["uid"]
    except TypeError:
        return doc_or_uid


def ts_msg_hook(msg, file=sys.stdout):
    t = f"{datetime.datetime.now():%H:%M:%S.%f}"
    msg_fmt = "{: <17s} -> {!s: <15s} args: {}, kwargs: {}, run: {}".format(
        msg.command,
        msg.obj.name if hasattr(msg.obj, "name") else msg.obj,
        msg.args,
        msg.kwargs,
        f"'{msg.run}'" if isinstance(msg.run, str) else msg.run,
    )
    print(f"{t} {msg_fmt}", file=file)


def make_decorator(wrapper):
    """
    Turn a generator instance wrapper into a generator function decorator.

    The functions named <something>_wrapper accept a generator instance and
    return a mutated generator instance.

    Example of a 'wrapper':
    >>> plan = count([det])  # returns a generator instance
    >>> revised_plan = some_wrapper(plan)  # returns a new instance

    Example of a decorator:
    >>> some_decorator = make_decorator(some_wrapper)  # returns decorator
    >>> customized_count = some_decorator(count)  # returns generator func
    >>> plan = customized_count([det])  # returns a generator instance

    This turns a 'wrapper' into a decorator, which accepts a generator
    function and returns a generator function.
    """

    @wraps(wrapper)
    def dec_outer(*args, **kwargs):
        def dec(gen_func):
            @wraps(gen_func)
            def dec_inner(*inner_args, **inner_kwargs):
                plan = gen_func(*inner_args, **inner_kwargs)
                plan = wrapper(plan, *args, **kwargs)
                return (yield from plan)

            return dec_inner

        return dec

    return dec_outer


def apply_to_dict_recursively(d, f):
    """Recursively apply function to a document

    This modifies the dict in place and returns it.

    Parameters
    ----------
    d: dict
        e.g. event_model Document
    f: function
       any func to be performed on d recursively
    """
    for key, val in d.items():
        if hasattr(val, "items"):
            d[key] = apply_to_dict_recursively(d=val, f=f)
        d[key] = f(val)
    return d


def _L2norm(x, y):
    "works on (3, 5) and ((0, 3), (4, 0))"
    return np.sqrt(np.sum((np.asarray(x) - np.asarray(y)) ** 2))


def merge_axis(objs):
    """Merge possibly related axis

    This function will take a list of objects and separate it into

     - list of completely independent objects (most settable things and
       detectors) that do not have coupled motion.
     - list of devices who have children who are coupled (PseudoPositioner
       ducked by looking for 'RealPosition' as an attribute)

    Both of these lists will only contain objects directly passed in
    in objs

     - map between parents and objects passed in.  Each value
       of the map is a map between the strings
       {'real', 'pseudo', 'independent'} and a list of objects.  All
       of the objects in the (doubly nested) map are in the input.

    Parameters
    ----------
    objs : Iterable[OphydObj]
        The input devices

    Returns
    -------
    independent_objs : List[OphydObj]
        Independent 'simple' axis

    complex_objs : List[PseudoPositioner]
        Independent objects which have interdependent children

    coupled : Dict[PseudoPositioner, Dict[str, List[OphydObj]]]
        Mapping of interdependent axis passed in.
    """

    def get_parent(o):
        return check_supports(o, HasParent).parent

    independent_objs = set()
    maybe_coupled = set()
    complex_objs = set()
    for o in objs:
        parent = o.parent
        if hasattr(o, "RealPosition"):
            complex_objs.add(o)
        elif parent is not None and hasattr(parent, "RealPosition"):
            maybe_coupled.add(o)
        else:
            independent_objs.add(o)
    coupled = {}

    for parent, children in groupby(get_parent, maybe_coupled).items():
        real_p = set(parent.real_positioners)
        pseudo_p = set(parent.pseudo_positioners)
        type_map = {"real": [], "pseudo": [], "unrelated": []}
        for c in children:
            if c in real_p:
                type_map["real"].append(c)
            elif c in pseudo_p:
                type_map["pseudo"].append(c)
            else:
                type_map["unrelated"].append(c)
        coupled[parent] = type_map

    return (independent_objs, complex_objs, coupled)


def merge_cycler(cyc):
    """Specify movements of sets of interdependent axes atomically.

    Inspect the keys of ``cyc`` (which are Devices) to indentify those
    which are interdependent (part of the same
    PseudoPositioner) and merge those independent entries into
    a single entry.

    This also validates that the user has not passed conflicting
    interdependent axis (such as a real and pseudo axis from the same
    PseudoPositioner)

    Parameters
    ----------
    cyc : Cycler[OphydObj, Sequence]
       A cycler as would be passed to :func:`scan_nd`

    Returns
    -------
    Cycler[OphydObj, Sequence]
       A cycler as would be passed to :func:`scan_nd` with the same
       or fewer keys than the input.

    """

    def my_name(obj):
        """Get the attribute name of this device on its parent Device"""
        parent = obj.parent
        return next(iter([nm for nm in parent.component_names if getattr(parent, nm) is obj]))

    io, co, gb = merge_axis(cyc.keys)

    # only simple non-coupled objects, declare victory and bail!
    if len(co) == len(gb) == 0:
        return cyc

    input_data = cyc.by_key()
    output_data = [cycler(i, input_data[i]) for i in io | co]

    for parent, type_map in gb.items():
        if parent in co and (type_map["pseudo"] or type_map["real"]):
            raise ValueError(
                "A PseudoPostiioner and its children were both "
                "passed in.  We do not yet know how to merge "
                "these inputs, failing."
            )

        if type_map["real"] and type_map["pseudo"]:
            raise ValueError("Passed in a mix of real and pseudo axis. Can not cope, failing")
        pseudo_axes = type_map["pseudo"]
        if len(pseudo_axes) > 1:
            p_cyc = reduce(
                operator.add,
                (cycler(my_name(c), input_data[c]) for c in type_map["pseudo"]),
            )
            output_data.append(cycler(parent, list(p_cyc)))
        elif len(pseudo_axes) == 1:
            (c,) = pseudo_axes
            output_data.append(cycler(c, input_data[c]))

        for c in type_map["real"] + type_map["unrelated"]:
            output_data.append(cycler(c, input_data[c]))

    return reduce(operator.add, output_data)


def _rearrange_into_parallel_dicts(readings):
    data = {}
    timestamps = {}
    for key, payload in readings.items():
        data[key] = payload["value"]
        timestamps[key] = payload["timestamp"]
    return data, timestamps


def is_movable(obj):
    """Check if object satisfies bluesky 'Movable' and `Readable` interfaces.

    Parameters
    ----------
    obj : Object
        Object to test.

    Returns
    -------
    boolean
        True if movable, False otherwise.
    """
    return isinstance(obj, Movable) and isinstance(obj, Readable)


def get_hinted_fields(obj) -> List[str]:
    if isinstance(obj, HasHints):
        return obj.hints.get("fields", [])
    else:
        return []


already_warned: Dict[Any, bool] = {}


def warn_if_msg_args_or_kwargs(msg, meth, args, kwargs):
    if args or kwargs and not already_warned.get(msg.command):
        already_warned[msg.command] = True
        error_msg = f"""\
About to call {meth.__name__}() with args {args} and kwargs {kwargs}.
In the future the passing of Msg.args and Msg.kwargs down to hardware from
Msg("{msg.command}") may be deprecated. If you have a use case for these,
we would like to know about it, so please open an issue at
https://github.com/bluesky/bluesky/issues"""
        warnings.warn(error_msg)  # noqa: B028


def maybe_update_hints(hints: Dict[str, Hints], obj):
    if isinstance(obj, HasHints):
        hints[obj.name] = obj.hints


async def iterate_maybe_async(iterator: SyncOrAsyncIterator[T]) -> AsyncIterator[T]:
    if inspect.isasyncgen(iterator):
        async for v in iterator:
            yield v
    else:
        for v in iterator:  # type: ignore
            yield v


async def maybe_collect_asset_docs(
    msg, obj, index: Optional[int] = None, *args, **kwargs
) -> AsyncIterable[Union[Asset, StreamAsset]]:
    # The if/elif statement must be done in this order because isinstance for protocol
    # doesn't check for exclusive signatures, and WritesExternalAssets will also
    # return true for a WritesStreamAsset as they both contain collect_asset_docs
    if isinstance(obj, WritesStreamAssets):
        warn_if_msg_args_or_kwargs(msg, obj.collect_asset_docs, args, kwargs)
        async for stream_doc in iterate_maybe_async(obj.collect_asset_docs(index, *args, **kwargs)):
            yield stream_doc
    elif isinstance(obj, WritesExternalAssets):
        warn_if_msg_args_or_kwargs(msg, obj.collect_asset_docs, args, kwargs)
        async for doc in iterate_maybe_async(obj.collect_asset_docs(*args, **kwargs)):
            yield doc


async def maybe_await(ret: SyncOrAsync[T]) -> T:
    if inspect.isawaitable(ret):
        return await ret
    else:
        # Mypy does not understand how to narrow type to non-awaitable in this
        # instance, see https://github.com/python/mypy/issues/15520
        return ret  # type: ignore
