from collections import namedtuple
import asyncio
import os
import sys
import signal
import operator
import uuid
from functools import reduce
from weakref import ref, WeakKeyDictionary
import types
import inspect
from inspect import Parameter, Signature
import itertools
from collections import Iterable
import numpy as np
from cycler import cycler
import logging
import datetime
from functools import wraps, partial
import threading
import time
from tqdm import tqdm
from tqdm._utils import _environ_cols_wrapper, _term_move_up, _unicode
import warnings
try:
    # cytools is a drop-in replacement for toolz, implemented in Cython
    from cytools import groupby
except ImportError:
    from toolz import groupby

logger = logging.getLogger(__name__)


class Msg(namedtuple('Msg_base', ['command', 'obj', 'args', 'kwargs'])):
    __slots__ = ()

    def __new__(cls, command, obj=None, *args, **kwargs):
        return super(Msg, cls).__new__(cls, command, obj, args, kwargs)

    def __repr__(self):
        return '{}: ({}), {}, {}'.format(
            self.command, self.obj, self.args, self.kwargs)


class RunEngineControlException(Exception):
    pass


class RequestAbort(RunEngineControlException):
    pass


class RequestStop(RunEngineControlException):
    pass


class RunEngineInterrupted(Exception):
    pass


class NoReplayAllowed(Exception):
    pass


class IllegalMessageSequence(Exception):
    pass


class FailedPause(Exception):
    pass


class FailedStatus(Exception):
    'Exception to be raised if a SatusBase object reports done but failed'


class InvalidCommand(KeyError):
    pass


class PlanHalt(GeneratorExit):
    pass


class RampFail(RuntimeError):
    ...


PLAN_TYPES = (types.GeneratorType,)
try:
    from types import CoroutineType
except ImportError:
    # < py35
    pass
else:
    PLAN_TYPES = PLAN_TYPES + (CoroutineType, )
    del CoroutineType


def ensure_generator(plan):
    """
    Ensure that the input is a generator.

    Parameters
    ----------
    plan : iterable or iterator

    Returns
    -------
    gen : coroutine
    """
    if isinstance(plan, Msg):
        return single_gen(plan)
    gen = iter(plan)  # no-op on generators; needed for classes
    if not isinstance(gen, PLAN_TYPES):
        # If plan does not support .send, we must wrap it in a generator.
        gen = (msg for msg in gen)

    return gen


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
    return (yield msg)


class SignalHandler:
    def __init__(self, sig, log=None):
        self.sig = sig
        self.interrupted = False
        self.count = 0
        self.log = log

    def __enter__(self):
        self.interrupted = False
        self.released = False
        self.count = 0

        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self.interrupted = True
            self.count += 1
            if self.log is not None:
                self.log.debug('SignalHandler caught SIGINT; count is %r',
                               self.count)
            if self.count > 10:
                orig_func = self.original_handler
                self.release()
                orig_func(signum, frame)

        signal.signal(self.sig, handler)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False
        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True


class SigintHandler(SignalHandler):
    def __init__(self, RE):
        super().__init__(signal.SIGINT, log=RE.log)
        self.RE = RE
        self.last_sigint_time = None  # time most recent SIGINT was processed
        self.num_sigints_processed = 0  # count SIGINTs processed

    def __enter__(self):
        self.RE.loop.call_later(0.1, self.check_for_signals)
        return super().__enter__()

    def check_for_signals(self):
        # Check for pause requests from keyboard.
        if self.RE.state.is_running and (not self.RE._interrupted):
            if self.count > self.num_sigints_processed:
                self.num_sigints_processed = self.count
                self.log.debug("RunEngine caught a new SIGINT")
                self.last_sigint_time = time.time()

                if self.count == 1:
                    # Ctrl-C once -> request a deferred pause
                    if not self.RE._deferred_pause_requested:
                        self.RE.loop.call_soon(self.RE.request_pause, True)
                        print("A 'deferred pause' has been requested. The "
                              "RunEngine will pause at the next checkpoint. "
                              "To pause immediately, hit Ctrl+C again in the "
                              "next 10 seconds.")
                elif self.count > 1:
                    # - Ctrl-C twice within 10 seconds -> hard pause
                    self.log.debug("RunEngine detected two SIGINTs. "
                                   "A hard pause will be requested.")
                    self.RE.loop.call_soon(self.RE.request_pause, False)
            else:
                # No new SIGINTs to process.
                if self.num_sigints_processed > 0:
                    if time.time() - self.last_sigint_time > 10:
                        self.log.debug("It has been 10 seconds since the "
                                       "last SIGINT. Resetting SIGINT "
                                       "handler.")
                        # It's been 10 seconds since the last SIGINT. Reset.
                        self.num_sigints_processed = 0
                        self.count = 0
                        self.interrupted = False
                        self.last_sigint_time = None

        if not self.released:
            self.RE.loop.call_later(0.1, self.check_for_signals)


class CallbackRegistry:
    """
    See matplotlib.cbook.CallbackRegistry. This is a simplified since
    ``bluesky`` is python3.4+ only!
    """
    def __init__(self, ignore_exceptions=False, allowed_sigs=None):
        self.ignore_exceptions = ignore_exceptions
        self.allowed_sigs = allowed_sigs
        self.callbacks = dict()
        self._cid = 0
        self._func_cid_map = {}

    def __getstate__(self):
        # We cannot currently pickle the callables in the registry, so
        # return an empty dictionary.
        return {}

    def __setstate__(self, state):
        # re-initialise an empty callback registry
        self.__init__()

    def connect(self, sig, func):
        """Register ``func`` to be called when ``sig`` is generated

        Parameters
        ----------
        sig
        func

        Returns
        -------
        cid : int
            The callback index. To be used with ``disconnect`` to deregister
            ``func`` so that it will no longer be called when ``sig`` is
            generated
        """
        if self.allowed_sigs is not None:
            if sig not in self.allowed_sigs:
                raise ValueError("Allowed signals are {0}".format(
                    self.allowed_sigs))
        self._func_cid_map.setdefault(sig, WeakKeyDictionary())
        # Note proxy not needed in python 3.
        # TODO rewrite this when support for python2.x gets dropped.
        proxy = _BoundMethodProxy(func)
        if proxy in self._func_cid_map[sig]:
            return self._func_cid_map[sig][proxy]

        proxy.add_destroy_callback(self._remove_proxy)
        self._cid += 1
        cid = self._cid
        self._func_cid_map[sig][proxy] = cid
        self.callbacks.setdefault(sig, dict())
        self.callbacks[sig][cid] = proxy
        return cid

    def _remove_proxy(self, proxy):
        # need the list because `del self._func_cid_map[sig]` mutates the dict
        for sig, proxies in list(self._func_cid_map.items()):
            try:
                del self.callbacks[sig][proxies[proxy]]
            except KeyError:
                pass

            if len(self.callbacks[sig]) == 0:
                del self.callbacks[sig]
                del self._func_cid_map[sig]

    def disconnect(self, cid):
        """Disconnect the callback registered with callback id *cid*

        Parameters
        ----------
        cid : int
            The callback index and return value from ``connect``
        """
        for eventname, callbackd in self.callbacks.items():
            try:
                del callbackd[cid]
            except KeyError:
                continue
            else:
                for sig, functions in self._func_cid_map.items():
                    for function, value in list(functions.items()):
                        if value == cid:
                            del functions[function]
                return

    def process(self, sig, *args, **kwargs):
        """Process ``sig``

        All of the functions registered to receive callbacks on ``sig``
        will be called with ``args`` and ``kwargs``

        Parameters
        ----------
        sig
        args
        kwargs
        """
        if self.allowed_sigs is not None:
            if sig not in self.allowed_sigs:
                raise ValueError("Allowed signals are {0}".format(
                    self.allowed_sigs))
        exceptions = []
        if sig in self.callbacks:
            for cid, func in list(self.callbacks[sig].items()):
                try:
                    func(*args, **kwargs)
                except ReferenceError:
                    self._remove_proxy(func)
                except Exception as e:
                    if self.ignore_exceptions:
                        exceptions.append((e, sys.exc_info()[2]))
                    else:
                        raise
        return exceptions


class _BoundMethodProxy:
    '''
    Our own proxy object which enables weak references to bound and unbound
    methods and arbitrary callables. Pulls information about the function,
    class, and instance out of a bound method. Stores a weak reference to the
    instance to support garbage collection.
    @organization: IBM Corporation
    @copyright: Copyright (c) 2005, 2006 IBM Corporation
    @license: The BSD License
    Minor bugfixes by Michael Droettboom
    '''
    def __init__(self, cb):
        self._hash = hash(cb)
        self._destroy_callbacks = []
        try:
            try:
                self.inst = ref(cb.__self__, self._destroy)
            except TypeError:
                self.inst = None
            self.func = cb.__func__
            self.klass = cb.__self__.__class__
        except AttributeError:
            self.inst = None
            self.func = cb
            self.klass = None

    def add_destroy_callback(self, callback):
        self._destroy_callbacks.append(_BoundMethodProxy(callback))

    def _destroy(self, wk):
        for callback in self._destroy_callbacks:
            try:
                callback(self)
            except ReferenceError:
                pass

    def __getstate__(self):
        d = self.__dict__.copy()
        # de-weak reference inst
        inst = d['inst']
        if inst is not None:
            d['inst'] = inst()
        return d

    def __setstate__(self, statedict):
        self.__dict__ = statedict
        inst = statedict['inst']
        # turn inst back into a weakref
        if inst is not None:
            self.inst = ref(inst)

    def __call__(self, *args, **kwargs):
        '''
        Proxy for a call to the weak referenced object. Take
        arbitrary params to pass to the callable.
        Raises `ReferenceError`: When the weak reference refers to
        a dead object
        '''
        if self.inst is not None and self.inst() is None:
            raise ReferenceError
        elif self.inst is not None:
            # build a new instance method with a strong reference to the
            # instance

            mtd = types.MethodType(self.func, self.inst())

        else:
            # not a bound method, just return the func
            mtd = self.func
        # invoke the callable and return the result
        return mtd(*args, **kwargs)

    def __eq__(self, other):
        '''
        Compare the held function and instance with that held by
        another proxy.
        '''
        try:
            if self.inst is None:
                return self.func == other.func and other.inst is None
            else:
                return self.func == other.func and self.inst() == other.inst()
        except Exception:
            return False

    def __ne__(self, other):
        '''
        Inverse of __eq__.
        '''
        return not self.__eq__(other)

    def __hash__(self):
        return self._hash


# The following two code blocks are adapted from David Beazley's
# 'Python 3 Metaprogramming' https://www.youtube.com/watch?v=sPiWg5jSoZI


class StructMeta(type):
    def __new__(cls, name, bases, clsdict):
        clsobj = super().__new__(cls, name, bases, clsdict)
        args_params = [Parameter(name, Parameter.POSITIONAL_OR_KEYWORD)
                       for name in clsobj._fields]
        kwargs_params = [Parameter(name, Parameter.KEYWORD_ONLY, default=None)
                         for name in ['md']]
        sig = Signature(args_params + kwargs_params)
        setattr(clsobj, '__signature__', sig)
        return clsobj


class Struct(metaclass=StructMeta):
    "The _fields of any subclass become its attritubes and __init__ args."
    _fields = []

    def __init__(self, *args, **kwargs):
        # Now bind default values of optional arguments.
        # If it seems like there should be a cleaner way to do this, see
        # http://bugs.python.org/msg221104
        bound = self.__signature__.bind(*args, **kwargs)
        for name, param in self.__signature__.parameters.items():
            if (name not in bound.arguments and
                    param.default is not inspect._empty):
                bound.arguments[name] = param.default
        for name, val in bound.arguments.items():
            setattr(self, name, val)
        self.flyers = []

    def set(self, **kwargs):
        "Update attributes as keyword arguments."
        for attr, val in kwargs.items():
            setattr(self, attr, val)


SUBS_NAMES = ['all', 'start', 'stop', 'event', 'descriptor']


def normalize_subs_input(subs):
    "Accept a callable, a list, or a dict. Normalize to a dict of lists."
    normalized = {name: [] for name in SUBS_NAMES}
    if subs is None:
        pass
    elif callable(subs):
        normalized['all'].append(subs)
    elif hasattr(subs, 'items'):
        for key, funcs in list(subs.items()):
            if key not in SUBS_NAMES:
                raise KeyError("Keys must be one of {!r:0}".format(SUBS_NAMES))
            if callable(funcs):
                normalized[key].append(funcs)
            else:
                normalized[key].extend(funcs)
    elif isinstance(subs, Iterable):
        normalized['all'].extend(subs)
    else:
        raise ValueError("Subscriptions should be a callable, a list of "
                         "callables, or a dictionary mapping subscription "
                         "names to lists of callables.")
    # Validates that all entries are callables.
    for name, funcs in normalized.items():
        for func in funcs:
            if not callable(func):
                raise ValueError("subs values must be functions or lists "
                                 "of functions. The offending entry is\n "
                                 "{0}".format(func))
    return normalized


class DefaultSubs:
    """a class-level descriptor"""
    def __init__(self, default=None):
        self._value = normalize_subs_input(default)

    def __get__(self, instance, owner):
        return self._value

    def __set__(self, instance, value):
        self._value = normalize_subs_input(value)


class Subs:
    """a 'reusable' property"""
    def __init__(self, default=None):
        self.default = normalize_subs_input(default)
        self.data = WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self.data.get(instance, self.default)

    def __set__(self, instance, value):
        self.data[instance] = normalize_subs_input(value)


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
    lengths = []
    new_cyclers = []
    for c in cyclers:
        lengths.append(len(c))
    total_length = np.product(lengths)
    for i, (c, snake) in enumerate(zip(cyclers, snake_booleans)):
        num_tiles = np.product(lengths[:i])
        num_repeats = np.product(lengths[i+1:])
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
    '''If all devices can have their trigger method re-run on resume.

    Parameters
    ----------
    devices : list
        List of devices

    Returns
    -------
    safe_rewind : bool
       If all the device can safely re-triggered
    '''
    for d in devices:
        if hasattr(d, 'rewindable'):
            rewindable = d.rewindable.get()
            if not rewindable:
                return False
    return True


SEARCH_PATH = []
ENV_VAR = 'BLUESKY_HISTORY_PATH'
if ENV_VAR in os.environ:
    SEARCH_PATH.append(os.environ[ENV_VAR])
SEARCH_PATH.extend([os.path.expanduser('~/.config/bluesky/bluesky_history.db'),
                    '/etc/bluesky/bluesky_history.db'])


def get_history():
    """
    Return a dict-like object for stashing metadata.

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
        print("You do not have historydict installed, your metadata "
              "will not be persistent or have any history of the "
              "values.")
        return dict()
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
        except IOError as exc:
            print(exc)
            print("Failed to create metadata history file at %s" % path)
            print("Storing HistoryDict in memory; it will not persist "
                  "when session is ended.")
            return historydict.HistoryDict(':memory:')


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
    if backend == 'nbAgg':
        install_nb_kicker(loop=loop, update_rate=update_rate)
    elif backend in ('Qt4Agg', 'Qt5Agg'):
        install_qt_kicker(loop=loop, update_rate=update_rate)
    else:
        raise NotImplementedError("The matplotlib backend {} is not yet "
                                  "supported.".format(backend))


def install_qt_kicker(loop=None, update_rate=0.03):
    """Install a periodic callback to integrate Qt and asyncio event loops.

    If a version of the Qt bindings are not already imported, this function
    will do nothing.

    It is safe to call this function multiple times.

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
    if not any(p in sys.modules for p in ['PyQt4', 'pyside', 'PyQt5']):
        return
    import matplotlib.backends.backend_qt5
    from matplotlib.backends.backend_qt5 import _create_qApp
    from matplotlib._pylab_helpers import Gcf

    _create_qApp()
    qApp = matplotlib.backends.backend_qt5.qApp

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
    '''Run sub factory functions for a plan

    Factory functions should return lists, which will be added onto the
    subscription key (e.g., 'all' or 'start') specified in the factory
    definition.

    If the factory function returns None, the list will not be modified.
    '''
    factories = normalize_subs_input(factories)
    out = {k: list(itertools.filterfalse(lambda x: x is None,
                                         (sf(plan) for sf in v)))
           for k, v in factories.items()}
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


def register_transform(RE, *, prefix='<'):
    '''Register RunEngine IPython magic convenience transform

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
    '''
    import IPython
    from IPython.core.inputtransformer import StatelessInputTransformer

    @StatelessInputTransformer.wrap
    def tr_re(line):
        if line.startswith(prefix):
            line = line[len(prefix):].strip()
            return '{}({})'.format(RE, line)
        return line
    ip = IPython.get_ipython()
    ip.input_splitter.logical_line_transforms.append(tr_re())
    ip.input_transformer_manager.logical_line_transforms.append(tr_re())


class AsyncInput:
    """a input prompt that allows event loop to run in the background

    adapted from http://stackoverflow.com/a/35514777/1221924
    """
    def __init__(self, loop=None):
        self.loop = loop or asyncio.get_event_loop()
        self.q = asyncio.Queue(loop=self.loop)
        self.loop.add_reader(sys.stdin, self.got_input)

    def got_input(self):
        asyncio.ensure_future(self.q.put(sys.stdin.readline()), loop=self.loop)

    @asyncio.coroutine
    def __call__(self, prompt, end='\n', flush=False):
        print(prompt, end=end, flush=flush)
        return (yield from self.q.get()).rstrip('\n')


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
        return '-'.join([label, new_uid()[:truncate]])
    else:
        return new_uid()[:truncate]


def ensure_uid(doc_or_uid):
    """
    Accept a uid or a dict with a 'uid' key. Return the uid.
    """
    try:
        return doc_or_uid['uid']
    except TypeError:
        return doc_or_uid


def ts_msg_hook(msg):
    t = '{:%H:%M:%S.%f}'.format(datetime.datetime.now())
    msg_fmt = '{: <17s} -> {!s: <15s} args: {}, kwargs: {}'.format(
        msg.command,
        msg.obj.name if hasattr(msg.obj, 'name') else msg.obj,
        msg.args,
        msg.kwargs)
    print('{} {}'.format(t, msg_fmt))


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
        if hasattr(val, 'items'):
            d[key] = apply_to_dict_recursively(d=val, f=f)
        d[key] = f(val)
    return d


class ProgressBar:
    def __init__(self, status_objs, delay_draw=0.2):
        """
        Represent status objects with a progress bars.

        Parameters
        ----------
        status_objs : list
            Status objects
        delay_draw : float, optional
            To avoid flashing progress bars that will complete quickly after
            they are displayed, delay drawing until the progress bar has been
            around for awhile. Default is 0.2 seconds.
        """
        self.meters = []
        self.status_objs = []
        # Determine terminal width.
        self.ncols = _environ_cols_wrapper()(sys.stdout) or 79
        self.fp = sys.stdout
        self.creation_time = time.time()
        self.delay_draw = delay_draw
        self.drawn = False
        self.done = False
        self.lock = threading.RLock()

        # If the ProgressBar is not finished before the delay_draw time but
        # never again updated after the delay_draw time, we need to draw it
        # once.
        if delay_draw:
            threading.Thread(target=self._ensure_draw, daemon=True).start()

        # Create a closure over self.update for each status object that
        # implemets the 'watch' method.
        for st in status_objs:
            with self.lock:
                if hasattr(st, 'watch') and not st.done:
                    pos = len(self.meters)
                    self.meters.append('')
                    self.status_objs.append(st)
                    st.watch(partial(self.update, pos))

    def update(self, pos, *,
               name=None,
               current=None, initial=None, target=None,
               unit='units', precision=None,
               fraction=None,
               time_elapsed=None, time_remaining=None):
        if all(x is not None for x in (current, initial, target)):
            # Display a proper progress bar.
            total = round(_L2norm(target, initial), precision or 3)
            n = round(_L2norm(current, initial), precision or 3)
            # Compute this only if the status object did not provide it.
            if time_elapsed is None:
                time_elapsed = time.time() - self.creation_time
            # TODO Account for 'fraction', which might in some special cases
            # differ from the naive computation above.
            # TODO Account for 'time_remaining' which might in some special
            # cases differ from the naive computaiton performed by
            # format_meter.
            meter = tqdm.format_meter(n=n, total=total, elapsed=time_elapsed,
                                      unit=unit,
                                      prefix=name,
                                      ncols=self.ncols)
        else:
            # Simply display completeness.
            if name is None:
                name = ''
            if self.status_objs[pos].done:
                meter = name + ' [Complete.]'
            else:
                meter = name + ' [In progress. No progress bar available.]'
            meter += ' ' * (self.ncols - len(meter))
            meter = meter[:self.ncols]

        self.meters[pos] = meter
        self.draw()

    def draw(self):
        with self.lock:
            if (time.time() - self.creation_time) < self.delay_draw:
                return
            if self.done:
                return
            for meter in self.meters:
                tqdm.status_printer(self.fp)(meter)
                self.fp.write('\n')
            self.fp.write(_unicode(_term_move_up() * len(self.meters)))
            self.drawn = True

    def _ensure_draw(self):
        # Ensure that the progress bar is drawn at least once after the delay.
        time.sleep(self.delay_draw)
        with self.lock:
            if (not self.done) and (not self.drawn):
                self.draw()

    def clear(self):
        with self.lock:
            self.done = True
            if self.drawn:
                for meter in self.meters:
                    self.fp.write('\r')
                    self.fp.write(' ' * self.ncols)
                    self.fp.write('\r')
                    self.fp.write('\n')
                self.fp.write(_unicode(_term_move_up() * len(self.meters)))


class ProgressBarManager:
    def __init__(self, delay_draw=0.2):
        self.delay_draw = delay_draw
        self.pbar = None

    def __call__(self, status_objs_or_none):
        if status_objs_or_none is not None:
            # Start a new ProgressBar.
            if self.pbar is not None:
                warnings.warn("Previous ProgressBar never competed.")
                self.pbar.clear()
            self.pbar = ProgressBar(status_objs_or_none,
                                    delay_draw=self.delay_draw)
        else:
            # Clean up an old one.
            if self.pbar is None:
                warnings.warn("There is no Progress bar to clean up.")
            else:
                self.pbar.clear()
                self.pbar = None


def _L2norm(x, y):
    "works on (3, 5) and ((0, 3), (4, 0))"
    return np.sqrt(np.sum((np.asarray(x) - np.asarray(y))**2))


def merge_axis(objs):
    '''Merge possibly related axis

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
    '''
    def get_parent(o):
        return getattr(o, 'parent')

    independent_objs = set()
    maybe_coupled = set()
    complex_objs = set()
    for o in objs:
        parent = o.parent
        if hasattr(o, 'RealPosition'):
            complex_objs.add(o)
        elif (parent is not None and hasattr(parent, 'RealPosition')):
            maybe_coupled.add(o)
        else:
            independent_objs.add(o)
    coupled = {}

    for parent, children in groupby(get_parent, maybe_coupled).items():
        real_p = set(parent.real_positioners)
        pseudo_p = set(parent.pseudo_positioners)
        type_map = {'real': [], 'pseudo': [], 'unrelated': []}
        for c in children:
            if c in real_p:
                type_map['real'].append(c)
            elif c in pseudo_p:
                type_map['pseudo'].append(c)
            else:
                type_map['unrelated'].append(c)
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
        """Get the attribute name of this device on its parent Device
        """
        parent = obj.parent
        return next(iter([nm for nm in parent.component_names
                          if getattr(parent, nm) is obj]))

    io, co, gb = merge_axis(cyc.keys)

    # only simple non-coupled objects, declare victory and bail!
    if len(co) == len(gb) == 0:
        return cyc

    input_data = cyc.by_key()
    output_data = [cycler(i, input_data[i]) for i in io | co]

    for parent, type_map in gb.items():

        if parent in co and (type_map['pseudo'] or type_map['real']):
            raise ValueError("A PseudoPostiioner and its children were both "
                             "passed in.  We do not yet know how to merge "
                             "these inputs, failing.")

        if type_map['real'] and type_map['pseudo']:
            raise ValueError("Passed in a mix of real and pseudo axis.  "
                             "Can not cope, failing")
        pseudo_axes = type_map['pseudo']
        if len(pseudo_axes) > 1:
            p_cyc = reduce(operator.add,
                           (cycler(my_name(c), input_data[c])
                            for c in type_map['pseudo']))
            output_data.append(cycler(parent, list(p_cyc)))
        elif len(pseudo_axes) == 1:
            c, = pseudo_axes
            output_data.append(cycler(c, input_data[c]))

        for c in type_map['real'] + type_map['unrelated']:
            output_data.append(cycler(c, input_data[c]))

    return reduce(operator.add, output_data)
