import collections.abc
from collections import namedtuple
import asyncio
import os
import sys
import signal
import operator
import uuid
from functools import reduce
from typing import Any, Optional, Callable
from weakref import ref, WeakKeyDictionary
import types
import inspect
from inspect import Parameter, Signature
import itertools
import abc
from collections.abc import Iterable
import numpy as np
from cycler import cycler
import datetime
from functools import wraps, partial
import threading
import time
from tqdm import tqdm
from tqdm.utils import _screen_shape_wrapper, _term_move_up, _unicode
import warnings

import msgpack
import msgpack_numpy
import zict

try:
    # cytools is a drop-in replacement for toolz, implemented in Cython
    from cytools import groupby
except ImportError:
    from toolz import groupby


class Msg(namedtuple("Msg_base", ["command", "obj", "args", "kwargs", "run"])):
    """Namedtuple sub-class to encapsulate a message from the plan to the RE.

    This class provides 3 key features:

    1. dot access to the contents
    2. default values and a variadic signature for args / kwargs
    3. a nice repr
    """

    __slots__ = ()

    def __new__(cls, command, obj=None, *args, run=None, **kwargs):
        return super(Msg, cls).__new__(cls, command, obj, args, kwargs, run)

    def __repr__(self):
        return (f"Msg({self.command!r}, obj={self.obj!r}, "
                f"args={self.args}, kwargs={self.kwargs}, run={self.run!r})")


class RunEngineControlException(Exception):
    """Exception for signaling within the RunEngine."""


class RequestAbort(RunEngineControlException):
    """Request that the current run be aborted."""

    exit_status = 'abort'


class RequestStop(RunEngineControlException):
    """Request that the current run be stopped and marked successful."""

    exit_status = 'success'


class RunEngineInterrupted(Exception):
    pass


class NoReplayAllowed(Exception):
    pass


class IllegalMessageSequence(Exception):
    pass


class FailedPause(Exception):
    pass


class FailedStatus(Exception):
    """Exception to be raised if a SatusBase object reports done but failed"""


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
    """Context manager for signal handing

    If multiple signals come in quickly, they may not all be seen, quoting
    the libc manual:

      Remember that if there is a particular signal pending for your
      process, additional signals of that same type that arrive in the
      meantime might be discarded. For example, if a SIGINT signal is
      pending when another SIGINT signal arrives, your program will
      probably only see one of them when you unblock this signal.

    https://www.gnu.org/software/libc/manual/html_node/Checking-for-Pending-Signals.html
    """
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

            self.handle_signals()

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

    def handle_signals(self):
        ...


class SigintHandler(SignalHandler):
    def __init__(self, RE):
        super().__init__(signal.SIGINT, log=RE.log)
        self.RE = RE
        self.last_sigint_time = None  # time most recent SIGINT was processed
        self.num_sigints_processed = 0  # count SIGINTs processed

    def __enter__(self):
        return super().__enter__()

    def handle_signals(self):
        # Check for pause requests from keyboard.
        # TODO, there is a possible race condition between the two
        # pauses here
        if self.RE.state.is_running and (not self.RE._interrupted):
            if (self.last_sigint_time is None or
                    time.time() - self.last_sigint_time > 10):
                # reset the counter to 1
                # It's been 10 seconds since the last SIGINT. Reset.
                self.count = 1
                if self.last_sigint_time is not None:
                    self.log.debug("It has been 10 seconds since the "
                                   "last SIGINT. Resetting SIGINT "
                                   "handler.")
                # weeee push these to threads to not block the main thread
                threading.Thread(target=self.RE.request_pause,
                                 args=(True,)).start()
                print("A 'deferred pause' has been requested. The "
                      "RunEngine will pause at the next checkpoint. "
                      "To pause immediately, hit Ctrl+C again in the "
                      "next 10 seconds.")

                self.last_sigint_time = time.time()
            elif self.count == 2:
                print('trying a second time')
                # - Ctrl-C twice within 10 seconds -> hard pause
                self.log.debug("RunEngine detected two SIGINTs. "
                               "A hard pause will be requested.")

                threading.Thread(target=self.RE.request_pause,
                                 args=(False,)).start()
            self.last_sigint_time = time.time()


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
        # Following discussion with TC: weakref.WeakMethod can not be used to
        #   replace the custom 'BoundMethodProxy', because it does not accept
        #   the 'destroy callback' as a parameter. The 'destroy callback' is
        #   necessary to automatically unsubscribe CB registry from the callback
        #   when the class object is destroyed and this is the main purpose of
        #   BoundMethodProxy.
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
                # Here we need to delete the last reference to proxy (in 'self.callbacks[sig]')
                #   The respective entries in 'self._func_cid_map' are deleted automatically,
                #   since 'self._func_cid_map[sig]' entries are WeakKeyDictionary objects.
                del self.callbacks[sig][proxies[proxy]]
            except KeyError:
                pass

            # Remove dictionary items for signals with no assigned callbacks
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
                # This may or may not remove entries in 'self._func_cid_map'.
                del callbackd[cid]
            except KeyError:
                continue
            else:
                # Look for cid in 'self._func_cid_map' as well. It may still be there.
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
            # This branch is successful if 'cb' bound method and class method,
            #   but destroy_callback mechanism works only for bound methods,
            #   since cb.__self__ points to class instance only for
            #   bound methods, not for class methods. Therefore destroy_callback
            #   will not be called for class methods.
            try:
                self.inst = ref(cb.__self__, self._destroy)
            except TypeError:
                self.inst = None
            self.func = cb.__func__
            self.klass = cb.__self__.__class__

        except AttributeError:
            # 'cb' is a function, callable object or static method.
            # No weak reference is created, strong reference is stored instead.
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


class PersistentDict(collections.abc.MutableMapping):
    """
    A MutableMapping which syncs it contents to disk.

    The contents are stored as msgpack-serialized files, with one file per item
    in the mapping.

    Note that when an item is *mutated* it is not immediately synced:

    >>> d['sample'] = {"color": "red"}  # immediately synced
    >>> d['sample']['shape'] = 'bar'  # not immediately synced

    but that the full contents are synced to disk when the PersistentDict
    instance is garbage collected.
    """
    def __init__(self, directory):
        self._directory = directory
        self._file = zict.File(directory)
        self._func = zict.Func(self._dump, self._load, self._file)
        self._cache = {}
        self.reload()

        # Similar to flush() or _do_update(), but without reference to self
        # to avoid circular reference preventing collection.
        # NOTE: This still doesn't guarantee call on delete or gc.collect()!
        #       Explicitly call flush() if immediate write to disk required.
        def finalize(zfile, cache, dump):
            zfile.update((k, dump(v)) for k, v in cache.items())

        import weakref
        self._finalizer = weakref.finalize(
            self, finalize, self._file, self._cache, PersistentDict._dump)

    @property
    def directory(self):
        return self._directory

    def __setitem__(self, key, value):
        self._cache[key] = value
        self._func[key] = value

    def __getitem__(self, key):
        return self._cache[key]

    def __delitem__(self, key):
        del self._cache[key]
        del self._func[key]

    def __len__(self):
        return len(self._cache)

    def __repr__(self):
        return f"<{self.__class__.__name__} {dict(self)!r}>"

    def __iter__(self):
        yield from self._cache

    def popitem(self):
        key, value = self._cache.popitem()
        del self._func[key]
        return key, value

    @staticmethod
    def _dump(obj):
        "Encode as msgpack using numpy-aware encoder."
        # See https://github.com/msgpack/msgpack-python#string-and-binary-type
        # for more on use_bin_type.
        return msgpack.packb(
            obj,
            default=msgpack_numpy.encode,
            use_bin_type=True)

    @staticmethod
    def _load(file):
        return msgpack.unpackb(
            file,
            object_hook=msgpack_numpy.decode,
            raw=False)

    def flush(self):
        """Force a write of the current state to disk"""
        for k, v in self.items():
            self._func[k] = v

    def reload(self):
        """Force a reload from disk, overwriting current cache"""
        self._cache = dict(self._func.items())


SEARCH_PATH = []
ENV_VAR = 'BLUESKY_HISTORY_PATH'
if ENV_VAR in os.environ:
    SEARCH_PATH.append(os.environ[ENV_VAR])
SEARCH_PATH.extend([os.path.expanduser('~/.config/bluesky/bluesky_history.db'),
                    '/etc/bluesky/bluesky_history.db'])


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
    warnings.warn("bluesky.utils.install_qt_kicker is no longer necessary and "
                  "has no effect. Please remove your use of it. It may be "
                  "removed in a future release of bluesky.")


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


def register_transform(RE, *, prefix='<', ip=None):
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
    ip : IPython shell, optional
        If not passed, uses `IPython.get_ipython()` to get the current shell
    '''
    import IPython

    if ip is None:
        ip = IPython.get_ipython()

    if IPython.__version__ >= '7':
        def tr_re(lines):
            if len(lines) != 1:
                return lines
            line, = lines
            head, split, tail = line.partition(prefix)
            if split == prefix and head.strip() == '':
                line = f'{RE}({tail.strip()})\n'

            return [line]

        ip.input_transformers_post.append(tr_re)

    else:
        from IPython.core.inputtransformer import StatelessInputTransformer

        @StatelessInputTransformer.wrap
        def tr_re(line):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                return '{}({})'.format(RE, line)
            return line

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

    async def __call__(self, prompt, end='\n', flush=False):
        print(prompt, end=end, flush=flush)
        return (await self.q.get()).rstrip('\n')


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


def ts_msg_hook(msg, file=sys.stdout):
    t = '{:%H:%M:%S.%f}'.format(datetime.datetime.now())
    msg_fmt = "{: <17s} -> {!s: <15s} args: {}, kwargs: {}, run: {}".format(
        msg.command,
        msg.obj.name if hasattr(msg.obj, 'name') else msg.obj,
        msg.args,
        msg.kwargs,
        "'{}'".format(msg.run) if isinstance(msg.run, str) else msg.run)
    print('{} {}'.format(t, msg_fmt), file=file)


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


class ProgressBarBase(abc.ABC):
    def update(
            self,
            pos: Any,
            *,
            name: str = None,
            current: Any = None,
            initial: Any = None,
            target: Any = None,
            unit: str = "units",
            precision: Any = None,
            fraction: Any = None,
            time_elapsed: float = None,
            time_remaining: float = None,
    ):
        ...

    def clear(self):
        ...


class TerminalProgressBar(ProgressBarBase):
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
        self.ncols = _screen_shape_wrapper()(sys.stdout)[0] or 79
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
            # make sure we ignore overshoot to prevent tqdm from exploding.
            n = np.clip(round(_L2norm(current, initial), precision or 3), 0, total)
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


class ProgressBar(TerminalProgressBar):
    """
    Alias for backwards compatibility
    """

    ...


def default_progress_bar(status_objs_or_none) -> ProgressBarBase:
    return TerminalProgressBar(status_objs_or_none, delay_draw=0.2)


class ProgressBarManager:
    pbar_factory: Callable[[Any], ProgressBarBase]
    pbar: Optional[ProgressBarBase]

    def __init__(self,
                 pbar_factory: Callable[[Any], ProgressBarBase] = default_progress_bar):
        """
        Manages creation and tearing down of progress bars.

        Parameters
        ----------
        pbar_factory : Callable[[Any], ProgressBar], optional
            A function that creates a progress bar given an optional list of status objects,
            by default default_progress_bar
        """

        self.pbar_factory = pbar_factory
        self.pbar = None

    def __call__(self, status_objs_or_none):
        """
        Updates the manager with a new set of status, creates a new progress bar and
        cleans up the old one if needed.

        Parameters
        ----------
        status_objs_or_none : Set[Status], optional
            Optional list of status objects to be passed to the factory.
        """

        if status_objs_or_none is not None:
            # Start a new ProgressBar.
            if self.pbar is not None:
                warnings.warn("Previous ProgressBar never competed.")
                self.pbar.clear()
            self.pbar = self.pbar_factory(status_objs_or_none)
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


_qapp = None


class DuringTask:
    """This class waits on the event (which fully blocks the thread)."""

    def __init__(self):
        pass

    def block(self, blocking_event):
        """
        Wait plan to finish.

        Parameters
        ----------
        blocking_event : threading.Event

        """
        blocking_event.wait()


class DefaultDuringTask(DuringTask):
    """This class run the Qt main loop while waiting for the plan to finish.

    The default setting for the RunEngine's during_task parameter.

    This makes it possible for plots that use Matplotlib's Qt backend to update
    live during data acquisition.

    It solves the problem that Qt must be run from the main thread.
    If Matplotlib and a known Qt binding are already imported, run
    Matplotlib qApp until the task completes. If not, there is no need to
    handle qApp: just wait on the task.

    """

    def __init__(self):
        """
        Initialize backend.

        Currently only the Qt backend is supported. The function is
        initializing the 'teleporter' if Qt backend is used.

        """
        if 'matplotlib' in sys.modules:
            import matplotlib
            backend = matplotlib.get_backend().lower()
            if 'qt' in backend:
                from bluesky.callbacks.mpl_plotting import initialize_qt_teleporter
                initialize_qt_teleporter()

    def block(self, blocking_event):
        # docstring inherited
        global _qapp
        if 'matplotlib' not in sys.modules:
            # We are not using matplotlib + Qt. Just wait on the Event.
            blocking_event.wait()
        # Figure out if we are using matplotlib with which backend
        # without importing anything that is not already imported.
        else:
            import matplotlib
            backend = matplotlib.get_backend().lower()
            # if with a Qt backend, do the scary thing
            if 'qt' in backend:

                from matplotlib.backends.qt_compat import QtCore, QtWidgets
                app = QtWidgets.QApplication.instance()
                if app is None:
                    _qapp = app = QtWidgets.QApplication([b'bluesky'])
                assert app is not None
                event_loop = QtCore.QEventLoop()

                def start_killer_thread():
                    def exit_loop():
                        blocking_event.wait()
                        # If the above wait ends quickly, we need to avoid the race
                        # condition where this thread might try to exit the qApp
                        # before it even starts.  Therefore, we use QTimer, below,
                        # which will not start running until the qApp event loop is
                        # running.
                        event_loop.exit()

                    threading.Thread(target=exit_loop).start()

                # https://www.riverbankcomputing.com/pipermail/pyqt/2015-March/035674.html
                # adapted from code at
                # https://bitbucket.org/tortoisehg/thg/commits/550e1df5fbad
                if os.name == 'posix' and hasattr(signal, 'set_wakeup_fd'):
                    # Wake up Python interpreter via pipe so that SIGINT
                    # can be handled immediately.
                    # (http://qt-project.org/doc/qt-4.8/unix-signals.html)
                    # Updated docs:
                    # https://doc.qt.io/qt-5/unix-signals.html
                    import fcntl
                    rfd, wfd = os.pipe()
                    for fd in (rfd, wfd):
                        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                    wakeupsn = QtCore.QSocketNotifier(rfd,
                                                      QtCore.QSocketNotifier.Read)
                    origwakeupfd = signal.set_wakeup_fd(wfd)

                    def cleanup():
                        wakeupsn.setEnabled(False)
                        rfd = wakeupsn.socket()
                        wfd = signal.set_wakeup_fd(origwakeupfd)
                        os.close(int(rfd))
                        os.close(wfd)

                    def handleWakeup(inp):
                        # here Python signal handler will be invoked
                        # this book-keeping is to drain the pipe
                        wakeupsn.setEnabled(False)
                        rfd = wakeupsn.socket()
                        try:
                            os.read(int(rfd), 4096)
                        except OSError as inst:
                            print('failed to read wakeup fd: %s\n' % inst)

                        wakeupsn.setEnabled(True)

                    wakeupsn.activated.connect(handleWakeup)

                else:
                    # On Windows, non-blocking anonymous pipe or socket is
                    # not available.

                    def null():
                        ...

                    # we need to 'kick' the python interpreter so it sees
                    # system signals
                    # https://stackoverflow.com/a/4939113/380231
                    kick_timer = QtCore.QTimer()
                    kick_timer.timeout.connect(null)
                    kick_timer.start(50)

                    cleanup = kick_timer.stop

                # we also need to make sure that the qApp never sees
                # exceptions raised by python inside of a c++ callback (as
                # it will segfault itself because due to the way the
                # code is called there is no clear way to propagate that
                # back to the python code.
                vals = (None, None, None)

                old_sys_handler = sys.excepthook

                def my_exception_hook(exctype, value, traceback):
                    nonlocal vals
                    vals = (exctype, value, traceback)
                    event_loop.exit()
                    old_sys_handler(exctype, value, traceback)

                # this kill the Qt event loop when the plan is finished
                killer_timer = QtCore.QTimer()
                killer_timer.setSingleShot(True)
                killer_timer.timeout.connect(start_killer_thread)
                killer_timer.start(0)

                try:
                    sys.excepthook = my_exception_hook
                    event_loop.exec_()
                    # make sure any pending signals are processed
                    event_loop.processEvents()
                    if vals[1] is not None:
                        raise vals[1]
                finally:
                    try:
                        cleanup()
                    finally:
                        sys.excepthook = old_sys_handler
            elif 'ipympl' in backend or 'nbagg' in backend:
                Gcf = matplotlib._pylab_helpers.Gcf
                while True:
                    done = blocking_event.wait(.1)
                    for f_mgr in Gcf.get_all_fig_managers():
                        if f_mgr.canvas.figure.stale:
                            f_mgr.canvas.draw()
                    if done:
                        return
            else:
                # We are not using matplotlib + Qt. Just wait on the Event.
                blocking_event.wait()


def _rearrange_into_parallel_dicts(readings):
    data = {}
    timestamps = {}
    for key, payload in readings.items():
        data[key] = payload['value']
        timestamps[key] = payload['timestamp']
    return data, timestamps


def is_movable(obj):
    """Check if object satisfies bluesky 'movable' interface.

    Parameters
    ----------
    obj : Object
        Object to test.

    Returns
    -------
    boolean
        True if movable, False otherwise.
    """
    EXPECTED_ATTRS = (
        'name',
        'parent',
        'read',
        'describe',
        'read_configuration',
        'describe_configuration',
        'set',
    )
    return all(hasattr(obj, attr) for attr in EXPECTED_ATTRS)


class Movable(metaclass=abc.ABCMeta):
    """
    Abstract base class for objects that satisfy the bluesky 'movable' interface.

    Examples
    --------

    .. code-block:: python

        m = hw.motor
        # We need to detect if 'm' is a motor
        if isinstance(m, Movable):
            print(f"The object {m.name} is a motor")
    """
    @classmethod
    def __subclasshook__(cls, C):
        # If the following condition is True, the object C is recognized
        # to have Movable interface (e.g. a motor)
        msg = """The Movable abstract base class is deprecated and will be removed in a future
                 version of bluesky. Please use bluesky.utils.is_movable(obj) to test if an object
                 satisfies the movable interface."""
        warnings.warn(msg, DeprecationWarning)
        EXPECTED_ATTRS = (
            'name',
            'parent',
            'read',
            'describe',
            'read_configuration',
            'describe_configuration',
            'set',
            'stop',
        )
        return all(hasattr(C, attr) for attr in EXPECTED_ATTRS)
