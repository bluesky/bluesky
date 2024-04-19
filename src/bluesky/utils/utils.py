import abc
import asyncio
import collections.abc
import inspect
import operator
import os
import signal
import sys
import threading
import time
import types
import warnings
from functools import partial
from inspect import Parameter, Signature
from typing import (
    Any,
    Callable,
    List,
    Optional,
)
from weakref import WeakKeyDictionary, ref

import msgpack
import msgpack_numpy
import numpy as np
import zict
from tqdm import tqdm
from tqdm.utils import _screen_shape_wrapper, _term_move_up, _unicode

from bluesky._vendor.super_state_machine.errors import TransitionError
from bluesky.utils.helper_functions import _L2norm, normalize_subs_input


class RunEngineControlException(Exception):
    """Exception for signaling within the RunEngine."""


class RequestAbort(RunEngineControlException):
    """Request that the current run be aborted."""

    exit_status = "abort"


class RequestStop(RunEngineControlException):
    """Request that the current run be stopped and marked successful."""

    exit_status = "success"


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


class RampFail(RuntimeError): ...


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
                self.log.debug("SignalHandler caught SIGINT; count is %r", self.count)
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

    def handle_signals(self): ...


class SigintHandler(SignalHandler):
    def __init__(self, RE):
        super().__init__(signal.SIGINT, log=RE.log)
        self.RE = RE
        self.last_sigint_time = None  # time most recent SIGINT was processed

    def __enter__(self):
        return super().__enter__()

    def handle_signals(self):
        # Check for pause requests from keyboard.
        # TODO, there is a possible race condition between the two
        # pauses here
        if self.RE.state.is_running and (not self.RE._interrupted):
            if self.last_sigint_time is None or time.time() - self.last_sigint_time > 10:
                # reset the counter to 1
                # It's been 10 seconds since the last SIGINT. Reset.
                self.count = 1
                if self.last_sigint_time is not None:
                    self.log.debug("It has been 10 seconds since the last SIGINT. Resetting SIGINT handler.")

                # weeee push these to threads to not block the main thread
                def maybe_defer_pause():
                    try:
                        self.RE.request_pause(True)
                    except TransitionError:
                        ...

                threading.Thread(target=maybe_defer_pause).start()
                print(
                    "A 'deferred pause' has been requested. The "
                    "RunEngine will pause at the next checkpoint. "
                    "To pause immediately, hit Ctrl+C again in the "
                    "next 10 seconds."
                )

                self.last_sigint_time = time.time()
            elif self.count == 2:
                print("trying a second time")
                # - Ctrl-C twice within 10 seconds -> hard pause
                self.log.debug("RunEngine detected two SIGINTs. A hard pause will be requested.")

                # weeee push these to threads to not block the main thread
                def maybe_prompt_pause():
                    try:
                        self.RE.request_pause(False)
                    except TransitionError:
                        ...

                threading.Thread(target=maybe_prompt_pause).start()
            self.last_sigint_time = time.time()


class CallbackRegistry:
    """
    See matplotlib.cbook.CallbackRegistry. This is a simplified since
    ``bluesky`` is python3.4+ only!
    """

    def __init__(self, ignore_exceptions=False, allowed_sigs=None):
        self.ignore_exceptions = ignore_exceptions
        self.allowed_sigs = allowed_sigs
        self.callbacks = dict()  # noqa: C408
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
                raise ValueError(f"Allowed signals are {self.allowed_sigs}")
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
        self.callbacks.setdefault(sig, dict())  # noqa: C408
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
        for eventname, callbackd in self.callbacks.items():  # noqa: B007
            try:
                # This may or may not remove entries in 'self._func_cid_map'.
                del callbackd[cid]
            except KeyError:
                continue
            else:
                # Look for cid in 'self._func_cid_map' as well. It may still be there.
                for sig, functions in self._func_cid_map.items():  # noqa: B007
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
                raise ValueError(f"Allowed signals are {self.allowed_sigs}")
        exceptions = []
        if sig in self.callbacks:
            for cid, func in list(self.callbacks[sig].items()):  # noqa: B007
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
    """
    Our own proxy object which enables weak references to bound and unbound
    methods and arbitrary callables. Pulls information about the function,
    class, and instance out of a bound method. Stores a weak reference to the
    instance to support garbage collection.
    @organization: IBM Corporation
    @copyright: Copyright (c) 2005, 2006 IBM Corporation
    @license: The BSD License
    Minor bugfixes by Michael Droettboom
    """

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
        inst = d["inst"]
        if inst is not None:
            d["inst"] = inst()
        return d

    def __setstate__(self, statedict):
        self.__dict__ = statedict
        inst = statedict["inst"]
        # turn inst back into a weakref
        if inst is not None:
            self.inst = ref(inst)

    def __call__(self, *args, **kwargs):
        """
        Proxy for a call to the weak referenced object. Take
        arbitrary params to pass to the callable.
        Raises `ReferenceError`: When the weak reference refers to
        a dead object
        """
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
        """
        Compare the held function and instance with that held by
        another proxy.
        """
        try:
            if self.inst is None:
                return self.func == other.func and other.inst is None
            else:
                return self.func == other.func and self.inst() == other.inst()
        except Exception:
            return False

    def __ne__(self, other):
        """
        Inverse of __eq__.
        """
        return not self.__eq__(other)

    def __hash__(self):
        return self._hash


# The following two code blocks are adapted from David Beazley's
# 'Python 3 Metaprogramming' https://www.youtube.com/watch?v=sPiWg5jSoZI


class StructMeta(type):
    def __new__(cls, name, bases, clsdict):
        clsobj = super().__new__(cls, name, bases, clsdict)
        args_params = [Parameter(name, Parameter.POSITIONAL_OR_KEYWORD) for name in clsobj._fields]
        kwargs_params = [Parameter(name, Parameter.KEYWORD_ONLY, default=None) for name in ["md"]]
        sig = Signature(args_params + kwargs_params)
        clsobj.__signature__ = sig
        return clsobj


class Struct(metaclass=StructMeta):
    "The _fields of any subclass become its attritubes and __init__ args."

    _fields: List[str] = []

    def __init__(self, *args, **kwargs):
        # Now bind default values of optional arguments.
        # If it seems like there should be a cleaner way to do this, see
        # http://bugs.python.org/msg221104
        bound = self.__signature__.bind(*args, **kwargs)
        for name, param in self.__signature__.parameters.items():
            if name not in bound.arguments and param.default is not inspect._empty:
                bound.arguments[name] = param.default
        for name, val in bound.arguments.items():
            setattr(self, name, val)
        self.flyers = []

    def set(self, **kwargs):
        "Update attributes as keyword arguments."
        for attr, val in kwargs.items():
            setattr(self, attr, val)


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

        self._finalizer = weakref.finalize(self, finalize, self._file, self._cache, PersistentDict._dump)

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
        return msgpack.packb(obj, default=msgpack_numpy.encode, use_bin_type=True)

    @staticmethod
    def _load(file):
        return msgpack.unpackb(file, object_hook=msgpack_numpy.decode, raw=False)

    def flush(self):
        """Force a write of the current state to disk"""
        for k, v in self.items():
            self._func[k] = v

    def reload(self):
        """Force a reload from disk, overwriting current cache"""
        self._cache = dict(self._func.items())


class AsyncInput:
    """a input prompt that allows event loop to run in the background

    adapted from http://stackoverflow.com/a/35514777/1221924
    """

    def __init__(self, loop=None):
        self.loop = loop or asyncio.get_event_loop()
        if sys.version_info < (3, 10):
            self.q = asyncio.Queue(loop=self.loop)
        else:
            self.q = asyncio.Queue()
        self.loop.add_reader(sys.stdin, self.got_input)

    def got_input(self):
        asyncio.ensure_future(self.q.put(sys.stdin.readline()), loop=self.loop)

    async def __call__(self, prompt, end="\n", flush=False):
        print(prompt, end=end, flush=flush)
        return (await self.q.get()).rstrip("\n")


class ProgressBarBase(abc.ABC):  # noqa: B024
    def update(  # noqa: B027
        self,
        pos: Any,
        *,
        name: Optional[str] = None,
        current: Any = None,
        initial: Any = None,
        target: Any = None,
        unit: str = "units",
        precision: Any = None,
        fraction: Any = None,
        time_elapsed: Optional[float] = None,
        time_remaining: Optional[float] = None,
    ): ...

    def clear(self): ...  # noqa: B027


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
                if hasattr(st, "watch") and not st.done:
                    pos = len(self.meters)
                    self.meters.append("")
                    self.status_objs.append(st)
                    st.watch(partial(self.update, pos))

    def update(
        self,
        pos,
        *,
        name=None,
        current=None,
        initial=None,
        target=None,
        unit="units",
        precision=None,
        fraction=None,
        time_elapsed=None,
        time_remaining=None,
    ):
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
            meter = tqdm.format_meter(
                n=n,
                total=total,
                elapsed=time_elapsed,
                unit=unit,
                prefix=name,
                ncols=self.ncols,
            )
        else:
            # Simply display completeness.
            if name is None:
                name = ""
            if self.status_objs[pos].done:
                meter = name + " [Complete.]"
            else:
                meter = name + " [In progress. No progress bar available.]"
            meter += " " * (self.ncols - len(meter))
            meter = meter[: self.ncols]

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
                self.fp.write("\n")
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
                for meter in self.meters:  # noqa: B007
                    self.fp.write("\r")
                    self.fp.write(" " * self.ncols)
                    self.fp.write("\r")
                    self.fp.write("\n")
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

    def __init__(self, pbar_factory: Callable[[Any], ProgressBarBase] = default_progress_bar):
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
                warnings.warn("Previous ProgressBar never competed.")  # noqa: B028
                self.pbar.clear()
            self.pbar = self.pbar_factory(status_objs_or_none)
        else:
            # Clean up an old one.
            if self.pbar is None:
                warnings.warn("There is no Progress bar to clean up.")  # noqa: B028
            else:
                self.pbar.clear()
                self.pbar = None


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
        if "matplotlib" in sys.modules:
            import matplotlib

            backend = matplotlib.get_backend().lower()
            if "qt" in backend:
                from bluesky.callbacks.mpl_plotting import initialize_qt_teleporter

                initialize_qt_teleporter()

    def block(self, blocking_event):
        # docstring inherited
        global _qapp
        if "matplotlib" not in sys.modules:
            # We are not using matplotlib + Qt. Just wait on the Event.
            blocking_event.wait()
        # Figure out if we are using matplotlib with which backend
        # without importing anything that is not already imported.
        else:
            import matplotlib

            backend = matplotlib.get_backend().lower()
            # if with a Qt backend, do the scary thing
            if "qt" in backend:
                import functools

                from matplotlib.backends.qt_compat import QT_API, QtCore, QtWidgets

                @functools.lru_cache(None)
                def _enum(name):
                    """
                    Between PyQt5 and PyQt6 how the various enums were accessed changed from
                    all of the various names being in the module namespace to being nested under
                    the Enum name.  Thus in PyQt5 we access the QSocketNotifier Type enum values as ::

                        QtCore.QSocketNotifier.Read

                    but in PyQt6 we use::

                         QtCore.QSocketNotifier.Type.Read

                    rather than have this checking inline where we use it, this function lets us do::

                        _enum('QtCore.QSocketNotifier.Type').Read

                    and well get the right namespace to get the Enum member from.

                    We use the extra layer of indirection of `operator.attrgetter` so that
                    multi-level names work and we can rely on the Qt compat layer to get the
                    correct PyQt5 vs PyQt6

                    This is copied from Matplotlib.
                    """
                    # foo.bar.Enum.Entry (PyQt6) <=> foo.bar.Entry (non-PyQt6).
                    return operator.attrgetter(name if QT_API == "PyQt6" else name.rpartition(".")[0])(
                        sys.modules[QtCore.__package__]
                    )

                app = QtWidgets.QApplication.instance()
                if app is None:
                    _qapp = app = QtWidgets.QApplication([b"bluesky"])
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
                if (
                    os.name == "posix"
                    and hasattr(signal, "set_wakeup_fd")
                    and
                    # TODO also check if main interpreter
                    threading.current_thread() is threading.main_thread()
                ):
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
                    wakeupsn = QtCore.QSocketNotifier(rfd, _enum("QtCore.QSocketNotifier.Type").Read)
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
                            print("failed to read wakeup fd: %s\n" % inst)

                        wakeupsn.setEnabled(True)

                    wakeupsn.activated.connect(handleWakeup)

                else:
                    # On Windows, non-blocking anonymous pipe or socket is
                    # not available.

                    def null(): ...

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
                    (event_loop.exec() if hasattr(event_loop, "exec") else event_loop.exec_())
                    # make sure any pending signals are processed
                    event_loop.processEvents()
                    if vals[1] is not None:
                        raise vals[1]
                finally:
                    try:
                        cleanup()
                    finally:
                        sys.excepthook = old_sys_handler
            elif "ipympl" in backend or "nbagg" in backend:
                Gcf = matplotlib._pylab_helpers.Gcf
                while True:
                    done = blocking_event.wait(0.1)
                    for f_mgr in Gcf.get_all_fig_managers():
                        if f_mgr.canvas.figure.stale:
                            f_mgr.canvas.draw()
                    if done:
                        return
            else:
                # We are not using matplotlib + Qt. Just wait on the Event.
                blocking_event.wait()
