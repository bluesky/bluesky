import asyncio
import itertools
import types
import time as ttime
import sys
import logging
from warnings import warn
from itertools import count, tee
from collections import namedtuple, deque, defaultdict, Iterable
import uuid
import signal
from enum import Enum


import json
import jsonschema
from super_state_machine.machines import StateMachine
from super_state_machine.extras import PropertyMachine
from super_state_machine.errors import TransitionError
import numpy as np
from pkg_resources import resource_filename as rs_fn

from .utils import (CallbackRegistry, SignalHandler, normalize_subs_input)

logger = logging.getLogger(__name__)


def expiring_function(func, *args, **kwargs):
    """
    If timeout has not occurred, call func(*args, **kwargs).

    This is meant to used with the event loop's run_in_exector
    method. Outside that context, it doesn't make any sense.
    """
    def dummy(start_time, timeout):
        if loop.time() > start_time + timeout:
            return
        func(*args, **kwargs)
        return

    return dummy


class DocumentNames(Enum):
    stop = 'stop'
    start = 'start'
    descriptor = 'descriptor'
    event = 'event'
    bulk_events = 'bulk_events'

SCHEMA_PATH = 'schema'
SCHEMA_NAMES = {DocumentNames.start: 'run_start.json',
                DocumentNames.stop: 'run_stop.json',
                DocumentNames.event: 'event.json',
                DocumentNames.bulk_events: 'bulk_events.json',
                DocumentNames.descriptor: 'event_descriptor.json'}
fn = '{}/{{}}'.format(SCHEMA_PATH)
schemas = {}
for name, filename in SCHEMA_NAMES.items():
    with open(rs_fn('bluesky', fn.format(filename))) as fin:
        schemas[name] = json.load(fin)


loop = asyncio.get_event_loop()
loop.set_debug(True)


class Msg(namedtuple('Msg_base', ['command', 'obj', 'args', 'kwargs'])):
    __slots__ = ()

    def __new__(cls, command, obj=None, *args, **kwargs):
        return super(Msg, cls).__new__(cls, command, obj, args, kwargs)

    def __repr__(self):
        return '{}: ({}), {}, {}'.format(
            self.command, self.obj, self.args, self.kwargs)


class RunEngineStateMachine(StateMachine):
    """

    Attributes
    ----------
    is_idle
        State machine is in its idle state
    is_running
        State machine is in its running state
    is_paused
        State machine is paused.
    """

    class States(Enum):
        """state.name = state.value"""
        IDLE = 'idle'
        RUNNING = 'running'
        PAUSED = 'paused'

        @classmethod
        def states(cls):
            return [state.value for state in cls]

    class Meta:
        allow_empty = False
        initial_state = 'idle'
        transitions = {
            # Notice that 'transitions' and 'named_transitions' have
            # opposite to <--> from structure.
            # from_state : [valid_to_states]
            'idle': ['running'],
            'running': ['idle', 'paused'],
            'paused': ['idle', 'running'],
        }
        named_checkers = [
            ('can_pause', 'paused'),
        ]


class LoggingPropertyMachine(PropertyMachine):
    "expects object to have a `log` attribute"
    def __init__(self, machine_type):
        super().__init__(machine_type)

    def __set__(self, obj, value):
        old_value = self.__get__(obj)
        super().__set__(obj, value)
        value = self.__get__(obj)
        obj.log.info("Change state on %r from %r -> %r",
                     obj, old_value, value)


class RunEngine:

    _loop = loop  # just a convenient way to inspect the global event loop
    state = LoggingPropertyMachine(RunEngineStateMachine)
    _UNCACHEABLE_COMMANDS = ['pause', 'subscribe', 'unsubscribe']

    def __init__(self, md=None, *, md_validator=None):
        """
        The Run Engine execute messages and emits Documents.

        Parameters
        ----------
        md : dict-like, optional
            The default is a standard Python dictionary, but fancier objects
            can be used to store long-term history and persist it between
            sessions. The standard configuration instantiates a Run Engine with
            historydict.HistoryDict, a simple interface to a sqlite file. Any object
            supporting `__getitem__`, `__setitem__`, and `clear` will work.

        md_validator : callable, optional
            a function that raises and prevents starting a run if it deems
            the metadata to be invalid or incomplete
            Expected signature: f(md)
            Function should raise if md is invalid. What that means is
            completely up to the user. The function's return value is
            ignored.

        Attributes
        ----------
        state
            {'idle', 'running', 'paused'}
        md
            direct access to the dict-like persistent storage described above
        event_timeout
            number of seconds before Events yet unprocessed by callbacks are
            skipped
        ignore_callback_exceptions
            boolean, True by default

        Methods
        -------
        request_pause
            Pause the Run Engine at the next checkpoint.
        resume
            Start from the last checkpoint after a pause.
        abort
            Move from 'paused' to 'idle', stopping the run permanently.
        panic
            Force the Run Engine to stop and/or disallow resume.
        all_is_well
            Un-panic
        register_command
            Teach the Run Engine a new Message command.
        unregister_command
            Undo register_command.
        """
        super().__init__()

        # Make a logger for this specific RE instance, using the instance's
        # Python id, to keep from mixing output from separate instances.
        logger_name = "{name}_id{id}".format(name=__name__, id=id(self))
        self.log = logging.getLogger(logger_name)
        self.log.setLevel(logging.DEBUG)
        self.verbose = False  # a convenience property, setting log.disabled

        if md is None:
            md = {}
        self.md = md
        if md_validator is None:
            md_validator = _default_md_validator
        self.md_validator = md_validator

        # The RunEngine keeps track of a *lot* of state.
        # All flags and caches are defined here with a comment. Good luck.
        self._metadata_per_call = {}  # for all runs generated by one __call__
        self._metadata_per_run = {}  # for one run, incorporating Msg metadata
        self._panic = False
        self._bundling = False  # if we are in the middle of bundling readings
        self._bundle_name = None  # name given to event descriptor
        self._deferred_pause_requested = False  # pause at next 'checkpoint'
        self._sigint_handler = None  # intercepts Ctrl+C
        self._exception = None  # stored and then raised in the _run loop
        self._interrupted = False  # True if paused, aborted, or failed
        self._objs_read = deque()  # objects read in one Event
        self._read_cache = deque()  # cache of obj.read() in one Event
        self._staged = set()  # objects staged, not yet unstaged
        self._movable_objs_touched = set()  # objects we moved at any point
        self._uncollected = set()  # objects after kickoff(), before collect()
        self._flyer_stream_names = {}  # names given at kickoff() for collect()
        self._run_start_uid = None  # uid of currently-open run
        self._run_start_uids = list()  # run start uids generated by __call__
        self._describe_cache = dict()  # cache of all obj.describe() output
        self._config_desc_cache = dict()  # " obj.describe_configuration()
        self._config_values_cache = dict()  # " obj.read_configuration() values
        self._config_ts_cache = dict()  # " obj.read_configuration() timestamps
        self._descriptors = dict()  # cache of {(name, objs_frozen_set): uid}
        self._sequence_counters = dict()  # a seq_num counter per Descriptor
        self._teed_sequence_counters = dict()  # for if we redo datapoints
        self._pause_requests = dict()  # holding {<name>: callable}
        self._block_groups = defaultdict(set)  # sets of objs to wait for
        self._temp_callback_ids = set()  # ids from CallbackRegistry
        self._msg_cache = None  # may be used to hold recently processed msgs
        self._genstack = deque()  # stack of generators to work off of
        self._new_gen = True  # flag if we need to prime the generator
        self._exit_status = 'success'  # optimistic default
        self._reason = ''  # reason for abort
        self._task = None  # asyncio.Task associated with call to self._run
        self._plan = None  # the scan plan instance from __call__
        self._command_registry = {
            'create': self._create,
            'save': self._save,
            'read': self._read,
            'null': self._null,
            'set': self._set,
            'trigger': self._trigger,
            'sleep': self._sleep,
            'wait': self._wait,
            'checkpoint': self._checkpoint,
            'clear_checkpoint': self._clear_checkpoint,
            'pause': self._pause,
            'collect': self._collect,
            'kickoff': self._kickoff,
            'configure': self._configure,
            'stage': self._stage,
            'unstage': self._unstage,
            'subscribe': self._subscribe,
            'unsubscribe': self._unsubscribe,
            'open_run': self._open_run,
            'close_run': self._close_run,
            'wait_for': self._wait_for,
        }

        # public dispatcher for callbacks
        # The Dispatcher's public methods are exposed through the
        # RunEngine for user convenience.
        self.dispatcher = Dispatcher()
        self.ignore_callback_exceptions = True
        self.event_timeout = 0.1
        self.subscribe = self.dispatcher.subscribe
        self.unsubscribe = self.dispatcher.unsubscribe

        # private dispatcher of critical callbacks
        # which get a lossless Event stream and abort a run if they raise
        # Ths subscribe/unsubscribe are exposed through the RunEngine
        # as subscribe_lossless and unsubscribe_lossless, defined below
        # to provide special docstrings.
        self._lossless_dispatcher = Dispatcher()

        loop.call_soon(self._check_for_trouble)
        loop.call_soon(self._check_for_signals)

    @property
    def verbose(self):
        return not self.log.disabled

    @verbose.setter
    def verbose(self, value):
        self.log.disabled = not value

    @property
    def _run_is_open(self):
        return self._run_start_uid is not None

    def _clear_run_cache(self):
        self._run_start_uid = None
        self._metadata_per_run.clear()
        self._bundling = False
        self._objs_read.clear()
        self._read_cache.clear()
        self._uncollected.clear()
        self._flyer_stream_names.clear()
        self._describe_cache.clear()
        self._config_desc_cache.clear()
        self._config_values_cache.clear()
        self._config_ts_cache.clear()
        self._descriptors.clear()
        self._sequence_counters.clear()
        self._teed_sequence_counters.clear()
        self._block_groups.clear()

    def _clear_call_cache(self):
        self._metadata_per_call.clear()
        self._staged.clear()
        self._movable_objs_touched.clear()
        self._deferred_pause_requested = False
        self._genstack = deque()
        self._msg_cache = None
        self._new_gen = True
        self._exception = None
        self._run_start_uids.clear()
        self._exit_status = 'success'
        self._reason = ''
        self._task = None
        self._plan = None
        self._interrupted = False

        # Unsubscribe for per-run callbacks.
        for cid in self._temp_callback_ids:
            self.unsubscribe(cid)
        self._temp_callback_ids.clear()

    def reset(self):
        self._clear_run_cache()
        self._clear_call_cache()
        self.dispatcher.unsubscribe_all()

    @property
    def resumable(self):
        return self._msg_cache is not None

    @property
    def ignore_callback_exceptions(self):
        return self.dispatcher.ignore_exceptions

    @ignore_callback_exceptions.setter
    def ignore_callback_exceptions(self, val):
        self.dispatcher.ignore_exceptions = val

    def register_command(self, name, func):
        """
        Register a new Message command.

        Parameters
        ----------
        name : str
        func : callable
            This can be a function or a method. The signature is `f(msg)`.
        """
        self._command_registry[name] = func

    def unregister_command(self, name):
        """
        Unregister a Message command.

        Parameters
        ----------
        name : str
        """
        del self._command_registry[name]

    def panic(self):
        """
        Do not permit the RunEngine to run until all_is_well() is called.

        If the RunEngine is currently running, it will abort before processing
        any more Messages and enter the 'idle' state.

        If the RunEngine is currently paused, it will stay in the 'paused'
        state, and it will disallow resume() until all_is_well() is called.
        """
        self._panic = True
        self._task.cancel()

    def all_is_well(self):
        """
        Un-panic.

        If the panic occurred during a pause, the run can be resumed.
        """
        self._panic = False
        # The cycle where _check_for_trouble schedules a future call to itself
        # is broken when it raises a PanicError.
        loop.call_later(0.1, self._check_for_trouble)

    def request_pause(self, defer=False, name=None, callback=None):
        """
        Command the Run Engine to pause.

        This function is called by 'pause' Messages. It can also be called
        by other threads. It cannot be called on the main thread during a run,
        but it is called by SIGINT (i.e., Ctrl+C).

        If there current run has no checkpoint commands, this will cause the
        run to abort.

        Parameters
        ----------
        defer : bool, optional
            If False, pause immediately before processing any new messages.
            If True, pause at the next checkpoint.
            False by default.
        name : str, optional
            Identify the source/reason for this pause. Required if there is a
            callback, below.
        callback : callable, optional
            "Permission to resume." Until this callable returns True, the Run
            Engine will not be allowed to resume. If None,
        """
        # No matter what, this will be processed if we try to resume.
        if callback is not None:
            if name is None:
                raise ValueError("Pause requests with a callback must include "
                                 "a name.")
            self._pause_requests[name] = callback
        # Now to the right pause state if we can.
        if not defer:
            self._interrupted = True
            if self.state.can_pause:
                print("Pausing...")
                self.state = 'paused'
                if self.resumable:
                    loop.stop()
                    # During pause, all motors should be stopped.
                    for obj in self._movable_objs_touched:
                        try:
                            obj.stop()
                        except Exception:
                            logger.error("Failed to stop %r", obj)
                else:
                    print("No checkpoint; cannot pause. Aborting...")
                    self._exception = FailedPause()
                    self._task.cancel()
            else:
                print("Cannot pause from {0} state. "
                      "Ignoring request.".format(self.state))
        else:
            if self.state.is_running:
                self._deferred_pause_requested = True
                print("Deferred pause acknowledged. Continuing to checkpoint.")
            else:
                print("Cannot pause from {0} state. "
                      "Ignoring request.".format(self.state))

    def subscribe_lossless(self, name, func):
        """Register a callback function to consume documents.

        Functions registered here are considered "critical." They receive
        a lossless stream of Event documents. If they generate an exception
        they always abort the run. (In contrast, exceptions from normal
        subscriptions are ignored by default.)

        The Run Engine can execute callback functions at the start and end
        of a scan, and after the insertion of new Event Descriptors
        and Events.

        Parameters
        ----------
        name : {'start', 'descriptor', 'event', 'stop', 'all'}
        func : callable
            expecting signature like ``f(name, document)``
            where name is a string and document is a dict

        Returns
        -------
        token : int
            an integer token that can be used to unsubscribe
        """
        return self._lossless_dispatcher.subscribe(name, func)

    def unsubscribe_lossless(self, token):
        """Un-register a 'critical' callback function.

        Parameters
        ----------
        token : int
            an integer token returned by _subscribe_lossless
        """
        self._lossless_dispatcher.unsubscribe(token)

    # aliases for back-compatibility
    _unsubscribe_lossless = unsubscribe_lossless
    _subscribe_lossless = subscribe_lossless

    def __call__(self, plan, subs=None, *, raise_if_interrupted=False,
                 **metadata_kw):
        """Run the scan defined by ``plan``

        Any keyword arguments other than those listed below will be interpreted
        as metadata and recorded with the run.

        Parameters
        ----------
        plan : generator
            a generator or that yields ``Msg`` objects (or an iterable that
            returns such a generator)
        subs: callable, list, or dict, optional
            Temporary subscriptions (a.k.a. callbacks) to be used on this run.
            For convenience, any of the following are accepted:
            - a callable, which will be subscribed to 'all'
            - a list of callables, which again will be subscribed to 'all'
            - a dictionary, mapping specific subscriptions to callables or
              lists of callables; valid keys are {'all', 'start', 'stop',
              'event', 'descriptor'}
        raise_if_interrupted : boolean
            If the RunEngine is called from inside a script or a function, it
            can be useful to make it raise an exception to halt further
            execution of the script after a pause or a stop. If True, these
            interruptions (that would normally not raise any exception) will
            raise RunEngineInterrupted. False by default.

        Returns
        -------
        uids : list
            list of Header uids (a.k.a RunStart uids) of run(s)

        Examples
        --------
        # Simplest example:
        >>> RE(my_scan)
        # Examples using subscriptions (a.k.a. callbacks):
        >>> def print_data(doc):
        ...     print("Measured: %s" % doc['data'])
        ...
        >>> def celebrate(doc):
        ...     # Do nothing with the input.
        ...     print("The run is finished!!!")
        ...
        >>> RE(my_generator, subs={'event': print_data, 'stop': celebrate})
        """
        # First thing's first: if we are in the wrong state, raise.
        if not self.state.is_idle:
            raise RuntimeError("The RunEngine is in a %s state" % self.state)
        if self._panic:
            raise PanicError("RunEngine is panicked. The run "
                             "was aborted before it began. No records "
                             "of this run were created.")

        self._clear_call_cache()
        self.state = 'running'

        for name, funcs in normalize_subs_input(subs).items():
            for func in funcs:
                self._temp_callback_ids.add(self.subscribe(name, func))

        self._plan = plan
        self._metadata_per_call.update(metadata_kw)

        gen = iter(plan)  # no-op on generators; needed for classes
        if not isinstance(gen, types.GeneratorType):
            # If plan does not support .send, we must wrap it in a generator.
            gen = (msg for msg in gen)
        self._genstack.append(gen)
        self._new_gen = True
        with SignalHandler(signal.SIGINT) as self._sigint_handler:  # ^C
            self._task = loop.create_task(self._run())
            loop.run_forever()
            if self._task.done() and not self._task.cancelled():
                exc = self._task.exception()
                if exc is not None:
                    raise exc
            if raise_if_interrupted and self._interrupted:
                raise RunEngineInterrupted("RunEngine was interrupted.")

        return self._run_start_uids

    def resume(self):
        """Resume a run from the last checkpoint.

        Returns
        -------
        requests_or_uids : list
            If any pause requests have not been released, a list of their names
            is immediately returned. Otherwise, after the run completes, this
            returns the list of uids this plan knows about.
        """
        # The state machine does not capture the whole picture.
        if not self.state.is_paused:
            raise RuntimeError("The RunEngine is the {0} state. You can only "
                               "resume for the paused state."
                               "".format(self.state))
        if self._panic:
            raise PanicError("Run Engine is panicked. If are you sure all is "
                             "well, call the all_is_well() method.")

        # Check that all pause requests have been released.
        outstanding_requests = []
        for name, func in list(self._pause_requests.items()):
            if func():
                # We have permission to continue. Clear the request.
                del self._pause_requests[name]
            else:
                outstanding_requests.append(name)
        if outstanding_requests:
            return outstanding_requests

        self._interrupted = False
        self._rewind()
        self._resume_event_loop()
        return self._run_start_uids

    def _rewind(self):
        "Clean up in preparation for resuming from a pause or suspension."
        self._genstack.append((msg for msg in list(self._msg_cache)))
        self._new_gen = True
        self._msg_cache = deque()
        self._sequence_counters.clear()
        self._sequence_counters.update(self._teed_sequence_counters)
        # This is needed to 'cancel' an open bundling (e.g. create) if
        # the pause happens after a 'checkpoint', after a 'create', but before
        # the paired 'save'.
        self._bundling = False

    def _resume_event_loop(self):
        # may be called by 'resume' or 'abort'
        self.state = 'running'
        with SignalHandler(signal.SIGINT) as self._sigint_handler:  # ^C
            if self._task.done():
                return
            loop.run_forever()
            if self._task.done() and not self._task.cancelled():
                exc = self._task.exception()
                if exc is not None:
                    raise exc

    def request_suspend(self, fut):
        """
        Request that the run suspend itself until the future is finished.

        Parameters
        ----------
        fut : asyncio.Future
        """
        if not self.resumable:
            print("No checkpoint; cannot suspend. Aborting...")
            self._exception = FailedPause()
        else:
            print("Suspending....To get prompt hit Ctrl-C to pause the scan")
            wait_msg = Msg('wait_for', [fut, ])
            self._msg_cache.appendleft(wait_msg)
            self._rewind()

    def abort(self, reason=''):
        """
        Stop a running or paused scan and mark it as aborted.
        """
        if self.state.is_idle:
            raise TransitionError("RunEngine is already idle.")
        print("Aborting....")
        self._interrupted = True
        self._reason = reason
        self._exception = RequestAbort()
        self._task.cancel()
        if self.state == 'paused':
            self._resume_event_loop()

    def stop(self):
        if self.state.is_idle:
            raise TransitionError("RunEngine is already idle.")
        print("Stopping...")
        self._interrupted = True
        self._exception = RequestStop()
        if self.state == 'paused':
            self._resume_event_loop()

    @asyncio.coroutine
    def _run(self):
        response = None
        self._reason = ''
        try:
            while True:
                try:
                    # This 'yield from' must be here to ensure that this
                    # coroutine breaks out of its current bevior before trying
                    # to get the next message from the top of the generator
                    # stack in case there has been a pause requested.
                    # Without this the next message after the pause may be
                    # processed first on resume (instead of the first
                    # message in self._msg_cache).
                    yield from asyncio.sleep(0.0001)
                    if self._exception is not None:
                        raise self._exception
                    # Send last response;
                    # get new message but don't process it yet.
                    try:
                        msg = self._genstack[-1].send(
                            response if not self._new_gen else None)

                    except StopIteration:
                        self._genstack.pop()
                        if len(self._genstack):
                            continue
                        else:
                            raise
                    if (self._msg_cache is not None and
                            msg.command not in self._UNCACHEABLE_COMMANDS):
                        # We have a checkpoint.
                        self._msg_cache.append(msg)
                    self._new_gen = False
                    try:
                        coro = self._command_registry[msg.command]
                        self.log.debug("Processing %r", msg)
                        response = yield from coro(msg)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        self._genstack[-1].throw(e)
                    self.log.debug("Response: %r", response)
                except KeyboardInterrupt:
                    # This only happens if some external code captures SIGINT
                    # -- overriding the RunEngine -- and then raises instead
                    # of (properly) calling the RunEngine's handler.
                    # See https://github.com/NSLS-II/bluesky/pull/242
                    loop.call_soon(self.request_pause, False, 'SIGINT')
                    print(PAUSE_MSG)
        except (StopIteration, RequestStop):
            self._exit_status = 'success'
            yield from asyncio.sleep(0.001)  # TODO Do we need this?
        except (FailedPause, RequestAbort, asyncio.CancelledError):
            self._exit_status = 'abort'
            yield from asyncio.sleep(0.001)  # TODO Do we need this?
            logger.error("Run aborted")
            logger.error("%s", self._exception)
            if isinstance(self._exception, PanicError):
                logger.critical("RE panicked")
                self._exit_status = 'fail'
                raise self._exception
        except Exception as err:
            self._exit_status = 'fail'  # Exception raises during 'running'
            self._reason = str(err)
            logger.error("Run aborted")
            logger.error("%s", err)
            raise err
        finally:
            self.state = 'idle'
            # call stop() on every movable object we ever set() or kickoff()
            for obj in self._movable_objs_touched:
                try:
                    obj.stop()
                except Exception:
                    logger.error("Failed to stop %r", obj)
            # Try to collect any flyers that were kicked off but not finished.
            # Some might not support partial collection. We swallow errors.
            for obj in list(self._uncollected):
                try:
                    yield from self._collect(Msg('collect', obj))
                except Exception:
                    logger.error("Failed to collect %r", obj)
            # in case we were interrupted between 'stage' and 'unstage'
            for obj in list(self._staged):
                try:
                    obj.unstage()
                except Exception:
                    logger.error("Failed to unstage %r", obj)
                self._staged.remove(obj)
            sys.stdout.flush()
            # Emit RunStop if necessary.
            if self._run_is_open:
                try:
                    yield from self._close_run(Msg('close_run'))
                except Exception:
                    logger.error("Failed to close run %r", self._run_start_uid)
                    # Exceptions from the callbacks should be re-raised.
                    # Close the loop first.
                    for task in asyncio.Task.all_tasks(loop):
                        task.cancel()
                    loop.stop()
                    raise
            for task in asyncio.Task.all_tasks(loop):
                task.cancel()
            loop.stop()

    def _check_for_trouble(self):
        if self.state.is_running:
            # Check for panic.
            if self._panic:
                self._interrupted = True
                self._exit_status = 'fail'
                exc = PanicError("Something told the Run Engine to "
                                 "panic after the run began. "
                                 "Records were created, but the run "
                                 "was marked with "
                                 "exit_status='fail'.")
                self._exception = exc  # will stop _run coroutine

        loop.call_later(0.1, self._check_for_trouble)

    def _check_for_signals(self):
        # Check for pause requests from keyboard.
        if self.state.is_running:
            if self._sigint_handler.interrupted:
                self._sigint_handler.interrupted = False
                with SignalHandler(signal.SIGINT) as second_sigint_handler:
                    self.log.debug("RunEngine detected a SIGINT (Ctrl+C)")
                    # We know we will either pause or abort, so we can
                    # hold up the scan now. Wait until 0.5 seconds pass or
                    # we catch a second SIGINT, whichever happens first.
                    for i in range(10):
                        ttime.sleep(0.05)
                        if second_sigint_handler.interrupted:
                            self.log.debug("RunEngine detected as second SIGINT")
                            loop.call_soon(self.abort, "SIGINT (Ctrl+C)")
                            break
                    else:
                        loop.call_soon(self.request_pause, False, 'SIGINT')
                        print(PAUSE_MSG)

        loop.call_later(0.1, self._check_for_signals)

    @asyncio.coroutine
    def _wait_for(self, msg):
        futs = msg.obj
        yield from asyncio.wait(futs)

    @asyncio.coroutine
    def _open_run(self, msg):
        if self._run_is_open:
            raise IllegalMessageSequence("A 'close_run' message was not "
                                         "received before the 'open_run' "
                                         "message")
        self._clear_run_cache()
        self._run_start_uid = new_uid()
        self._run_start_uids.append(self._run_start_uid)
        self.log.debug("Starting new run: %s", self._run_start_uid)

        # Increment scan ID
        scan_id = self.md.get('scan_id', 0) + 1
        self.md['scan_id'] = scan_id
        self.log.debug("New transient id %d", scan_id)

        # Metadata can come from historydict, __call__, or the open_run Msg.
        self._metadata_per_run.update(self.md)
        # Set a plan_type here, but it could be overruled,
        # which is useful if multiple plans were chained together.
        plan_type = type(self._plan).__name__
        self._metadata_per_run.update({'plan_type': plan_type})
        self._metadata_per_run.update(self._metadata_per_call)
        self._metadata_per_run.update(msg.kwargs)

        # The metadata is final. Validate it now, at the last moment.
        # Use copy for some reasonable (admittedly not total) protection
        # against users mutating the md with their validator.
        self.md_validator(dict(self._metadata_per_run))

        doc = dict(uid=self._run_start_uid, time=ttime.time(),
                   **self._metadata_per_run)
        yield from self.emit(DocumentNames.start, doc)
        self.log.debug("Emitted RunStart")

    @asyncio.coroutine
    def _close_run(self, msg):
        if not self._run_is_open:
            raise IllegalMessageSequence("A 'close_run' message was received "
                                         "but there is no run open. If this "
                                         "occurred after a pause/resume, add "
                                         "a 'checkpoint' message after the "
                                         "'close_run' message.")
        self.log.debug("Stopping run %s", self._run_start_uid)
        doc = dict(run_start=self._run_start_uid,
                   time=ttime.time(), uid=new_uid(),
                   exit_status=self._exit_status,
                   reason=self._reason)
        self._clear_run_cache()
        yield from self.emit(DocumentNames.stop, doc)
        self.log.debug("Emitted RunStop")

    @asyncio.coroutine
    def _create(self, msg):
        if self._bundling:
            raise IllegalMessageSequence("A second 'create' message is not "
                                         "allowed until the current event "
                                         "bundle is closed with a 'save' "
                                         "message.")
        self._read_cache.clear()
        self._objs_read.clear()
        self._bundling = True
        self._bundle_name = None  # default
        command, obj, args, kwargs = msg
        try:
            self._bundle_name = kwargs['name']
        except KeyError:
            if len(args) == 1:
                self._bundle_name, = args

    @asyncio.coroutine
    def _read(self, msg):
        obj = msg.obj
        self._objs_read.append(obj)
        if obj not in self._describe_cache:
            # Validate that there is no data key name collision.
            data_keys = obj.describe()
            keys = data_keys.keys()  # that is, field names
            for known_obj, known_data_keys in self._describe_cache.items():
                known_keys = known_data_keys.keys()  # that is, field names
                if set(known_keys) & set(keys):
                    raise ValueError("Data keys (field names) from {0!r} "
                                     "collide with those from {1!r}"
                                     "".format(obj, known_obj))
            self._describe_cache[obj] = data_keys
            self._config_desc_cache[obj] = obj.describe_configuration()
            self._cache_config(obj)
        ret = obj.read(*msg.args, **msg.kwargs)
        self._read_cache.append(ret)
        return ret

    def _cache_config(self, obj):
        "Read the object's configuration and cache it."
        config_values = {}
        config_ts = {}
        for key, val in obj.read_configuration().items():
            config_values[key] = val['value']
            config_ts[key] = val['timestamp']
        self._config_values_cache[obj] = config_values
        self._config_ts_cache[obj] = config_ts

    @asyncio.coroutine
    def _save(self, msg):
        if not self._run_is_open:
            raise IllegalMessageSequence("A 'save' message was sent but no "
                                         "run is open.")
        if not self._bundling:
            raise IllegalMessageSequence("A 'create' message must be sent, to "
                                         "open an event bundle, before that "
                                         "bundle can be saved with 'save'.")
        # Short-circuit if nothing has been read. (Do not create empty Events.)
        if not self._objs_read:
            self._bundling = False
            self._bundle_name = None
            return
        # The Event Descriptor is uniquely defined by the set of objects
        # read in this Event grouping.
        objs_read = frozenset(self._objs_read)

        # Event Descriptor
        if (self._bundle_name, objs_read) not in self._descriptors:
            # We don't not have an Event Descriptor for this set.
            data_keys = {}
            config = {}
            object_keys = {}
            for obj in objs_read:
                dks = self._describe_cache[obj]
                name = obj.name
                # dks is an OrderedDict. Record that order as a list.
                object_keys[obj.name] = list(dks)
                for field, dk in dks.items():
                    dk['object_name'] = name
                data_keys.update(dks)
                config[name] = {}
                config[name]['data'] = self._config_values_cache[obj]
                config[name]['timestamps'] = self._config_ts_cache[obj]
                config[name]['data_keys'] = self._config_desc_cache[obj]
            descriptor_uid = new_uid()
            doc = dict(run_start=self._run_start_uid, time=ttime.time(),
                       data_keys=data_keys, uid=descriptor_uid,
                       configuration=config, name=self._bundle_name,
                       object_keys=object_keys)
            yield from self.emit(DocumentNames.descriptor, doc)
            self.log.debug("Emitted Event Descriptor")
            self._descriptors[(self._bundle_name, objs_read)] = descriptor_uid
        else:
            descriptor_uid = self._descriptors[(self._bundle_name, objs_read)]
        # This is a separate check because it can be reset on resume.
        if objs_read not in self._sequence_counters:
            self._sequence_counters[objs_read] = count(1)
        self._bundling = False
        self._bundle_name = None

        # Events
        seq_num = next(self._sequence_counters[objs_read])
        event_uid = new_uid()
        # Merge list of readings into single dict.
        readings = {k: v for d in self._read_cache for k, v in d.items()}
        for key in readings:
            readings[key]['value'] = _sanitize_np(readings[key]['value'])
        data, timestamps = _rearrange_into_parallel_dicts(readings)
        doc = dict(descriptor=descriptor_uid,
                   time=ttime.time(), data=data, timestamps=timestamps,
                   seq_num=seq_num, uid=event_uid)
        yield from self.emit(DocumentNames.event, doc)
        self.log.debug("Emitted Event")

    @asyncio.coroutine
    def _kickoff(self, msg):
        _, obj, args, kwargs = msg
        self._uncollected.add(obj)
        block_group = msg.kwargs.pop('block_group', None)

        # Stash a name that will be put in the ev. desc. when collected.
        stream_name = None  # default
        try:
            stream_name = kwargs['name']
        except KeyError:
            if len(args) == 1:
                stream_name, = args
        self._flyer_stream_names[obj] = stream_name

        self._movable_objs_touched.add(obj)
        ret = obj.kickoff(*msg.args, **msg.kwargs)

        if block_group:
            p_event = asyncio.Event()

            def done_callback():
                if not ret.success:
                    loop.call_soon_threadsafe(self._failed_status, ret)
                loop.call_soon_threadsafe(p_event.set)

            ret.finished_cb = done_callback
            self._block_groups[block_group].add(p_event.wait())

        return ret

    @asyncio.coroutine
    def _collect(self, msg):
        if not self._run_is_open:
            raise IllegalMessageSequence("A 'collect' message was sent but no "
                                         "run is open.")
        obj = msg.obj
        self._uncollected.remove(obj)
        stream_name = self._flyer_stream_names.pop(obj)

        # TODO Since Flyer.describe() return a *list* of data_keys it should
        # probably have a different method name (describe_all?) *and* have some
        # way of injecting a 'name' like the create msg does. For now, the name
        # is hard-coded to None.
        data_keys_list = obj.describe()
        bulk_data = {}
        for data_keys in data_keys_list:
            objs_read = frozenset(data_keys)
            if (stream_name, objs_read) not in self._descriptors:
                # We don't not have an Event Descriptor for this set.
                descriptor_uid = new_uid()
                doc = dict(run_start=self._run_start_uid, time=ttime.time(),
                           data_keys=data_keys, uid=descriptor_uid,
                           name=stream_name)
                yield from self.emit(DocumentNames.descriptor, doc)
                self.log.debug("Emitted Event Descriptor")
                self._descriptors[(stream_name, objs_read)] = descriptor_uid
                self._sequence_counters[objs_read] = count(1)
            else:
                descriptor_uid = self._descriptors[(stream_name, objs_read)]

            bulk_data[descriptor_uid] = []

        for ev in obj.collect():
            objs_read = frozenset(ev['data'])
            seq_num = next(self._sequence_counters[objs_read])
            descriptor_uid = self._descriptors[(None, objs_read)]
            event_uid = new_uid()

            reading = ev['data']
            for key in ev['data']:
                reading[key] = _sanitize_np(reading[key])
            ev['data'] = reading
            ev['descriptor'] = descriptor_uid
            ev['seq_num'] = seq_num
            ev['uid'] = event_uid

            bulk_data[descriptor_uid].append(ev)

        yield from self.emit(DocumentNames.bulk_events, bulk_data)
        self.log.debug("Emitted bulk events")

    @asyncio.coroutine
    def _null(self, msg):
        pass

    @asyncio.coroutine
    def _set(self, msg):
        block_group = msg.kwargs.pop('block_group', None)
        self._movable_objs_touched.add(msg.obj)
        ret = msg.obj.set(*msg.args, **msg.kwargs)
        if block_group:
            p_event = asyncio.Event()

            def done_callback():

                self.log.debug("The object %r reports set is done "
                               "with status %r",
                               msg.obj, ret.success)

                if not ret.success:
                    loop.call_soon_threadsafe(self._failed_status, ret)
                loop.call_soon_threadsafe(p_event.set)

            ret.finished_cb = done_callback
            self._block_groups[block_group].add(p_event.wait())

        return ret

    @asyncio.coroutine
    def _trigger(self, msg):
        block_group = msg.kwargs.pop('block_group', None)
        ret = msg.obj.trigger(*msg.args, **msg.kwargs)

        if block_group:
            p_event = asyncio.Event()

            def done_callback():
                self.log.debug("The object %r reports trigger is "
                               "done with status %r.",
                               msg.obj, ret.success)

                if not ret.success:
                    loop.call_soon_threadsafe(self._failed_status, ret)

                loop.call_soon_threadsafe(p_event.set)

            ret.finished_cb = done_callback
            self._block_groups[block_group].add(p_event.wait())

        return ret

    @asyncio.coroutine
    def _wait(self, msg):
        # Block progress until every object that was trigged
        # triggered with the keyword argument `block=group` is done.
        group = msg.kwargs.get('group', msg.args[0])
        objs = list(self._block_groups.pop(group, []))
        if objs:
            yield from self._wait_for(Msg('wait_for', objs))

    def _failed_status(self, ret):
        """
        This is called a status object finishes but has failed.

        This will interrupt the run the next time the _run coroutine
        gets control of the event loop. The argument `ret` is included
        in the error message.

        Parameters
        ----------
        ret : StatusBase
            a status object that has failed
        """
        self._exception = FailedStatus(ret)
        # self._task.cancel()  # TODO -- should we kill ASAP?

    @asyncio.coroutine
    def _sleep(self, msg):
        yield from asyncio.sleep(*msg.args)

    @asyncio.coroutine
    def _pause(self, msg):
        self.request_pause(*msg.args, **msg.kwargs)

    @asyncio.coroutine
    def _checkpoint(self, msg):
        if self._bundling:
            raise IllegalMessageSequence("Cannot 'checkpoint' after 'create' "
                                         "and before 'save'. Aborting!")
        self._msg_cache = deque()

        # Keep a safe separate copy of the sequence counters to use if we
        # rewind and retake some data points.
        for key, counter in list(self._sequence_counters.items()):
            counter_copy1, counter_copy2 = tee(counter)
            self._sequence_counters[key] = counter_copy1
            self._teed_sequence_counters[key] = counter_copy2

        if self._deferred_pause_requested:
            self._interrupted = True
            self.state = 'paused'
            loop.stop()

    @asyncio.coroutine
    def _clear_checkpoint(self, msg):
        # clear message cache
        self._msg_cache = None
        # clear stashed
        self._teed_sequence_counters.clear()

    @asyncio.coroutine
    def _configure(self, msg):
        if self._bundling:
            raise IllegalMessageSequence(
                "Cannot configure after 'create' but before 'save'"
                "Aborting!")
        _, obj, args, kwargs = msg

        # Invalidate any event descriptors that include this object.
        # New event descriptors, with this new configuration, will
        # be created for any future event documents.
        for name, obj_set in list(self._descriptors):
            if obj in obj_set:
                del self._descriptors[(name, obj_set)]

        old, new = obj.configure(*args, **kwargs)

        self._cache_config(obj)
        return old, new

    @asyncio.coroutine
    def _stage(self, msg):
        _, obj, args, kwargs = msg
        # If an object has no 'stage' method, assume there is nothing to do.
        if not hasattr(obj, 'stage'):
            return
        result = obj.stage()
        self._staged.add(obj)  # add first in case of failure below
        return result

    @asyncio.coroutine
    def _unstage(self, msg):
        _, obj, args, kwargs = msg
        # If an object has no 'unstage' method, assume there is nothing to do.
        if not hasattr(obj, 'unstage'):
            return
        result = obj.unstage()
        self._staged.remove(obj)
        return result

    @asyncio.coroutine
    def _subscribe(self, msg):
        """
        Add a subscription after the run has started.

        This, like subscriptions passed to __call__, will be removed at the
        end by the RunEngine.
        """
        self.log.debug("Adding subsription %r", msg)
        _, obj, args, kwargs = msg
        token = self.subscribe(*args, **kwargs)
        self._temp_callback_ids.add(token)
        return token

    @asyncio.coroutine
    def _unsubscribe(self, msg):
        """
        Remove a subscription during a call -- useful for a multi-run call
        where subscriptions are wanted for some runs but not others.
        """
        self.log.debug("Removing subscription %r", msg)
        _, obj, args, kwargs = msg
        try:
            token = kwargs['token']
        except KeyError:
            token, = args
        self.unsubscribe(token)
        self._temp_callback_ids.remove(token)

    @asyncio.coroutine
    def emit(self, name, doc):
        "Process blocking callbacks and schedule non-blocking callbacks."
        jsonschema.validate(doc, schemas[name])
        self._lossless_dispatcher.process(name, doc)
        if name != DocumentNames.event:
            self.dispatcher.process(name, doc)
        else:
            start_time = loop.time()
            dummy = expiring_function(self.dispatcher.process, name, doc)
            loop.run_in_executor(None, dummy, start_time, self.event_timeout)


class Dispatcher:
    """Dispatch documents to user-defined consumers on the main thread."""

    def __init__(self):
        self.cb_registry = CallbackRegistry(allowed_sigs=DocumentNames)
        self._counter = count()
        self._token_mapping = dict()

    def process(self, name, doc):
        exceptions = self.cb_registry.process(name, name.name, doc)
        for exc, traceback in exceptions:
            warn("A %r was raised during the processing of a %s "
                 "Document. The error will be ignored to avoid "
                 "interrupting data collection. To investigate, "
                 "set RunEngine.ignore_callback_exceptions = False "
                 "and run again." % (exc, name.name))

    def subscribe(self, name, func):
        """
        Register a callback function to consume documents.

        The Run Engine can execute callback functions at the start and end
        of a scan, and after the insertion of new Event Descriptors
        and Events.

        Parameters
        ----------
        name: {'start', 'descriptor', 'event', 'stop', 'all'}
        func: callable
            expecting signature like ``f(name, document)``
            where name is a string and document is a dict

        Returns
        -------
        token : int
            an integer token that can be used to unsubscribe
        """
        if name == 'all':
            private_tokens = []
            for key in DocumentNames:
                private_tokens.append(self.cb_registry.connect(key, func))
            public_token = next(self._counter)
            self._token_mapping[public_token] = private_tokens
            return public_token

        if name not in DocumentNames:
            name = DocumentNames[name]
        private_token = self.cb_registry.connect(name, func)
        public_token = next(self._counter)
        self._token_mapping[public_token] = [private_token]
        return public_token

    def unsubscribe(self, token):
        """
        Unregister a callback function using its integer ID.

        Parameters
        ----------
        token : int
            the integer token issued by `subscribe`
        """
        for private_token in self._token_mapping[token]:
            self.cb_registry.disconnect(private_token)

    def unsubscribe_all(self):
        """Unregister ALL callbacks from the dispatcher
        """
        for public_token in self._token_mapping.keys():
            self.unsubscribe(public_token)

    @property
    def ignore_exceptions(self):
        return self.cb_registry.ignore_exceptions

    @ignore_exceptions.setter
    def ignore_exceptions(self, val):
        self.cb_registry.ignore_exceptions = val


def new_uid():
    return str(uuid.uuid4())


def _sanitize_np(val):
    "Convert any numpy objects into built-in Python types."
    if isinstance(val, np.generic):
        if np.isscalar(val):
            return val.item()
        return val.tolist()
    return val


def _rearrange_into_parallel_dicts(readings):
    data = {}
    timestamps = {}
    for key, payload in readings.items():
        data[key] = payload['value']
        timestamps[key] = payload['timestamp']
    return data, timestamps


def _run_engine_log_template(metadata):
    template = []
    for key in metadata:
        template.append("{key}: {{{key}}}".format(key=key))
    return '\n'.join(template)


class RequestAbort(Exception):
    pass


class RequestStop(Exception):
    pass


class PanicError(Exception):
    pass


class RunEngineInterrupted(Exception):
    pass


class IllegalMessageSequence(Exception):
    pass


class FailedPause(Exception):
    pass


class FailedStatus(Exception):
    'Exception to be raised if a SatusBase object reports done but failed'


PAUSE_MSG = """
Your RunEngine is entering a paused state. These are your options for changing
the state of the RunEngine:

resume()  --> will resume the scan
 abort()  --> will kill the scan with an 'aborted' state to indicate
              the scan was interrupted
  stop()  --> will kill the scan with a 'finished' state to indicate
              the scan stopped normally

Pro Tip: Next time, if you want to abort, tap Ctrl+C twice quickly.
"""


def _default_md_validator(md):
    for field in ['beamline_id', 'owner', 'group']:
        if field not in md:
            raise KeyError("The field '{0}' was not specified as is "
                           "required.".format(field))
        if 'sample' in md and not (hasattr(md['sample'], 'keys')
                                   or isinstance(md['sample'], str)):
            raise ValueError(
                "You specified 'sample' metadata. We give this field special "
                "significance in order to make your data easily searchable. "
                "Therefore, you must make 'sample' a string or a  "
                "dictionary, like so: "
                "GOOD: sample='dirt' "
                "GOOD: sample={'color': 'red', 'number': 5} "
                "BAD: sample=[1, 2] ")
