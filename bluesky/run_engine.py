import asyncio
import time as ttime
import sys
import logging
from warnings import warn
from itertools import count, tee
from collections import deque, defaultdict, ChainMap
import uuid
import signal
from enum import Enum
import functools


import jsonschema
from event_model import DocumentNames, schemas
from super_state_machine.machines import StateMachine
from super_state_machine.extras import PropertyMachine
from super_state_machine.errors import TransitionError
import numpy as np

from .utils import (CallbackRegistry, SignalHandler, normalize_subs_input,
                    AsyncInput, new_uid, sanitize_np)
from . import Msg
from .plan_tools import ensure_generator
from .plans import single_gen


def expiring_function(func, loop, *args, **kwargs):
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

    state = LoggingPropertyMachine(RunEngineStateMachine)
    _UNCACHEABLE_COMMANDS = ['pause', 'subscribe', 'unsubscribe', 'stage',
                             'unstage', 'monitor', 'unmonitor', 'open_run',
                             'close_run']

    def __init__(self, md=None, *, loop=None, md_validator=None):
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

        loop : asyncio event loop
            e.g., ``asyncio.get_event_loop()`` or ``asyncio.new_event_loop()``

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

        msg_hook
            callable that receives all messages before they are processed
            (useful for logging or other development purposes); expected
            signature is ``f(msg)`` where ``msg`` is a ``bluesky.Msg``, a
            kind of namedtuple; default is None

        suspenders
            read-only collection of `bluesky.suspenders.SuspenderBase` objects
            which can suspend and resume execution; see related methods.


        Methods
        -------
        request_pause
            Pause the Run Engine at the next checkpoint.
        resume
            Start from the last checkpoint after a pause.
        abort
            Move from 'paused' to 'idle', stopping the run permanently.
        register_command
            Teach the Run Engine a new Message command.
        unregister_command
            Undo register_command.
        """
        if loop is None:
            loop = asyncio.get_event_loop()
        self._loop = loop

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

        self.msg_hook = None
        self.record_interruptions = False

        # The RunEngine keeps track of a *lot* of state.
        # All flags and caches are defined here with a comment. Good luck.
        self._metadata_per_call = {}  # for all runs generated by one __call__
        self._bundling = False  # if we are in the middle of bundling readings
        self._bundle_name = None  # name given to event descriptor
        self._deferred_pause_requested = False  # pause at next 'checkpoint'
        self._sigint_handler = None  # intercepts Ctrl+C
        self._last_sigint_time = None  # time most recent SIGINT was processed
        self._num_sigints_processed = 0  # count SIGINTs processed
        self._exception = None  # stored and then raised in the _run loop
        self._interrupted = False  # True if paused, aborted, or failed
        self._objs_read = deque()  # objects read in one Event
        self._read_cache = deque()  # cache of obj.read() in one Event
        self._staged = set()  # objects staged, not yet unstaged
        self._objs_seen = set()  # all objects seen
        self._movable_objs_touched = set()  # objects we moved at any point
        self._uncollected = set()  # objects after kickoff(), before collect()
        self._run_start_uid = None  # uid of currently-open run
        self._run_start_uids = list()  # run start uids generated by __call__
        self._interruptions_desc_uid = None  # uid for a special Event Desc.
        self._interruptions_counter = count(1)  # seq_num, special Event stream
        self._describe_cache = dict()  # cache of all obj.describe() output
        self._config_desc_cache = dict()  # " obj.describe_configuration()
        self._config_values_cache = dict()  # " obj.read_configuration() values
        self._config_ts_cache = dict()  # " obj.read_configuration() timestamps
        self._descriptors = dict()  # cache of {(name, objs_frozen_set): uid}
        self._monitor_params = dict()  # cache of {obj: (cb, kwargs)}
        self._sequence_counters = dict()  # a seq_num counter per Descriptor
        self._teed_sequence_counters = dict()  # for if we redo datapoints
        self._suspenders = set()  # set holding suspenders
        self._groups = defaultdict(set)  # sets of objs to wait for
        self._temp_callback_ids = set()  # ids from CallbackRegistry
        self._msg_cache = deque()  # history of processed msgs for rewinding
        self._plan_stack = deque()  # stack of generators to work off of
        self._response_stack = deque([None])  # resps to send into the plans
        self._exit_status = 'success'  # optimistic default
        self._reason = ''  # reason for abort
        self._task = None  # asyncio.Task associated with call to self._run
        self._failed_status_tasks = deque()  # Tasks from self._failed_status
        self._plan = None  # the scan plan instance from __call__
        self._command_registry = {
            'create': self._create,
            'save': self._save,
            'read': self._read,
            'monitor': self._monitor,
            'unmonitor': self._unmonitor,
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
            'complete': self._complete,
            'configure': self._configure,
            'stage': self._stage,
            'unstage': self._unstage,
            'subscribe': self._subscribe,
            'unsubscribe': self._unsubscribe,
            'open_run': self._open_run,
            'close_run': self._close_run,
            'wait_for': self._wait_for,
            'input': self._input,
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

        self.loop.call_soon(self._check_for_signals)

    @property
    def loop(self):
        return self._loop

    @property
    def suspenders(self):
        return tuple(self._suspenders)

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
        "Clean up for a new run."
        self._run_start_uid = None
        self._bundling = False
        self._objs_read.clear()
        self._read_cache.clear()
        self._uncollected.clear()
        self._describe_cache.clear()
        self._config_desc_cache.clear()
        self._config_values_cache.clear()
        self._config_ts_cache.clear()
        self._descriptors.clear()
        self._sequence_counters.clear()
        self._teed_sequence_counters.clear()
        self._groups.clear()
        self._interruptions_desc_uid = None
        self._interruptions_counter = count(1)

    def _clear_call_cache(self):
        "Clean up for a new __call__ (which may encompass multiple runs)."
        self._metadata_per_call.clear()
        self._staged.clear()
        self._objs_seen.clear()
        self._movable_objs_touched.clear()
        self._deferred_pause_requested = False
        self._plan_stack = deque()
        self._msg_cache = deque()
        self._response_stack = deque([None])
        self._exception = None
        self._run_start_uids.clear()
        self._exit_status = 'success'
        self._reason = ''
        self._task = None
        self._failed_status_tasks.clear()
        self._plan = None
        self._interrupted = False
        self._last_sigint_time = None
        self._num_sigints_processed = 0

        # Unsubscribe for per-run callbacks.
        for cid in self._temp_callback_ids:
            self.unsubscribe(cid)
        self._temp_callback_ids.clear()

    def reset(self):
        """
        Clean up caches and unsubscribe lossy subscriptions.

        Lossless subscriptions are not unsubscribed.
        """
        self._clear_run_cache()
        self._clear_call_cache()
        self.dispatcher.unsubscribe_all()

    @property
    def resumable(self):
        "i.e., can the plan in progress by rewound"
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

    def request_pause(self, defer=False):
        """
        Command the Run Engine to pause.

        This function is called by 'pause' Messages. It can also be called
        by other threads. It cannot be called on the main thread during a run,
        but it is called by SIGINT (i.e., Ctrl+C).

        If there current run has no checkpoint (via the 'clear_checkpoint'
        message), this will cause the run to abort.

        Parameters
        ----------
        defer : bool, optional
            If False, pause immediately before processing any new messages.
            If True, pause at the next checkpoint.
            False by default.
        """
        if defer:
            self._deferred_pause_requested = True
            print("Deferred pause acknowledged. Continuing to checkpoint.")
            return

        # We are pausing. Cancel any deferred pause previously requested.
        self._deferred_pause_requested = False
        self._interrupted = True
        print("Pausing...")
        self.state = 'paused'
        self._record_interruption('pause')
        if not self.resumable:
            # cannot resume, so we cannot pause.  Abort the scan
            print("No checkpoint; cannot pause. Aborting...")
            self._exception = FailedPause()
            self._task.cancel()
            for task in self._failed_status_tasks:
                task.cancel()
            return
        # stop accepting new tasks in the event loop (existing tasks will
        # still be processed)
        self.loop.stop()
        # Remove any monitoring callbacks, but keep refs in
        # self._monitor_params to re-instate them later.
        for obj, (cb, kwargs) in list(self._monitor_params.items()):
            obj.clear_sub(cb)
        # During pause, all motors should be stopped. Call stop() on every
        # object we ever set().
        self._stop_movable_objects()
        # Notify Devices of the pause in case they want to clean up.
        for obj in self._objs_seen:
            if hasattr(obj, 'pause'):
                obj.pause()

    def _record_interruption(self, content):
        """
        Emit an event in the 'interruptions' event stream.

        If we are not inside a run or if self.record_interruptions is False,
        nothing is done.
        """
        if self._interruptions_desc_uid is not None:
            # We are inside a run and self.record_interruptions is True.
            doc = dict(descriptor=self._interruptions_desc_uid,
                       time=ttime.time(), uid=new_uid(),
                       seq_num=next(self._interruptions_counter),
                       data={'interruption': content},
                       timestamps={'interruption': ttime.time()})
            jsonschema.validate(doc, schemas[DocumentNames.event])
            self._lossless_dispatcher.process(DocumentNames.event, doc)
            # Unlike the RunEngine.emit coroutine, here both dispatchers
            # get a lossless stream of all documents.
            self.dispatcher.process(DocumentNames.event, doc)

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

        futs = []
        for sup in self.suspenders:
            f_lst, justification = sup.get_futures()
            print("At least one suspender is tripped. Execution will begin "
                  "when all suspenders are ready.")
            print("Suspending....To get prompt hit Ctrl-C twice to pause.")
            if f_lst:
                futs.extend(f_lst)
                print(justification)

        self._clear_call_cache()
        self._clear_run_cache()  # paranoia, in case of previous bad exit
        self.state = 'running'

        for name, funcs in normalize_subs_input(subs).items():
            for func in funcs:
                self._temp_callback_ids.add(self.subscribe(name, func))

        self._plan = plan
        self._metadata_per_call.update(metadata_kw)

        gen = ensure_generator(plan)

        self._plan_stack.append(gen)
        self._response_stack.append(None)
        if futs:
            self._plan_stack.append(single_gen(Msg('wait_for', None, futs)))
            self._response_stack.append(None)

        # Intercept ^C.
        with SignalHandler(signal.SIGINT, self.log) as self._sigint_handler:
            self._task = self.loop.create_task(self._run())
            self.loop.run_forever()

            if self._task.done() and not self._task.cancelled():
                exc = self._task.exception()
                if exc is not None:
                    raise exc
            if raise_if_interrupted and self._interrupted:
                raise RunEngineInterrupted("RunEngine was interrupted.")

        return self._run_start_uids

    def resume(self):
        """Resume a paused plan from the last checkpoint.

        Returns
        -------
        uids : list
            list of Header uids (a.k.a RunStart uids) of run(s)
        """
        # The state machine does not capture the whole picture.
        if not self.state.is_paused:
            raise RuntimeError("The RunEngine is the {0} state. You can only "
                               "resume for the paused state."
                               "".format(self.state))

        self._interrupted = False
        self._record_interruption('resume')
        new_plan = self._rewind()
        self._plan_stack.append(new_plan)
        self._response_stack.append(None)
        # Re-instate monitoring callbacks.
        for obj, (cb, kwargs) in self._monitor_params.items():
            obj.subscribe(cb, **kwargs)
        # Notify Devices of the resume in case they want to clean up.
        for obj in self._objs_seen:
            if hasattr(obj, 'resume'):
                obj.resume()
        self._resume_event_loop()
        return self._run_start_uids

    def _rewind(self):
        '''Clean up in preparation for resuming from a pause or suspension.

        Returns
        -------
        new_plan : generator
             A new plan made from the messages in the message cache

        '''
        new_plan = ensure_generator(list(self._msg_cache))
        self._msg_cache = deque()
        self._sequence_counters.clear()
        self._sequence_counters.update(self._teed_sequence_counters)
        # This is needed to 'cancel' an open bundling (e.g. create) if
        # the pause happens after a 'checkpoint', after a 'create', but before
        # the paired 'save'.
        self._bundling = False
        return new_plan

    def _resume_event_loop(self):
        # may be called by 'resume' or 'abort'
        self.state = 'running'
        self._last_sigint_time = None
        self._num_sigints_processed = 0
        # Intercept ^C
        with SignalHandler(signal.SIGINT, self.log) as self._sigint_handler:
            if self._task.done():
                return
            self.loop.run_forever()
            if self._task.done() and not self._task.cancelled():
                exc = self._task.exception()
                if exc is not None:
                    raise exc

    def install_suspender(self, suspender):
        """
        Install a 'suspender', which can suspend and resume execution.

        Parameters
        ----------
        suspender : `bluesky.suspenders.SuspenderBase`

        See Also
        --------
        `RunEngine.remove_suspender`
        """
        self._suspenders.add(suspender)
        suspender.install(self)

    def remove_suspender(self, suspender):
        """
        Uninstall a suspender.

        Parameters
        ----------
        suspender : `bluesky.suspenders.SuspenderBase`

        See Also
        --------
        `RunEngine.install_suspender`
        """
        if suspender in self._suspenders:
            suspender.remove()
        self._suspenders.discard(suspender)

    def request_suspend(self, fut, *, pre_plan=None, post_plan=None,
                        justification=None):
        """
        Request that the run suspend itself until the future is finished.

        The two plans will be run before and after waiting for the future.
        This enable doing things like opening and closing shutters and
        resetting cameras around a suspend.

        Parameters
        ----------
        fut : asyncio.Future
        pre_plan : iterable, optional
            Plan to execute just before suspending
        post_plan : iterable, optional
            Plan to execute just before resuming
        justification : str, optional
            explanation of why the suspension has been requested
        """
        if not self.resumable:
            print("No checkpoint; cannot suspend. Aborting...")
            self._exception = FailedPause()
        else:
            print("Suspending....To get prompt hit Ctrl-C twice to pause.")
            if justification is not None:
                print("Justification for this suspension:\n%s" % justification)
            self._record_interruption('suspend')
            # Stash a copy in a local var to re-instating the monitors.
            for obj, (cb, kwargs) in list(self._monitor_params.items()):
                obj.clear_sub(cb)
            # During suspend, all motors should be stopped. Call stop() on
            # every object we ever set().
            self._stop_movable_objects()
            # Notify Devices of the pause in case they want to clean up.
            for obj in self._objs_seen:
                if hasattr(obj, 'pause'):
                    obj.pause()

            # rewind to the last checkpoint
            new_plan = self._rewind()
            # queue up the cached messages
            self._plan_stack.append(new_plan)
            self._response_stack.append(None)
            # if there is a post plan add it between the wait
            # and the cached messages
            if post_plan is not None:
                self._plan_stack.append(ensure_generator(post_plan))
                self._response_stack.append(None)
            # add the wait on the future to the stack
            self._plan_stack.append(single_gen(Msg('wait_for', None, [fut, ])))
            self._response_stack.append(None)
            # if there is a pre plan add on top of the wait
            if pre_plan is not None:
                self._plan_stack.append(ensure_generator(pre_plan))
                self._response_stack.append(None)
            # The event loop is still running. The pre_plan will be processed,
            # and then the RunEngine will be hung up on processing the
            # 'wait_for' message until `fut` is set.

    def abort(self, reason=''):
        """
        Stop a running or paused plan and mark it as aborted.
        """
        if self.state.is_idle:
            raise TransitionError("RunEngine is already idle.")
        print("Aborting....")
        self._interrupted = True
        self._reason = reason
        self._exception = RequestAbort()
        self._task.cancel()
        for task in self._failed_status_tasks:
            task.cancel()
        if self.state == 'paused':
            self._resume_event_loop()

    def stop(self):
        """
        Stop a running or paused plan, but mark it as successful (not aborted).
        """
        if self.state.is_idle:
            raise TransitionError("RunEngine is already idle.")
        print("Stopping...")
        self._interrupted = True
        self._exception = RequestStop()
        if self.state == 'paused':
            self._resume_event_loop()

    def _stop_movable_objects(self):
        "Call obj.stop() for all objects we have moved. Log any exceptions."
        for obj in self._movable_objs_touched:
            try:
                stop = obj.stop
            except AttributeError:
                self.log.debug("No 'stop' method available on %r", obj)
            else:
                try:
                    stop()
                except Exception as exc:
                    self.log.error("Failed to stop %r. Error: %r", obj, exc)

    @asyncio.coroutine
    def _run(self):
        """Pull messages from the plan, process them, send results back.

        Upon exit, clean up.
        - Call stop() on all objects that were 'set' or 'kickoff'.
        - Try to collect any uncollected flyers.
        - Try to unstage any devices left staged by the plan.
        - Try to remove any monitoring subscriptions left on by the plan.
        - If interrupting the middle of a run, try to emit a RunStop document.
        """
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
                    yield from asyncio.sleep(0.0001, loop=self.loop)
                    # Send last response;
                    # get new message but don't process it yet.
                    try:
                        if self._exception is not None:
                            # throw the message at the current plan
                            try:
                                msg = self._plan_stack[-1].throw(
                                    self._exception)
                            except Exception as e:
                                # deal with the case where the top
                                # plan is a re-wind/suspender injected
                                # plan, all plans in stack get a
                                # chance to take a bite at this apple.

                                # if the exact same exception comes
                                # back up, then the plan did not
                                # handle it and the next plan gets to
                                # try.
                                if e is self._exception:
                                    self._plan_stack.pop()
                                    if len(self._plan_stack):
                                        continue
                                    else:
                                        raise
                        else:
                            resp = self._response_stack.pop()
                            msg = self._plan_stack[-1].send(resp)

                    except StopIteration:
                        self._plan_stack.pop()
                        if len(self._plan_stack):
                            continue
                        else:
                            raise
                    # If we are here one of the plans handled the exception
                    # and wants to do something with it
                    self._exception = None
                    if self.msg_hook is not None:
                        self.msg_hook(msg)
                    self._objs_seen.add(msg.obj)
                    if (self._msg_cache is not None and
                            msg.command not in self._UNCACHEABLE_COMMANDS):
                        # We have a checkpoint.
                        self._msg_cache.append(msg)
                    try:
                        coro = self._command_registry[msg.command]
                        response = yield from coro(msg)
                        self._response_stack.append(response)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        self._exception = e
                        continue
                except KeyboardInterrupt:
                    # This only happens if some external code captures SIGINT
                    # -- overriding the RunEngine -- and then raises instead
                    # of (properly) calling the RunEngine's handler.
                    # See https://github.com/NSLS-II/bluesky/pull/242
                    print("An unknown external library has improperly raised "
                          "KeyboardInterrupt. Intercepting and triggering "
                          "a hard pause instead.")
                    self.loop.call_soon(self.request_pause, False)
                    print(PAUSE_MSG)
        except (StopIteration, RequestStop):
            self._exit_status = 'success'
            # TODO Is the sleep here necessasry?
            yield from asyncio.sleep(0.001, loop=self.loop)
        except (FailedPause, RequestAbort, asyncio.CancelledError):
            self._exit_status = 'abort'
            # TODO Is the sleep here necessasry?
            yield from asyncio.sleep(0.001, loop=self.loop)
            self.log.error("Run aborted")
            self.log.error("%r", self._exception)
        except Exception as err:
            self._exit_status = 'fail'  # Exception raises during 'running'
            self._reason = str(err)
            self.log.error("Run aborted")
            self.log.error("%r", err)
            raise err
        finally:
            # call stop() on every movable object we ever set()
            self._stop_movable_objects()
            # Try to collect any flyers that were kicked off but not finished.
            # Some might not support partial collection. We swallow errors.
            for obj in list(self._uncollected):
                try:
                    yield from self._collect(Msg('collect', obj))
                except Exception as exc:
                    self.log.error("Failed to collect %r. Error: %r", obj, exc)
            # in case we were interrupted between 'stage' and 'unstage'
            for obj in list(self._staged):
                try:
                    obj.unstage()
                except Exception as exc:
                    self.log.error("Failed to unstage %r. Error: %r", obj, exc)
                self._staged.remove(obj)
            # Clear any uncleared monitoring callbacks.
            for obj, (cb, kwargs) in list(self._monitor_params.items()):
                try:
                    obj.clear_sub(cb)
                except Exception as exc:
                    self.log.error("Failed to stop monitoring %r. Error: %r",
                                   obj, exc)
                else:
                    del self._monitor_params[obj]
            sys.stdout.flush()
            # Emit RunStop if necessary.
            if self._run_is_open:
                try:
                    yield from self._close_run(Msg('close_run'))
                except Exception as exc:
                    self.log.error("Failed to close run %r. Error: %r",
                                   self._run_start_uid, exc)
                    # Exceptions from the callbacks should be re-raised.
                    # Close the loop first.
                    self._clear_run_cache()
                    for task in asyncio.Task.all_tasks(self.loop):
                        task.cancel()
                    self.loop.stop()
                    self.state = 'idle'
                    raise

            for task in asyncio.Task.all_tasks(self.loop):
                task.cancel()
            for p in self._plan_stack:
                try:
                    p.close()
                except RuntimeError as e:
                    print('The plan {!r} tried to yield a value on close.  '
                          'Please fix your plan.')
            self.loop.stop()
            self.state = 'idle'
        print('exited run loop')

    def _check_for_signals(self):
        # Check for pause requests from keyboard.
        if self.state.is_running:
            count = self._sigint_handler.count
            if count > self._num_sigints_processed:
                self._num_sigints_processed = count
                self.log.debug("RunEngine caught a new SIGINT")
                self._last_sigint_time = ttime.time()

                if count == 1:
                    # Ctrl-C once -> request a deferred pause
                    if not self._deferred_pause_requested:
                        self.loop.call_soon(self.request_pause, True)
                        print("A 'deferred pause' has been requested. The "
                              "RunEngine will pause at the next checkpoint. "
                              "To pause immediately, hit Ctrl+C again in the "
                              "next 10 seconds.")
                elif count > 1:
                    # - Ctrl-C twice within 10 seconds -> hard pause
                    # - Ctrl-C twice with 10 seconds and then again within
                    #       0.5 seconds -> abort
                    #
                    # We know we will either pause or abort, so we can
                    # hold up the scan now. Wait until 0.5 seconds pass or
                    # we catch a second SIGINT, whichever happens first.
                    self.log.debug("RunEngine detected a second SIGINT -- "
                                   "waiting 0.5 seconds for a third.")
                    for i in range(10):
                        ttime.sleep(0.05)
                        if self._sigint_handler.count > 2:
                            self.log.debug("RunEngine detected a third "
                                            "SIGINT -- aborting.")
                            self.loop.call_soon(self.abort, "SIGINT (Ctrl+C)")
                            break
                    else:
                        self.log.debug("RunEngine detected two SIGINTs. "
                                       "A hard pause will be requested.")
                        self.loop.call_soon(self.request_pause, False)
                        print(PAUSE_MSG)
            else:
                # No new SIGINTs to process.
                if self._num_sigints_processed > 0:
                    if ttime.time() - self._last_sigint_time > 10:
                        self.log.debug("It has been 10 seconds since the "
                                       "last SIGINT. Resetting SIGINT "
                                       "handler.")
                        # It's been 10 seconds since the last SIGINT. Reset.
                        self._num_sigints_processed = 0
                        self._sigint_handler.count = 0
                        self._sigint_handler.interrupted = False
                        self._last_sigint_time = None

        self.loop.call_later(0.1, self._check_for_signals)

    @asyncio.coroutine
    def _wait_for(self, msg):
        """Instruct the RunEngine to wait until msg.obj has completed. Better
        yet, see the docstring for ``asyncio.wait`` for what msg.obj should
        be...

        TODO: Get someone who knows how this works to check this note, since
        it is almost assuredly total bs

        Expected message object is:

            Msg('wait_for', None, futures, **kwargs)

        Where ``obj`` and **kwargs are the position and keyword-only arguments
        for ``asyncio.await``
        """
        futs, = msg.args
        yield from asyncio.wait(futs, loop=self.loop, **msg.kwargs)

    @asyncio.coroutine
    def _open_run(self, msg):
        """Instruct the RunEngine to start a new "run"

        Expected message object is:

            Msg('open_run', None, **kwargs)

        where **kwargs are any additional metadata that should go into
        the RunStart document
        """
        if self._run_is_open:
            raise IllegalMessageSequence("A 'close_run' message was not "
                                         "received before the 'open_run' "
                                         "message")
        self._clear_run_cache()
        self._run_start_uid = new_uid()
        self._run_start_uids.append(self._run_start_uid)
        self.log.debug("Starting new with uid %r", self._run_start_uid)

        # Increment scan ID
        scan_id = self.md.get('scan_id', 0) + 1
        self.md['scan_id'] = scan_id

        # For metadata below, info about plan passed to self.__call__ for.
        plan_type = type(self._plan).__name__
        plan_name = getattr(self._plan, '__name__', '')

        # Combine metadata, in order of decreasing precedence:
        md = ChainMap(self._metadata_per_call,  # from kwargs to self.__call__
                      msg.kwargs,  # from 'open_run' Msg
                      {'plan_type': plan_type,  # computed from self._plan
                       'plan_name': plan_name},
                      self.md)  # stateful, persistent metadata
        # The metadata is final. Validate it now, at the last moment.
        # Use copy for some reasonable (admittedly not total) protection
        # against users mutating the md with their validator.
        self.md_validator(dict(md))

        doc = dict(uid=self._run_start_uid, time=ttime.time(), **md)
        yield from self.emit(DocumentNames.start, doc)
        self.log.debug("Emitted RunStart (uid=%r)", doc['uid'])
        yield from self._reset_checkpoint_state()

        # Emit an Event Descriptor for recording any interruptions as Events.
        if self.record_interruptions:
            self._interruptions_desc_uid = new_uid()
            dk = {'dtype': 'string', 'shape': None, 'source': 'RunEngine'}
            interruptions_desc = dict(time=ttime.time(),
                                      uid=self._interruptions_desc_uid,
                                      name='interruptions',
                                      data_keys={'interruption': dk},
                                      run_start=self._run_start_uid)
            yield from self.emit(DocumentNames.descriptor, interruptions_desc)

    @asyncio.coroutine
    def _close_run(self, msg):
        """Instruct the RunEngine to write the RunStop document

        Expected message object is:

            Msg('close_run')
        """
        if not self._run_is_open:
            raise IllegalMessageSequence("A 'close_run' message was received "
                                         "but there is no run open. If this "
                                         "occurred after a pause/resume, add "
                                         "a 'checkpoint' message after the "
                                         "'close_run' message.")
        self.log.debug("Stopping run %r", self._run_start_uid)
        # Clear any uncleared monitoring callbacks.
        for obj, (cb, kwargs) in list(self._monitor_params.items()):
            obj.clear_sub(cb)
            del self._monitor_params[obj]
        doc = dict(run_start=self._run_start_uid,
                   time=ttime.time(), uid=new_uid(),
                   exit_status=self._exit_status,
                   reason=self._reason)
        self._clear_run_cache()
        yield from self.emit(DocumentNames.stop, doc)
        self.log.debug("Emitted RunStop (uid=%r)", doc['uid'])
        yield from self._reset_checkpoint_state()

    @asyncio.coroutine
    def _create(self, msg):
        """Trigger the run engine to start bundling future obj.read() calls for
         an Event document

        Expected message object is:

            Msg('create', None, name='primary')
            Msg('create')

        Note that the `name` kwarg will be the 'name' field of the resulting
        descriptor. So descriptor['name'] = msg.kwargs['name'].

        Also note that changing the 'name' of the Event will create a new
        Descriptor document.
        """
        if not self._run_is_open:
            raise IllegalMessageSequence("Cannot bundle readings without "
                                         "an open run. That is, 'create' must "
                                         "be preceded by 'open_run'.")
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
        """
        Add a reading to the open event bundle.

        Expected message object is:

            Msg('read', obj)
        """
        obj = msg.obj
        # actually _read_ the object
        ret = obj.read(*msg.args, **msg.kwargs)

        if self._bundling:
            # if the object is not in the _describe_cache, cache it
            if obj not in self._describe_cache:
                # Validate that there is no data key name collision.
                data_keys = obj.describe()
                self._describe_cache[obj] = data_keys
                self._config_desc_cache[obj] = obj.describe_configuration()
                self._cache_config(obj)

            # check that current read collides with nothing else in
            # current event
            cur_keys = set(self._describe_cache[obj].keys())
            for read_obj in self._objs_read:
                # that is, field names
                known_keys = self._describe_cache[read_obj].keys()
                if set(known_keys) & cur_keys:
                    raise ValueError("Data keys (field names) from {0!r} "
                                     "collide with those from {1!r}"
                                     "".format(obj, read_obj))

            # add this object to the cache of things we have read
            self._objs_read.append(obj)

            # stash the results
            self._read_cache.append(ret)

        return ret

    def _cache_config(self, obj):
        "Read the object's configuration and cache it."
        config_values = {}
        config_ts = {}
        for key, val in obj.read_configuration().items():
            config_values[key] = sanitize_np(val['value'])
            config_ts[key] = val['timestamp']
        self._config_values_cache[obj] = config_values
        self._config_ts_cache[obj] = config_ts

    @asyncio.coroutine
    def _monitor(self, msg):
        """
        Monitor a signal. Emit event documents asynchronously.

        A descriptor document is emitted immediately. Then, a closure is
        defined that emits Event documents associated with that descriptor
        from a separate thread. This process is not related to the main
        bundling process (create/read/save).

        Expected message object is:

            Msg('monitor', obj, **kwargs)
            Msg('monitor', obj, name='event-stream-name', **kwargs)

        where kwargs are passed through to ``obj.subscribe()``
        """
        obj = msg.obj
        if msg.args:
            raise ValueError("The 'monitor' Msg does not accept positional "
                             "arguments.")
        name = msg.kwargs.get('name')
        if not self._run_is_open:
            raise IllegalMessageSequence("A 'monitor' message was sent but no "
                                         "run is open.")
        if obj in self._monitor_params:
            raise IllegalMessageSequence("A 'monitor' message was sent for {}"
                                         "which is already monitored".format(
                                             obj))
        descriptor_uid = new_uid()
        data_keys = obj.describe()
        config = obj.read_configuration()
        for key, val in list(config.items()):
            val['value'] = sanitize_np(val['value'])
        object_keys = {obj.name: list(data_keys)}
        desc_doc = dict(run_start=self._run_start_uid, time=ttime.time(),
                        data_keys=data_keys, uid=descriptor_uid,
                        configuration=config, name=name,
                        object_keys=object_keys)
        self.log.debug("Emitted Event Descriptor with name %r containing "
                       "data keys %r (uid=%r)", name, data_keys.keys(),
                       descriptor_uid)
        seq_num_counter = count()

        def emit_event(*args, **kwargs):
            # Ignore the inputs. Use this call as a signal to call read on the
            # object, a crude way to be sure we get all the info we need.
            data, timestamps = _rearrange_into_parallel_dicts(obj.read())
            doc = dict(descriptor=descriptor_uid,
                       time=ttime.time(), data=data, timestamps=timestamps,
                       seq_num=next(seq_num_counter), uid=new_uid())
            jsonschema.validate(doc, schemas[DocumentNames.event])
            self._lossless_dispatcher.process(DocumentNames.event, doc)
            # Unlike the RunEngine.emit coroutine, here both dispatchers
            # get a lossless stream of all documents. Monitors are already
            # "lossy"; we will not mix in our own lossiness here. If
            # monitors are generating too much data, they should be
            # implemented as flyers.
            self.dispatcher.process(DocumentNames.event, doc)

        self._monitor_params[obj] = emit_event, msg.kwargs
        obj.subscribe(emit_event, **msg.kwargs)
        yield from self.emit(DocumentNames.descriptor, desc_doc)
        yield from self._reset_checkpoint_state()

    @asyncio.coroutine
    def _unmonitor(self, msg):
        """
        Stop monitoring; i.e., remove the callback emitting event documents.

        Expected message object is:

            Msg('unmonitor', obj)
        """
        obj = msg.obj
        if obj not in self._monitor_params:
            raise IllegalMessageSequence("Cannot 'unmonitor' %r; it is not "
                                         "being monitored." % obj)
        cb, kwargs = self._monitor_params[obj]
        obj.clear_sub(cb)
        del self._monitor_params[obj]
        yield from self._reset_checkpoint_state()

    @asyncio.coroutine
    def _save(self, msg):
        """Save the event that is currently being bundled

        Expected message object is:

            Msg('save')
        """
        if not self._bundling:
            raise IllegalMessageSequence("A 'create' message must be sent, to "
                                         "open an event bundle, before that "
                                         "bundle can be saved with 'save'.")
        if not self._run_is_open:
            # sanity check -- this should be caught by 'create' which makes
            # this code path impossible
            raise IllegalMessageSequence("A 'save' message was sent but no "
                                         "run is open.")
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
            self.log.debug("Emitted Event Descriptor with name %r containing "
                           "data keys %r (uid=%r)", self._bundle_name,
                           data_keys.keys(), descriptor_uid)
            self._descriptors[(self._bundle_name, objs_read)] = descriptor_uid
        else:
            descriptor_uid = self._descriptors[(self._bundle_name, objs_read)]
        # This is a separate check because it can be reset on resume.
        if objs_read not in self._sequence_counters:
            counter = count(1)
            counter_copy1, counter_copy2 = tee(counter)
            self._sequence_counters[objs_read] = counter_copy1
            self._teed_sequence_counters[objs_read] = counter_copy2
        self._bundling = False
        self._bundle_name = None

        # Events
        seq_num = next(self._sequence_counters[objs_read])
        event_uid = new_uid()
        # Merge list of readings into single dict.
        readings = {k: v for d in self._read_cache for k, v in d.items()}
        for key in readings:
            readings[key]['value'] = sanitize_np(readings[key]['value'])
        data, timestamps = _rearrange_into_parallel_dicts(readings)
        doc = dict(descriptor=descriptor_uid,
                   time=ttime.time(), data=data, timestamps=timestamps,
                   seq_num=seq_num, uid=event_uid)
        yield from self.emit(DocumentNames.event, doc)
        self.log.debug("Emitted Event with data keys %r (uid=%r)", data.keys(),
                       event_uid)

    @asyncio.coroutine
    def _kickoff(self, msg):
        """Start a flyscan object

        Parameters
        ----------
        msg : Msg

        Special kwargs for the 'Msg' object in this function:
        group : str
            The blocking group to this flyer to

        Expected message object is:

        If `flyer_object` has a `kickoff` function that takes no arguments:

            Msg('kickoff', flyer_object)
            Msg('kickoff', flyer_object, group=<name>)

        If `flyer_object` has a `kickoff` function that takes
        `(start, stop, steps)` as its function arguments:

            Msg('kickoff', flyer_object, start, stop, step)
            Msg('kickoff', flyer_object, start, stop, step, group=<name>)
        """
        if not self._run_is_open:
            raise IllegalMessageSequence("A 'kickoff' message was sent but no "
                                         "run is open.")
        _, obj, args, kwargs = msg
        self._uncollected.add(obj)
        group = msg.kwargs.pop('group', None)

        ret = obj.kickoff(*msg.args, **msg.kwargs)

        if group:
            p_event = asyncio.Event(loop=self.loop)

            def done_callback():
                if not ret.success:
                    task = self.loop.call_soon_threadsafe(self._failed_status,
                                                          ret)
                    self._failed_status_tasks.append(task)
                self.loop.call_soon_threadsafe(p_event.set)

            ret.finished_cb = done_callback
            self._groups[group].add(p_event.wait())

        return ret

    @asyncio.coroutine
    def _complete(self, msg):
        """
        Tell a flyer, 'stop collecting, whenver you are ready'.

        The flyer returns a status object. Some flyers respond to this
        command by stopping collection and returning a finished status
        object immedately. Other flyers finish their given course and
        finish whenever they finish, irrespective of when this command is
        issued.

        Expected message object is:

            Msg('complete', flyer, group=<GROUP>)

        where <GROUP> is a hashable identifier.
        """
        group = msg.kwargs.pop('group', None)
        ret = msg.obj.complete(*msg.args, **msg.kwargs)

        if group is not None:
            p_event = asyncio.Event(loop=self.loop)

            def done_callback():
                if not ret.success:
                    task = self.loop.call_soon_threadsafe(self._failed_status,
                                                          ret)
                    self._failed_status_tasks.append(task)
                self.loop.call_soon_threadsafe(p_event.set)

            ret.finished_cb = done_callback
            self._groups[group].add(p_event.wait())

        return ret

    @asyncio.coroutine
    def _collect(self, msg):
        """
        Collect data cached by a flyer and emit descriptor and event documents.

        Expect message object is

            Msg('collect', obj)
            Msg('collect', obj, stream=True)
        """
        obj = msg.obj
        if obj not in self._uncollected:
            raise IllegalMessageSequence("The flyer %r was never kicked off "
                                         "(or already collected)." % obj)
        if not self._run_is_open:
            # sanity check -- 'kickoff' should catch this and make this
            # code path impossible
            raise IllegalMessageSequence("A 'collect' message was sent but no "
                                         "run is open.")
        self._uncollected.remove(obj)

        named_data_keys = obj.describe_collect()
        # e.g., {name_for_desc1: data_keys_for_desc1,
        #        name for_desc2: data_keys_for_desc2, ...}
        bulk_data = {}
        local_descriptors = {}  # hashed on obj_read, not (name, objs_read)
        for stream_name, data_keys in named_data_keys.items():
            objs_read = frozenset(data_keys)
            if (stream_name, objs_read) not in self._descriptors:
                # We don't not have an Event Descriptor for this set.
                descriptor_uid = new_uid()
                doc = dict(run_start=self._run_start_uid, time=ttime.time(),
                           data_keys=data_keys, uid=descriptor_uid,
                           name=stream_name)
                yield from self.emit(DocumentNames.descriptor, doc)
                self.log.debug("Emitted Event Descriptor with name %r "
                               "containing data keys %r (uid=%r)", stream_name,
                               data_keys.keys(), descriptor_uid)
                self._descriptors[(stream_name, objs_read)] = descriptor_uid
                self._sequence_counters[objs_read] = count(1)
            else:
                descriptor_uid = self._descriptors[(stream_name, objs_read)]

            local_descriptors[objs_read] = descriptor_uid

            bulk_data[descriptor_uid] = []

        # If stream is True, run 'event' subscription per document.
        # If stream is False, run 'bulk_events' subscription once.
        stream = msg.kwargs.pop('stream', False)
        for ev in obj.collect():
            objs_read = frozenset(ev['data'])
            seq_num = next(self._sequence_counters[objs_read])
            descriptor_uid = local_descriptors[objs_read]
            event_uid = new_uid()

            reading = ev['data']
            for key in ev['data']:
                reading[key] = sanitize_np(reading[key])
            ev['data'] = reading
            ev['descriptor'] = descriptor_uid
            ev['seq_num'] = seq_num
            ev['uid'] = event_uid

            if stream:
                self.log.debug("Emitted Event with data keys %r (uid=%r)",
                               ev['data'].keys(), ev['uid'])
                yield from self.emit(DocumentNames.event, ev)
            else:
                bulk_data[descriptor_uid].append(ev)

        if not stream:
            yield from self.emit(DocumentNames.bulk_events, bulk_data)
            self.log.debug("Emitted bulk events for descriptors with uids "
                           "%r", bulk_data.keys())

    @asyncio.coroutine
    def _null(self, msg):
        """
        A no-op message, mainly for debugging and testing.
        """
        pass

    @asyncio.coroutine
    def _set(self, msg):
        """
        Set a device and cache the returned status object.

        Also, note that the device has been touched so it can be stopped upon
        exit.

        Expected messgage object is

            Msg('set', obj, *args, **kwargs)

        where arguments are passed through to `obj.set(*args, **kwargs)`.
        """
        group = msg.kwargs.pop('group', None)
        self._movable_objs_touched.add(msg.obj)
        ret = msg.obj.set(*msg.args, **msg.kwargs)
        if group:
            p_event = asyncio.Event(loop=self.loop)

            def done_callback():

                self.log.debug("The object %r reports set is done "
                               "with status %r",
                               msg.obj, ret.success)

                if not ret.success:
                    task = self.loop.call_soon_threadsafe(self._failed_status,
                                                          ret)
                    self._failed_status_tasks.append(task)
                self.loop.call_soon_threadsafe(p_event.set)

            ret.finished_cb = done_callback
            self._groups[group].add(p_event.wait())

        return ret

    @asyncio.coroutine
    def _trigger(self, msg):
        """
        Trigger a device and cache the returned status object.

        Expected message object is:

            Msg('trigger', obj)
        """
        group = msg.kwargs.pop('group', None)
        ret = msg.obj.trigger(*msg.args, **msg.kwargs)

        if group:
            p_event = asyncio.Event(loop=self.loop)

            def done_callback():
                self.log.debug("The object %r reports trigger is "
                               "done with status %r.",
                               msg.obj, ret.success)

                if not ret.success:
                    task = self.loop.call_soon_threadsafe(self._failed_status,
                                                          ret)
                    self._failed_status_tasks.append(task)

                self.loop.call_soon_threadsafe(p_event.set)

            ret.finished_cb = done_callback
            self._groups[group].add(p_event.wait())

        return ret

    @asyncio.coroutine
    def _wait(self, msg):
        """Block progress until every object that was triggered or set
        with the keyword argument `group=<GROUP>` is done.

        Expected message object is:

            Msg('wait', group=<GROUP>)

        where ``<GROUP>`` is any hashable key.
        """
        if msg.args:
            group, = msg.args
        else:
            group = msg.kwargs['group']
        futs = list(self._groups.pop(group, []))
        if futs:
            yield from self._wait_for(Msg('wait_for', None, futs))

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
        """Sleep the event loop

        Expected message object is:

            Msg('sleep', None, sleep_time)

        where `sleep_time` is in seconds
        """
        yield from asyncio.sleep(*msg.args, loop=self.loop)

    @asyncio.coroutine
    def _pause(self, msg):
        """Request the run engine to pause

        Expected message object is:

            Msg('pause', defer=False, name=None, callback=None)

        See RunEngine.request_pause() docstring for explanation of the three
        keyword arugments in the `Msg` signature
        """
        self.request_pause(*msg.args, **msg.kwargs)

    @asyncio.coroutine
    def _checkpoint(self, msg):
        """Instruct the RunEngine to create a checkpoint so that we can rewind
        to this point if necessary

        Expected message object is:

            Msg('checkpoint')
        """
        if self._bundling:
            raise IllegalMessageSequence("Cannot 'checkpoint' after 'create' "
                                         "and before 'save'. Aborting!")

        yield from self._reset_checkpoint_state()

        if self._deferred_pause_requested:
            # We are at a checkpoint; we are done deferring the pause.
            # Give the _check_for_signals courtine time to look for
            # additional SIGINTs that would trigger an abort.
            yield from asyncio.sleep(0.5, loop=self.loop)
            self.request_pause(defer=False)

    @asyncio.coroutine
    def _reset_checkpoint_state(self):
        if self._msg_cache is None:
            return

        self._msg_cache = deque()

        # Keep a safe separate copy of the sequence counters to use if we
        # rewind and retake some data points.
        for key, counter in list(self._sequence_counters.items()):
            counter_copy1, counter_copy2 = tee(counter)
            self._sequence_counters[key] = counter_copy1
            self._teed_sequence_counters[key] = counter_copy2

    @asyncio.coroutine
    def _clear_checkpoint(self, msg):
        """Clear a set checkpoint

        Expected message object is:

            Msg('clear_checkpoint')
        """
        # clear message cache
        self._msg_cache = None
        # clear stashed
        self._teed_sequence_counters.clear()

    @asyncio.coroutine
    def _configure(self, msg):
        """Configure an object

        Expected message object is:

            Msg('configure', object, *args, **kwargs)

        which results in this call:

            object.configure(*args, **kwargs)
        """
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
        """Instruct the RunEngine to stage the object

        Expected message object is:

            Msg('stage', object)
        """
        _, obj, args, kwargs = msg
        # If an object has no 'stage' method, assume there is nothing to do.
        if not hasattr(obj, 'stage'):
            return []
        result = obj.stage()
        self._staged.add(obj)  # add first in case of failure below
        yield from self._reset_checkpoint_state()
        return result

    @asyncio.coroutine
    def _unstage(self, msg):
        """Instruct the RunEngine to unstage the object

        Expected message object is:

            Msg('unstage', object)
        """
        _, obj, args, kwargs = msg
        # If an object has no 'unstage' method, assume there is nothing to do.
        if not hasattr(obj, 'unstage'):
            return []
        result = obj.unstage()
        # use `discard()` to ignore objects that are not in the staged set.
        self._staged.discard(obj)
        yield from self._reset_checkpoint_state()
        return result

    @asyncio.coroutine
    def _subscribe(self, msg):
        """
        Add a subscription after the run has started.

        This, like subscriptions passed to __call__, will be removed at the
        end by the RunEngine.

        Expected message object is:

            Msg('subscribe', None, document_name, callback_function)

        where `document_name` is one of:

            {'start', 'descriptor', 'event', 'stop', 'all'}

        and `callback_function` is expected to have a signature of:

            ``f(name, document)``

            where name is one of the ``document_name`` options and ``document``
            is one of the document dictionaries in the event model.

        See the docstring of bluesky.run_engine.Dispatcher.subscribe() for more
        information.
        """
        self.log.debug("Adding subsription %r", msg)
        _, obj, args, kwargs = msg
        token = self.subscribe(*args, **kwargs)
        self._temp_callback_ids.add(token)
        yield from self._reset_checkpoint_state()
        return token

    @asyncio.coroutine
    def _unsubscribe(self, msg):
        """
        Remove a subscription during a call -- useful for a multi-run call
        where subscriptions are wanted for some runs but not others.

        Expected message object is:

            Msg('unsubscribe', None, TOKEN)
            Msg('unsubscribe', token=TOKEN)

        where ``TOKEN`` is the return value from ``RunEngine._subscribe()``
        """
        self.log.debug("Removing subscription %r", msg)
        _, obj, args, kwargs = msg
        try:
            token = kwargs['token']
        except KeyError:
            token, = args
        self.unsubscribe(token)
        self._temp_callback_ids.remove(token)
        yield from self._reset_checkpoint_state()

    @asyncio.coroutine
    def _input(self, msg):
        """
        Process a 'input' Msg. Excpected Msg:

            Msg('input', None)
            Msg('input', None, prompt='>')  # customize prompt
        """
        prompt = msg.kwargs.get('prompt', '')
        async_input = AsyncInput(self.loop)
        async_input = functools.partial(async_input, end='', flush=True)
        return (yield from async_input(prompt))

    @asyncio.coroutine
    def emit(self, name, doc):
        "Process blocking callbacks and schedule non-blocking callbacks."
        jsonschema.validate(doc, schemas[name])
        self._lossless_dispatcher.process(name, doc)
        if name != DocumentNames.event:
            self.dispatcher.process(name, doc)
        else:
            start_time = self.loop.time()
            dummy = expiring_function(self.dispatcher.process, self.loop, name,
                                      doc)
            self.loop.run_in_executor(None, dummy, start_time,
                                      self.event_timeout)


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


def _rearrange_into_parallel_dicts(readings):
    data = {}
    timestamps = {}
    for key, payload in readings.items():
        data[key] = payload['value']
        timestamps[key] = payload['timestamp']
    return data, timestamps


class RequestAbort(Exception):
    pass


class RequestStop(Exception):
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

Pro Tip: Next time, if you want to abort, tap Ctrl+C three times quickly.
"""


def _default_md_validator(md):
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
