import asyncio
from datetime import datetime
import sys
import logging
from warnings import warn
from inspect import Parameter, Signature
from itertools import count
from collections import deque, defaultdict, ChainMap
from enum import Enum
import functools
import inspect
from contextlib import ExitStack
import threading
import weakref
from .bundlers import RunBundler

import concurrent

from event_model import DocumentNames, schema_validators
from super_state_machine.machines import StateMachine
from super_state_machine.extras import PropertyMachine
from super_state_machine.errors import TransitionError

try:
    from asyncio import current_task
except ImportError:
    # handle py < 3,7
    from asyncio.tasks import Task
    current_task = Task.current_task
    del Task

from .utils import (
    AsyncInput,
    CallbackRegistry,
    FailedPause,
    FailedStatus,
    IllegalBundlingSequence,
    IllegalMessageSequence,
    IllegalRunSequence,
    InvalidCommand,
    Msg,
    NoReplayAllowed,
    PlanHalt,
    RequestAbort,
    RequestStop,
    RunEngineInterrupted,
    SigintHandler,
    default_during_task,
    ensure_generator,
    normalize_subs_input,
    single_gen,
)


class _RunEnginePanic(Exception):
    ...


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

        PAUSING = 'pausing'
        PAUSED = 'paused'

        HALTING = 'halting'
        STOPPING = 'stopping'
        ABORTING = 'aborting'

        SUSPENDING = 'suspending'

        PANICKED = 'panicked'

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
            'idle': ['running', 'panicked'],
            'running': ['idle', 'pausing', 'halting', 'stopping',
                        'aborting', 'suspending', 'panicked'],
            'pausing': ['paused', 'idle', 'halting', 'aborting', 'panicked'],
            'suspending': ['running', 'halting', 'aborting', 'panicked'],
            'paused': ['idle', 'running', 'halting', 'stopping', 'aborting',
                       'panicked'],
            'halting': ['idle', 'panicked'],
            'stopping': ['idle', 'panicked'],
            'aborting': ['idle', 'panicked'],
            'panicked': []
        }
        named_checkers = [
            ('can_pause', 'paused'),
        ]


class LoggingPropertyMachine(PropertyMachine):
    """expects object to have a `log` attribute
    and a `state_hook` attribute that is ``None`` or a callable with signature
    ``f(value, old_value)``"""
    def __init__(self, machine_type):
        super().__init__(machine_type)

    def __set__(self, obj, value):
        own = type(obj)
        old_value = self.__get__(obj, own)
        with obj._state_lock:
            super().__set__(obj, value)
        value = self.__get__(obj, own)
        obj.log.info("Change state on %r from %r -> %r",
                     obj, old_value, value)
        if obj.state_hook is not None:
            obj.state_hook(value, old_value)

    def __get__(self, instance, owner):
        if instance is None:
            return super().__get__(instance, owner)
        with instance._state_lock:
            return super().__get__(instance, owner)


# See RunEngine.__call__.
_call_sig = Signature(
    [Parameter('self', Parameter.POSITIONAL_ONLY),
     Parameter('plan', Parameter.POSITIONAL_ONLY),
     Parameter('subs', Parameter.POSITIONAL_ONLY, default=None),
     Parameter('metadata_kw', Parameter.VAR_KEYWORD)])


def default_scan_id_source(md):
    return md.get('scan_id', 0) + 1


def _state_locked(func):
    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        with self._state_lock:
            return func(self, *args, **kwargs)

    return inner


def _extract_run_key(msg):
    return msg.run or _extract_run_key.__default_run
_extract_run_key.__default_run = object()


class RunEngine:
    """The Run Engine execute messages and emits Documents.

    Parameters
    ----------
    md : dict-like, optional
        The default is a standard Python dictionary, but fancier
        objects can be used to store long-term history and persist
        it between sessions. The standard configuration
        instantiates a Run Engine with historydict.HistoryDict, a
        simple interface to a sqlite file. Any object supporting
        `__getitem__`, `__setitem__`, and `clear` will work.

    loop : asyncio event loop
        e.g., ``asyncio.get_event_loop()`` or ``asyncio.new_event_loop()``

    preprocessors : list, optional
        Generator functions that take in a plan (generator instance) and
        modify its messages on the way out. Suitable examples include
        the functions in the module ``bluesky.plans`` with names ending in
        'wrapper'.  Functions are composed in order: the preprocessors
        ``[f, g]`` are applied like ``f(g(plan))``.

    context_managers : list, optional
        Context managers that will be entered when we run a plan. The context
        managers will be composed in order, much like the preprocessors. If
        this argument is omitted, we will use a user-oriented handler for
        SIGINT. The elements of this list will be passed this ``RunEngine``
        instance as their only argument. You may pass an empty list if you
        would like a ``RunEngine`` with no signal handling and no context
        managers.

    md_validator : callable, optional
        a function that raises and prevents starting a run if it deems
        the metadata to be invalid or incomplete
        Expected signature: f(md)
        Function should raise if md is invalid. What that means is
        completely up to the user. The function's return value is
        ignored.

    scan_id_source : callable, optional
        a function that will be used to calculate scan_id. Default is to
        increment scan_id by 1 each time. However you could pass in a
        customized function to get a scan_id from any source.
        Expected signature: f(md)
        Expected return: updated scan_id value

    during_task : callable, optional
        Function to be run to block the main thread during `RE.__call__`

        The required signature is ::

              def blocking_func(ev: Threading.Event) -> None:
                  "Returns when ev is set"

        The default value handles the cases of:
           - Matplotlib is not imported (just wait on the event)
           - Matplotlib is imported, but not using a Qt, notebook or ipympl
             backend (just wait on the event)
           - Matplotlib is imported and using a Qt backend (run the Qt app
             on the main thread until the run finishes)
           - Matplotlib is imported and using a nbagg or ipympl backend (
             wait on the event and poll to push updates to the browser)

    Attributes
    ----------
    md
        Direct access to the dict-like persistent storage described above

    record_interruptions
        False by default. Set to True to generate an extra event stream
        that records any interruptions (pauses, suspensions).

    state
        {'idle', 'running', 'paused'}

    suspenders
        Read-only collection of `bluesky.suspenders.SuspenderBase` objects
        which can suspend and resume execution; see related methods.

    preprocessors : list
        Generator functions that take in a plan (generator instance) and
        modify its messages on the way out. Suitable examples include
        the functions in the module ``bluesky.plans`` with names ending in
        'wrapper'.  Functions are composed in order: the preprocessors
        ``[f, g]`` are applied like ``f(g(plan))``.

    msg_hook
        Callable that receives all messages before they are processed
        (useful for logging or other development purposes); expected
        signature is ``f(msg)`` where ``msg`` is a ``bluesky.Msg``, a
        kind of namedtuple; default is None.

    state_hook
        Callable with signature ``f(new_state, old_state)`` that will be
        called whenever the RunEngine's state attribute is updated; default
        is None

    waiting_hook
        Callable with signature ``f(status_object)`` that will be called
        whenever the RunEngine is waiting for long-running commands
        (trigger, set, kickoff, complete) to complete. This hook is useful to
        incorporate a progress bar.

    ignore_callback_exceptions
        Boolean, False by default.

    loop : asyncio event loop
        e.g., ``asyncio.get_event_loop()`` or ``asyncio.new_event_loop()``

    max_depth
        Maximum stack depth; set this to prevent users from calling the
        RunEngine inside a function (which can result in unexpected
        behavior and breaks introspection tools). Default is None.
        For built-in Python interpreter, set to 2. For IPython, set to 11
        (tested on IPython 5.1.0; other versions may vary).

    pause_msg : str
        The message printed when a run is interrupted. This message
        includes instructions of changing the state of the RunEngine.
        It is set to ``bluesky.run_engine.PAUSE_MSG`` by default and
        can be modified based on needs.

    commands:
        The list of commands available to Msg.

    """

    _state = LoggingPropertyMachine(RunEngineStateMachine)
    _UNCACHEABLE_COMMANDS = ['pause', 'subscribe', 'unsubscribe', 'stage',
                             'unstage', 'monitor', 'unmonitor', 'open_run',
                             'close_run', 'install_suspender',
                             'remove_suspender']

    @property
    def state(self):
        return self._state

    def __init__(self, md=None, *, loop=None, preprocessors=None,
                 context_managers=None, md_validator=None,
                 scan_id_source=default_scan_id_source,
                 during_task=default_during_task):
        if loop is None:
            loop = get_bluesky_event_loop()
        self._th = _ensure_event_loop_running(loop)
        self._state_lock = threading.RLock()
        self._loop = loop
        self._during_task = during_task

        # When set, RunEngine.__call__ should stop blocking.
        self._blocking_event = threading.Event()

        # When cleared, RunEngine._run will pause until set.
        self._run_permit = asyncio.Event(loop=loop)
        self.loop.call_soon_threadsafe(self._run_permit.set)

        # Make a logger for this specific RE instance, using the instance's
        # Python id, to keep from mixing output from separate instances.
        logger_name = "bluesky.RE.{id}".format(id=id(self))
        self.log = logging.getLogger(logger_name)

        if md is None:
            md = {}
        self.md = md
        self.md.setdefault('versions', {})

        try:
            import ophyd
            self.md['versions']['ophyd'] = ophyd.__version__
        except ImportError:
            self.log.debug("Failed to import ophyd.")

        from ._version import get_versions
        self.md['versions']['bluesky'] = get_versions()['version']
        del get_versions

        if preprocessors is None:
            preprocessors = []
        self.preprocessors = preprocessors
        if context_managers is None:
            context_managers = [SigintHandler]
        self.context_managers = context_managers
        if md_validator is None:
            md_validator = _default_md_validator
        self.md_validator = md_validator
        self.scan_id_source = scan_id_source

        self.max_depth = None
        self.msg_hook = None
        self.state_hook = None
        self.waiting_hook = None
        self.record_interruptions = False
        self.pause_msg = PAUSE_MSG

        # The RunEngine keeps track of a *lot* of state.
        # All flags and caches are defined here with a comment. Good luck.

        self._run_bundlers = {}  # a mapping of open run -> bundlers
        self._metadata_per_call = {}  # for all runs generated by one __call__
        self._deferred_pause_requested = False  # pause at next 'checkpoint'
        self._exception = None  # stored and then raised in the _run loop
        self._interrupted = False  # True if paused, aborted, or failed
        self._staged = set()  # objects staged, not yet unstaged
        self._objs_seen = set()  # all objects seen
        self._movable_objs_touched = set()  # objects we moved at any point
        self._run_start_uids = list()  # run start uids generated by __call__
        self._suspenders = set()  # set holding suspenders
        self._groups = defaultdict(set)  # sets of Events to wait for
        self._status_objs = defaultdict(set)  # status objects to wait for
        self._temp_callback_ids = set()  # ids from CallbackRegistry
        self._msg_cache = deque()  # history of processed msgs for rewinding
        self._rewindable_flag = True  # if the RE is allowed to replay msgs
        self._plan_stack = deque()  # stack of generators to work off of
        self._response_stack = deque()  # resps to send into the plans
        self._exit_status = 'success'  # optimistic default
        self._reason = ''  # reason for abort
        self._task = None  # asyncio.Task associated with call to self._run
        self._task_fut = None  # future proxy to the task above
        self._status_tasks = deque()  # from self._status_object_completed
        self._pardon_failures = None  # will hold an asyncio.Event
        self._plan = None  # the plan instance from __call__
        self._command_registry = {
            'create': self._create,
            'save': self._save,
            'drop': self._drop,
            'read': self._read,
            'monitor': self._monitor,
            'unmonitor': self._unmonitor,
            'null': self._null,
            'stop': self._stop,
            'set': self._set,
            'trigger': self._trigger,
            'sleep': self._sleep,
            'wait': self._wait,
            'checkpoint': self._checkpoint,
            'clear_checkpoint': self._clear_checkpoint,
            'rewindable': self._rewindable,
            'pause': self._pause,
            'resume': self._resume,
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
            'install_suspender': self._install_suspender,
            'remove_suspender': self._remove_suspender, }

        # public dispatcher for callbacks
        # The Dispatcher's public methods are exposed through the
        # RunEngine for user convenience.
        self.dispatcher = Dispatcher()
        self.ignore_callback_exceptions = False

        # aliases for back-compatibility
        self.subscribe_lossless = self.dispatcher.subscribe
        self.unsubscribe_lossless = self.dispatcher.unsubscribe
        self._subscribe_lossless = self.dispatcher.subscribe
        self._unsubscribe_lossless = self.dispatcher.unsubscribe

    @property
    def commands(self):
        '''
        The list of commands available to Msg.

        See Also
        --------
        :meth:`RunEngine.register_command`
        :meth:`RunEngine.unregister_command`
        :meth:`RunEngine.print_command_registry`

        Examples
        --------
        >>> from bluesky import RunEngine
        >>> RE = RunEngine()
        >>> # to list commands
        >>> RE.commands
        '''
        # return as a list, not lazy loader, no surprises...
        return list(self._command_registry.keys())

    def print_command_registry(self, verbose=False):
        '''
        This conveniently prints the command registry of available
        commands.

        Parameters
        ----------
        Verbose : bool, optional
        verbose print. Default is False

        See Also
        --------
        :meth:`RunEngine.register_command`
        :meth:`RunEngine.unregister_command`
        :attr:`RunEngine.commands`

        Examples
        --------
        >>> from bluesky import RunEngine
        >>> RE = RunEngine()
        >>> # Print a very verbose list of currently registered commands
        >>> RE.print_command_registry(verbose=True)
        '''
        commands = "List of available commands\n"

        for command, func in self._command_registry.items():
            docstring = func.__doc__
            if not verbose:
                docstring = docstring.split("\n")[0]
            commands = commands + "{} : {}\n".format(command, docstring)

        return commands

    def subscribe(self, func, name='all'):
        """
        Register a callback function to consume documents.

        .. versionchanged :: 0.10.0
            The order of the arguments was swapped and the ``name``
            argument has been given a default value, ``'all'``. Because the
            meaning of the arguments is unambiguous (they must be a callable
            and a string, respectively) the old order will be supported
            indefinitely, with a warning.

        Parameters
        ----------
        func: callable
            expecting signature like ``f(name, document)``
            where name is a string and document is a dict
        name : {'all', 'start', 'descriptor', 'event', 'stop'}, optional
            the type of document this function should receive ('all' by
            default)

        Returns
        -------
        token : int
            an integer ID that can be used to unsubscribe

        See Also
        --------
        :meth:`RunEngine.unsubscribe`
        """
        # pass through to the Dispatcher, spelled out verbosely here to make
        # sphinx happy -- tricks with __doc__ aren't enough to fool it
        return self.dispatcher.subscribe(func, name)

    def unsubscribe(self, token):
        """
        Unregister a callback function its integer ID.

        Parameters
        ----------
        token : int
            the integer ID issued by :meth:`RunEngine.subscribe`

        See Also
        --------
        :meth:`RunEngine.subscribe`
        """
        # pass through to the Dispatcher, spelled out verbosely here to make
        # sphinx happy -- tricks with __doc__ aren't enough to fool it
        return self.dispatcher.unsubscribe(token)

    @property
    def rewindable(self):
        return self._rewindable_flag

    @rewindable.setter
    def rewindable(self, v):
        cur_state = self._rewindable_flag
        self._rewindable_flag = bool(v)
        if self.resumable and self._rewindable_flag != cur_state:
            self._reset_checkpoint_state()

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

    def _clear_run_cache(self):
        "Clean up for a new run."
        self._groups.clear()
        self._status_objs.clear()
        self._interruptions_desc_uid = None
        self._interruptions_counter = count(1)

    @_state_locked
    def _clear_call_cache(self):
        "Clean up for a new __call__ (which may encompass multiple runs)."
        self._metadata_per_call.clear()
        self._staged.clear()
        self._objs_seen.clear()
        self._movable_objs_touched.clear()
        self._deferred_pause_requested = False
        self._plan_stack = deque()
        self._msg_cache = deque()
        self._response_stack = deque()
        self._exception = None
        self._run_start_uids.clear()
        self._exit_status = 'success'
        self._reason = ''
        self._task = None
        self._task_fut = None
        self._status_tasks.clear()
        self._pardon_failures = asyncio.Event(loop=self.loop)
        self._plan = None
        self._interrupted = False

        # Unsubscribe for per-run callbacks.
        for cid in self._temp_callback_ids:
            self.unsubscribe(cid)
        self._temp_callback_ids.clear()

    def reset(self):
        """
        Clean up caches and unsubscribe subscriptions.

        Lossless subscriptions are not unsubscribed.
        """
        if self._state != 'idle':
            self.halt()
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

        See Also
        --------
        :meth:`RunEngine.unregister_command`
        :meth:`RunEngine.print_command_registry`
        :attr:`RunEngine.commands`
        """
        self._command_registry[name] = func

    def unregister_command(self, name):
        """
        Unregister a Message command.

        Parameters
        ----------
        name : str

        See Also
        --------
        :meth:`RunEngine.register_command`
        :meth:`RunEngine.print_command_registry`
        :attr:`RunEngine.commands`
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
        future = asyncio.run_coroutine_threadsafe(
            self._request_pause_coro(defer),
            loop=self.loop)
        # TODO add a timeout here?
        return future.result()

    async def _request_pause_coro(self, defer=False):
        # We are pausing. Cancel any deferred pause previously requested.
        if defer:
            self._deferred_pause_requested = True
            print("Deferred pause acknowledged. Continuing to checkpoint.")
            return

        print("Pausing...")

        self._deferred_pause_requested = False
        self._interrupted = True
        self._state = 'pausing'
        for current_run in self._run_bundlers.values():
            current_run.record_interruption('pause')

        self._task.cancel()

    def __call__(self, *args, **metadata_kw):
        """Execute a plan.

        Any keyword arguments will be interpreted as metadata and recorded with
        any run(s) created by executing the plan. Notice that the plan
        (required) and extra subscriptions (optional) must be given as
        positional arguments.

        Parameters
        ----------
        plan : generator (positional only)
            a generator or that yields ``Msg`` objects (or an iterable that
            returns such a generator)
        subs : callable, list, or dict, optional (positional only)
            Temporary subscriptions (a.k.a. callbacks) to be used on this run.
            For convenience, any of the following are accepted:

            * a callable, which will be subscribed to 'all'
            * a list of callables, which again will be subscribed to 'all'
            * a dictionary, mapping specific subscriptions to callables or
              lists of callables; valid keys are {'all', 'start', 'stop',
              'event', 'descriptor'}

        Returns
        -------
        uids : list
            list of uids (i.e. RunStart Document uids) of run(s)
        """
        if self.state == 'panicked':
            raise RuntimeError("The RunEngine is panicked and "
                               "cannot be recovered. "
                               "You must restart bluesky.")
        # This scheme lets us make 'plan' and 'subs' POSITIONAL ONLY, reserving
        # all keyword arguments for user metdata.
        arguments = _call_sig.bind(self, *args, **metadata_kw).arguments
        plan = arguments['plan']
        subs = arguments.get('subs', None)
        metadata_kw = arguments.get('metadata_kw', {})
        if 'raise_if_interrupted' in metadata_kw:
            warn("The 'raise_if_interrupted' flag has been removed. The "
                 "RunEngine now always raises RunEngineInterrupted if it is "
                 "interrupted. The 'raise_if_interrupted' keyword argument, "
                 "like all keyword arguments, will be interpreted as "
                 "metadata.")
        # Check that the RE is not being called from inside a function.
        if self.max_depth is not None:
            frame = inspect.currentframe()
            depth = len(inspect.getouterframes(frame))
            if depth > self.max_depth:
                text = MAX_DEPTH_EXCEEDED_ERR_MSG.format(self.max_depth, depth)
                raise RuntimeError(text)

        # If we are in the wrong state, raise.
        if not self._state.is_idle:
            raise RuntimeError("The RunEngine is in a %s state" % self._state)

        futs = []
        tripped_justifications = []
        for sup in self.suspenders:
            f_lst, justification = sup.get_futures()
            if f_lst:
                futs.extend(f_lst)
                tripped_justifications.append(justification)

        if tripped_justifications:
            print("At least one suspender has tripped. The plan will begin "
                  "when all suspenders are ready. Justification:")
            for i, justification in enumerate(tripped_justifications):
                print('    {}. {}'.format(i + 1, justification))

            print()
            print("Suspending... To get to the prompt, "
                  "hit Ctrl-C twice to pause.")

        self._clear_call_cache()
        self._clear_run_cache()  # paranoia, in case of previous bad exit

        for name, funcs in normalize_subs_input(subs).items():
            for func in funcs:
                self._temp_callback_ids.add(self.subscribe(func, name))

        self._plan = plan  # this ref is just used for metadata introspection
        self._metadata_per_call.update(metadata_kw)

        gen = ensure_generator(plan)
        for wrapper_func in self.preprocessors:
            gen = wrapper_func(gen)

        self._plan_stack.append(gen)
        self._response_stack.append(None)
        if futs:
            self._plan_stack.append(single_gen(Msg('wait_for', None, futs)))
            self._response_stack.append(None)
        self.log.info("Executing plan %r", self._plan)

        def _build_task():
            # make sure _run will block at the top
            self._run_permit.clear()
            self._blocking_event.clear()
            self._task_fut = asyncio.run_coroutine_threadsafe(self._run(),
                                                              loop=self.loop)

            def set_blocking_event(future):
                self._blocking_event.set()

            self._task_fut.add_done_callback(set_blocking_event)

        self._resume_task(init_func=_build_task)

        if self._interrupted:
            raise RunEngineInterrupted(self.pause_msg) from None

        return tuple(self._run_start_uids)

    __call__.__signature__ = _call_sig

    def resume(self):
        """Resume a paused plan from the last checkpoint.

        Returns
        -------
        uids : list
            list of Header uids (a.k.a RunStart uids) of run(s)
        """
        if self.state == 'panicked':
            raise RuntimeError("The RunEngine is panicked and "
                               "cannot be recovered. "
                               "You must restart bluesky.")

        # The state machine does not capture the whole picture.
        if not self._state.is_paused:
            raise TransitionError("The RunEngine is the {0} state. "
                                  "You can only resume for the paused state."
                                  "".format(self._state))

        self._interrupted = False
        for current_run in self._run_bundlers.values():
            current_run.record_interruption('resume')
        new_plan = self._rewind()
        self._plan_stack.append(new_plan)
        self._response_stack.append(None)
        # Notify Devices of the resume in case they want to clean up.
        for obj in self._objs_seen:
            if hasattr(obj, 'resume'):
                obj.resume()
        self._resume_task()
        if self._interrupted:
            raise RunEngineInterrupted(self.pause_msg) from None
        return tuple(self._run_start_uids)

    def _rewind(self):
        '''Clean up in preparation for resuming from a pause or suspension.

        Returns
        -------
        new_plan : generator
             A new plan made from the messages in the message cache

        '''
        len_msg_cache = len(self._msg_cache)
        new_plan = ensure_generator(list(self._msg_cache))
        self._msg_cache = deque()
        if len_msg_cache:
            for current_run in self._run_bundlers.values():
                current_run.rewind()

        return new_plan

    def _resume_task(self, *, init_func=None):
        # Clear the blocking Event so that we can wait on it below.
        # The task will set it when it is done, as it was previously
        # configured to do it __call__.
        self._blocking_event.clear()

        # Handle all context managers
        with ExitStack() as stack:
            for mgr in self.context_managers:
                stack.enter_context(mgr(self))

            if init_func is not None:
                init_func()

            if self._task_fut is None or self._task_fut.done():
                return

            # The _run task is waiting on this Event. Let is continue.
            self.loop.call_soon_threadsafe(self._run_permit.set)
            try:
                # Block until plan is complete or exception is raised.
                try:
                    self._during_task(self._blocking_event)
                except KeyboardInterrupt:
                    import ctypes
                    self._interrupted = True
                    # we can not interrupt a python thread from the outside
                    # but there is an API to schedule an exception to be raised
                    # the next time that thread would interpret byte code.
                    # The documentation of this function includes the sentence
                    #
                    #   To prevent naive misuse, you must write your
                    #   own C extension to call this.
                    #
                    # Here we cheat a bit and use ctypes.
                    num_threads = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_ulong(self._th.ident),
                        ctypes.py_object(_RunEnginePanic))
                    # however, if the thread is in a system call (such
                    # as sleep or I/O) there is no way to interrupt it
                    # (per decree of Guido) thus we give it a second
                    # to sort it's self out
                    task_finished = self._blocking_event.wait(1)
                    # before giving up and putting the RE in a
                    # non-recoverable panicked state.
                    if not task_finished or num_threads != 1:
                        self._state = 'panicked'
                except Exception as raised_er:
                    self.halt()
                    self._interrupted = True
                    raise raised_er
            finally:
                if self._task_fut.done():
                    # get exceptions from the main task
                    try:
                        exc = self._task_fut.exception()
                    except (asyncio.CancelledError,
                            concurrent.futures.CancelledError):
                        exc = None
                    # if the main task exception is not None, re-raise
                    # it (unless it is a canceled error)
                    if (exc is not None
                            and not isinstance(exc, _RunEnginePanic)):
                        raise exc

    def install_suspender(self, suspender):
        """
        Install a 'suspender', which can suspend and resume execution.

        Parameters
        ----------
        suspender : `bluesky.suspenders.SuspenderBase`

        See Also
        --------
        :meth:`RunEngine.remove_suspender`
        :meth:`RunEngine.clear_suspenders`
        """
        self._suspenders.add(suspender)
        suspender.install(self)

    async def _install_suspender(self, msg):
        """
        See :meth: `RunEngine.install_suspender`

        Expected message object is:

            Msg('install_suspender', None, suspender)
        """
        suspender = msg.args[0]
        self.install_suspender(suspender)

    def remove_suspender(self, suspender):
        """
        Uninstall a suspender.

        Parameters
        ----------
        suspender : `bluesky.suspenders.SuspenderBase`

        See Also
        --------
        :meth:`RunEngine.install_suspender`
        :meth:`RunEngine.clear_suspenders`
        """
        if suspender in self._suspenders:
            suspender.remove()
        self._suspenders.discard(suspender)

    async def _remove_suspender(self, msg):
        """
        See :meth: `RunEngine.remove_suspender`

        Expected message object is:

            Msg('remove_suspender', None, suspender)
        """
        suspender = msg.args[0]
        self.remove_suspender(suspender)

    def clear_suspenders(self):
        """
        Uninstall all suspenders.

        See Also
        --------
        :meth:`RunEngine.install_suspender`
        :meth:`RunEngine.remove_suspender`
        """
        for sus in self.suspenders:
            self.remove_suspender(sus)

    def request_suspend(self, fut, *, pre_plan=None, post_plan=None,
                        justification=None):
        """Request that the run suspend itself until the future is finished.

        The two plans will be run before and after waiting for the future.
        This enable doing things like opening and closing shutters and
        resetting cameras around a suspend.

        Parameters
        ----------
        fut : asyncio.Future

        pre_plan : iterable or callable, optional
           Plan to execute just before suspending. If callable, must
           take no arguments.

        post_plan : iterable or callable, optional
            Plan to execute just before resuming. If callable, must
            take no arguments.

        justification : str, optional
            explanation of why the suspension has been requested

        """

        print("Suspending....To get prompt hit Ctrl-C twice to pause.")
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("Suspension occurred at {}.".format(ts))

        async def _request_suspend(pre_plan, post_plan, justification):
            if not self.resumable:
                print("No checkpoint; cannot suspend.")
                print("Aborting: running cleanup and marking "
                      "exit_status as 'abort'...")
                self._interrupted = True
                with self._state_lock:
                    self._exception = FailedPause()
                was_paused = self._state == 'paused'
                self._state = 'aborting'
                if not was_paused:
                    self._task.cancel()

            if justification is not None:
                print("Justification for this suspension:\n%s" % justification)
            for current_run in self._run_bundlers.values():
                current_run.record_interruption('resume')
            # During suspend, all motors should be stopped. Call stop() on
            # every object we ever set().
            self._stop_movable_objects(success=True)
            # Notify Devices of the pause in case they want to clean up.
            for obj in self._objs_seen:
                if hasattr(obj, 'pause'):
                    try:
                        obj.pause()
                    except NoReplayAllowed:
                        self._reset_checkpoint_state_meth()

            # rewind to the last checkpoint
            new_plan = self._rewind()
            # queue up the cached messages
            self._plan_stack.append(new_plan)
            self._response_stack.append(None)

            self._plan_stack.append(single_gen(
                Msg('rewindable', None, self.rewindable)))
            self._response_stack.append(None)

            # if there is a post plan add it between the wait
            # and the cached messages
            if post_plan is not None:
                if callable(post_plan):
                    post_plan = post_plan()
                self._plan_stack.append(ensure_generator(post_plan))
                self._response_stack.append(None)

            # tell the devices they are ready to go again
            self._plan_stack.append(single_gen(Msg('resume', None, )))
            self._response_stack.append(None)

            # add the wait on the future to the stack
            self._plan_stack.append(single_gen(Msg('wait_for', None, [fut, ])))
            self._response_stack.append(None)

            # if there is a pre plan add on top of the wait
            if pre_plan is not None:
                if callable(pre_plan):
                    pre_plan = pre_plan()
                self._plan_stack.append(ensure_generator(pre_plan))
                self._response_stack.append(None)

            self._plan_stack.append(single_gen(
                Msg('rewindable', None, False)))
            self._response_stack.append(None)
            # The event loop is still running. The pre_plan will be processed,
            # and then the RunEngine will be hung up on processing the
            # 'wait_for' message until `fut` is set.
            if not self._state == 'paused':
                self._state = 'suspending'
                # bump the _run task out of what ever it is awaiting
                self._task.cancel()

        self.loop.call_soon_threadsafe(
            self.loop.create_task,
            _request_suspend(pre_plan, post_plan, justification))

    def abort(self, reason=''):
        """
        Stop a running or paused plan and mark it as aborted.

        See Also
        --------
        :meth:`RunEngine.halt`
        :meth:`RunEngine.stop`
        """
        return self.__interrupter_helper(self._abort_coro(reason))

    async def _abort_coro(self, reason):
        if self._state.is_idle:
            raise TransitionError("RunEngine is already idle.")
        print("Aborting: running cleanup and marking "
              "exit_status as 'abort'...")
        self._interrupted = True
        self._reason = reason

        self._exit_status = 'abort'

        was_paused = self._state == 'paused'
        self._state = 'aborting'
        if was_paused:
            with self._state_lock:
                self._exception = RequestAbort()
        else:
            self._task.cancel()
        for task in self._status_tasks:
            task.cancel()

        return tuple(self._run_start_uids)

    def stop(self):
        """
        Stop a running or paused plan, but mark it as successful (not aborted).

        See Also
        --------
        :meth:`RunEngine.abort`
        :meth:`RunEngine.halt`
        """
        return self.__interrupter_helper(self._stop_coro())

    async def _stop_coro(self):
        if self._state.is_idle:
            raise TransitionError("RunEngine is already idle.")
        print("Stopping: running cleanup and marking exit_status "
              "as 'success'...")

        self._interrupted = True
        was_paused = self._state == 'paused'
        self._state = 'stopping'
        if not was_paused:
            self._task.cancel()
        return tuple(self._run_start_uids)

    def halt(self):
        '''
        Stop the running plan and do not allow the plan a chance to clean up.

        See Also
        --------
        :meth:`RunEngine.abort`
        :meth:`RunEngine.stop`
        '''
        return self.__interrupter_helper(self._halt_coro())

    def __interrupter_helper(self, coro):
        if self.state == 'panicked':
            raise RuntimeError("The RunEngine is panicked and "
                               "cannot be recovered. "
                               "You must restart bluesky.")

        coro_event = threading.Event()
        task = None

        def end_cb(fut):
            coro_event.set()

        def start_task():
            nonlocal task
            task = self.loop.create_task(coro)
            task.add_done_callback(end_cb)

        was_paused = self._state == 'paused'
        self.loop.call_soon_threadsafe(start_task)
        coro_event.wait()
        if was_paused:
            self._resume_task()

        return task.result()

    async def _halt_coro(self):
        if self._state.is_idle:
            raise TransitionError("RunEngine is already idle.")
        print("Halting: skipping cleanup and marking exit_status as "
              "'abort'...")
        self._interrupted = True
        with self._state_lock:
            self._exception = PlanHalt()
            self._exit_status = 'abort'
        was_paused = self._state == 'paused'
        self._state = 'halting'
        if not was_paused:
            self._task.cancel()
        return tuple(self._run_start_uids)

    def _stop_movable_objects(self, *, success=True):
        "Call obj.stop() for all objects we have moved. Log any exceptions."
        for obj in self._movable_objs_touched:
            try:
                stop = obj.stop
            except AttributeError:
                self.log.debug("No 'stop' method available on %r", obj)
            else:
                try:
                    stop(success=success)
                except Exception:
                    self.log.exception("Failed to stop %r.", obj)

    async def _run(self):
        """Pull messages from the plan, process them, send results back.

        Upon exit, clean up.
        - Call stop() on all objects that were 'set' or 'kickoff'.
        - Try to collect any uncollected flyers.
        - Try to unstage any devices left staged by the plan.
        - Try to remove any monitoring subscriptions left on by the plan.
        - If interrupting the middle of a run, try to emit a RunStop document.
        """
        await self._run_permit.wait()
        # grab the current task.  We need to do this here because the
        # object returned by `run_coroutine_threadsafe` is a future
        # that acts as a proxy that does not have the correct behavior
        # when `.cancel` is called on it.
        with self._state_lock:
            self._task = current_task(self.loop)
        stashed_exception = None
        debug = logging.getLogger('{}.msg'.format(self.log.name)).debug
        self._reason = ''
        # sentinel to decide if need to add to the response stack or not
        sentinel = object()
        exit_reason = ''
        try:
            self._state = 'running'
            while True:
                if self._state in ('pausing', 'suspending'):
                    if not self.resumable:
                        self._run_permit.set()
                        stashed_exception = FailedPause()
                        for task in self._status_tasks:
                            task.cancel()
                        self._state = 'aborting'
                        continue
                # currently only using 'suspending' to get us into the
                # block above, we do not have a 'suspended' state
                # (yet)
                if self._state == 'suspending':
                    self._state = 'running'
                if not self._run_permit.is_set():
                    # A pause has been requested. First, put everything in a
                    # resting state.
                    assert self._state == 'pausing'
                    # Remove any monitoring callbacks, but keep refs in
                    # self._monitor_params to re-instate them later.
                    for current_run in self._run_bundlers.values():
                        await current_run.suspend_monitors()
                    # During pause, all motors should be stopped. Call stop()
                    # on every object we ever set().
                    self._stop_movable_objects(success=True)
                    # Notify Devices of the pause in case they want to
                    # clean up.
                    for obj in self._objs_seen:
                        if hasattr(obj, 'pause'):
                            try:
                                obj.pause()
                            except NoReplayAllowed:
                                self._reset_checkpoint_state_meth()
                    self._state = 'paused'
                    # Let RunEngine.__call__ return...
                    self._blocking_event.set()

                    await self._run_permit.wait()
                    # Restore any monitors
                    for current_run in self._run_bundlers.values():
                        await current_run.restore_monitors()
                    if self._state == 'paused':
                        # may be called by 'resume', 'stop', 'abort', 'halt'
                        self._state = 'running'

                    # If we are here, we have come back to life either to
                    # continue (resume) or to clean up before exiting.

                assert len(self._response_stack) == len(self._plan_stack)
                # set resp to the sentinel so that if we fail in the sleep
                # we do not add an extra response
                resp = sentinel
                try:
                    # the new response to be added
                    new_response = None

                    # This 'await' must be here to ensure that
                    # this coroutine breaks out of its current behavior
                    # before trying to get the next message from the
                    # top of the generator stack in case there has
                    # been a pause requested.  Without this the next
                    # message after the pause may be processed first
                    # on resume (instead of the first message in
                    # self._msg_cache).

                    # This sleep has to be inside of this try block so
                    # that any of the 'async' exceptions get thrown in the
                    # correct place
                    await asyncio.sleep(0, loop=self.loop)
                    # always pop off a result, we are either sending it back in
                    # or throwing an exception in, in either case the left hand
                    # side of the yield in the plan will be moved past
                    resp = self._response_stack.pop()
                    # if any status tasks have failed, grab the exceptions.
                    # give priority to things pushed in from outside
                    with self._state_lock:
                        if self._exception is not None:
                            stashed_exception = self._exception
                            self._exception = None
                    # The case where we have a stashed exception
                    if (stashed_exception is not None or
                            isinstance(resp, Exception)):
                        # throw the exception at the current plan
                        try:
                            msg = self._plan_stack[-1].throw(
                                stashed_exception or resp)
                        except Exception as e:
                            # The current plan did not handle it,
                            # maybe the next plan (if any) would like
                            # to try
                            self._plan_stack.pop()
                            # we have killed the current plan, do not give
                            # it a new response
                            resp = sentinel
                            if len(self._plan_stack):
                                stashed_exception = e
                                continue
                            # no plans left and still an unhandled exception
                            # re-raise to exit the infinite loop
                            else:
                                raise
                        # clear the stashed exception, the top plan
                        # handled it.
                        else:
                            stashed_exception = None
                    # The normal case of clean operation
                    else:
                        try:
                            msg = self._plan_stack[-1].send(resp)
                        # We have exhausted the top generator
                        except StopIteration:
                            # pop the dead generator go back to the top
                            self._plan_stack.pop()
                            # we have killed the current plan, do not give
                            # it a new response
                            resp = sentinel
                            if len(self._plan_stack):
                                continue
                            # or reraise to get out of the infinite loop
                            else:
                                raise
                        # Any other exception that comes out of the plan
                        except Exception as e:
                            # pop the dead plan, stash the exception and
                            # go to the top of the loop
                            self._plan_stack.pop()
                            # we have killed the current plan, do not give
                            # it a new response
                            resp = sentinel
                            if len(self._plan_stack):
                                stashed_exception = e
                                continue
                            # or reraise to get out of the infinite loop
                            else:
                                raise

                    # if we have a message hook, call it
                    if self.msg_hook is not None:
                        self.msg_hook(msg)
                    debug(msg)

                    # update the running set of all objects we have seen
                    self._objs_seen.add(msg.obj)

                    # if this message can be cached for rewinding, cache it
                    if (self._msg_cache is not None and
                            self._rewindable_flag and
                            msg.command not in self._UNCACHEABLE_COMMANDS):
                        # We have a checkpoint.
                        self._msg_cache.append(msg)

                    # try to look up the coroutine to execute the command
                    try:
                        coro = self._command_registry[msg.command]
                    # replace KeyError with a local sub-class and go
                    # to top of the loop
                    except KeyError:
                        # TODO make this smarter
                        new_response = InvalidCommand(msg.command)
                        continue

                    # try to finally run the command the user asked for
                    try:
                        # this is one of two places that 'async'
                        # exceptions (coming in via throw) can be
                        # raised
                        new_response = await coro(msg)

                    # special case `CancelledError` and let the outer
                    # exception block deal with it.
                    except asyncio.CancelledError:
                        raise
                    # any other exception, stash it and go to the top of loop
                    except Exception as e:
                        new_response = e
                        continue
                    # normal use, if it runs cleanly, stash the response and
                    # go to the top of the loop
                    else:
                        continue

                except KeyboardInterrupt:
                    # This only happens if some external code captures SIGINT
                    # -- overriding the RunEngine -- and then raises instead
                    # of (properly) calling the RunEngine's handler.
                    # See https://github.com/NSLS-II/bluesky/pull/242
                    print("An unknown external library has improperly raised "
                          "KeyboardInterrupt. Intercepting and triggering "
                          "a HALT.")
                    await self._halt_coro()
                except asyncio.CancelledError as e:
                    if self._state == 'pausing':
                        # if we got a CancelledError and we are in the
                        # 'pausing' state clear the run permit and
                        # bounce to the top
                        self._run_permit.clear()
                        continue
                    if self._state in ('halting', 'stopping', 'aborting'):
                        # if we got this while just keep going in tear-down
                        exception_map = {'halting': PlanHalt,
                                         'stopping': RequestStop,
                                         'aborting': RequestAbort}
                        # if the exception is not set bounce to the top
                        if stashed_exception is None:
                            stashed_exception = exception_map[self.state]
                        continue
                    if self._state == 'suspending':
                        # just bounce to the top
                        continue
                    # if we are handling this twice, raise and leave the plans
                    # alone
                    if stashed_exception is e:
                        raise e
                    # the case where FailedPause, RequestAbort or a coro
                    # raised error is not already stashed in _exception
                    if stashed_exception is None:
                        stashed_exception = e
                finally:
                    # if we poped a response and did not pop a plan, we need
                    # to put the new response back on the stack
                    if resp is not sentinel:
                        self._response_stack.append(new_response)

        except (StopIteration, RequestStop):
            self._exit_status = 'success'
            # TODO Is the sleep here necessary?
            await asyncio.sleep(0, loop=self.loop)
        except (FailedPause, RequestAbort, asyncio.CancelledError,
                PlanHalt):
            self._exit_status = 'abort'
            # TODO Is the sleep here necessary?
            await asyncio.sleep(0, loop=self.loop)
            self.log.exception("Run aborted")
        except GeneratorExit as err:
            self._exit_status = 'fail'  # Exception raises during 'running'
            exit_reason = str(err)
            raise ValueError from err
        except Exception as err:
            self._exit_status = 'fail'  # Exception raises during 'running'
            exit_reason = str(err)
            self.log.exception("Run aborted")
            raise err
        finally:
            if not exit_reason:
                exit_reason = self._reason
            # Some done_callbacks may still be alive in other threads.
            # Block them from creating new 'failed status' tasks on the loop.
            self._pardon_failures.set()
            # call stop() on every movable object we ever set()
            self._stop_movable_objects(success=True)
            for current_run in self._run_bundlers.values():
                # Clear any uncleared monitoring callbacks.
                current_run.clear_monitors()
                # Try to collect any flyers that were kicked off but
                # not finished.  Some might not support partial
                # collection. We swallow errors.
                await current_run.backstop_collect()
            # in case we were interrupted between 'stage' and 'unstage'
            for obj in list(self._staged):
                try:
                    obj.unstage()
                except Exception:
                    self.log.exception("Failed to unstage %r.", obj)
                self._staged.remove(obj)

            sys.stdout.flush()
            # Emit RunStop if necessary.
            for key, current_run in self._run_bundlers.items():
                if current_run.run_is_open:
                    try:
                        await current_run.close_run(
                            Msg('close_run',
                                exit_status=self._exit_status,
                                reason=exit_reason,
                                run_id=key))
                    except Exception:
                        self.log.error(
                            "Failed to close run %r.", current_run)
            self._run_bundlers.clear()

            for p in self._plan_stack:
                try:
                    p.close()
                except RuntimeError:
                    print('The plan {!r} tried to yield a value on close.  '
                          'Please fix your plan.'.format(p))

            self._state = 'idle'

        self.log.info("Cleaned up from plan %r", self._plan)
        if isinstance(stashed_exception, asyncio.CancelledError):
            raise stashed_exception

    async def _wait_for(self, msg):
        """Instruct the RunEngine to wait for futures

        Expected message object is:

            Msg('wait_for', None, awaitable_factories, **kwargs)

        The keyword arguments will be passed through to `asyncio.wait`.

        The callables in awaitable_factories must have the signature ::

           def fut_fac() -> awaitable:
               'This must work multiple times'

        """
        futs, = msg.args
        futs = [f() for f in futs]
        await asyncio.wait(futs, loop=self.loop, **msg.kwargs)

    async def _open_run(self, msg):
        """Instruct the RunEngine to start a new "run"

        Expected message object is:

            Msg('open_run', None, **kwargs)

        where **kwargs are any additional metadata that should go into
        the RunStart document
        """
        # TODO extract this from the Msg
        run_key = _extract_run_key(msg)
        if run_key in self._run_bundlers:
            raise IllegalRunSequence(msg, "A 'close_run' message was not "
                                     "received before the 'open_run' "
                                     "message")

        # Run scan_id calculation method
        self.md['scan_id'] = self.scan_id_source(self.md)

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

        current_run = self._run_bundlers[run_key] = RunBundler(
            md, self.record_interruptions, self.emit, self.emit_sync, self.log,
            loop=self.loop)

        new_uid = await current_run.open_run(msg)
        self._run_start_uids.append(new_uid)
        return new_uid

    async def _close_run(self, msg):
        """Instruct the RunEngine to write the RunStop document

        Expected message object is:

            Msg('close_run', None, exit_status=None, reason=None)

        if *exit_stats* and *reason* are not provided, use the values
        stashed on the RE.
        """
        # TODO extract this from the Msg
        run_key = _extract_run_key(msg)
        try:
            current_run = self._run_bundlers[run_key]
        except KeyError as ke:
            raise IllegalRunSequence(msg, "A 'close_run' message was received "
                                     "but there is no run open. If this "
                                     "occurred after a pause/resume, add a "
                                     "'checkpoint' message after the "
                                     "'close_run' message.") from ke

        ret = (await current_run.close_run(msg))
        del self._run_bundlers[run_key]
        return ret

    async def _create(self, msg):
        """Trigger the run engine to start bundling future obj.read() calls for
         an Event document

        Expected message object is:

            Msg('create', None, name='primary')
            Msg('create', name='primary')

        Note that the `name` kwarg will be the 'name' field of the resulting
        descriptor. So descriptor['name'] = msg.kwargs['name'].

        Also note that changing the 'name' of the Event will create a new
        Descriptor document.
        """
        run_key = _extract_run_key(msg)
        try:
            current_run = self._run_bundlers[run_key]
        except KeyError as ke:
            raise IllegalRunSequence(msg, "Cannot bundle readings without an "
                                     "open run. That is, 'create' must be "
                                     "preceded by 'open_run'.") from ke

        return (await current_run.create(msg))

    async def _read(self, msg):
        """
        Add a reading to the open event bundle.

        Expected message object is:

            Msg('read', obj)
        """
        obj = msg.obj
        # actually _read_ the object
        ret = obj.read(*msg.args, **msg.kwargs)

        if ret is None:
            raise RuntimeError(
                f"The read of {obj.name} returned None. "
                "This is a bug in your object implementation, "
                "`read` must return a dictionary.")
        run_key = _extract_run_key(msg)
        try:
            current_run = self._run_bundlers[run_key]
        except KeyError:
            ...
        else:
            await current_run.read(msg, ret)

        return ret

    async def _monitor(self, msg):
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

        run_key = _extract_run_key(msg)
        try:
            current_run = self._run_bundlers[run_key]
        except KeyError as ke:
            raise IllegalRunSequence(msg, "A 'monitor' message was sent but "
                                     "no run is open.") from ke

        await current_run.monitor(msg)
        await self._reset_checkpoint_state_coro()

    async def _unmonitor(self, msg):
        """
        Stop monitoring; i.e., remove the callback emitting event documents.

        Expected message object is:

            Msg('unmonitor', obj)
        """
        run_key = _extract_run_key(msg)
        try:
            current_run = self._run_bundlers[run_key]
        except KeyError as ke:
            raise IllegalRunSequence(msg,
                "A 'unmonitor' message was sent but no "
                "run is open.") from ke
        await current_run.unmonitor(msg)
        await self._reset_checkpoint_state_coro()

    async def _save(self, msg):
        """Save the event that is currently being bundled

        Expected message object is:

            Msg('save')
        """
        run_key = _extract_run_key(msg)
        try:
            current_run = self._run_bundlers[run_key]
        except KeyError as ke:
            # sanity check -- this should be caught by 'create' which makes
            # this code path impossible
            raise IllegalRunSequence(msg, "A 'save' message was sent but no "
                                     "run is open.") from ke

        await current_run.save(msg)

    async def _drop(self, msg):
        """Drop the event that is currently being bundled

        Expected message object is:

            Msg('drop')
        """
        run_key = _extract_run_key(msg)
        try:
            current_run = self._run_bundlers[run_key]
        except KeyError as ke:
            raise IllegalRunSequence(msg, "A 'drop' message was sent but no "
                                     "run is open.") from ke

        await current_run.drop(msg)

    async def _kickoff(self, msg):
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
        run_key = _extract_run_key(msg)
        try:
            current_run = self._run_bundlers[run_key]
        except KeyError as ke:
            raise IllegalRunSequence(msg, "A 'kickoff' message was sent but "
                                     "no run is open.") from ke

        _, obj, args, kwargs, _ = msg
        kwargs = dict(msg.kwargs)
        group = kwargs.pop("group", None)

        ret = obj.kickoff(*msg.args, **kwargs)
        p_event = asyncio.Event(loop=self.loop)
        pardon_failures = self._pardon_failures

        await current_run.kickoff(msg)

        def done_callback():
            self.log.debug(
                "The object %r reports 'kickoff' is done " "with status %r",
                msg.obj,
                ret.success,
            )
            task = self._loop.call_soon_threadsafe(
                self._status_object_completed, ret, p_event, pardon_failures
            )
            self._status_tasks.append(task)

        try:
            ret.add_callback(done_callback)
        except AttributeError:
            # for ophyd < v0.8.0
            ret.finished_cb = done_callback
        self._groups[group].add(p_event.wait)
        self._status_objs[group].add(ret)

        return ret

    async def _complete(self, msg):
        """
        Tell a flyer, 'stop collecting, whenever you are ready'.

        The flyer returns a status object. Some flyers respond to this
        command by stopping collection and returning a finished status
        object immediately. Other flyers finish their given course and
        finish whenever they finish, irrespective of when this command is
        issued.

        Expected message object is:

            Msg('complete', flyer, group=<GROUP>)

        where <GROUP> is a hashable identifier.
        """
        run_key = _extract_run_key(msg)
        try:
            current_run = self._run_bundlers[run_key]
        except KeyError as ke:
            raise IllegalMessageSequence("A 'complete' message was sent but no "
                                         "run is open.") from ke

        await current_run.complete(msg)
        kwargs = dict(msg.kwargs)
        group = kwargs.pop("group", None)
        ret = msg.obj.complete(*msg.args, **kwargs)

        p_event = asyncio.Event(loop=self.loop)
        pardon_failures = self._pardon_failures

        def done_callback():
            self.log.debug(
                "The object %r reports 'complete' is done " "with status %r",
                msg.obj,
                ret.success,
            )
            task = self._loop.call_soon_threadsafe(
                self._status_object_completed, ret, p_event, pardon_failures
            )
            self._status_tasks.append(task)

        try:
            ret.add_callback(done_callback)
        except AttributeError:
            # for ophyd < v0.8.0
            ret.finished_cb = done_callback
        self._groups[group].add(p_event.wait)
        self._status_objs[group].add(ret)
        return ret

    async def _collect(self, msg):
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
        run_key = _extract_run_key(msg)
        try:
            current_run = self._run_bundlers[run_key]
        except KeyError as ke:
            # sanity check -- 'kickoff' should catch this and make this
            # code path impossible
            raise IllegalRunSequence(msg, "A 'collect' message was sent but "
                                     "no run is open.") from ke

        return (await current_run.collect(msg))

    async def _null(self, msg):
        """
        A no-op message, mainly for debugging and testing.
        """
        pass

    async def _set(self, msg):
        """
        Set a device and cache the returned status object.

        Also, note that the device has been touched so it can be stopped upon
        exit.

        Expected message object is

            Msg('set', obj, *args, **kwargs)

        where arguments are passed through to `obj.set(*args, **kwargs)`.
        """
        kwargs = dict(msg.kwargs)
        group = kwargs.pop('group', None)
        self._movable_objs_touched.add(msg.obj)
        ret = msg.obj.set(*msg.args, **kwargs)
        p_event = asyncio.Event(loop=self.loop)
        pardon_failures = self._pardon_failures

        def done_callback():
            self.log.debug("The object %r reports set is done "
                           "with status %r", msg.obj, ret.success)
            task = self._loop.call_soon_threadsafe(
                self._status_object_completed, ret, p_event, pardon_failures)
            self._status_tasks.append(task)

        try:
            ret.add_callback(done_callback)
        except AttributeError:
            # for ophyd < v0.8.0
            ret.finished_cb = done_callback
        self._groups[group].add(p_event.wait)
        self._status_objs[group].add(ret)

        return ret

    async def _trigger(self, msg):
        """
        Trigger a device and cache the returned status object.

        Expected message object is:

            Msg('trigger', obj)
        """
        kwargs = dict(msg.kwargs)
        group = kwargs.pop('group', None)
        ret = msg.obj.trigger(*msg.args, **kwargs)
        p_event = asyncio.Event(loop=self.loop)
        pardon_failures = self._pardon_failures

        def done_callback():
            self.log.debug("The object %r reports trigger is "
                           "done with status %r.", msg.obj, ret.success)
            task = self._loop.call_soon_threadsafe(
                self._status_object_completed, ret, p_event, pardon_failures)
            self._status_tasks.append(task)

        try:
            ret.add_callback(done_callback)
        except AttributeError:
            # for ophyd < v0.8.0
            ret.finished_cb = done_callback
        self._groups[group].add(p_event.wait)
        self._status_objs[group].add(ret)

        return ret

    async def _wait(self, msg):
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
            status_objs = self._status_objs.pop(group)
            try:
                if self.waiting_hook is not None:
                    # Notify the waiting_hook function that the RunEngine is
                    # waiting for these status_objs to complete. Users can use
                    # the information these encapsulate to create a progress
                    # bar.
                    self.waiting_hook(status_objs)
                await self._wait_for(Msg('wait_for', None, futs))
            finally:
                if self.waiting_hook is not None:
                    # Notify the waiting_hook function that we have moved on by
                    # sending it `None`. If all goes well, it could have
                    # inferred this from the status_obj, but there are edge
                    # cases.
                    self.waiting_hook(None)

    def _status_object_completed(self, ret, p_event, pardon_failures):
        """
        Created as a task on the loop when a status object is finished

        Parameters
        ----------
        ret : status object
        p_event : asyncio.Event
            held in the RunEngine's self._groups cache for waiting
        pardon_failuers : asyncio.Event
            tells us whether the __call__ this status object is over
        """
        if not ret.success and not pardon_failures.is_set():
            # TODO: need a better channel to move this information back
            # to the run task.
            with self._state_lock:
                self._exception = FailedStatus(ret)
        p_event.set()

    async def _sleep(self, msg):
        """Sleep the event loop

        Expected message object is:

            Msg('sleep', None, sleep_time)

        where `sleep_time` is in seconds
        """
        await asyncio.sleep(*msg.args, loop=self.loop)

    async def _pause(self, msg):
        """Request the run engine to pause

        Expected message object is:

            Msg('pause', defer=False, name=None, callback=None)

        See RunEngine.request_pause() docstring for explanation of the three
        keyword arguments in the `Msg` signature
        """
        await self._request_pause_coro(*msg.args, **msg.kwargs)

    async def _resume(self, msg):
        """Request the run engine to resume

        Expected message object is:

            Msg('resume', defer=False, name=None, callback=None)

        See RunEngine.resume() docstring for explanation of the three
        keyword arguments in the `Msg` signature
        """
        # Re-instate monitoring callbacks.
        for current_run in self._run_bundlers.values():
            await current_run.restore_monitors()
        # Notify Devices of the resume in case they want to clean up.
        for obj in self._objs_seen:
            if hasattr(obj, 'resume'):
                obj.resume()

    async def _checkpoint(self, msg):
        """Instruct the RunEngine to create a checkpoint so that we can rewind
        to this point if necessary

        Expected message object is:

            Msg('checkpoint')
        """
        for current_run in self._run_bundlers.values():
            if current_run.bundling:
                raise IllegalBundlingSequence(msg, "Cannot 'checkpoint' after "
                                              "'create' and before 'save'. "
                                              "Aborting!")

        await self._reset_checkpoint_state_coro()

        if self._deferred_pause_requested:
            # We are at a checkpoint; we are done deferring the pause.
            # Give the _check_for_signals coroutine time to look for
            # additional SIGINTs that would trigger an abort.
            await asyncio.sleep(0.5, loop=self.loop)
            await self._request_pause_coro(defer=False)

    def _reset_checkpoint_state(self):
        self._reset_checkpoint_state_meth()

    def _reset_checkpoint_state_meth(self):
        if self._msg_cache is None:
            return

        self._msg_cache = deque()
        for current_run in self._run_bundlers.values():
            current_run.reset_checkpoint_state()

    async def _reset_checkpoint_state_coro(self):
        self._reset_checkpoint_state()

    async def _clear_checkpoint(self, msg):
        """Clear a set checkpoint

        Expected message object is:

            Msg('clear_checkpoint')
        """
        # clear message cache
        self._msg_cache = None
        # clear stashed
        for current_run in self._run_bundlers.values():
            await current_run.clear_checkpoint(msg)

    async def _rewindable(self, msg):
        '''Set rewindable state of RunEngine

        Expected message object is:

            Msg('rewindable', None, bool or None)
        '''

        rw_flag, = msg.args
        if rw_flag is not None:
            self.rewindable = rw_flag

        return self.rewindable

    async def _configure(self, msg):
        """Configure an object

        Expected message object is:

            Msg('configure', object, *args, **kwargs)

        which results in this call:

            object.configure(*args, **kwargs)
        """
        run_key = _extract_run_key(msg)
        try:
            current_run = self._run_bundlers[run_key]
        except KeyError:
            current_run = None
        else:
            if current_run.bundling:
                raise IllegalBundlingSequence(
                    msg,
                    "Cannot configure after 'create' but before 'save'. "
                    "Aborting!")

        _, obj, args, kwargs, _ = msg

        old, new = obj.configure(*args, **kwargs)
        if current_run:
            await current_run.configure(msg)
        return old, new

    async def _stage(self, msg):
        """Instruct the RunEngine to stage the object

        Expected message object is:

            Msg('stage', object)
        """
        _, obj, args, kwargs, _ = msg
        # If an object has no 'stage' method, assume there is nothing to do.
        if not hasattr(obj, 'stage'):
            return []
        result = obj.stage()
        self._staged.add(obj)  # add first in case of failure below
        await self._reset_checkpoint_state_coro()
        return result

    async def _unstage(self, msg):
        """Instruct the RunEngine to unstage the object

        Expected message object is:

            Msg('unstage', object)
        """
        _, obj, args, kwargs, _ = msg
        # If an object has no 'unstage' method, assume there is nothing to do.
        if not hasattr(obj, 'unstage'):
            return []
        result = obj.unstage()
        # use `discard()` to ignore objects that are not in the staged set.
        self._staged.discard(obj)
        await self._reset_checkpoint_state_coro()
        return result

    async def _stop(self, msg):
        """
        Stop a device.

        Expected message object is:

            Msg('stop', obj)
        """
        return msg.obj.stop()  # nominally, this returns None

    async def _subscribe(self, msg):
        """
        Add a subscription after the run has started.

        This, like subscriptions passed to __call__, will be removed at the
        end by the RunEngine.

        Expected message object is:

            Msg('subscribe', None, callback_function, document_name)

        where `document_name` is one of:

            {'start', 'descriptor', 'event', 'stop', 'all'}

        and `callback_function` is expected to have a signature of:

            ``f(name, document)``

            where name is one of the ``document_name`` options and ``document``
            is one of the document dictionaries in the event model.

        See the docstring of bluesky.run_engine.Dispatcher.subscribe() for more
        information.
        """
        self.log.debug("Adding subscription %r", msg)
        _, obj, args, kwargs, _ = msg
        token = self.subscribe(*args, **kwargs)
        self._temp_callback_ids.add(token)
        await self._reset_checkpoint_state_coro()
        return token

    async def _unsubscribe(self, msg):
        """
        Remove a subscription during a call -- useful for a multi-run call
        where subscriptions are wanted for some runs but not others.

        Expected message object is:

            Msg('unsubscribe', None, TOKEN)
            Msg('unsubscribe', token=TOKEN)

        where ``TOKEN`` is the return value from ``RunEngine._subscribe()``
        """
        self.log.debug("Removing subscription %r", msg)
        _, obj, args, kwargs, _ = msg
        try:
            token = kwargs['token']
        except KeyError:
            token, = args
        self.unsubscribe(token)
        self._temp_callback_ids.remove(token)
        await self._reset_checkpoint_state_coro()

    async def _input(self, msg):
        """
        Process a 'input' Msg. Expected Msg:

            Msg('input', None)
            Msg('input', None, prompt='>')  # customize prompt
        """
        prompt = msg.kwargs.get('prompt', '')
        async_input = AsyncInput(self.loop)
        async_input = functools.partial(async_input, end='', flush=True)
        return (await async_input(prompt))

    def emit_sync(self, name, doc):
        "Process blocking callbacks and schedule non-blocking callbacks."
        schema_validators[name].validate(doc)
        self.dispatcher.process(name, doc)

    async def emit(self, name, doc):
        self.emit_sync(name, doc)


class Dispatcher:
    """Dispatch documents to user-defined consumers on the main thread."""

    def __init__(self):
        self.cb_registry = CallbackRegistry(allowed_sigs=DocumentNames)
        self._counter = count()
        self._token_mapping = dict()

    def process(self, name, doc):
        """
        Dispatch document ``doc`` of type ``name`` to the callback registry.

        Parameters
        ----------
        name : {'start', 'descriptor', 'event', 'stop'}
        doc : dict
        """
        exceptions = self.cb_registry.process(name, name.name, doc)
        for exc, traceback in exceptions:
            warn("A %r was raised during the processing of a %s "
                 "Document. The error will be ignored to avoid "
                 "interrupting data collection. To investigate, "
                 "set RunEngine.ignore_callback_exceptions = False "
                 "and run again." % (exc, name.name))

    def subscribe(self, func, name='all'):
        """
        Register a callback function to consume documents.

        .. versionchanged :: 0.10.0
            The order of the arguments was swapped and the ``name``
            argument has been given a default value, ``'all'``. Because the
            meaning of the arguments is unambiguous (they must be a callable
            and a string, respectively) the old order will be supported
            indefinitely, with a warning.

        .. versionchanged :: 0.10.0
            The order of the arguments was swapped and the ``name``
            argument has been given a default value, ``'all'``. Because the
            meaning of the arguments is unambiguous (they must be a callable
            and a string, respectively) the old order will be supported
            indefinitely, with a warning.

        Parameters
        ----------
        func: callable
            expecting signature like ``f(name, document)``
            where name is a string and document is a dict
        name : {'all', 'start', 'descriptor', 'event', 'stop'}, optional
            the type of document this function should receive ('all' by
            default).

        Returns
        -------
        token : int
            an integer ID that can be used to unsubscribe

        See Also
        --------
        :meth:`Dispatcher.unsubscribe`
            an integer token that can be used to unsubscribe
        """
        if callable(name) and isinstance(func, str):
            name, func = func, name
            warn("The order of the arguments has been changed. Because the "
                 "meaning of the arguments is unambiguous, the old usage will "
                 "continue to work indefinitely, but the new usage is "
                 "encouraged: call subscribe(func, name) instead of "
                 "subscribe(name, func). Additionally, the 'name' argument "
                 "has become optional. Its default value is 'all'.")
        if name == 'all':
            private_tokens = []
            for key in DocumentNames:
                private_tokens.append(self.cb_registry.connect(key, func))
            public_token = next(self._counter)
            self._token_mapping[public_token] = private_tokens
            return public_token

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
            the integer ID issued by :meth:`Dispatcher.subscribe`

        See Also
        --------
        :meth:`Dispatcher.subscribe`
        """
        for private_token in self._token_mapping[token]:
            self.cb_registry.disconnect(private_token)

    def unsubscribe_all(self):
        """Unregister all callbacks from the dispatcher
        """
        for public_token in self._token_mapping.keys():
            self.unsubscribe(public_token)

    @property
    def ignore_exceptions(self):
        return self.cb_registry.ignore_exceptions

    @ignore_exceptions.setter
    def ignore_exceptions(self, val):
        self.cb_registry.ignore_exceptions = val


PAUSE_MSG = """
Your RunEngine is entering a paused state. These are your options for changing
the state of the RunEngine:

RE.resume()    Resume the plan.
RE.abort()     Perform cleanup, then kill plan. Mark exit_stats='aborted'.
RE.stop()      Perform cleanup, then kill plan. Mark exit_status='success'.
RE.halt()      Emergency Stop: Do not perform cleanup --- just stop.
"""


MAX_DEPTH_EXCEEDED_ERR_MSG = """
RunEngine.max_depth is set to {}; depth of {} was detected.

The RunEngine should not be called from inside another function. Doing so
breaks introspection tools and can result in unexpected behavior in the event
of an interruption. See documentation for more information and what to do
instead:

http://nsls-ii.github.io/bluesky/plans_intro.html#combining-plans
"""


def _default_md_validator(md):
    if 'sample' in md and not (hasattr(md['sample'], 'keys') or
                               isinstance(md['sample'], str)):
        raise ValueError(
            "You specified 'sample' metadata. We give this field special "
            "significance in order to make your data easily searchable. "
            "Therefore, you must make 'sample' a string or a  "
            "dictionary, like so: "
            "GOOD: sample='dirt' "
            "GOOD: sample={'color': 'red', 'number': 5} "
            "BAD: sample=[1, 2] ")


def _ensure_event_loop_running(loop):
    """
    Run an asyncio event loop forever on a background thread.

    This is idempotent: if the loop is already running nothing will be done.
    """
    if not loop.is_running():
        th = threading.Thread(target=loop.run_forever, daemon=True)
        th.start()
        _ensure_event_loop_running.loop_to_thread[loop] = th
    else:
        th = _ensure_event_loop_running.loop_to_thread[loop]
    return th


_ensure_event_loop_running.loop_to_thread = weakref.WeakKeyDictionary()

_bluesky_event_loop = None


def get_bluesky_event_loop():
    global _bluesky_event_loop
    if _bluesky_event_loop is None:
        _bluesky_event_loop = asyncio.new_event_loop()
    return _bluesky_event_loop


def set_bluesky_event_loop(loop):
    global _bluesky_event_loop
    _bluesky_event_loop = loop
