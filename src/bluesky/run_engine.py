import asyncio
import concurrent
import inspect
import threading
import typing
from contextlib import ExitStack
from warnings import warn

from bluesky._vendor.super_state_machine.errors import TransitionError

from .bundlers import RunBundler
from .log import ComposableLogAdapter, logger

# Everything this module used to define now lives in plan_executor, and is
# exported from there. These are imported, and deliberately left out of the
# __all__ below, so that `from bluesky.run_engine import Dispatcher` and its
# like keep working for code written before the split.
from .plan_executor import (  # noqa: F401
    NO_PLAN_RETURN,
    UNCACHEABLE_COMMANDS,
    Dispatcher,
    DocumentNames,
    LoggingPropertyMachine,
    PlanExecutor,
    PlanSession,
    RunEngineMetadata,
    RunEngineResult,
    RunEngineStateMachine,
    WaitForTimeoutError,
    _ensure_event_loop_running,
    _panicked_state,
    announce_state_change,
    announce_suspend,
    autoawait_in_bluesky_event_loop,
    call_in_bluesky_event_loop,
    default_scan_id_source,
    get_bluesky_event_loop,
    in_bluesky_event_loop,
    set_bluesky_event_loop,
)
from .protocols import SyncOrAsync
from .utils import (
    DefaultDuringTask,
    DuringTask,
    FailedPause,
    FailedStatus,
    IllegalMessageSequence,
    InvalidCommand,
    Msg,
    NoReplayAllowed,
    PlanHalt,
    RequestAbort,
    RequestStop,
    RunEngineInterrupted,
    SigintHandler,
    Subscribers,
    single_gen,
)

# What this module defines, plus the names it has always passed through from
# bluesky.utils and event_model. The names that moved to plan_executor are
# exported from there, not here.
__all__ = [
    "MAX_DEPTH_EXCEEDED_ERR_MSG",
    "PAUSE_MSG",
    "DocumentNames",
    "FailedPause",
    "FailedStatus",
    "IllegalMessageSequence",
    "InvalidCommand",
    "Msg",
    "NoReplayAllowed",
    "PlanHalt",
    "RequestAbort",
    "RequestStop",
    "RunEngine",
    "RunEngineInterrupted",
    "TransitionError",
]


class _RunEnginePanic(Exception): ...


#: What `RunEngine.state` reports once this engine has panicked.
_PANICKED_STATE = _panicked_state()


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


class RunEngine:
    """The Run Engine execute messages and emits Documents.

    Parameters
    ----------
    md : MutableMapping[str, Any], optional
        The default is a standard Python dictionary, but fancier
        objects can be used to store long-term history and persist
        it between sessions. Any object adhering to the MutableMapping
        Protocol will work.

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
        Expected signature: f(md: MutableMapping[str, Any])
        Function should raise if md is invalid. What that means is
        completely up to the user. The function's return value is
        ignored.

    md_normalizer : callable, optional
        a function that, similar to md_validator, raises and prevents starting
        a run if it deems the metadata to be invalid or incomplete.
        If it succeeds, it returns the normalized/transformed version of
        the original metadata.
        Expected signature: f(md: MutableMapping[str, Any]) -> MutableMapping[str, Any]
        Function should raise if md is invalid. What that means is
        completely up to the user.
        Expected return: normalized metadata

    scan_id_source : callable, optional
        a (possibly async) function that will be used to calculate scan_id.
        Default is to increment scan_id by 1 each time. However you could pass
        in a customized function to get a scan_id from any source.
        Expected signature: f(md)
        Expected return: updated scan_id value

    during_task : reference to an object of class DuringTask, optional
        Class methods: ``block()`` to be run to block
        the main thread during `RE.__call__`

        The required signatures for the class methods ::

              def block(ev: Threading.Event) -> None:
                  "Returns when ev is set"

        The default value handles the cases of:
           - Matplotlib is not imported (just wait on the event)
           - Matplotlib is imported, but not using a Qt, notebook or ipympl
             backend (just wait on the event)
           - Matplotlib is imported and using a Qt backend (run the Qt app
             on the main thread until the run finishes)
           - Matplotlib is imported and using a nbagg or ipympl backend (
             wait on the event and poll to push updates to the browser)

    call_returns_result : bool, default False
        A flag that controls the return value of __call__
        If ``True``, the ``RunEngine`` will return a :class:``RunEngineResult``
        object that contains information about the plan that was run.
        If ``False``, the ``RunEngine`` will return a tuple of uids.
        Defaults to ``False`` to preserve the old ``RunEngine`` behavior,
        but the default is expected to change to ``True`` in the future.

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

    call_returns_result
        Boolean, False by default. If False, RunEngine will return uuid list
        after running a plan. If True, RunEngine will return a RunEngineResult
        object that contains the plan result, error status, and uuid list.

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

    # Aliases of the module-level constants, kept so that
    # RunEngine.NO_PLAN_RETURN and RunEngine._UNCACHEABLE_COMMANDS keep working.
    NO_PLAN_RETURN = NO_PLAN_RETURN
    _UNCACHEABLE_COMMANDS = UNCACHEABLE_COMMANDS

    #: Overridable by subclasses; copied onto the session on construction.
    RunBundler = RunBundler

    @property
    def state(self):
        # 'panicked' is the RunEngine's own state, not the session's: it means
        # this engine's loop thread is wedged. It is a one-way latch, so it
        # trumps whatever the plan's state machine last recorded.
        if self._is_panicked:
            return _PANICKED_STATE
        return self._session.state

    @property
    def deferred_pause_requested(self):
        """
        The property returns ``True`` if deferred pause was requested, but
        not processed. The deferred pause is processed at the next checkpoint.
        If the pause is requested past the last checkpoint, the plan runs
        to completion and this property returns ``True`` until the next
        plan is started. Starting the next plan clears deferred pause request.

        Returns
        -------
        boolean
            Indicates if deferred pause was requested, but not processed.
        """
        return self._executor._deferred_pause_requested

    def __init__(
        self,
        md: RunEngineMetadata | None = None,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        preprocessors: list | None = None,
        context_managers: list | None = None,
        md_validator: typing.Callable | None = None,
        md_normalizer: typing.Callable | None = None,
        scan_id_source: typing.Callable[[RunEngineMetadata], SyncOrAsync[int]] = default_scan_id_source,
        during_task: DuringTask | None = None,
        call_returns_result: bool = False,
    ):
        if loop is None:
            loop = asyncio.new_event_loop()
        set_bluesky_event_loop(loop)
        self._th = _ensure_event_loop_running(loop)
        self._loop = loop
        # When set, RunEngine.__call__ should stop blocking.
        self._blocking_event = threading.Event()
        # Set when this engine's loop thread could not be shut down and the
        # engine is unusable. Written once, from the main thread, and never
        # cleared; see _resume_task. A plain bool rather than a state machine
        # transition because the loop it would have to be marshalled onto is
        # precisely the thing that has stopped responding.
        self._is_panicked = False

        # Make a logger for this specific RE instance, using the instance's
        # Python id, to keep from mixing output from separate instances.
        log = ComposableLogAdapter(logger, {"RE": self})

        # Everything that outlives a single plan lives on the session, which
        # this RunEngine drives but does not otherwise own. The properties
        # below forward to it, so RE.md, RE.state and friends are unchanged.
        self._session = PlanSession(
            md,
            loop=loop,
            preprocessors=preprocessors,
            md_validator=md_validator,
            md_normalizer=md_normalizer,
            scan_id_source=scan_id_source,
            log=log,
            on_pause=self._blocking_event.set,
            # Honour a RunBundler overridden on a RunEngine subclass, and let
            # Msg('RE_class') keep reporting the RunEngine rather than the
            # executor that happens to be running the plan.
            run_bundler_cls=type(self).RunBundler,
            run_engine_cls=type(self),
        )

        if context_managers is None:
            context_managers = [SigintHandler]
        self.context_managers = context_managers

        self.max_depth = None
        self.pause_msg = PAUSE_MSG

        if during_task is None:
            during_task = DefaultDuringTask()
        self._during_task = during_task

        self._call_returns_result = call_returns_result  # should __call__ return UIDs or plan value
        self._task_fut = None  # future proxy to the task running the plan

        # Everything belonging to the execution of a single plan lives on an
        # executor. A new one is built for each __call__ and kept afterwards,
        # so that a paused plan can be resumed and a finished one inspected.
        # The forwarding properties installed at the bottom of this module
        # keep RE._msg_cache, RE._task and the rest pointing at it. The session
        # built one as it was constructed, so adopt that rather than replacing
        # it with an identical one.
        self._executor = self._session.executor

        # aliases for back-compatibility
        self.subscribe_lossless = self.dispatcher.subscribe
        self.unsubscribe_lossless = self.dispatcher.unsubscribe
        self._subscribe_lossless = self.dispatcher.subscribe
        self._unsubscribe_lossless = self.dispatcher.unsubscribe

    def _rebuild_command_registry(self):
        """Recompose the executor's vocabulary after a registration change."""
        self._executor.rebuild_command_registry()

    # ------------------------------------------------------------------
    # Forwarded to the session, which owns everything that outlives a plan.

    @property
    def log(self):
        return self._session.log

    @property
    def md(self):
        return self._session.md

    @md.setter
    def md(self, value):
        self._session.md = value

    @property
    def dispatcher(self):
        return self._session.dispatcher

    @property
    def preprocessors(self):
        return self._session.preprocessors

    @preprocessors.setter
    def preprocessors(self, value):
        self._session.preprocessors = value

    @property
    def md_validator(self):
        return self._session.md_validator

    @md_validator.setter
    def md_validator(self, value):
        self._session.md_validator = value

    @property
    def md_normalizer(self):
        return self._session.md_normalizer

    @md_normalizer.setter
    def md_normalizer(self, value):
        self._session.md_normalizer = value

    @property
    def scan_id_source(self):
        return self._session.scan_id_source

    @scan_id_source.setter
    def scan_id_source(self, value):
        self._session.scan_id_source = value

    @property
    def msg_hook(self):
        return self._session.msg_hook

    @msg_hook.setter
    def msg_hook(self, value):
        self._session.msg_hook = value

    @property
    def state_hook(self):
        return self._session.state_hook

    @state_hook.setter
    def state_hook(self, value):
        self._session.state_hook = value

    @property
    def waiting_hook(self):
        return self._session.waiting_hook

    @waiting_hook.setter
    def waiting_hook(self, value):
        self._session.waiting_hook = value

    @property
    def record_interruptions(self):
        return self._session.record_interruptions

    @record_interruptions.setter
    def record_interruptions(self, value):
        self._session.record_interruptions = value

    @property
    def _require_stream_declaration(self):
        return self._session._require_stream_declaration

    @_require_stream_declaration.setter
    def _require_stream_declaration(self, value):
        self._session._require_stream_declaration = value

    @property
    def commands(self):
        """
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
        """
        # return as a list, not lazy loader, no surprises...
        return list(self._executor.command_registry.keys())

    def print_command_registry(self, verbose=False):
        """
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
        """
        commands = "List of available commands\n"

        for command, func in self._executor.command_registry.items():
            docstring = func.__doc__
            if not verbose:
                docstring = docstring.split("\n")[0]
            commands = commands + f"{command} : {docstring}\n"

        return commands

    def subscribe(self, func, name="all"):
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
        return self._executor.rewindable_flag

    @rewindable.setter
    def rewindable(self, v):
        # Setting this discards the message cache and resets every open
        # bundler's checkpoint state, which belongs to the plan in progress.
        # It is not marshalled onto the event loop, and does not need to be:
        # this is a plain sync setter with no await in it, so against every
        # other task on the loop it is atomic. Reaching a RunEngine from a
        # second thread is not supported, and while a plan is running the
        # main thread is blocked inside __call__ and cannot get here anyway.
        cur_state = self._executor.rewindable_flag
        self._executor.rewindable_flag = bool(v)
        if self.resumable and self._executor.rewindable_flag != cur_state:
            self._executor._reset_checkpoint_state()

    @property
    def loop(self):
        return self._loop

    @property
    def suspenders(self):
        return self._session.suspenders

    @property
    def verbose(self):
        return not self.log.disabled

    @verbose.setter
    def verbose(self, value):
        self.log.disabled = not value

    @property
    def call_returns_result(self):
        return self._call_returns_result

    def _new_executor(self, subs=None):
        """Start a fresh executor, discarding the state of the previous plan.

        Building a new one is how the caches are cleared: there is no list of
        things to remember to reset. The session owns the construction, and
        tears down the previous plan's temporary subscriptions as it goes.
        """
        self._executor = self._session.new_executor(subs)
        self._task_fut = None

    def _clear_run_cache(self):
        "Deprecated. Clean up for a new run."
        self._new_executor()

    def _clear_call_cache(self):
        "Deprecated. Clean up for a new __call__."
        self._new_executor()

    def reset(self):
        """
        Clean up caches and unsubscribe subscriptions.

        Lossless subscriptions are not unsubscribed.
        """
        if self._session.state != "idle":
            self.halt()
        self._new_executor()
        self.dispatcher.unsubscribe_all()

    @property
    def resumable(self):
        "i.e., can the plan in progress by rewound"
        return self._executor.resumable

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
        self._session.register_command(name, func)

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
        self._session.unregister_command(name)

    async def _request_pause_coro(self, defer=False):
        """Pause without blocking the caller.

        Kept because bluesky-queueserver calls it directly: its worker cannot
        use the blocking `request_pause` below, which never returns if the
        event loop is wedged, and there is no public non-blocking equivalent.
        """
        await self._executor.request_pause(defer)

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
        if self.state == "panicked":
            raise RuntimeError("The RunEngine is panicked and cannot be recovered. You must restart bluesky.")
        future = asyncio.run_coroutine_threadsafe(self._executor.request_pause(defer), loop=self.loop)
        # TODO add a timeout here?
        return future.result()

    def _create_result(self, plan_return):
        """
        Create a RunEngineResult to return from __call__, using
        plan_return and internal state
        """
        return self._executor.result(plan_return)

    def __call__(
        self,
        plan: typing.Iterable[Msg],
        subs: Subscribers | None = None,
        /,
        **metadata_kw: typing.Any,
    ) -> RunEngineResult | tuple[str, ...]:
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
        uids : tuple
            list of uids (i.e. RunStart Document uids) of run(s)
            if :attr:`RunEngine._call_returns_result` is ``False``
        result : :class:`RunEngineResult`
            if :attr:`RunEngine._call_returns_result` is ``True``
        """
        if self.state == "panicked":
            raise RuntimeError("The RunEngine is panicked and cannot be recovered. You must restart bluesky.")
        if "raise_if_interrupted" in metadata_kw:
            warn(  # noqa: B028
                "The 'raise_if_interrupted' flag has been removed. The "
                "RunEngine now always raises RunEngineInterrupted if it is "
                "interrupted. The 'raise_if_interrupted' keyword argument, "
                "like all keyword arguments, will be interpreted as "
                "metadata."
            )
        # Check that the RE is not being called from inside a function.
        if self.max_depth is not None:
            frame = inspect.currentframe()
            depth = len(inspect.getouterframes(frame))
            if depth > self.max_depth:
                text = MAX_DEPTH_EXCEEDED_ERR_MSG.format(self.max_depth, depth)
                raise RuntimeError(text)

        # If we are in the wrong state, raise.
        if not self._session.state.is_idle:
            raise RuntimeError(f"The RunEngine is in a {self._session.state} state")

        futs = []
        tripped_justifications = []
        for sup in self.suspenders:
            f_lst, justification = sup.get_futures()
            if f_lst:
                futs.extend(f_lst)
                tripped_justifications.append(justification)

        if tripped_justifications:
            print(
                "At least one suspender has tripped. The plan will begin "
                "when all suspenders are ready. Justification:"
            )
            for i, justification in enumerate(tripped_justifications):
                print(f"    {i + 1}. {justification}")

            print()
            print("Suspending... To get to the prompt, hit Ctrl-C twice to pause.")

        self._new_executor(subs)
        self.log.info("Executing plan %r", plan)

        def _build_task():
            # make sure _run will block at the top
            self._executor.block_run()
            self._blocking_event.clear()
            self._task_fut = asyncio.run_coroutine_threadsafe(
                self._executor.run(
                    plan,
                    metadata=metadata_kw,
                    # Wait for any already-tripped suspenders before starting.
                    prologue=single_gen(Msg("wait_for", None, futs)) if futs else None,
                ),
                loop=self.loop,
            )

            def set_blocking_event(future):
                self._blocking_event.set()

            self._task_fut.add_done_callback(set_blocking_event)

        plan_return = self._resume_task(init_func=_build_task)

        if self._executor.interrupted:
            raise RunEngineInterrupted(self.pause_msg) from None

        if self._call_returns_result:
            run_engine_result = self._create_result(plan_return)
            return run_engine_result
        else:
            return tuple(self._executor.run_start_uids)

    def resume(self):
        """Resume a paused plan from the last checkpoint.

        Returns
        -------
        uids : list
            list of uids (i.e. RunStart Document uids) of run(s)
            if :attr:`RunEngine._call_returns_result` is ``False``
        result : :class:`RunEngineResult`
            if :attr:`RunEngine._call_returns_result` is ``True``
        """
        if self.state == "panicked":
            raise RuntimeError("The RunEngine is panicked and cannot be recovered. You must restart bluesky.")

        # The state machine does not capture the whole picture.
        if not self._session.state.is_paused:
            raise TransitionError(
                f"The RunEngine is the {self._session.state} state. You can only resume for the paused state."
            )

        asyncio.run_coroutine_threadsafe(self._executor.prepare_resume(), self._loop).result()
        plan_return = self._resume_task()
        if self._executor.interrupted:
            raise RunEngineInterrupted(self.pause_msg) from None

        if self._call_returns_result:
            run_engine_result = self._create_result(plan_return)
            return run_engine_result
        else:
            return tuple(self._executor.run_start_uids)

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
                try:
                    return self._task_fut.result()
                except concurrent.futures.CancelledError:
                    return NO_PLAN_RETURN
            # The _run task is waiting on this Event. Let is continue.
            self.loop.call_soon_threadsafe(self._executor.permit_run)
            try:
                # Block until plan is complete or exception is raised.
                try:
                    self._during_task.block(self._blocking_event)
                except KeyboardInterrupt:
                    import ctypes

                    self._executor.interrupted = True
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
                        ctypes.c_ulong(self._th.ident), ctypes.py_object(_RunEnginePanic)
                    )
                    # however, if the thread is in a system call (such
                    # as sleep or I/O) there is no way to interrupt it
                    # (per decree of Guido) thus we give it a second
                    # to sort it's self out
                    task_finished = self._blocking_event.wait(1)
                    # before giving up and putting the RE in a
                    # non-recoverable panicked state.
                    if not task_finished or num_threads != 1:
                        old_state = self._session.state
                        self._is_panicked = True
                        # The session's machine is untouched -- it belongs to
                        # the loop -- so announce the change by hand, so that
                        # a state_hook driving a display still sees the panic
                        # rather than watching the state freeze.
                        announce_state_change(self._session, old_state, "panicked")
                except Exception as raised_er:
                    self.halt()
                    self._executor.interrupted = True
                    raise raised_er
            finally:
                if self._task_fut.done():
                    # get exceptions from the main task
                    try:
                        exc = self._task_fut.exception()
                    except (asyncio.CancelledError, concurrent.futures.CancelledError):
                        exc = None
                    # Only try to get a result if there wasn't an error,
                    # (other than a cancelled error)
                    if exc is None:
                        try:
                            plan_return = self._task_fut.result()
                        except concurrent.futures.CancelledError:
                            plan_return = NO_PLAN_RETURN
                    # we have something in exc
                    else:
                        # special case the panic exception that we put in above
                        if isinstance(exc, _RunEnginePanic):
                            plan_return = NO_PLAN_RETURN
                        # otherwise re-raise it
                        else:
                            raise exc
                else:
                    plan_return = None
            return plan_return

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
        self._session.install_suspender(suspender)

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
        self._session.remove_suspender(suspender)

    def clear_suspenders(self):
        """
        Uninstall all suspenders.

        See Also
        --------
        :meth:`RunEngine.install_suspender`
        :meth:`RunEngine.remove_suspender`
        """
        self._session.clear_suspenders()

    def request_suspend(self, fut, *, pre_plan=None, post_plan=None, justification=None):
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
        # Announce on the calling thread, so the message arrives when the
        # caller asked rather than whenever the loop gets to it. Then straight
        # to the plan: going by way of the session would only find the
        # executor this already has, and would announce it a second time.
        announce_suspend()
        asyncio.run_coroutine_threadsafe(
            self._executor.request_suspend(
                fut, pre_plan=pre_plan, post_plan=post_plan, justification=justification
            ),
            self.loop,
        )

    def abort(self, reason=""):
        """
        Stop a running or paused plan and mark it as aborted.

        Returns
        -------
        uids : tuple
            list of uids (i.e. RunStart Document uids) of run(s)
            if :attr:`RunEngine._call_returns_result` is ``False``
        result : :class:`RunEngineResult`
            if :attr:`RunEngine._call_returns_result` is ``True``

        See Also
        --------
        :meth:`RunEngine.halt`
        :meth:`RunEngine.stop`
        """
        return self.__interrupter_helper(self._executor.abort(reason))

    def stop(self):
        """
        Stop a running or paused plan, but mark it as successful (not aborted).

        Returns
        -------
        uids : tuple
            list of uids (i.e. RunStart Document uids) of run(s)
            if :attr:`RunEngine._call_returns_result` is ``False``
        result : :class:`RunEngineResult`
            if :attr:`RunEngine._call_returns_result` is ``True``

        See Also
        --------
        :meth:`RunEngine.abort`
        :meth:`RunEngine.halt`
        """
        return self.__interrupter_helper(self._executor.stop())

    def halt(self):
        """
        Stop the running plan and do not allow the plan a chance to clean up.

        Returns
        -------
        uids : tuple
            list of uids (i.e. RunStart Document uids) of run(s)
            if :attr:`RunEngine._call_returns_result` is ``False``
        result : :class:`RunEngineResult`
            if :attr:`RunEngine._call_returns_result` is ``True``

        See Also
        --------
        :meth:`RunEngine.abort`
        :meth:`RunEngine.stop`
        """
        return self.__interrupter_helper(self._executor.halt())

    def __interrupter_helper(self, coro):
        if self.state == "panicked":
            coro.close()
            raise RuntimeError("The RunEngine is panicked and cannot be recovered. You must restart bluesky.")

        coro_event = threading.Event()
        task = None

        def end_cb(fut):
            coro_event.set()

        def start_task():
            nonlocal task
            task = self.loop.create_task(coro)
            task.add_done_callback(end_cb)

        was_paused = self._session.state == "paused"
        self.loop.call_soon_threadsafe(start_task)
        coro_event.wait()
        # Re-raise anything the coroutine raised, e.g. a TransitionError.
        task.result()
        # Describe the outcome before resuming, since resuming lets the plan
        # run its cleanup and change what we would report.
        result = self._interrupted_result()
        if was_paused:
            self._resume_task()

        return result

    def _interrupted_result(self):
        """What abort(), stop() and halt() return."""
        if self._call_returns_result:
            return self._create_result(NO_PLAN_RETURN)
        return tuple(self._executor.run_start_uids)

    # Emission belongs to the executor now, which hands a document to the
    # session's subscribers and then to the plan's. Both of these are kept for
    # callers written against the older pair. `emit` stays a coroutine while
    # `PlanExecutor.emit` is not; awaiting it never suspended, because the body
    # never awaited anything, so the difference costs nothing.
    def emit_sync(self, name, doc):
        """Give a document to every subscriber."""
        self._executor.emit(name, doc)

    async def emit(self, name, doc):
        """Give a document to every subscriber."""
        self._executor.emit(name, doc)


# Names the RunEngine used to hold itself, which now belong to the executor for
# the plan being run. Mapped to the executor attribute they forward to.
#
# These are private, but they are read and written by tests and by downstream
# code, so they keep working. Forwarding silently is deliberate: the test suite
# turns warnings into errors, so a DeprecationWarning here would break callers
# rather than warn them. One can be added once the ecosystem has moved to
# reading RunEngine._executor, or to using a PlanExecutor directly.

# Forwards with a caller we can point at, inside bluesky or outside it.
_FORWARDS_WITH_CALLERS = {
    "_task": "_task",
    "_run_bundlers": "_run_bundlers",
    "_run_start_uids": "run_start_uids",
    "_seen_wait_and_move_on_keys": "_seen_wait_and_move_on_keys",
    "_deferred_pause_requested": "_deferred_pause_requested",
    "_command_registry": "command_registry",
}

# Forwards with no caller we could find, carried as insurance rather than
# because anything needs them. Searched across bluesky, its tests and docs,
# bluesky-queueserver and blueapi when this was written; the pull request that
# introduced this split records what was searched and what was found. Delete
# this dict and its term in the union below if you would rather not carry them.
_FORWARDS_WITHOUT_KNOWN_CALLERS = {
    "_run_permit": "_run_permit",
    "_pardon_failures": "_pardon_failures",
    "_plan": "_plan",
    "_plan_stack": "_plan_stack",
    "_response_stack": "_response_stack",
    "_msg_cache": "_msg_cache",
    "_rewindable_flag": "rewindable_flag",
    "_metadata_per_call": "_metadata_per_call",
    "_run_tracing_spans": "_run_tracing_spans",
    "_staged": "_staged",
    "_objs_seen": "_objs_seen",
    "_movable_objs_touched": "_movable_objs_touched",
    "_groups": "_groups",
    "_status_objs": "_status_objs",
    "_exception": "exception",
    "_interrupted": "interrupted",
    "_exit_status": "exit_status",
    "_reason": "reason",
}

_EXECUTOR_FORWARDS = _FORWARDS_WITH_CALLERS | _FORWARDS_WITHOUT_KNOWN_CALLERS


def _forward_to_executor(name: str) -> property:
    """A property reading and writing ``name`` on the current executor."""

    def getter(self):
        return getattr(self._executor, name)

    def setter(self, value):
        setattr(self._executor, name, value)

    return property(getter, setter, doc=f"Forwards to :attr:`PlanExecutor.{name}`.")


for _old_name, _new_name in _EXECUTOR_FORWARDS.items():
    setattr(RunEngine, _old_name, _forward_to_executor(_new_name))
del _old_name, _new_name
