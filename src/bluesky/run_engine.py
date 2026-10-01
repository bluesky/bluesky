import asyncio
import concurrent
import inspect
import threading
import typing
import weakref
from collections.abc import Callable
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from inspect import iscoroutine
from warnings import warn

from bluesky._vendor.super_state_machine.errors import TransitionError
from bluesky._vendor.super_state_machine.extras import ProxyString

from ._loop import run_coro_on_loop, running_on
from .bundlers import RunBundler

# Re-exported, and left out of __all__, so existing imports keep working.
from .dispatcher import Dispatcher, DocumentNames  # noqa: F401
from .log import ComposableLogAdapter, logger

# Re-exported as well.
from .plan_runner import (  # noqa: F401
    NO_PLAN_RETURN,
    UNCACHEABLE_COMMANDS,
    LoggingPropertyMachine,
    PlanRunner,
    RunEngineMetadata,
    RunEngineStateMachine,
    WaitForTimeoutError,
    announce_state_change,
    default_scan_id_source,
    do_nothing,
)
from .plan_session import PlanSession  # noqa: F401
from .protocols import SyncOrAsync, T
from .suspenders import SUBSCRIPTION_TIMEOUT
from .suspension import SuspensionReason, join_justifications
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
)

# Names moved to plan_runner are exported from there.
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


def _panicked_state() -> ProxyString:
    """A 'panicked' ProxyString, whose ``is_*`` checks answer like any other state."""
    machine = RunEngineStateMachine()
    machine.set_("panicked")
    return ProxyString("panicked", machine)


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


def _hook_or_none(hook):
    """What a user set, or ``None`` if they never set one."""
    return None if hook is do_nothing else hook


@dataclass
class RunEngineResult:
    """
    Information about the plan that was run

    Attributes
    ----------
    run_start_uids : list
        A list of the UIDs generated during the plan (if any)
    plan_result :
        The return value of the top-level plan that was run
    exit_status : str
    interrupted : bool
        True if the plan was halted, stopped or aborted
    reason : str
        A text description of the reason why the plan was aborted (if aborted)
    exception :
        The `~bluesky.utils.RequestStop`, `~bluesky.utils.RequestAbort` or
        `~bluesky.utils.PlanHalt` instance that ended a paused plan, if any.
    """

    run_start_uids: tuple[str, ...]
    plan_result: typing.Any
    exit_status: str
    interrupted: bool
    reason: str
    exception: BaseException | None


def _runs_on_loop(*, timeout: float | None = None):
    """Run the decorated `RunEngine` method's body on the engine's loop, and wait.

    Called on the loop, from a plan body or a subscriber, it runs the body there
    and then. ``timeout`` bounds only the wait: the body may still run after the
    caller gives up.
    """

    def decorate(method):
        @wraps(method)
        def crossing(self, *args, **kwargs):
            self._raise_if_panicked()
            if running_on(self.loop):
                return method(self, *args, **kwargs)

            async def work():
                return method(self, *args, **kwargs)

            return run_coro_on_loop(work(), self.loop, timeout=timeout)

        return crossing

    return decorate


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
        The names of the commands available to Msg.

    """

    # Aliases of the module-level constants, for existing callers.
    NO_PLAN_RETURN = NO_PLAN_RETURN
    _UNCACHEABLE_COMMANDS = UNCACHEABLE_COMMANDS

    #: Overridable by subclasses; passed to the session.
    RunBundler = RunBundler

    def _raise_if_panicked(self):
        """Raise if the loop thread is wedged, rather than schedule work it will never run."""
        if self._is_panicked:
            raise RuntimeError("The RunEngine is panicked and cannot be recovered. You must restart bluesky.")

    @property
    def state(self):
        # The engine's own one-way latch; overrides the plan's state.
        if self._is_panicked:
            return _PANICKED_STATE
        return self._runner.state

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
        return self._runner.deferred_pause_requested

    def __init__(
        self,
        md: RunEngineMetadata | None = None,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        preprocessors: list | None = None,
        context_managers: list | None = None,
        md_validator: Callable | None = None,
        md_normalizer: Callable | None = None,
        scan_id_source: Callable[[RunEngineMetadata], SyncOrAsync[int]] = default_scan_id_source,
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

        # Make a logger for this specific RE instance, using the instance's
        # Python id, to keep from mixing output from separate instances.
        log = ComposableLogAdapter(logger, {"RE": self})

        # Set, once, when the loop thread could not be shut down. A plain bool:
        # the loop that would own a state transition is what has stopped.
        self._is_panicked = False

        # The session holds everything that outlives a plan; properties below
        # forward to it.
        self._session = PlanSession(
            md,
            loop=loop,
            log=log,
            # Honour a RunBundler overridden on a subclass, and log state
            # changes and answer Msg('RE_class') as this RunEngine.
            run_bundler_cls=type(self).RunBundler,
            identity=self,
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

        if preprocessors is not None:
            self._session.preprocessors = preprocessors
        if md_validator is not None:
            self._session.md_validator = md_validator
        if md_normalizer is not None:
            self._session.md_normalizer = md_normalizer
        self._session.scan_id_source = scan_id_source
        # Prints; the only half that knows a terminal is watching.
        hooks = self._session.hooks
        hooks.pause_requested = self._announce_pause
        hooks.stop_requested = self._announce_stop
        hooks.suspension_began = self._announce_suspended
        hooks.suspender_joined = self._announce_joined
        hooks.suspender_recovered = self._announce_recovered
        hooks.suspension_refused = self._announce_refused
        hooks.hold_began = self._announce_held
        hooks.plan_paused = self._paused
        # Holds every runner's plan until `_resume_task` has entered the
        # context managers (and so installed SigintHandler). Shut again on a
        # pause.
        self._proceed_permitted = asyncio.Event()
        hooks.may_proceed = self._proceed_permitted.wait

        # The current runner, kept after its plan ends so it can be resumed or
        # inspected. Idle, not None, before the first plan.
        self._runner = self._session._idle_runner()

        # aliases for back-compatibility
        self.subscribe_lossless = self.dispatcher.subscribe
        self.unsubscribe_lossless = self.dispatcher.unsubscribe
        self._subscribe_lossless = self.dispatcher.subscribe
        self._unsubscribe_lossless = self.dispatcher.unsubscribe

    # ------------------------------------------------------------------
    # Forwarded to the session.

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
        # Published for existing callers; the session keeps it private.
        return self._session._dispatcher

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

    # `None` means unset here; `PlanHooks` stores `do_nothing` instead.

    @property
    def msg_hook(self):
        return _hook_or_none(self._session.hooks.msg_received)

    @msg_hook.setter
    def msg_hook(self, value):
        self._session.hooks.msg_received = value

    @property
    def state_hook(self):
        return _hook_or_none(self._session.hooks.state_changed)

    @state_hook.setter
    def state_hook(self, value):
        self._session.hooks.state_changed = value

    @property
    def waiting_hook(self):
        return _hook_or_none(self._session.hooks.waiting_on)

    @waiting_hook.setter
    def waiting_hook(self, value):
        self._session.hooks.waiting_on = value

    @property
    def record_interruptions(self):
        return self._session.record_interruptions

    @record_interruptions.setter
    def record_interruptions(self, value):
        self._session.record_interruptions = value

    # The old spelling of the session's strict_pre_declare, still set by
    # existing code (test_new_examples.py).
    @property
    def _require_stream_declaration(self):
        return self._session.strict_pre_declare

    @_require_stream_declaration.setter
    def _require_stream_declaration(self, value):
        self._session.strict_pre_declare = value

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
        # Names only, in registration order, as before.
        return list(self._session.commands)

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

        for command, docstring in self._session.commands.items():
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
        # The running plan's live value, else the session's default for the
        # next plan.
        if not self._runner.state.is_idle:
            return self._runner.rewindable
        return self._session.rewindable

    @rewindable.setter
    def rewindable(self, v):
        # Both: the session's default for later plans, the runner's for now.
        # Not marshalled onto the loop: a sync setter is atomic there.
        self._session.rewindable = bool(v)
        self._runner.rewindable = bool(v)

    @property
    def loop(self):
        return self._loop

    @property
    def suspenders(self):
        """Every suspender that can suspend the plan in progress: session-scoped and plan-scoped."""
        return tuple(set(self._session.suspenders) | set(self._runner.suspenders))

    @property
    def verbose(self):
        # The adapter, not the logger, was asked here and written to below.
        # `logging.LoggerAdapter` has no `disabled` of its own, so reading it
        # raised until a write had made one, and a write silenced nothing:
        # every level check logging makes goes to `self.logger.disabled`.
        # Disabling reaches the whole `bluesky` logger, which is what it has
        # always claimed to do -- the adapter is per-engine, the logger is not.
        return not self.log.logger.disabled

    @verbose.setter
    def verbose(self, value):
        self.log.logger.disabled = not value

    @property
    def call_returns_result(self):
        return self._call_returns_result

    def _new_runner(self, plan=None, *, metadata=None, subs=None):
        """Replace the runner, discarding the previous plan's state.

        With no plan, the new runner is idle. With one, the plan is held at
        `PlanHooks.may_proceed` until `_resume_task` releases it. The outgoing
        runner's task is cancelled.
        """
        # One plan at a time is this engine's rule, not the session's.
        if not self._runner.state.is_idle:
            raise RuntimeError(
                f"{self._runner!r} is still running a plan, in the "
                f"'{self._runner.state}' state. A RunEngine runs one plan at a time."
            )

        async def build():
            # On the loop, in one crossing, so the plan cannot start before this
            # is arranged. Shut first, so the new plan is held.
            self._proceed_permitted.clear()
            outgoing = self._runner._task
            if outgoing is not None and not outgoing.done():
                # Cancel and wait for an unfinished task. A finished one is not
                # awaited, as that would re-raise the previous plan's error.
                outgoing.cancel()
                with suppress(asyncio.CancelledError):
                    await outgoing
            if plan is None:
                return self._session._idle_runner(), None
            runner = self._session.start(plan, metadata=metadata, subs=subs)

            # A future the main thread can read, carrying the plan's outcome.
            reachable: concurrent.futures.Future = concurrent.futures.Future()

            def finished(task: asyncio.Task) -> None:
                if task.cancelled():
                    reachable.cancel()
                elif task.exception() is not None:
                    reachable.set_exception(task.exception())
                else:
                    reachable.set_result(task.result())
                self._blocking_event.set()

            runner._task.add_done_callback(finished)
            return runner, reachable

        # `run_coro_on_loop` re-raises here, so a malformed plan raises to the
        # caller.
        self._runner, self._task_fut = run_coro_on_loop(build(), self.loop)

    def reset(self):
        """
        Clean up caches and unsubscribe subscriptions.

        Lossless subscriptions are not unsubscribed.
        """
        self._raise_if_panicked()
        if self._runner.state != "idle":
            self.halt()
        self._new_runner()
        self.dispatcher.unsubscribe_all()

    @property
    def resumable(self):
        "i.e., can the plan in progress by rewound"
        return self._runner.resumable

    @property
    def ignore_callback_exceptions(self):
        return self.dispatcher.ignore_exceptions

    @ignore_callback_exceptions.setter
    def ignore_callback_exceptions(self, val):
        # Reaches running plans too.
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
        """Pause without blocking the caller. Used by bluesky-queueserver."""
        await self._runner.pause(defer)

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
        self._raise_if_panicked()
        return run_coro_on_loop(self._runner.pause(defer), self.loop)

    def _create_result(self, plan_return) -> RunEngineResult:
        """Describe how the plan finished, to return from `__call__`."""
        return RunEngineResult(
            tuple(self._runner.run_start_uids),
            plan_return,
            self._runner.exit_status,
            self._runner.interrupted,
            self._runner.exit_reason,
            self._runner.exit_exception,
        )

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
        self._raise_if_panicked()
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
        if not self._runner.state.is_idle:
            raise RuntimeError(f"The RunEngine is in a {self._runner.state} state")

        # A malformed plan raises here. The plan is held at
        # `PlanHooks.may_proceed` until `_resume_task`.
        self._new_runner(plan, metadata=metadata_kw, subs=subs)
        self.log.info("Executing plan %r", plan)

        plan_return = self._resume_task()

        if self._runner.interrupted:
            raise RunEngineInterrupted(self.pause_msg) from None

        if self._call_returns_result:
            run_engine_result = self._create_result(plan_return)
            return run_engine_result
        else:
            return tuple(self._runner.run_start_uids)

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
        self._raise_if_panicked()

        # The state machine does not capture the whole picture.
        if not self._runner.state.is_paused:
            raise TransitionError(
                f"The RunEngine is the {self._runner.state} state. You can only resume for the paused state."
            )

        plan_return = self._resume_task(release=self._runner.resume)
        if self._runner.interrupted:
            raise RunEngineInterrupted(self.pause_msg) from None

        if self._call_returns_result:
            run_engine_result = self._create_result(plan_return)
            return run_engine_result
        else:
            return tuple(self._runner.run_start_uids)

    def _paused(self):
        """Hold the plan at `PlanHooks.may_proceed`, and release the main thread."""
        self._proceed_permitted.clear()
        self._blocking_event.set()

    def _resume_task(self, *, release=None):
        # Clear the blocking Event so that we can wait on it below.
        # The task will set it when it is done, as it was previously
        # configured to do it __call__.
        self._blocking_event.clear()

        # Handle all context managers
        with ExitStack() as stack:
            for mgr in self.context_managers:
                stack.enter_context(mgr(self))

            # Inside the context managers, so SigintHandler is installed first.
            async def proceed():
                if release is not None:
                    await release()
                self._proceed_permitted.set()

            run_coro_on_loop(proceed(), self.loop)

            if self._task_fut is None:
                # No task was ever started; nothing to wait on or return.
                return self.NO_PLAN_RETURN
            if self._task_fut.done():
                try:
                    return self._task_fut.result()
                except concurrent.futures.CancelledError:
                    return NO_PLAN_RETURN
            try:
                # Block until plan is complete or exception is raised.
                try:
                    self._during_task.block(self._blocking_event)
                except KeyboardInterrupt:
                    import ctypes

                    self._runner.interrupted = True
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
                        old_state = self._runner.state
                        self._is_panicked = True
                        # The runner's machine lives on the dead loop, so
                        # announce the change by hand.
                        announce_state_change(self, self._session.hooks, old_state, "panicked")
                except Exception as raised_er:
                    self.halt()
                    self._runner.interrupted = True
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

    def _announce_held(self, reasons: typing.Mapping[typing.Hashable, SuspensionReason], at_start: bool) -> None:
        """Say what is holding the plan up, and that it will wait for it."""
        verb = "begin" if at_start else "continue"
        print(
            f"At least one suspender has tripped. The plan will {verb} "
            "when all suspenders are ready. Justification:"
        )
        for i, reason in enumerate(reasons.values()):
            print(f"    {i + 1}. {reason.justification}")
        print()
        print("Suspending... To get to the prompt, hit Ctrl-C twice to pause.")

    def _announce_pause(self, deferred: bool) -> None:
        """Say a pause was asked for."""
        print("Deferred pause acknowledged. Continuing to checkpoint." if deferred else "Pausing...")

    def _announce_stop(self, success: bool, finalize: bool) -> None:
        """Say the plan was asked to end, and how."""
        verb = "Stopping" if success else "Aborting" if finalize else "Halting"
        cleanup = "running cleanup" if finalize else "skipping cleanup"
        exit_status = "success" if success else "abort"
        print(f"{verb}: {cleanup} and marking exit_status as {exit_status!r}...")

    def _announce_refused(self, reasons: typing.Mapping[typing.Hashable, SuspensionReason]) -> None:
        """Say a suspension could not begin, so the plan is aborted."""
        print("No checkpoint; cannot suspend.")
        print("Aborting: running cleanup and marking exit_status as 'abort'...")

    def _announce_joined(self, key: typing.Hashable, reason: SuspensionReason) -> None:
        """Say another suspender has joined the suspension."""
        print(f"Another suspender has also tripped. Justification: {reason.justification}")

    def _announce_recovered(self, key: typing.Hashable, reason: SuspensionReason) -> None:
        """Say a suspender has recovered, and when its reason will be dropped."""
        sleep = reason.settle_time
        # Set on every recovered reason.
        assert sleep is not None
        ts = (datetime.now() + timedelta(seconds=sleep)).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"Suspender {key!r} reports a return to nominal "
            f"conditions. Will sleep for {sleep} seconds and then "
            f"release suspension at {ts}."
        )

    def _announce_suspended(self, reasons: typing.Mapping[typing.Hashable, SuspensionReason]) -> None:
        """Say a suspension has begun, and how to get back to a prompt."""
        print("Suspending....To get prompt hit Ctrl-C twice to pause.")
        print(f"Suspension occurred at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        justification = join_justifications(reasons)
        if justification:
            print(f"Justification for this suspension:\n{justification}")

    @_runs_on_loop(timeout=SUBSCRIPTION_TIMEOUT)
    def install_suspender(self, suspender):
        """
        Install a session-scoped 'suspender', which can suspend and resume execution.

        It holds up every plan this engine runs until removed. One installed
        with ``Msg('install_suspender', None, suspender)`` holds up only that
        plan, and ends with it.

        Parameters
        ----------
        suspender : `bluesky.suspenders.SuspenderBase`

        See Also
        --------
        :meth:`RunEngine.remove_suspender`
        :meth:`RunEngine.clear_suspenders`
        """
        # Reaches a running plan too, through its suspension's parent.
        self._session.install_suspender(suspender)

    @_runs_on_loop(timeout=SUBSCRIPTION_TIMEOUT)
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

    @_runs_on_loop(timeout=SUBSCRIPTION_TIMEOUT)
    def clear_suspenders(self):
        """
        Uninstall all suspenders, session-scoped and the running plan's plan-scoped ones.

        See Also
        --------
        :meth:`RunEngine.install_suspender`
        :meth:`RunEngine.remove_suspender`
        """
        # The session cannot remove plan-scoped ones, so ask each.
        self._session.clear_suspenders()
        self._runner.clear_suspenders()

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
        return self.__interrupter_helper(self._runner.abort(reason))

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
        return self.__interrupter_helper(self._runner.stop())

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
        return self.__interrupter_helper(self._runner.halt())

    def __interrupter_helper(self, coro):
        if self._is_panicked:
            # Close it, or it warns "never awaited" at collection.
            coro.close()
        self._raise_if_panicked()

        was_paused = self._runner.state == "paused"
        # No timeout: giving up would leave the plan running.
        run_coro_on_loop(coro, self.loop)
        # Before a paused plan's cleanup can change it.
        result = self._interrupted_result()
        if was_paused:
            self._resume_task()

        return result

    def _interrupted_result(self):
        """What abort(), stop() and halt() return."""
        if self._call_returns_result:
            return self._create_result(NO_PLAN_RETURN)
        return tuple(self._runner.run_start_uids)

    def emit_sync(self, name, doc):
        "Process blocking callbacks and schedule non-blocking callbacks."

        # Process the doc, already validated against the schema in event-model
        self.dispatcher.process(name, doc)

    async def emit(self, name, doc):
        self.emit_sync(name, doc)


# Private names that moved to the runner, and the test file that reads each.
# Silent, because the test suite makes warnings errors.
_RUNNER_FORWARDS = {
    "_task": "_task",  # tests/test_run_engine.py
    "_run_bundlers": "_run_bundlers",  # tests/test_run_engine.py
    "_run_start_uids": "run_start_uids",  # tests/test_plan_runner.py
    "_seen_wait_and_move_on_keys": "_seen_wait_and_move_on_keys",  # tests/test_flyer.py
    "_command_registry": "_command_registry",  # tests/test_run_engine.py
    "_msg_cache": "_msg_cache",  # tests/test_run_engine.py
    "_exception": "_exception",  # tests/test_suspensions.py
    "_exit_status": "exit_status",  # tests/test_plan_runner.py
}


def _forward_to_runner(name: str) -> property:
    """A property reading and writing ``name`` on the current runner."""

    def getter(self):
        return getattr(self._runner, name)

    def setter(self, value):
        setattr(self._runner, name, value)

    return property(getter, setter, doc=f"Forwards to :attr:`PlanRunner.{name}`.")


for _old_name, _new_name in _RUNNER_FORWARDS.items():
    setattr(RunEngine, _old_name, _forward_to_runner(_new_name))
del _old_name, _new_name


# Driving the loop from outside it: a background thread, the prompt, IPython.


def _ensure_event_loop_running(loop):
    """
    Run an asyncio event loop forever on a background thread.

    This is idempotent: if the loop is already running nothing will be done.
    """
    if not loop.is_running():
        th = threading.Thread(target=loop.run_forever, daemon=True, name="bluesky-run-engine")
        th.start()
        _ensure_event_loop_running.loop_to_thread[loop] = th
    else:
        th = _ensure_event_loop_running.loop_to_thread[loop]
    return th


_ensure_event_loop_running.loop_to_thread = weakref.WeakKeyDictionary()  # type: ignore

_bluesky_event_loop = None


def get_bluesky_event_loop():
    return _bluesky_event_loop


def set_bluesky_event_loop(loop):
    global _bluesky_event_loop
    _bluesky_event_loop = loop


def in_bluesky_event_loop() -> bool:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Ok, no running loop
        return False
    else:
        # Check if running loop is bluesky event loop
        return loop is _bluesky_event_loop


def call_in_bluesky_event_loop(coro: typing.Awaitable[T], timeout: float | None = None) -> T:
    if _bluesky_event_loop is None or not _bluesky_event_loop.is_running():
        # Quell "coroutine never awaited" warnings
        if iscoroutine(coro):
            coro.close()
        raise RuntimeError("Bluesky event loop not running")
    return run_coro_on_loop(coro, _bluesky_event_loop, timeout=timeout)


def autoawait_in_bluesky_event_loop(ip=None):
    if ip is None:
        import IPython

        ip = IPython.get_ipython()  # type: ignore
    assert ip, "Couldn't import IPython"
    ip.loop_runner = call_in_bluesky_event_loop
