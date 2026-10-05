"""The environment plans are executed in, and what outlives any one of them."""

import asyncio
import typing
from logging import LoggerAdapter

import event_model

from .bundlers import RunBundler, maybe_await
from .dispatcher import Dispatcher
from .log import ComposableLogAdapter, logger
from .plan_runner import (
    PlanEnvironment,
    PlanHooks,
    PlanRunner,
    RunEngineMetadata,
    _default_event_loop,
    _default_md_normalizer,
    _default_md_validator,
    default_scan_id_source,
)
from .protocols import SyncOrAsync
from .suspenders import SuspenderBase
from .suspension import Suspension, SuspensionReason

__all__ = ["PlanSession"]


class PlanSession:
    """The environment that plans are executed in.

    .. warning::

       This API is provisional: names, signatures and behaviour may change
       without a deprecation period. :class:`~bluesky.run_engine.RunEngine`, which is built on it, is not
       provisional.

    Holds what outlives any one plan: persistent metadata, subscriptions,
    suspenders and hooks. Each :meth:`start` builds a :class:`~bluesky.plan_runner.PlanRunner` for one
    plan::

        runner = session.start(my_plan())
        result = await runner

    Keeps no reference to its runners, so several plans may run at once.

    Parameters
    ----------
    md : MutableMapping[str, Any], optional
        Persistent metadata. Defaults to a ``dict``; any MutableMapping works.
    loop : asyncio event loop, optional
        The loop plans run on. Defaults to the running loop, or the one a
        :class:`~bluesky.run_engine.RunEngine` has already established.
    log : logging.LoggerAdapter, optional
        Where this session and its runners log to.
    run_bundler_cls : type, optional
        The bundler used to compose each open run's documents.
    identity : object, optional
        What state changes are logged against and ``Msg('RE_class')`` reports
        the class of. A :class:`~bluesky.run_engine.RunEngine` passes itself; by default each runner
        answers for itself.

    Attributes
    ----------
    preprocessors
        Generator functions applied to each plan: ``[f, g]`` gives
        ``f(g(plan))``.
    md_validator
        Raises to prevent opening a run whose metadata is invalid.
    md_normalizer
        Like ``md_validator``, but returns the normalized metadata.
    scan_id_source
        A (possibly async) function computing ``scan_id``.
    ignore_exceptions
        Whether a raising subscriber is warned about rather than raised.
    hooks
        The :class:`~bluesky.plan_runner.PlanHooks` every runner shares. Read-only: set their fields,
        and a change reaches a running plan.
    suspenders
        The session-scoped suspenders. Plan-scoped ones are not here.
    suspension_reasons
        The reasons holding up every plan now, keyed by who tripped them.
    md
        Persistent metadata, and the ``scan_id`` counter. Each plan gets a
        copy.

    ``preprocessors``, ``md_validator``, ``md_normalizer``,
    ``record_interruptions``, ``strict_pre_declare`` and ``rewindable`` (the
    initial value of :attr:`~bluesky.plan_runner.PlanRunner.rewindable`) are read by :meth:`start`, so a change
    takes effect for the next plan.
    """

    def __init__(
        self,
        md: RunEngineMetadata | None = None,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        log: LoggerAdapter | None = None,
        run_bundler_cls: type[RunBundler] = RunBundler,
        identity: typing.Any = None,
    ) -> None:
        if loop is None:
            loop = _default_event_loop()
        self._loop = loop

        self.log = log if log is not None else ComposableLogAdapter(logger, {"RE": self})

        # Set before the environment that holds it is built.
        if md is None:
            md = {}
        md.setdefault("versions", {})

        try:
            import ophyd

            md["versions"]["ophyd"] = ophyd.__version__
        except ImportError:
            self.log.debug("Failed to import ophyd.")

        try:
            import ophyd_async

            md["versions"]["ophyd_async"] = ophyd_async.__version__
        except ImportError:
            self.log.debug("Failed to import ophyd_async.")

        from ._version import __version__

        md["versions"]["bluesky"] = __version__
        md["versions"]["event_model"] = event_model.__version__

        self.scan_id_source: typing.Callable[[RunEngineMetadata], SyncOrAsync[int]] = default_scan_id_source
        # Serialises scan id allocation across concurrent runners.
        self._scan_id_lock = asyncio.Lock()

        # Shared with every runner by reference, so a change reaches a running plan.
        self._hooks = PlanHooks()

        # Read by `start` into each plan's frozen `PlanEnvironment`.
        self.md = md
        self.preprocessors: list = []
        self.md_validator: typing.Callable = _default_md_validator
        self.md_normalizer: typing.Callable = _default_md_normalizer
        self.run_bundler_cls = run_bundler_cls
        self.identity = identity
        self.record_interruptions = False
        self.strict_pre_declare = False
        self.rewindable = True

        self._suspenders: set[SuspenderBase] = set()
        # Parent of every plan's suspension.
        self._suspension = Suspension(loop)

        # User-registered and unregistered commands, applied to each new runner.
        self._registered_commands: dict[str, typing.Callable] = {}
        self._unregistered_commands: set[str] = set()

        self._dispatcher = Dispatcher()

    async def _next_scan_id(self) -> int:
        """Compute, store in ``md`` and return the ``scan_id`` for an opening run."""
        async with self._scan_id_lock:
            scan_id = await maybe_await(self.scan_id_source(self.md))
            self.md["scan_id"] = scan_id
            return scan_id

    @property
    def suspenders(self) -> tuple[SuspenderBase, ...]:
        """Read-only collection of installed suspenders."""
        return tuple(self._suspenders)

    @property
    def hooks(self) -> PlanHooks:
        """The hooks every runner shares. Set their fields; a change reaches a running plan."""
        return self._hooks

    @property
    def suspension_reasons(self) -> typing.Mapping[typing.Hashable, SuspensionReason]:
        """The reasons holding up every plan this session runs, keyed by who tripped them."""
        return self._suspension.reasons

    def register_command(self, name: str, func: typing.Callable) -> None:
        """Register a new Message command for the plans started after this.

        Parameters
        ----------
        name : str
        func : callable
            This can be a function or a method. The signature is ``f(msg)``.
        """
        self._registered_commands[name] = func
        self._unregistered_commands.discard(name)

    def unregister_command(self, name: str) -> None:
        """Unregister a Message command.

        Parameters
        ----------
        name : str
        """
        # Built-ins can be unregistered too, so check every known name.
        if name not in self.commands:
            raise KeyError(name)
        self._registered_commands.pop(name, None)
        self._unregistered_commands.add(name)

    @property
    def commands(self) -> dict[str, str]:
        """The commands the next plan will understand, name to docstring."""
        registry: dict[str, typing.Callable] = dict(PlanRunner._DEFAULT_COMMANDS)
        registry.update(self._registered_commands)
        return {
            name: func.__doc__ or "" for name, func in registry.items() if name not in self._unregistered_commands
        }

    def start(self, plan, *, metadata=None, subs=None) -> "PlanRunner":
        """Build a runner for ``plan``, and return it without keeping a reference.

        Parameters
        ----------
        plan : iterable of Msg
            The plan the new runner will run. Malformed plans raise here.
        metadata : dict, optional
            Metadata for every run the plan opens.
        subs : callable, list, or dict, optional
            Subscriptions for this plan only, in the forms
            :meth:`RunEngine.__call__ <bluesky.run_engine.RunEngine.__call__>` accepts.
        """
        return self._build(plan, metadata=metadata, subs=subs)

    def _idle_runner(self) -> "PlanRunner":
        """A runner with no plan, which reports 'idle'."""
        return self._build(None)

    def _build(self, plan, *, metadata=None, subs=None) -> "PlanRunner":
        """Compose a runner for ``plan`` from the current settings."""
        # The plan's suspension is a child of the session's.
        suspension = Suspension(self._loop, parent=self._suspension)

        return PlanRunner(
            plan,
            # Frozen once built, so a plan's environment cannot change under it.
            PlanEnvironment(
                loop=self._loop,
                log=self.log,
                # A copy: a change to `session.md` takes effect for the next plan.
                md=dict(self.md),
                next_scan_id=self._next_scan_id,
                md_validator=self.md_validator,
                md_normalizer=self.md_normalizer,
                run_bundler_cls=self.run_bundler_cls,
                record_interruptions=self.record_interruptions,
                strict_pre_declare=self.strict_pre_declare,
            ),
            suspension,
            self.hooks,
            # The plan's dispatcher is a child of the session's.
            Dispatcher(parent=self._dispatcher),
            preprocessors=self.preprocessors,
            initially_rewindable=self.rewindable,
            metadata=metadata,
            subs=subs,
            identity=self.identity,
            commands=dict(self._registered_commands),
            without_commands=self._unregistered_commands,
        )

    def subscribe(self, func: typing.Callable, name: str = "all") -> int:
        """Register a callback to consume documents from every plan.

        Parameters
        ----------
        func : callable
            Expecting a signature like ``f(name, document)``, where name is a
            string and document is a dict.
        name : {'all', 'start', 'descriptor', 'event', 'stop'}, optional
            The type of document this function should receive ('all' by
            default).

        Returns
        -------
        token : int
            An integer ID that can be passed to :meth:`unsubscribe`.

        See Also
        --------
        :meth:`PlanSession.unsubscribe`
        """
        return self._dispatcher.subscribe(func, name)

    def unsubscribe(self, token: int) -> None:
        """Unregister a callback by the integer ID :meth:`PlanSession.subscribe` returned.

        See Also
        --------
        :meth:`PlanSession.subscribe`
        """
        self._dispatcher.unsubscribe(token)

    def unsubscribe_all(self) -> None:
        """Unregister every callback registered on this session, but not a plan's own."""
        self._dispatcher.unsubscribe_all()

    @property
    def ignore_exceptions(self) -> bool:
        """Whether a raising subscriber is warned about rather than raised, for every plan."""
        return self._dispatcher.ignore_exceptions

    @ignore_exceptions.setter
    def ignore_exceptions(self, val: bool) -> None:
        self._dispatcher.ignore_exceptions = val

    def install_suspender(self, suspender: SuspenderBase) -> None:
        """Install a suspender that holds up every plan this session runs.

        It stays subscribed between plans, so it sees a condition that is
        already bad when a plan starts.
        """
        self._suspenders.add(suspender)
        suspender.install(self._suspension)

    def remove_suspender(self, suspender: SuspenderBase) -> None:
        """Uninstall a session suspender."""
        if suspender in self._suspenders:
            suspender.remove()
        self._suspenders.discard(suspender)

    def clear_suspenders(self) -> None:
        """Uninstall all suspenders."""
        for suspender in self.suspenders:
            self.remove_suspender(suspender)
