# `RunEngine` before and after it drives a session

For review only; the last commit deletes this file. `RunEngine` at the commit before this one against `RunEngine` here, method by method. It replaces reading the added lines of the diff.

## Changed (26)

### `state`

```diff
--- before.state
+++ after.state
@@ -1,3 +1,6 @@
     @property
     def state(self):
-        return self._state
+        # The engine's own one-way latch; overrides the plan's state.
+        if self._is_panicked:
+            return _PANICKED_STATE
+        return self._runner.state
```

### `deferred_pause_requested`

```diff
--- before.deferred_pause_requested
+++ after.deferred_pause_requested
@@ -1,15 +1,15 @@
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
-        return self._deferred_pause_requested
+        return self._runner.deferred_pause_requested
```

### `__init__`

```diff
--- before.__init__
+++ after.__init__
@@ -1,183 +1,87 @@
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
-        self._state_lock = threading.RLock()
         self._loop = loop
         # When set, RunEngine.__call__ should stop blocking.
         self._blocking_event = threading.Event()
 
-        self._run_tracing_spans: list[Span] = []
-
-        # When cleared, RunEngine._run will pause until set.
-        self._run_permit = None
-
-        setup_event = threading.Event()
-
-        def setup_run_permit():
-            self._run_permit = asyncio.Event()
-            self._run_permit.set()
-            setup_event.set()
-
-        self.loop.call_soon_threadsafe(setup_run_permit)
-        setup_event.wait()
-
         # Make a logger for this specific RE instance, using the instance's
         # Python id, to keep from mixing output from separate instances.
-        self.log = ComposableLogAdapter(logger, {"RE": self})
+        log = ComposableLogAdapter(logger, {"RE": self})
 
-        if md is None:
-            md = {}
-        self.md = md
-        self.md.setdefault("versions", {})
+        # Set, once, when the loop thread could not be shut down. A plain bool:
+        # the loop that would own a state transition is what has stopped.
+        self._is_panicked = False
 
-        try:
-            import ophyd
+        # The session holds everything that outlives a plan; properties below
+        # forward to it.
+        self._session = PlanSession(
+            md,
+            loop=loop,
+            log=log,
+            # Honour a RunBundler overridden on a subclass, and log state
+            # changes and answer Msg('RE_class') as this RunEngine.
+            run_bundler_cls=type(self).RunBundler,
+            identity=self,
+        )
 
-            self.md["versions"]["ophyd"] = ophyd.__version__
-        except ImportError:
-            self.log.debug("Failed to import ophyd.")
-
-        try:
-            import ophyd_async
-
-            self.md["versions"]["ophyd_async"] = ophyd_async.__version__
-        except ImportError:
-            self.log.debug("Failed to import ophyd_async.")
-
-        from ._version import __version__
-
-        self.md["versions"]["bluesky"] = __version__
-        self.md["versions"]["event_model"] = event_model.__version__
-
-        if preprocessors is None:
-            preprocessors = []
-        self.preprocessors = preprocessors
         if context_managers is None:
             context_managers = [SigintHandler]
         self.context_managers = context_managers
-        if md_validator is None:
-            md_validator = _default_md_validator
-        self.md_validator = md_validator
-        if md_normalizer is None:
-            md_normalizer = _default_md_normalizer
-        self.md_normalizer = md_normalizer
-        self.scan_id_source = scan_id_source
 
         self.max_depth = None
-        self.msg_hook = None
-        self.state_hook = None
-        self.waiting_hook = None
-        self.record_interruptions = False
         self.pause_msg = PAUSE_MSG
-        self.NO_PLAN_RETURN = object()
 
         if during_task is None:
             during_task = DefaultDuringTask()
         self._during_task = during_task
 
-        # The RunEngine keeps track of a *lot* of state.
-        # All flags and caches are defined here with a comment. Good luck.
         self._call_returns_result = call_returns_result  # should __call__ return UIDs or plan value
-        self._run_bundlers: dict[typing.Any, RunBundler] = {}  # a mapping of open run -> bundlers
-        self._metadata_per_call: dict[typing.Any, typing.Any] = {}  # for all runs generated by one __call__
-        self._deferred_pause_requested = False  # pause at next 'checkpoint'
-        self._exception: type[BaseException] | BaseException | None = (
-            None  # stored and then raised in the _run loop
-        )
-        self._interrupted = False  # True if paused, aborted, or failed
-        self._staged: set[typing.Any] = set()  # objects staged, not yet unstaged
-        self._objs_seen: set[typing.Any] = set()  # all objects seen
-        self._movable_objs_touched: set[typing.Any] = set()  # objects we moved at any point
-        self._run_start_uids: list[typing.Any] = list()  # run start uids generated by __call__  # noqa: C408
-        self._suspenders: set[typing.Any] = set()  # the installed suspenders
-        # What the suspenders trip, and what a held plan waits on.
-        self._suspension = Suspension(self._loop)
-        # True from a hold or suspension opening until its wait ends.
-        self._held_up = False
-        # Suspends the plan when the suspension trips. One per plan.
-        self._supervisor: asyncio.Task | None = None
-        self._groups: defaultdict[str, set[Callable[[], asyncio.Future]]] = defaultdict(
-            set
-        )  # sets of Events to wait for
-        self._status_objs: defaultdict[typing.Any, set[typing.Any]] = defaultdict(
-            set
-        )  # status objects to wait for
-        self._temp_callback_ids: set[typing.Any] = set()  # ids from CallbackRegistry
-        self._seen_wait_and_move_on_keys: set[typing.Any] = (
-            set()
-        )  # group ids that have been passed to _wait_and_move_on
-        self._msg_cache: deque[typing.Any] = deque()  # history of processed msgs for rewinding
-        self._rewindable_flag: bool = True  # if the RE is allowed to replay msgs
-        self._plan_stack: deque[typing.Any] = deque()  # stack of generators to work off of
-        self._response_stack: deque[typing.Any] = deque()  # resps to send into the plans
-        self._exit_status = "success"  # optimistic default
-        self._reason = ""  # reason for abort
-        self._task = None  # asyncio.Task associated with call to self._run
-        self._task_fut = None  # future proxy to the task above
-        self._pardon_failures = None  # will hold an asyncio.Event
-        self._plan: typing.Iterable[Msg] | None = None  # the plan instance from __call__
-        self._require_stream_declaration = False
-        self._command_registry = {
-            "declare_stream": self._declare_stream,
-            "create": self._create,
-            "save": self._save,
-            "drop": self._drop,
-            "read": self._read,
-            "locate": self._locate,
-            "monitor": self._monitor,
-            "unmonitor": self._unmonitor,
-            "null": self._null,
-            "RE_class": self._RE_class,
-            "stop": self._stop,
-            "set": self._set,
-            "trigger": self._trigger,
-            "sleep": self._sleep,
-            "wait": self._wait,
-            "checkpoint": self._checkpoint,
-            "clear_checkpoint": self._clear_checkpoint,
-            "rewindable": self._rewindable,
-            "pause": self._pause,
-            "_resume_from_suspender": self._resume,
-            "_start_suspender": self._start_suspender,
-            "prepare": self._prepare,
-            "collect": self._collect,
-            "kickoff": self._kickoff,
-            "complete": self._complete,
-            "configure": self._configure,
-            "stage": self._stage,
-            "unstage": self._unstage,
-            "subscribe": self._subscribe,
-            "unsubscribe": self._unsubscribe,
-            "open_run": self._open_run,
-            "close_run": self._close_run,
-            "wait_for": self._wait_for,
-            "input": self._input,
-            "install_suspender": self._install_suspender,
-            "remove_suspender": self._remove_suspender,
-        }
+        self._task_fut = None  # future proxy to the task running the plan
 
-        # public dispatcher for callbacks
-        # The Dispatcher's public methods are exposed through the
-        # RunEngine for user convenience.
-        self.dispatcher = Dispatcher()
-        self.ignore_callback_exceptions = False
+        if preprocessors is not None:
+            self._session.preprocessors = preprocessors
+        if md_validator is not None:
+            self._session.md_validator = md_validator
+        if md_normalizer is not None:
+            self._session.md_normalizer = md_normalizer
+        self._session.scan_id_source = scan_id_source
+        # Prints; the only half that knows a terminal is watching.
+        hooks = self._session.hooks
+        hooks.pause_requested = self._announce_pause
+        hooks.stop_requested = self._announce_stop
+        hooks.suspension_began = self._announce_suspended
+        hooks.suspender_joined = self._announce_joined
+        hooks.suspender_recovered = self._announce_recovered
+        hooks.suspension_refused = self._announce_refused
+        hooks.hold_began = self._announce_held
+        hooks.plan_paused = self._paused
+        # Holds every runner's plan until `_resume_task` has entered the
+        # context managers (and so installed SigintHandler). Shut again on a
+        # pause.
+        self._proceed_permitted = asyncio.Event()
+        hooks.may_proceed = self._proceed_permitted.wait
+
+        # The current runner, kept after its plan ends so it can be resumed or
+        # inspected. Idle, not None, before the first plan.
+        self._runner = self._session._idle_runner()
 
         # aliases for back-compatibility
         self.subscribe_lossless = self.dispatcher.subscribe
         self.unsubscribe_lossless = self.dispatcher.unsubscribe
         self._subscribe_lossless = self.dispatcher.subscribe
         self._unsubscribe_lossless = self.dispatcher.unsubscribe
```

### `commands`

```diff
--- before.commands
+++ after.commands
@@ -1,20 +1,20 @@
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
-        # return as a list, not lazy loader, no surprises...
-        return list(self._command_registry.keys())
+        # Names only, in registration order, as before.
+        return list(self._session.commands)
```

### `print_command_registry`

```diff
--- before.print_command_registry
+++ after.print_command_registry
@@ -1,32 +1,31 @@
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
 
-        for command, func in self._command_registry.items():
-            docstring = func.__doc__
+        for command, docstring in self._session.commands.items():
             if not verbose:
                 docstring = docstring.split("\n")[0]
             commands = commands + f"{command} : {docstring}\n"
 
         return commands
```

### `rewindable`

```diff
--- before.rewindable
+++ after.rewindable
@@ -1,9 +1,13 @@
     @property
     def rewindable(self):
-        return self._rewindable_flag
+        # The running plan's live value, else the session's default for the
+        # next plan.
+        if not self._runner.state.is_idle:
+            return self._runner.rewindable
+        return self._session.rewindable
     @rewindable.setter
     def rewindable(self, v):
-        cur_state = self._rewindable_flag
-        self._rewindable_flag = bool(v)
-        if self.resumable and self._rewindable_flag != cur_state:
-            self._reset_checkpoint_state()
+        # Both: the session's default for later plans, the runner's for now.
+        # Not marshalled onto the loop: a sync setter is atomic there.
+        self._session.rewindable = bool(v)
+        self._runner.rewindable = bool(v)
```

### `suspenders`

```diff
--- before.suspenders
+++ after.suspenders
@@ -1,3 +1,4 @@
     @property
     def suspenders(self):
-        return tuple(self._suspenders)
+        """Every suspender that can suspend the plan in progress: session-scoped and plan-scoped."""
+        return tuple(set(self._session.suspenders) | set(self._runner.suspenders))
```

### `reset`

```diff
--- before.reset
+++ after.reset
@@ -1,11 +1,11 @@
     def reset(self):
         """
         Clean up caches and unsubscribe subscriptions.
 
         Lossless subscriptions are not unsubscribed.
         """
-        if self._state != "idle":
+        self._raise_if_panicked()
+        if self._runner.state != "idle":
             self.halt()
-        self._clear_run_cache()
-        self._clear_call_cache()
+        self._new_runner()
         self.dispatcher.unsubscribe_all()
```

### `resumable`

```diff
--- before.resumable
+++ after.resumable
@@ -1,4 +1,4 @@
     @property
     def resumable(self):
         "i.e., can the plan in progress by rewound"
-        return self._msg_cache is not None
+        return self._runner.resumable
```

### `ignore_callback_exceptions`

```diff
--- before.ignore_callback_exceptions
+++ after.ignore_callback_exceptions
@@ -1,6 +1,7 @@
     @property
     def ignore_callback_exceptions(self):
         return self.dispatcher.ignore_exceptions
     @ignore_callback_exceptions.setter
     def ignore_callback_exceptions(self, val):
+        # Reaches running plans too.
         self.dispatcher.ignore_exceptions = val
```

### `register_command`

```diff
--- before.register_command
+++ after.register_command
@@ -1,17 +1,17 @@
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
-        self._command_registry[name] = func
+        self._session.register_command(name, func)
```

### `unregister_command`

```diff
--- before.unregister_command
+++ after.unregister_command
@@ -1,15 +1,15 @@
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
-        del self._command_registry[name]
+        self._session.unregister_command(name)
```

### `request_pause`

```diff
--- before.request_pause
+++ after.request_pause
@@ -1,23 +1,20 @@
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
-        if self.state == "panicked":
-            raise RuntimeError("The RunEngine is panicked and cannot be recovered. You must restart bluesky.")
-        future = asyncio.run_coroutine_threadsafe(self._request_pause_coro(defer), loop=self.loop)
-        # TODO add a timeout here?
-        return future.result()
+        self._raise_if_panicked()
+        return run_coro_on_loop(self._runner.pause(defer), self.loop)
```

### `_request_pause_coro`

```diff
--- before._request_pause_coro
+++ after._request_pause_coro
@@ -1,19 +1,3 @@
     async def _request_pause_coro(self, defer=False):
-        # We are pausing. Cancel any deferred pause previously requested.
-        if not self.state.can_pause:
-            raise TransitionError(f"Run Engine is in '{self.state}' state and can not be paused.")
-
-        if defer:
-            self._deferred_pause_requested = True
-            self._announce_pause(True)
-            return
-
-        self._announce_pause(False)
-
-        self._deferred_pause_requested = False
-        self._interrupted = True
-        self._state = "pausing"
-        for current_run in self._run_bundlers.values():
-            current_run.record_interruption("pause")
-
-        self._task.cancel()
+        """Pause without blocking the caller. Used by bluesky-queueserver."""
+        await self._runner.pause(defer)
```

### `_create_result`

```diff
--- before._create_result
+++ after._create_result
@@ -1,14 +1,10 @@
-    def _create_result(self, plan_return):
-        """
-        Create a RunEngineResult to return from __call__, using
-        plan_return and internal state
-        """
-        rs = RunEngineResult(
-            tuple(self._run_start_uids),
+    def _create_result(self, plan_return) -> RunEngineResult:
+        """Describe how the plan finished, to return from `__call__`."""
+        return RunEngineResult(
+            tuple(self._runner.run_start_uids),
             plan_return,
-            self._exit_status,
-            self._interrupted,
-            self._reason,
-            self._exception,
+            self._runner.exit_status,
+            self._runner.interrupted,
+            self._runner.exit_reason,
+            self._runner.exit_exception,
         )
-        return rs
```

### `__call__`

```diff
--- before.__call__
+++ after.__call__
@@ -1,100 +1,73 @@
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
-        if self.state == "panicked":
-            raise RuntimeError("The RunEngine is panicked and cannot be recovered. You must restart bluesky.")
+        self._raise_if_panicked()
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
-        if not self._state.is_idle:
-            raise RuntimeError(f"The RunEngine is in a {self._state} state")
+        if not self._runner.state.is_idle:
+            raise RuntimeError(f"The RunEngine is in a {self._runner.state} state")
 
-        self._clear_call_cache()
-        self._clear_run_cache()  # paranoia, in case of previous bad exit
+        # A malformed plan raises here. The plan is held at
+        # `PlanHooks.may_proceed` until `_resume_task`.
+        self._new_runner(plan, metadata=metadata_kw, subs=subs)
+        self.log.info("Executing plan %r", plan)
 
-        for name, funcs in normalize_subs_input(subs).items():
-            for func in funcs:
-                self._temp_callback_ids.add(self.subscribe(func, name))
+        plan_return = self._resume_task()
 
-        self._plan = plan  # this ref is just used for metadata introspection
-        self._metadata_per_call.update(metadata_kw)
-
-        gen = ensure_generator(plan)
-        for wrapper_func in self.preprocessors:
-            gen = wrapper_func(gen)
-
-        self._plan_stack.append(gen)
-        self._response_stack.append(None)
-        # After the plan is on the stack, so a hold goes in front of it.
-        self._arrange_suspension(at_start=True)
-        self.log.info("Executing plan %r", self._plan)
-
-        def _build_task():
-            # make sure _run will block at the top
-            self._run_permit.clear()
-            self._blocking_event.clear()
-            self._task_fut = asyncio.run_coroutine_threadsafe(self._run(), loop=self.loop)
-
-            def set_blocking_event(future):
-                self._blocking_event.set()
-
-            self._task_fut.add_done_callback(set_blocking_event)
-
-        plan_return = self._resume_task(init_func=_build_task)
-
-        if self._interrupted:
+        if self._runner.interrupted:
             raise RunEngineInterrupted(self.pause_msg) from None
 
         if self._call_returns_result:
             run_engine_result = self._create_result(plan_return)
             return run_engine_result
         else:
-            return tuple(self._run_start_uids)
+            return tuple(self._runner.run_start_uids)
```

### `resume`

```diff
--- before.resume
+++ after.resume
@@ -1,42 +1,28 @@
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
-        if self.state == "panicked":
-            raise RuntimeError("The RunEngine is panicked and cannot be recovered. You must restart bluesky.")
+        self._raise_if_panicked()
 
         # The state machine does not capture the whole picture.
-        if not self._state.is_paused:
+        if not self._runner.state.is_paused:
             raise TransitionError(
-                f"The RunEngine is the {self._state} state. You can only resume for the paused state."
+                f"The RunEngine is the {self._runner.state} state. You can only resume for the paused state."
             )
 
-        self._interrupted = False
-        for current_run in self._run_bundlers.values():
-            current_run.record_interruption("resume")
-        new_plan = self._rewind()
-        self._plan_stack.append(new_plan)
-        self._response_stack.append(None)
-        # Re-read the suspension on the way back in, for a trip during the pause.
-        self._arrange_suspension(at_start=False)
-        # Notify Devices of the resume in case they want to clean up.
-        for obj in self._objs_seen:
-            if isinstance(obj, Pausable):
-                fut = asyncio.run_coroutine_threadsafe(maybe_await(obj.resume()), self._loop)
-                fut.result()
-        plan_return = self._resume_task()
-        if self._interrupted:
+        plan_return = self._resume_task(release=self._runner.resume)
+        if self._runner.interrupted:
             raise RunEngineInterrupted(self.pause_msg) from None
 
         if self._call_returns_result:
             run_engine_result = self._create_result(plan_return)
             return run_engine_result
         else:
-            return tuple(self._run_start_uids)
+            return tuple(self._runner.run_start_uids)
```

### `_resume_task`

```diff
--- before._resume_task
+++ after._resume_task
@@ -1,82 +1,89 @@
-    def _resume_task(self, *, init_func=None):
+    def _resume_task(self, *, release=None):
         # Clear the blocking Event so that we can wait on it below.
         # The task will set it when it is done, as it was previously
         # configured to do it __call__.
         self._blocking_event.clear()
 
         # Handle all context managers
         with ExitStack() as stack:
             for mgr in self.context_managers:
                 stack.enter_context(mgr(self))
 
-            if init_func is not None:
-                init_func()
+            # Inside the context managers, so SigintHandler is installed first.
+            async def proceed():
+                if release is not None:
+                    await release()
+                self._proceed_permitted.set()
+
+            run_coro_on_loop(proceed(), self.loop)
 
             if self._task_fut is None:
                 # No task was ever started; nothing to wait on or return.
                 return self.NO_PLAN_RETURN
             if self._task_fut.done():
                 try:
                     return self._task_fut.result()
                 except concurrent.futures.CancelledError:
-                    return self.NO_PLAN_RETURN
-            # The _run task is waiting on this Event. Let is continue.
-            self.loop.call_soon_threadsafe(self._run_permit.set)
+                    return NO_PLAN_RETURN
             try:
                 # Block until plan is complete or exception is raised.
                 try:
                     self._during_task.block(self._blocking_event)
                 except KeyboardInterrupt:
                     import ctypes
 
-                    self._interrupted = True
+                    self._runner.interrupted = True
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
-                        self._state = "panicked"
+                        old_state = self._runner.state
+                        self._is_panicked = True
+                        # The runner's machine lives on the dead loop, so
+                        # announce the change by hand.
+                        announce_state_change(self, self._session.hooks, old_state, "panicked")
                 except Exception as raised_er:
                     self.halt()
-                    self._interrupted = True
+                    self._runner.interrupted = True
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
-                            plan_return = self.NO_PLAN_RETURN
+                            plan_return = NO_PLAN_RETURN
                     # we have something in exc
                     else:
                         # special case the panic exception that we put in above
                         if isinstance(exc, _RunEnginePanic):
-                            plan_return = self.NO_PLAN_RETURN
+                            plan_return = NO_PLAN_RETURN
                         # otherwise re-raise it
                         else:
                             raise exc
                 else:
                     plan_return = None
             return plan_return
```

### `_announce_suspended`

```diff
--- before._announce_suspended
+++ after._announce_suspended
@@ -1,7 +1,7 @@
     def _announce_suspended(self, reasons: typing.Mapping[typing.Hashable, SuspensionReason]) -> None:
-        """Say that a suspension is beginning, and what is holding it."""
+        """Say a suspension has begun, and how to get back to a prompt."""
         print("Suspending....To get prompt hit Ctrl-C twice to pause.")
         print(f"Suspension occurred at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
         justification = join_justifications(reasons)
         if justification:
             print(f"Justification for this suspension:\n{justification}")
```

### `install_suspender`

```diff
--- before.install_suspender
+++ after.install_suspender
@@ -1,16 +1,20 @@
     @_runs_on_loop(timeout=SUBSCRIPTION_TIMEOUT)
     def install_suspender(self, suspender):
         """
-        Install a 'suspender', which can suspend and resume execution.
+        Install a session-scoped 'suspender', which can suspend and resume execution.
+
+        It holds up every plan this engine runs until removed. One installed
+        with ``Msg('install_suspender', None, suspender)`` holds up only that
+        plan, and ends with it.
 
         Parameters
         ----------
         suspender : `bluesky.suspenders.SuspenderBase`
 
         See Also
         --------
         :meth:`RunEngine.remove_suspender`
         :meth:`RunEngine.clear_suspenders`
         """
-        self._suspenders.add(suspender)
-        suspender.install(self._suspension)
+        # Reaches a running plan too, through its suspension's parent.
+        self._session.install_suspender(suspender)
```

### `remove_suspender`

```diff
--- before.remove_suspender
+++ after.remove_suspender
@@ -1,17 +1,15 @@
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
-        if suspender in self._suspenders:
-            suspender.remove()
-        self._suspenders.discard(suspender)
+        self._session.remove_suspender(suspender)
```

### `clear_suspenders`

```diff
--- before.clear_suspenders
+++ after.clear_suspenders
@@ -1,13 +1,13 @@
     @_runs_on_loop(timeout=SUBSCRIPTION_TIMEOUT)
     def clear_suspenders(self):
         """
-        Uninstall all suspenders.
+        Uninstall all suspenders, session-scoped and the running plan's plan-scoped ones.
 
         See Also
         --------
         :meth:`RunEngine.install_suspender`
         :meth:`RunEngine.remove_suspender`
         """
-        for suspender in tuple(self._suspenders):
-            suspender.remove()
-        self._suspenders.clear()
+        # The session cannot remove plan-scoped ones, so ask each.
+        self._session.clear_suspenders()
+        self._runner.clear_suspenders()
```

### `abort`

```diff
--- before.abort
+++ after.abort
@@ -1,18 +1,18 @@
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
-        return self.__interrupter_helper(self._abort_coro(reason))
+        return self.__interrupter_helper(self._runner.abort(reason))
```

### `stop`

```diff
--- before.stop
+++ after.stop
@@ -1,18 +1,18 @@
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
-        return self.__interrupter_helper(self._stop_coro())
+        return self.__interrupter_helper(self._runner.stop())
```

### `halt`

```diff
--- before.halt
+++ after.halt
@@ -1,18 +1,18 @@
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
-        return self.__interrupter_helper(self._halt_coro())
+        return self.__interrupter_helper(self._runner.halt())
```

### `__interrupter_helper`

```diff
--- before.__interrupter_helper
+++ after.__interrupter_helper
@@ -1,23 +1,15 @@
     def __interrupter_helper(self, coro):
-        if self.state == "panicked":
+        if self._is_panicked:
+            # Close it, or it warns "never awaited" at collection.
             coro.close()
-            raise RuntimeError("The RunEngine is panicked and cannot be recovered. You must restart bluesky.")
+        self._raise_if_panicked()
 
-        coro_event = threading.Event()
-        task = None
-
-        def end_cb(fut):
-            coro_event.set()
-
-        def start_task():
-            nonlocal task
-            task = self.loop.create_task(coro)
-            task.add_done_callback(end_cb)
-
-        was_paused = self._state == "paused"
-        self.loop.call_soon_threadsafe(start_task)
-        coro_event.wait()
+        was_paused = self._runner.state == "paused"
+        # No timeout: giving up would leave the plan running.
+        run_coro_on_loop(coro, self.loop)
+        # Before a paused plan's cleanup can change it.
+        result = self._interrupted_result()
         if was_paused:
             self._resume_task()
 
-        return task.result()
+        return result
```

## New (16)

### `_raise_if_panicked`

```python
    def _raise_if_panicked(self):
        """Raise if the loop thread is wedged, rather than schedule work it will never run."""
        if self._is_panicked:
            raise RuntimeError("The RunEngine is panicked and cannot be recovered. You must restart bluesky.")
```

### `log`

```python
    @property
    def log(self):
        return self._session.log
```

### `md`

```python
    @property
    def md(self):
        return self._session.md
    @md.setter
    def md(self, value):
        self._session.md = value
```

### `dispatcher`

```python
    @property
    def dispatcher(self):
        # Published for existing callers; the session keeps it private.
        return self._session._dispatcher
```

### `preprocessors`

```python
    @property
    def preprocessors(self):
        return self._session.preprocessors
    @preprocessors.setter
    def preprocessors(self, value):
        self._session.preprocessors = value
```

### `md_validator`

```python
    @property
    def md_validator(self):
        return self._session.md_validator
    @md_validator.setter
    def md_validator(self, value):
        self._session.md_validator = value
```

### `md_normalizer`

```python
    @property
    def md_normalizer(self):
        return self._session.md_normalizer
    @md_normalizer.setter
    def md_normalizer(self, value):
        self._session.md_normalizer = value
```

### `scan_id_source`

```python
    @property
    def scan_id_source(self):
        return self._session.scan_id_source
    @scan_id_source.setter
    def scan_id_source(self, value):
        self._session.scan_id_source = value
```

### `msg_hook`

```python
    @property
    def msg_hook(self):
        return _hook_or_none(self._session.hooks.msg_received)
    @msg_hook.setter
    def msg_hook(self, value):
        self._session.hooks.msg_received = value
```

### `state_hook`

```python
    @property
    def state_hook(self):
        return _hook_or_none(self._session.hooks.state_changed)
    @state_hook.setter
    def state_hook(self, value):
        self._session.hooks.state_changed = value
```

### `waiting_hook`

```python
    @property
    def waiting_hook(self):
        return _hook_or_none(self._session.hooks.waiting_on)
    @waiting_hook.setter
    def waiting_hook(self, value):
        self._session.hooks.waiting_on = value
```

### `record_interruptions`

```python
    @property
    def record_interruptions(self):
        return self._session.record_interruptions
    @record_interruptions.setter
    def record_interruptions(self, value):
        self._session.record_interruptions = value
```

### `_require_stream_declaration`

```python
    @property
    def _require_stream_declaration(self):
        return self._session.strict_pre_declare
    @_require_stream_declaration.setter
    def _require_stream_declaration(self, value):
        self._session.strict_pre_declare = value
```

### `_new_runner`

```python
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
```

### `_paused`

```python
    def _paused(self):
        """Hold the plan at `PlanHooks.may_proceed`, and release the main thread."""
        self._proceed_permitted.clear()
        self._blocking_event.set()
```

### `_interrupted_result`

```python
    def _interrupted_result(self):
        """What abort(), stop() and halt() return."""
        if self._call_returns_result:
            return self._create_result(NO_PLAN_RETURN)
        return tuple(self._runner.run_start_uids)
```

## Removed (56)

`_clear_run_cache`, `_clear_call_cache`, `_rewind` (now on `PlanRunner`), `_install_suspender` (now on `PlanRunner`), `_remove_suspender` (now on `PlanRunner`), `_supervise_suspension` (now on `PlanRunner`), `_pause_objects` (now on `PlanRunner`), `_start_suspender` (now on `PlanRunner`), `_arrange_suspension`, `_abort_coro`, `_stop_coro`, `_halt_coro`, `_stop_movable_objects` (now on `PlanRunner`), `_destroy_open_run_tracing_spans` (now on `PlanRunner`), `_run` (now on `PlanRunner`), `_wait_for` (now on `PlanRunner`), `_open_run` (now on `PlanRunner`), `_close_run` (now on `PlanRunner`), `_close_run_trace` (now on `PlanRunner`), `_create` (now on `PlanRunner`), `_declare_stream` (now on `PlanRunner`), `_read` (now on `PlanRunner`), `_locate` (now on `PlanRunner`), `_monitor` (now on `PlanRunner`), `_unmonitor` (now on `PlanRunner`), `_save` (now on `PlanRunner`), `_drop` (now on `PlanRunner`), `_prepare` (now on `PlanRunner`), `_kickoff` (now on `PlanRunner`), `_complete` (now on `PlanRunner`), `_collect` (now on `PlanRunner`), `_null` (now on `PlanRunner`), `_RE_class` (now on `PlanRunner`), `_set` (now on `PlanRunner`), `_trigger` (now on `PlanRunner`), `_call_waiting_hook`, `_wait` (now on `PlanRunner`), `_status_object_completed` (now on `PlanRunner`), `_sleep` (now on `PlanRunner`), `_pause` (now on `PlanRunner`), `_resume_objects` (now on `PlanRunner`), `_resume`, `_checkpoint` (now on `PlanRunner`), `_reset_checkpoint_state` (now on `PlanRunner`), `_reset_checkpoint_state_meth`, `_reset_checkpoint_state_coro`, `_clear_checkpoint` (now on `PlanRunner`), `_rewindable` (now on `PlanRunner`), `_configure` (now on `PlanRunner`), `_add_status_to_group` (now on `PlanRunner`), `_stage` (now on `PlanRunner`), `_unstage` (now on `PlanRunner`), `_stop` (now on `PlanRunner`), `_subscribe` (now on `PlanRunner`), `_unsubscribe` (now on `PlanRunner`), `_input` (now on `PlanRunner`).

## Unchanged (13)

`subscribe`, `unsubscribe`, `loop`, `verbose`, `call_returns_result`, `_announce_held`, `_announce_pause`, `_announce_stop`, `_announce_refused`, `_announce_joined`, `_announce_recovered`, `emit_sync`, `emit`.

## Module level

### `RunEngineResult`

```diff
--- before.RunEngineResult
+++ after.RunEngineResult
@@ -1,25 +1,26 @@
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
-        The exception generated by the plan, if any
+        The `~bluesky.utils.RequestStop`, `~bluesky.utils.RequestAbort` or
+        `~bluesky.utils.PlanHalt` instance that ended a paused plan, if any.
     """
 
     run_start_uids: tuple[str, ...]
     plan_result: typing.Any
     exit_status: str
     interrupted: bool
     reason: str
-    exception: Exception | None
+    exception: BaseException | None
```

### `_runs_on_loop`

```diff
--- before._runs_on_loop
+++ after._runs_on_loop
@@ -1,22 +1,23 @@
 def _runs_on_loop(*, timeout: float | None = None):
     """Run the decorated `RunEngine` method's body on the engine's loop, and wait.
 
     Called on the loop, from a plan body or a subscriber, it runs the body there
     and then. ``timeout`` bounds only the wait: the body may still run after the
     caller gives up.
     """
 
     def decorate(method):
-        @functools.wraps(method)
+        @wraps(method)
         def crossing(self, *args, **kwargs):
-            if running_on(self._loop):
+            self._raise_if_panicked()
+            if running_on(self.loop):
                 return method(self, *args, **kwargs)
 
             async def work():
                 return method(self, *args, **kwargs)
 
-            return run_coro_on_loop(work(), self._loop, timeout=timeout)
+            return run_coro_on_loop(work(), self.loop, timeout=timeout)
 
         return crossing
 
     return decorate
```

### `call_in_bluesky_event_loop`

```diff
--- before.call_in_bluesky_event_loop
+++ after.call_in_bluesky_event_loop
@@ -1,11 +1,7 @@
 def call_in_bluesky_event_loop(coro: typing.Awaitable[T], timeout: float | None = None) -> T:
     if _bluesky_event_loop is None or not _bluesky_event_loop.is_running():
         # Quell "coroutine never awaited" warnings
         if iscoroutine(coro):
             coro.close()
         raise RuntimeError("Bluesky event loop not running")
-    fut: concurrent.futures.Future = asyncio.run_coroutine_threadsafe(
-        coro,  # type: ignore
-        loop=_bluesky_event_loop,
-    )
-    return fut.result(timeout=timeout)
+    return run_coro_on_loop(coro, _bluesky_event_loop, timeout=timeout)
```

### `_panicked_state` (new)

```python
def _panicked_state() -> ProxyString:
    """A 'panicked' ProxyString, whose ``is_*`` checks answer like any other state."""
    machine = RunEngineStateMachine()
    machine.set_("panicked")
    return ProxyString("panicked", machine)
```

### `_hook_or_none` (new)

```python
def _hook_or_none(hook):
    """What a user set, or ``None`` if they never set one."""
    return None if hook is do_nothing else hook
```

### `_forward_to_runner` (new)

```python
def _forward_to_runner(name: str) -> property:
    """A property reading and writing ``name`` on the current runner."""

    def getter(self):
        return getattr(self._runner, name)

    def setter(self, value):
        setattr(self._runner, name, value)

    return property(getter, setter, doc=f"Forwards to :attr:`PlanRunner.{name}`.")
```

Removed: `WaitForTimeoutError`, `RunEngineStateMachine`, `LoggingPropertyMachine`, `default_scan_id_source`, `_state_locked`, `_called`, `_set_span_msg_attributes`, `_default_md_validator`, `_default_md_normalizer`.
