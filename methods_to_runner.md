# Methods moved from `RunEngine` to `PlanRunner`

For review only; the last commit deletes this file. Each block is a `RunEngine` method at the commit before this one against its `PlanRunner` counterpart here, grouped as the commit message groups them.

## Changed

### `_run`

```diff
--- RunEngine._run
+++ PlanRunner._run
@@ -1,333 +1,326 @@
     async def _run(self):
-        """Pull messages from the plan, process them, send results back.
+        """Run the plan; the task built in ``__init__``.
+
+        Notes
+        -----
+        Pull messages from the plan, process them, send results back.
 
         Upon exit, clean up.
         - Call stop() on all objects that were 'set' or 'kickoff'.
         - Try to collect any uncollected flyers.
         - Try to unstage any devices left staged by the plan.
         - Try to remove any monitoring subscriptions left on by the plan.
         - If interrupting the middle of a run, try to emit a RunStop document.
         """
-        await self._run_permit.wait()
-        # grab the current task.  We need to do this here because the
-        # object returned by `run_coroutine_threadsafe` is a future
-        # that acts as a proxy that does not have the correct behavior
-        # when `.cancel` is called on it.
-        with self._state_lock:
-            self._task = asyncio.current_task(self.loop)
+        # Before leaving 'idle', and outside the try: a cancel here ends a
+        # plan that never ran, not one that aborted.
+        await maybe_await(self._hooks.may_proceed())
+        self._arrange_permission(at_start=True)
         stashed_exception = None
         debug = msg_logger.debug
-        self._reason = ""
+        self.exit_status = "success"
+        self.exit_reason = ""
+        self.exit_exception = None
         # sentinel to decide if need to add to the response stack or not
         sentinel = object()
-        plan_return = self.NO_PLAN_RETURN
-        exit_reason = ""
+        plan_return = NO_PLAN_RETURN
         try:
             self._state = "running"
             while True:
-                if self._state in ("pausing", "suspending"):
+                if self.state in ("pausing", "suspending"):
                     if not self.resumable:
                         self._run_permit.set()
                         stashed_exception = FailedPause()
 
                         self._state = "aborting"
                         continue
                 # currently only using 'suspending' to get us into the
                 # block above, we do not have a 'suspended' state
                 # (yet)
-                if self._state == "suspending":
+                if self.state == "suspending":
                     self._state = "running"
                 if not self._run_permit.is_set():
                     # A pause has been requested. First, put everything in a
                     # resting state.
-                    assert self._state == "pausing"
+                    assert self.state == "pausing"
                     # Remove any monitoring callbacks, but keep refs in
                     # self._monitor_params to re-instate them later.
                     for current_run in self._run_bundlers.values():
                         await current_run.suspend_monitors()
-                    # During pause, all motors should be stopped. Call stop()
-                    # on every object we ever set().
                     await self._stop_movable_objects(success=True)
-                    # Notify Devices of the pause in case they want to
-                    # clean up.
                     await self._pause_objects()
                     self._state = "paused"
                     # Let RunEngine.__call__ return...
-                    self._blocking_event.set()
+                    self._hooks.plan_paused()
 
                     await self._run_permit.wait()
+                    # See `PlanHooks.may_proceed`.
+                    await maybe_await(self._hooks.may_proceed())
                     # Restore any monitors
                     for current_run in self._run_bundlers.values():
                         await current_run.restore_monitors()
-                    if self._state == "paused":
+                    if self.state == "paused":
                         # may be called by 'resume', 'stop', 'abort', 'halt'
                         self._state = "running"
 
                     # If we are here, we have come back to life either to
                     # continue (resume) or to clean up before exiting.
 
                 assert len(self._response_stack) == len(self._plan_stack)
                 # set resp to the sentinel so that if we fail in the sleep
                 # we do not add an extra response
                 resp = sentinel
                 try:
                     # the new response to be added
                     new_response = None
 
                     # This 'await' must be here to ensure that this coroutine
                     # breaks out of its current behavior before trying to get
                     # the next message from the top of the generator stack in
                     # case there has been a pause requested.  Without this the
                     # next message after the pause may be processed first on
                     # resume (instead of the first message in self._msg_cache).
                     # This await also gives the co-routine for requesting
                     # suspends a chance to run.
 
                     # This sleep has to be inside of this try block so that any
                     # of the 'async' exceptions get thrown in the correct
                     # place.
 
                     # If we are handling an exception, then burn through the
                     # current plan stack before rather than allowing a pause or
                     # suspension to try and finish firing.
                     if stashed_exception is None:
                         await asyncio.sleep(0)
                     # always pop off a result, we are either sending it back in
                     # or throwing an exception in, in either case the left hand
                     # side of the yield in the plan will be moved past
                     resp = self._response_stack.pop()
                     # if any status tasks have failed, grab the exceptions.
                     # give priority to things pushed in from outside
-                    with self._state_lock:
-                        if self._exception is not None:
-                            stashed_exception = self._exception
-                            self._exception = None
+                    if self._exception is not None:
+                        stashed_exception = self._exception
+                        self._exception = None
                     # The case where we have a stashed exception
                     if stashed_exception is not None or isinstance(resp, Exception):
                         # throw the exception at the current plan
                         try:
                             msg = self._plan_stack[-1].throw(stashed_exception or resp)
                         except Exception as e:
                             # The current plan did not handle it,
                             # maybe the next plan (if any) would like
                             # to try
                             self._plan_stack.pop()
                             # we have killed the current plan, do not give
                             # it a new response
                             resp = sentinel
                             # If there is at least one plan left in the stack,
                             # stash the new exception go back to top
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
-                    if self.msg_hook is not None:
-                        self.msg_hook(msg)
+                    self._hooks.msg_received(msg)
                     debug(
                         "%s(%r, *%r **%r, run=%r)",
                         msg.command,
                         msg.obj,
                         msg.args,
                         msg.kwargs,
                         msg.run,
                         extra={"msg_command": msg.command},
                     )
 
                     # update the running set of all objects we have seen
                     self._objs_seen.add(msg.obj)
 
                     # if this message can be cached for rewinding, cache it
                     if (
                         self._msg_cache is not None
-                        and self._rewindable_flag
+                        and self.rewindable
                         and not self._held_up
-                        and msg.command not in self._UNCACHEABLE_COMMANDS
+                        and msg.command not in UNCACHEABLE_COMMANDS
                     ):
                         # We have a checkpoint.
                         self._msg_cache.append(msg)
 
                     # try to look up the coroutine to execute the command
                     if (
                         coro := self._command_registry.get(msg.command, key_absence_sentinel := object())
                     ) is key_absence_sentinel:
                         # flag invalid command
                         # and return to the top of the loop
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
-                    print(
+                    self._env.log.warning(
                         "An unknown external library has improperly raised "
-                        "KeyboardInterrupt. Intercepting and triggering "
-                        "a HALT."
+                        "KeyboardInterrupt. Intercepting and triggering a HALT."
                     )
-                    await self._halt_coro()
+                    await self.halt()
                 except asyncio.CancelledError as e:
-                    if self._state == "pausing":
+                    if self.state == "pausing":
                         # if we got a CancelledError and we are in the
-                        # 'pausing' state clear the run permit and
+                        # 'pausing' state clear the run suspension and
                         # bounce to the top
                         self._run_permit.clear()
                         continue
-                    if self._state in ("halting", "stopping", "aborting"):
+                    if self.state in ("halting", "stopping", "aborting"):
                         # if we got this while just keep going in tear-down
                         exception_map = {"halting": PlanHalt, "stopping": RequestStop, "aborting": RequestAbort}
                         # if the exception is not set bounce to the top
                         if stashed_exception is None:
                             stashed_exception = exception_map[self.state]
                         continue
-                    if self._state == "suspending":
+                    if self.state == "suspending":
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
 
         except StopIteration as e:
-            self._exit_status = "success"
+            self.exit_status = "success"
             plan_return = e.value
             # TODO Is the sleep here necessary?
             await asyncio.sleep(0)
         except RequestStop:
-            self._exit_status = "success"
+            self.exit_status = "success"
             # TODO Is the sleep here necessary?
             await asyncio.sleep(0)
         except (FailedPause, RequestAbort, asyncio.CancelledError, PlanHalt):
-            self._exit_status = "abort"
+            self.exit_status = "abort"
             # TODO Is the sleep here necessary?
             await asyncio.sleep(0)
-            self.log.exception("Run aborted")
+            self._env.log.exception("Run aborted")
         except GeneratorExit as err:
-            self._exit_status = "fail"  # Exception raises during 'running'
-            exit_reason = str(err)
+            self.exit_status = "fail"  # Exception raises during 'running'
+            self.exit_reason = str(err)
             raise ValueError from err
         except Exception as err:
-            self._exit_status = "fail"  # Exception raises during 'running'
-            exit_reason = str(err)
-            self.log.exception("Run aborted")
+            self.exit_status = "fail"  # Exception raises during 'running'
+            self.exit_reason = str(err)
+            self._env.log.exception("Run aborted")
             raise err
         finally:
-            if not exit_reason:
-                exit_reason = self._reason
             # Some done_callbacks may still be alive in other threads.
             # Block them from creating new 'failed status' tasks on the loop.
             self._pardon_failures.set()
             # call stop() on every movable object we ever set()
             await self._stop_movable_objects(success=True)
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
-                    self.log.exception("Failed to unstage %r.", obj)
+                    self._env.log.exception("Failed to unstage %r.", obj)
                 self._staged.remove(obj)
 
-            sys.stdout.flush()
             # Emit RunStop if necessary.
             for key, current_run in self._run_bundlers.items():
                 if current_run.run_is_open:
                     try:
                         await current_run.close_run(
-                            Msg("close_run", exit_status=self._exit_status, reason=exit_reason, run_id=key)
+                            Msg("close_run", exit_status=self.exit_status, reason=self.exit_reason, run_id=key)
                         )
                     except Exception:
-                        self.log.error("Failed to close run %r.", current_run)
+                        self._env.log.error("Failed to close run %r.", current_run)
             self._run_bundlers.clear()
 
             for p in self._plan_stack:
                 try:
                     p.close()
                 except RuntimeError:
-                    print(f"The plan {p!r} tried to yield a value on close.  Please fix your plan.")
-
-            # Stop the supervisor; the next plan starts another.
+                    self._env.log.warning("The plan %r tried to yield a value on close.  Please fix your plan.", p)
+
+            self.clear_suspenders()
             if self._supervisor is not None:
                 self._supervisor.cancel()
-                self._supervisor = None
 
             self._state = "idle"
 
-        self.log.info("Cleaned up from plan %r", self._plan)
+        self._env.log.info("Cleaned up from plan %r", self._plan)
         if isinstance(stashed_exception, asyncio.CancelledError):
             raise stashed_exception
         return plan_return
```

### `resume`

```diff
--- RunEngine.resume
+++ PlanRunner.resume
@@ -1,42 +1,15 @@
-    def resume(self):
-        """Resume a paused plan from the last checkpoint.
+    async def resume(self) -> None:
+        """Continue a paused plan from its last checkpoint. On the loop.
 
-        Returns
-        -------
-        uids : list
-            list of uids (i.e. RunStart Document uids) of run(s)
-            if :attr:`RunEngine._call_returns_result` is ``False``
-        result : :class:`RunEngineResult`
-            if :attr:`RunEngine._call_returns_result` is ``True``
+        Rewinds, tells devices, and releases the plan. If the suspension is
+        tripped, the plan waits for it to clear, with no pre- or post-plans.
         """
-        if self.state == "panicked":
-            raise RuntimeError("The RunEngine is panicked and cannot be recovered. You must restart bluesky.")
-
-        # The state machine does not capture the whole picture.
-        if not self._state.is_paused:
-            raise TransitionError(
-                f"The RunEngine is the {self._state} state. You can only resume for the paused state."
-            )
-
-        self._interrupted = False
+        self.interrupted = False
         for current_run in self._run_bundlers.values():
             current_run.record_interruption("resume")
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
-            raise RunEngineInterrupted(self.pause_msg) from None
-
-        if self._call_returns_result:
-            run_engine_result = self._create_result(plan_return)
-            return run_engine_result
-        else:
-            return tuple(self._run_start_uids)
+        self._push_plan(self._rewind())
+        await self._resume_objects()
+        # Ahead of the replayed messages.
+        self._arrange_permission(at_start=False)
+        # Last, so the plan wakes to all of the above.
+        self._run_permit.set()
```

### `_open_run`

```diff
--- RunEngine._open_run
+++ PlanRunner._open_run
@@ -1,55 +1,56 @@
-    async def _open_run(self, msg):
+    async def _open_run(self, msg: Msg) -> typing.Any:
         """Instruct the RunEngine to start a new "run"
 
         Expected message object is:
 
             Msg('open_run', None, **kwargs)
 
         where **kwargs are any additional metadata that should go into
         the RunStart document
         """
         _span = tracer.start_span(f"{_SPAN_NAME_PREFIX} run")
         _set_span_msg_attributes(_span, msg)
 
         self._run_tracing_spans.append(_span)
 
         # TODO extract this from the Msg
         run_key = msg.run
         if run_key in self._run_bundlers:
             raise IllegalMessageSequence("A 'close_run' message was not received before the 'open_run' message")
 
-        # Run scan_id calculation method
-        self.md["scan_id"] = await maybe_await(self.scan_id_source(self.md))
+        # Use the id returned: another runner may have written md since.
+        scan_id = await maybe_await(self._env.next_scan_id())
 
         # For metadata below, info about plan passed to self.__call__ for.
         plan_type = type(self._plan).__name__
         plan_name = getattr(self._plan, "__name__", "")
 
         # Combine metadata, in order of decreasing precedence:
         md = ChainMap(
             self._metadata_per_call,  # from kwargs to self.__call__
             msg.kwargs,  # from 'open_run' Msg
             {
                 "plan_type": plan_type,  # computed from self._plan
                 "plan_name": plan_name,
+                "scan_id": scan_id,  # from the session, for this run alone
             },
-            self.md,
+            self._env.md,
         )  # stateful, persistent metadata
         # The metadata is final. Validate it now, at the last moment.
-        self.md_validator(dict(md))
+        self._env.md_validator(dict(md))
 
         # Apply normalizer at the same level of the validator
-        validated = self.md_normalizer(copy.deepcopy(md))
+        validated = self._env.md_normalizer(copy.deepcopy(md))
 
-        current_run = self._run_bundlers[run_key] = type(self).RunBundler(
+        current_run = self._run_bundlers[run_key] = self._env.run_bundler_cls(
             validated,
-            self.record_interruptions,
-            self.emit,
-            self.emit_sync,
-            self.log,
-            strict_pre_declare=self._require_stream_declaration,
+            self._env.record_interruptions,
+            self._emit_async,
+            self._emit_sync,
+            self._env.log,
+            strict_pre_declare=self._env.strict_pre_declare,
         )
 
         new_uid = await current_run.open_run(msg)
-        self._run_start_uids.append(new_uid)
+        self.run_start_uids.append(new_uid)
         return new_uid
```

### `_add_status_to_group`

```diff
--- RunEngine._add_status_to_group
+++ PlanRunner._add_status_to_group
@@ -1,16 +1,19 @@
     def _add_status_to_group(self, obj: typing.Any, status_object: Status, group: str, action: str) -> None:
-        fut = self._loop.create_future()
+        loop = self._env.loop
+        fut = loop.create_future()
         pardon_failures = self._pardon_failures
+        settle = functools.partial(self._status_object_completed, status_object, fut, pardon_failures, obj, action)
 
-        def done_callback(status: Status):
-            self.log.debug("The object %r reports %r is done with status %r.", obj, action, status_object.success)
-            call_soon_or_now(self._loop, self._status_object_completed, status_object, fut, pardon_failures)
+        # A sync ophyd Status calls back on the thread that completed it, so
+        # hand the work to the loop.
+        def done_callback(*args: typing.Any, **kwargs: typing.Any) -> None:
+            call_soon_or_now(loop, settle)
 
         try:
             status_object.add_callback(done_callback)
         except AttributeError:
             # for ophyd < v0.8.0
             status_object.finished_cb = done_callback  # type: ignore
 
         self._groups[group].add(lambda: fut)
         self._status_objs[group].add(status_object)
```

### `_status_object_completed`

```diff
--- RunEngine._status_object_completed
+++ PlanRunner._status_object_completed
@@ -1,28 +1,38 @@
-    def _status_object_completed(self, ret, fut: asyncio.Future, pardon_failures):
+    def _status_object_completed(
+        self,
+        ret,
+        fut: asyncio.Future,
+        pardon_failures: asyncio.Event,
+        obj: typing.Any = None,
+        action: str | None = None,
+    ) -> None:
         """
-        Task to run when a status object is finished.
+        Task to run when a status object is finished. On the event loop.
 
         Parameters
         ----------
         ret : status object
-        p_event : asyncio.Event
+        p_event : asyncio.Future
             held in the RunEngine's self._groups cache for waiting
         pardon_failuers : asyncio.Event
             tells us whether the __call__ this status object is over
+        obj : object, optional
+            the device the status object came from, for logging
+        action : str, optional
+            what the device was asked to do, for logging
         """
+        self._env.log.debug("The object %r reports %r is done with status %r.", obj, action, ret.success)
         if not ret.success and not pardon_failures.is_set():
             # TODO: need a better channel to move this information back
             # to the run task.
-            with self._state_lock:
-                try:
-                    exc = ret.exception(timeout=0)
-                    raise FailedStatus(ret) from exc
-                except Exception as e:
-                    self._exception = e
-                    fut.set_exception(e)
-                    # We have set the exception, but we don't mind if
-                    # no-one collects it from the future, so fetch it ourselves to
-                    # squash "Future exception was never retrieved" at teardown.
-                    fut.exception()
+            try:
+                exc = ret.exception(timeout=0)
+                raise FailedStatus(ret) from exc
+            except Exception as e:
+                self._exception = e
+                fut.set_exception(e)
+                # Retrieve it ourselves, to squash "Future exception was never
+                # retrieved".
+                fut.exception()
         else:
             fut.set_result(None)
```

### `_install_suspender`

```diff
--- RunEngine._install_suspender
+++ PlanRunner._install_suspender
@@ -1,11 +1,11 @@
-    async def _install_suspender(self, msg):
+    async def _install_suspender(self, msg: Msg) -> typing.Any:
         """
-        See :meth: `RunEngine.install_suspender`
+        Install a plan-scoped suspender, removed when the plan ends.
 
         Expected message object is:
 
             Msg('install_suspender', None, suspender)
         """
         suspender = msg.args[0]
-        self._suspenders.add(suspender)
+        self._plan_suspenders.add(suspender)
         suspender.install(self._suspension)
```

### `_remove_suspender`

```diff
--- RunEngine._remove_suspender
+++ PlanRunner._remove_suspender
@@ -1,12 +1,18 @@
-    async def _remove_suspender(self, msg):
+    async def _remove_suspender(self, msg: Msg) -> typing.Any:
         """
-        See :meth: `RunEngine.remove_suspender`
+        Remove a plan-scoped suspender this plan installed.
 
         Expected message object is:
 
             Msg('remove_suspender', None, suspender)
         """
         suspender = msg.args[0]
-        if suspender in self._suspenders:
-            suspender.remove()
-        self._suspenders.discard(suspender)
+        if suspender not in self._plan_suspenders:
+            warn(
+                f"{suspender!r} is not installed on this plan, so "
+                "Msg('remove_suspender') ignored it. A plan can only remove a "
+                "suspender it installed itself.",
+                stacklevel=2,
+            )
+            return
+        self._drop_plan_suspender(suspender)
```

### `clear_suspenders`

```diff
--- RunEngine.clear_suspenders
+++ PlanRunner.clear_suspenders
@@ -1,13 +1,4 @@
-    @_runs_on_loop(timeout=SUBSCRIPTION_TIMEOUT)
-    def clear_suspenders(self):
-        """
-        Uninstall all suspenders.
-
-        See Also
-        --------
-        :meth:`RunEngine.install_suspender`
-        :meth:`RunEngine.remove_suspender`
-        """
-        for suspender in tuple(self._suspenders):
-            suspender.remove()
-        self._suspenders.clear()
+    def clear_suspenders(self) -> None:
+        """Uninstall every plan-scoped suspender of this plan."""
+        for suspender in list(self._plan_suspenders):
+            self._drop_plan_suspender(suspender)
```

### `suspenders`

```diff
--- RunEngine.suspenders
+++ PlanRunner.suspenders
@@ -1,3 +1,4 @@
     @property
-    def suspenders(self):
-        return tuple(self._suspenders)
+    def suspenders(self) -> tuple[SuspenderBase, ...]:
+        """The suspenders this plan installed, which end with it."""
+        return tuple(self._plan_suspenders)
```

### `state`

```diff
--- RunEngine.state
+++ PlanRunner.state
@@ -1,3 +1,4 @@
     @property
     def state(self):
+        """This plan's state. One of {'idle', 'running', 'paused', ...}."""
         return self._state
```

### `_RE_class`

```diff
--- RunEngine._RE_class
+++ PlanRunner._RE_class
@@ -1,5 +1,5 @@
-    async def _RE_class(self, msg):
+    async def _RE_class(self, msg: Msg) -> typing.Any:
         """
         A no-op message, mainly for debugging and testing.
         """
-        return type(self)
+        return type(self._identity)
```

### `_start_suspender`

```diff
--- RunEngine._start_suspender
+++ PlanRunner._start_suspender
@@ -1,68 +1,68 @@
-    async def _start_suspender(self, msg):
+    async def _start_suspender(self, msg: Msg) -> typing.Any:
         """
         An internal message to do the initial work of starting a suspender
         """
         (opening,) = msg.args
         for current_run in self._run_bundlers.values():
             current_run.record_interruption(join_justifications(opening) or "suspended")
         try:
             # During suspend, all motors should be stopped. Call stop() on
             # every object we ever set().
             await self._stop_movable_objects(success=True)
             # Notify Devices of the pause in case they want to clean up.
             await self._pause_objects()
         except asyncio.CancelledError:
             # Paused before the suspension began, so the resume holds instead.
             self._held_up = False
             raise
         # rewind to the last checkpoint
         rewind_plan = self._rewind()
         was_rewindable = self.rewindable
         # Every reason in this suspension, in the order it tripped.
         known = dict(opening)
 
-        def joined_or_cleared():
+        def joined_or_cleared() -> typing.Awaitable[SuspensionChange]:
             return self._suspension.wait_until(lambda c: not c.now or c.now.keys() - known.keys())
 
-        def suspender_helper_inner_plan():
+        def suspender_helper_inner_plan() -> typing.Generator[Msg, typing.Any, None]:
             try:
                 # none of this should run again.
                 yield Msg("rewindable", None, False)
                 # if there is a pre plan add on top of the wait
                 for reason in opening.values():
                     if reason.pre_plan is not None:
                         yield from ensure_generator(_called(reason.pre_plan))
                 # wait for every reason to clear; one that joins runs its pre plan
                 woken = False
                 while self._suspension.reasons:
                     joining = {key: reason for key, reason in self._suspension.reasons.items() if key not in known}
                     for key, reason in joining.items():
-                        self._announce_joined(key, reason)
+                        self._hooks.suspender_joined(key, reason)
                         known[key] = reason
                         if reason.pre_plan is not None:
                             yield from ensure_generator(_called(reason.pre_plan))
                     if not joining:
                         if woken:
                             # Woken by a pause, and resumed still tripped.
-                            self._announce_held(self._suspension.reasons, at_start=False)
+                            self._hooks.hold_began(self._suspension.reasons, False)
                         yield Msg("wait_for", None, [joined_or_cleared])
                     woken = not joining
             finally:
                 # Released: a trip from here on is a new suspension.
                 self._held_up = False
+            self._hooks.suspension_ended()
             # do the work we need to do to resume
             yield Msg(
                 "_resume_from_suspender",
                 None,
             )
             # if there is a post plan, run it
             for reason in reversed(known.values()):
                 if reason.post_plan is not None:
                     yield from ensure_generator(_called(reason.post_plan))
             # put rewindable back the way it was
             yield Msg("rewindable", None, was_rewindable)
             yield from rewind_plan
 
         # add the above helper to the plan stack
-        self._plan_stack.append(suspender_helper_inner_plan())
-        self._response_stack.append(None)
+        self._push_plan(suspender_helper_inner_plan())
```

### `_supervise_suspension`

```diff
--- RunEngine._supervise_suspension
+++ PlanRunner._supervise_suspension
@@ -1,38 +1,34 @@
-    async def _supervise_suspension(self):
-        """Suspend the running plan when the suspension trips, and announce recoveries while held up."""
+    async def _supervise_suspension(self) -> None:
+        """Suspend the running plan when the suspension trips, and report recoveries while held up."""
         # From none, so a reason tripped before this first runs is still added.
         since: Reasons = {}
         while True:
             change = await self._suspension.wait_until(lambda c: c.added or c.recovered, since=since)
             since = change.now
             # Not while pausing or paused, whose resume holds instead, nor while ending.
-            # REVIEW: The rule proposed in 1.1 of the PR body: a pause only pauses a suspension, and nothing
-            #   opens one while the plan is stopping, aborting or halting.
             running = self._state == "running"
             if self._held_up and running:
                 for key, reason in change.recovered.items():
-                    self._announce_recovered(key, reason)
+                    self._hooks.suspender_recovered(key, reason)
             if not change.now or self._held_up or not running:
                 continue
 
-            self._announce_suspended(change.now)
+            self._hooks.suspension_began(change.now)
 
             if not self.resumable:
                 # No checkpoint to rewind to, so the plan cannot be held: end it.
-                self._announce_refused(change.now)
-                self._interrupted = True
-                with self._state_lock:
-                    self._exception = FailedPause()
+                self._hooks.suspension_refused(change.now)
+                self.interrupted = True
+                self._exception = FailedPause()
                 self._state = "aborting"
-                self._task.cancel()
+                self._get_run_task().cancel()
                 continue
 
             # add starting the suspender logic to the stack
-            self._plan_stack.append(single_gen(Msg("_start_suspender", None, dict(change.now))))
-            self._response_stack.append(None)
+            self._push_plan(single_gen(Msg("_start_suspender", None, dict(change.now))))
 
             # Held up until the suspension's wait ends, so a further trip joins it.
             self._held_up = True
             self._state = "suspending"
             # bump the _run task out of what ever it is awaiting
-            self._task.cancel()
+            self._get_run_task().cancel()
```

## Renamed

### `_arrange_suspension` → `_arrange_permission`

```diff
--- RunEngine._arrange_suspension
+++ PlanRunner._arrange_permission
@@ -1,27 +1,25 @@
-    def _arrange_suspension(self, at_start: bool):
+    def _arrange_permission(self, at_start: bool) -> None:
         """Hold the plan in band while the suspension is tripped; at the start, supervise it too."""
         if at_start:
-            # Called off the loop, so the task is made on it.
-            async def start_supervisor():
-                self._supervisor = self._loop.create_task(self._supervise_suspension())
-
-            run_coro_on_loop(start_supervisor(), self._loop)
+            self._supervisor = self._loop.create_task(self._supervise_suspension())
             if not self._suspension.reasons:
                 # Decided now, so that any later trip suspends.
                 return
         if self._held_up:
             return
 
-        def hold_inner_plan(at_start):
+        def hold_inner_plan(at_start: bool) -> typing.Generator[Msg, typing.Any, None]:
             # wait until the suspension clears, with no pre- or post-plans
+            held = bool(self._suspension.reasons)
             while self._suspension.reasons:
-                self._announce_held(self._suspension.reasons, at_start)
+                self._hooks.hold_began(self._suspension.reasons, at_start)
                 # A pause ends the wait early; the plan then continues, not begins.
                 at_start = False
                 yield Msg("wait_for", None, [lambda: self._suspension.wait_until(lambda c: not c.now)])
             self._held_up = False
+            if held:
+                self._hooks.hold_ended()
 
         # On a resume, the hold looks in band, so a trip before the plan moves still holds it.
         self._held_up = True
-        self._plan_stack.append(hold_inner_plan(at_start))
-        self._response_stack.append(None)
+        self._push_plan(hold_inner_plan(at_start))
```

### `_resume` → `_resume_from_suspender`

```diff
--- RunEngine._resume
+++ PlanRunner._resume_from_suspender
@@ -1,13 +1,13 @@
-    async def _resume(self, msg):
+    async def _resume_from_suspender(self, msg: Msg) -> typing.Any:
         """The suspension is over: tell the devices.
 
         Expected message object is:
 
             Msg('_resume_from_suspender')
 
         Sent by the helper plan `_start_suspender` pushes, between the hold and
-        the post-plan. Nothing to do with `RunEngine.resume`.
+        the post-plan. Nothing to do with `PlanRunner.resume`.
 
         Monitors are untouched: a suspension never stopped them.
         """
         await self._resume_objects()
```

### `_request_pause_coro` → `pause`

```diff
--- RunEngine._request_pause_coro
+++ PlanRunner.pause
@@ -1,19 +1,20 @@
-    async def _request_pause_coro(self, defer=False):
+    async def pause(self, defer: bool = False) -> None:
+        """Bring the plan to rest at a resting point. On the loop."""
         # We are pausing. Cancel any deferred pause previously requested.
         if not self.state.can_pause:
             raise TransitionError(f"Run Engine is in '{self.state}' state and can not be paused.")
 
         if defer:
             self._deferred_pause_requested = True
-            self._announce_pause(True)
+            self._hooks.pause_requested(True)
             return
 
-        self._announce_pause(False)
+        self._hooks.pause_requested(False)
 
         self._deferred_pause_requested = False
-        self._interrupted = True
+        self.interrupted = True
         self._state = "pausing"
         for current_run in self._run_bundlers.values():
             current_run.record_interruption("pause")
 
-        self._task.cancel()
+        self._get_run_task().cancel()
```

### `_reset_checkpoint_state_meth` → `_reset_checkpoint_state`

```diff
--- RunEngine._reset_checkpoint_state_meth
+++ PlanRunner._reset_checkpoint_state
@@ -1,7 +1,8 @@
-    def _reset_checkpoint_state_meth(self):
+    def _reset_checkpoint_state(self) -> None:
+        """Forget the messages cached for a rewind, here and in every run."""
         if self._msg_cache is None:
             return
 
         self._msg_cache = deque()
         for current_run in self._run_bundlers.values():
             current_run.reset_checkpoint_state()
```

### `_stop_coro` → `stop`

```diff
--- RunEngine._stop_coro
+++ PlanRunner.stop
@@ -1,20 +1,3 @@
-    async def _stop_coro(self):
-        if self._state.is_idle:
-            raise TransitionError("RunEngine is already idle.")
-        self._announce_stop(True, True)
-
-        self._interrupted = True
-        was_paused = self._state == "paused"
-        self._state = "stopping"
-        if was_paused:
-            with self._state_lock:
-                self._exception = RequestStop()
-        else:
-            self._task.cancel()
-
-        if self._call_returns_result:
-            plan_return = self.NO_PLAN_RETURN
-            run_engine_result = self._create_result(plan_return)
-            return run_engine_result
-        else:
-            return tuple(self._run_start_uids)
+    async def stop(self) -> None:
+        """End the plan; it cleans up, and its runs close as a success. As `RunEngine.stop`."""
+        await self._end(success=True, finalize=True)
```

### `_abort_coro` → `abort`

```diff
--- RunEngine._abort_coro
+++ PlanRunner.abort
@@ -1,24 +1,6 @@
-    async def _abort_coro(self, reason):
-        if self._state.is_idle:
-            raise TransitionError("RunEngine is already idle.")
-        self._announce_stop(False, True)
-        self._interrupted = True
-        self._reason = reason
+    async def abort(self, reason: str = "") -> None:
+        """End the plan; it cleans up, and its runs close as aborted. As `RunEngine.abort`.
 
-        self._exit_status = "abort"
-        self._destroy_open_run_tracing_spans()
-
-        was_paused = self._state == "paused"
-        self._state = "aborting"
-        if was_paused:
-            with self._state_lock:
-                self._exception = RequestAbort()
-        else:
-            self._task.cancel()
-
-        if self._call_returns_result:
-            plan_return = self.NO_PLAN_RETURN
-            run_engine_result = self._create_result(plan_return)
-            return run_engine_result
-        else:
-            return tuple(self._run_start_uids)
+        ``reason`` is recorded on the RunStop of every run still open.
+        """
+        await self._end(success=False, finalize=True, reason=reason)
```

### `_halt_coro` → `halt`

```diff
--- RunEngine._halt_coro
+++ PlanRunner.halt
@@ -1,21 +1,3 @@
-    async def _halt_coro(self):
-        if self._state.is_idle:
-            raise TransitionError("RunEngine is already idle.")
-        self._announce_stop(False, False)
-        self._destroy_open_run_tracing_spans()
-        self._interrupted = True
-        was_paused = self._state == "paused"
-        self._state = "halting"
-        if was_paused:
-            with self._state_lock:
-                self._exception = PlanHalt()
-                self._exit_status = "abort"
-        else:
-            self._task.cancel()
-
-        if self._call_returns_result:
-            plan_return = self.NO_PLAN_RETURN
-            run_engine_result = self._create_result(plan_return)
-            return run_engine_result
-        else:
-            return tuple(self._run_start_uids)
+    async def halt(self) -> None:
+        """End the plan without letting it clean up; its runs close as aborted. As `RunEngine.halt`."""
+        await self._end(success=False, finalize=False)
```

### `emit_sync` → `_emit_sync`

```diff
--- RunEngine.emit_sync
+++ PlanRunner._emit_sync
@@ -1,5 +1,3 @@
-    def emit_sync(self, name, doc):
-        "Process blocking callbacks and schedule non-blocking callbacks."
-
-        # Process the doc, already validated against the schema in event-model
-        self.dispatcher.process(name, doc)
+    def _emit_sync(self, name, doc) -> None:
+        """Dispatch a document now, on the loop."""
+        self._dispatcher.process(name, doc)
```

## Moved with mechanical renames only

<details><summary><code>_checkpoint</code></summary>

```diff
--- RunEngine._checkpoint
+++ PlanRunner._checkpoint
@@ -1,20 +1,20 @@
-    async def _checkpoint(self, msg):
+    async def _checkpoint(self, msg: Msg) -> typing.Any:
         """Instruct the RunEngine to create a checkpoint so that we can rewind
         to this point if necessary
 
         Expected message object is:
 
             Msg('checkpoint')
         """
         for current_run in self._run_bundlers.values():
             if current_run.bundling:
                 raise IllegalMessageSequence("Cannot 'checkpoint' after 'create' and before 'save'. Aborting!")
 
-        await self._reset_checkpoint_state_coro()
+        self._reset_checkpoint_state()
 
         if self._deferred_pause_requested:
             # We are at a checkpoint; we are done deferring the pause.
             # Give the _check_for_signals coroutine time to look for
             # additional SIGINTs that would trigger an abort.
             await asyncio.sleep(0.5)
-            await self._request_pause_coro(defer=False)
+            await self.pause(defer=False)
```

</details>

<details><summary><code>_close_run</code></summary>

```diff
--- RunEngine._close_run
+++ PlanRunner._close_run
@@ -1,21 +1,18 @@
-    async def _close_run(self, msg):
+    async def _close_run(self, msg: Msg) -> typing.Any:
         """Instruct the RunEngine to write the RunStop document
 
         Expected message object is:
 
             Msg('close_run', None, exit_status=None, reason=None)
 
         if *exit_stats* and *reason* are not provided, use the values
         stashed on the RE.
         """
         # TODO extract this from the Msg
         run_key = msg.run
-        if (
-            current_run := self._run_bundlers.get(run_key, key_absence_sentinel := object())
-        ) is key_absence_sentinel:
-            ims_msg = "A 'close_run' message was not received before the 'open_run' message"
-            raise IllegalMessageSequence(ims_msg)
+        ims_msg = "A 'close_run' message was not received before the 'open_run' message"
+        current_run = self._bundler_for(run_key, ims_msg)
         ret = await current_run.close_run(msg)
         del self._run_bundlers[run_key]
         self._close_run_trace(msg)
         return ret
```

</details>

<details><summary><code>_close_run_trace</code></summary>

```diff
--- RunEngine._close_run_trace
+++ PlanRunner._close_run_trace
@@ -1,10 +1,10 @@
-    def _close_run_trace(self, msg: Msg):
-        exit_status = msg.kwargs.get("exit_status", self._exit_status)
-        reason = msg.kwargs.get("reason", self._reason)
+    def _close_run_trace(self, msg: Msg) -> None:
+        exit_status = msg.kwargs.get("exit_status", self.exit_status)
+        reason = msg.kwargs.get("reason", self.exit_reason)
         try:
             _span: Span = self._run_tracing_spans.pop()
             _span.set_attribute("exit_status", exit_status if exit_status is not None else "None")
             _span.set_attribute("reason", reason if reason is not None else "None")
             _span.end()
         except IndexError:
             logger.warning("No open traces left to close!")
```

</details>

<details><summary><code>_collect</code></summary>

```diff
--- RunEngine._collect
+++ PlanRunner._collect
@@ -1,20 +1,16 @@
     @tracer.start_as_current_span(f"{_SPAN_NAME_PREFIX} collect")
-    async def _collect(self, msg):
+    async def _collect(self, msg: Msg) -> typing.Any:
         """
         Collect data cached by a flyer and emit documents
 
         Expected message object is:
 
             Msg('collect', flyer_object)
             Msg('collect', flyer_object, stream=True, return_payload=False, name="a_name")
         """
         _set_span_msg_attributes(trace.get_current_span(), msg)
         run_key = msg.run
-        if (
-            current_run := self._run_bundlers.get(run_key, key_absence_sentinel := object())
-        ) is key_absence_sentinel:
-            # TODO add test exercising this path
-            ims_msg = "A 'collect' message was sent but no run is open."
-            raise IllegalMessageSequence(ims_msg)
-
+        # TODO add test exercising this path
+        ims_msg = "A 'collect' message was sent but no run is open."
+        current_run = self._bundler_for(run_key, ims_msg)
         return await current_run.collect(msg)
```

</details>

<details><summary><code>_configure</code></summary>

```diff
--- RunEngine._configure
+++ PlanRunner._configure
@@ -1,25 +1,21 @@
-    async def _configure(self, msg):
+    async def _configure(self, msg: Msg) -> typing.Any:
         """Configure an object
 
         Expected message object is:
 
             Msg('configure', object, *args, **kwargs)
 
         which results in this call:
 
             object.configure(*args, **kwargs)
         """
-        run_key = msg.run
-        if (
-            current_run := self._run_bundlers.get(run_key, key_absence_sentinel := object())
-        ) is key_absence_sentinel:
-            current_run = None
-        elif current_run.bundling:
+        current_run = self._run_bundlers.get(msg.run)
+        if current_run is not None and current_run.bundling:
             ims_msg = "Cannot configure after 'create' but before 'save' Aborting!"
             raise IllegalMessageSequence(ims_msg)
         _, obj, args, kwargs, _ = msg
 
         old, new = obj.configure(*args, **kwargs)
         if current_run:
             await current_run.configure(msg)
         return old, new
```

</details>

<details><summary><code>_create</code></summary>

```diff
--- RunEngine._create
+++ PlanRunner._create
@@ -1,24 +1,19 @@
-    async def _create(self, msg):
+    async def _create(self, msg: Msg) -> typing.Any:
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
         run_key = msg.run
-        if (
-            current_run := self._run_bundlers.get(run_key, key_absence_sentinel := object())
-        ) is key_absence_sentinel:
-            ims_msg = (
-                "Cannot bundle readings without an open run. That is, 'create' must be preceded by 'open_run'."
-            )
-            raise IllegalMessageSequence(ims_msg)
+        ims_msg = "Cannot bundle readings without an open run. That is, 'create' must be preceded by 'open_run'."
+        current_run = self._bundler_for(run_key, ims_msg)
         return await current_run.create(msg)
```

</details>

<details><summary><code>_declare_stream</code></summary>

```diff
--- RunEngine._declare_stream
+++ PlanRunner._declare_stream
@@ -1,25 +1,20 @@
-    async def _declare_stream(self, msg):
+    async def _declare_stream(self, msg: Msg) -> typing.Any:
         """Trigger the run engine to start bundling future obj.describe() calls for
          an Event document
 
         Expected message object is:
 
             Msg('declare_stream', None, name='primary')
             Msg('declare_stream', name='primary')
             Msg('create', name='primary', collect=True)
 
         Note that the `name` kwarg will be the 'name' field of the resulting
         descriptor. So descriptor['name'] = msg.kwargs['name'].
 
         If `collect` is set to True (default false) then `describe_collect` will be called
         on declare_stream, rather than `describe`.
         """
         run_key = msg.run
-        if (
-            current_run := self._run_bundlers.get(run_key, key_absence_sentinel := object())
-        ) is key_absence_sentinel:
-            ims_msg = (
-                "Cannot bundle readings without an open run. That is, 'create' must be preceded by 'open_run'."
-            )
-            raise IllegalMessageSequence(ims_msg)
+        ims_msg = "Cannot bundle readings without an open run. That is, 'create' must be preceded by 'open_run'."
+        current_run = self._bundler_for(run_key, ims_msg)
         return await current_run.declare_stream(msg)
```

</details>

<details><summary><code>_drop</code></summary>

```diff
--- RunEngine._drop
+++ PlanRunner._drop
@@ -1,15 +1,11 @@
-    async def _drop(self, msg):
+    async def _drop(self, msg: Msg) -> typing.Any:
         """Drop the event that is currently being bundled
 
         Expected message object is:
 
             Msg('drop')
         """
         run_key = msg.run
-        if (
-            current_run := self._run_bundlers.get(run_key, key_absence_sentinel := object())
-        ) is key_absence_sentinel:
-            ims_msg = "A 'drop' message was sent but no run is open."
-            raise IllegalMessageSequence(ims_msg)
-        else:
-            await current_run.drop(msg)
+        ims_msg = "A 'drop' message was sent but no run is open."
+        current_run = self._bundler_for(run_key, ims_msg)
+        await current_run.drop(msg)
```

</details>

<details><summary><code>_input</code></summary>

```diff
--- RunEngine._input
+++ PlanRunner._input
@@ -1,11 +1,11 @@
-    async def _input(self, msg):
+    async def _input(self, msg: Msg) -> typing.Any:
         """
         Process a 'input' Msg. Expected Msg:
 
             Msg('input', None)
             Msg('input', None, prompt='>')  # customize prompt
         """
         prompt = msg.kwargs.get("prompt", "")
-        async_input = AsyncInput(self.loop)
-        async_input = functools.partial(async_input, end="", flush=True)
-        return await async_input(prompt)
+        async_input = AsyncInput(self._env.loop)
+        ask = functools.partial(async_input, end="", flush=True)
+        return await ask(prompt)
```

</details>

<details><summary><code>_kickoff</code></summary>

```diff
--- RunEngine._kickoff
+++ PlanRunner._kickoff
@@ -1,38 +1,35 @@
-    async def _kickoff(self, msg):
+    async def _kickoff(self, msg: Msg) -> typing.Any:
         """Start a flyscan object
 
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
         run_key = msg.run
-        if (
-            current_run := self._run_bundlers.get(run_key, key_absence_sentinel := object())
-        ) is key_absence_sentinel:
-            ims_msg = "A 'kickoff' message was sent but no run is open."
-            raise IllegalMessageSequence(ims_msg)
+        ims_msg = "A 'kickoff' message was sent but no run is open."
+        current_run = self._bundler_for(run_key, ims_msg)
 
         _, obj, args, kwargs, _ = msg
         obj = check_supports(obj, Flyable)
         kwargs = dict(msg.kwargs)
         group = kwargs.pop("group", None)
         warn_if_msg_args_or_kwargs(msg, obj.kickoff, msg.args, kwargs)
         ret = obj.kickoff(*msg.args, **kwargs)
         await current_run.kickoff(msg)
 
         self._add_status_to_group(obj=obj, status_object=ret, group=group, action="kickoff")
 
         return ret
```

</details>

<details><summary><code>_monitor</code></summary>

```diff
--- RunEngine._monitor
+++ PlanRunner._monitor
@@ -1,26 +1,22 @@
-    async def _monitor(self, msg):
+    async def _monitor(self, msg: Msg) -> typing.Any:
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
 
         run_key = msg.run
-        if (
-            current_run := self._run_bundlers.get(run_key, key_absence_sentinel := object())
-        ) is key_absence_sentinel:
-            ims_msg = "A 'monitor' message was sent but no run is open."
-            raise IllegalMessageSequence(ims_msg)
-        else:
-            await current_run.monitor(msg)
-        await self._reset_checkpoint_state_coro()
+        ims_msg = "A 'monitor' message was sent but no run is open."
+        current_run = self._bundler_for(run_key, ims_msg)
+        await current_run.monitor(msg)
+        self._reset_checkpoint_state()
```

</details>

<details><summary><code>_pause</code></summary>

```diff
--- RunEngine._pause
+++ PlanRunner._pause
@@ -1,11 +1,11 @@
-    async def _pause(self, msg):
+    async def _pause(self, msg: Msg) -> typing.Any:
         """Request the run engine to pause
 
         Expected message object is:
 
             Msg('pause', defer=False, name=None, callback=None)
 
         See RunEngine.request_pause() docstring for explanation of the three
         keyword arguments in the `Msg` signature
         """
-        await self._request_pause_coro(*msg.args, **msg.kwargs)
+        await self.pause(*msg.args, **msg.kwargs)
```

</details>

<details><summary><code>_pause_objects</code></summary>

```diff
--- RunEngine._pause_objects
+++ PlanRunner._pause_objects
@@ -1,13 +1,9 @@
-    async def _pause_objects(self):
-        """Tell every object the plan has touched that it is being held.
-
-        `bluesky.protocols.Pausable` and not ``hasattr(obj, "pause")``: the
-        protocol requires ``resume`` too, and something told a hold has begun
-        must be something that can be told it has ended.
-        """
+    async def _pause_objects(self) -> None:
+        """Tell every `Pausable` object the plan has come to rest."""
         for obj in self._objs_seen:
             if isinstance(obj, Pausable):
                 try:
                     await maybe_await(obj.pause())
                 except NoReplayAllowed:
-                    self._reset_checkpoint_state_meth()
+                    # The device cannot be replayed through, so drop the cache.
+                    self._reset_checkpoint_state()
```

</details>

<details><summary><code>_read</code></summary>

```diff
--- RunEngine._read
+++ PlanRunner._read
@@ -1,26 +1,24 @@
-    async def _read(self, msg):
+    async def _read(self, msg: Msg) -> typing.Any:
         """
         Add a reading to the open event bundle.
 
         Expected message object is:
 
             Msg('read', obj)
         """
         obj = check_supports(msg.obj, Readable)
         # actually _read_ the object
         warn_if_msg_args_or_kwargs(msg, obj.read, msg.args, msg.kwargs)
         ret = await maybe_await(obj.read(*msg.args, **msg.kwargs))
 
         if ret is None:
             raise RuntimeError(
                 f"The read of {obj.name} returned None. "
                 "This is a bug in your object implementation, "
                 "`read` must return a dictionary."
             )
-        run_key = msg.run
-        if (
-            current_run := self._run_bundlers.get(run_key, key_absence_sentinel := object())
-        ) is not key_absence_sentinel:
+        current_run = self._run_bundlers.get(msg.run)
+        if current_run is not None:
             await current_run.read(msg, ret)
 
         return ret
```

</details>

<details><summary><code>_resume_objects</code></summary>

```diff
--- RunEngine._resume_objects
+++ PlanRunner._resume_objects
@@ -1,5 +1,8 @@
-    async def _resume_objects(self):
-        """The plan is moving again: tell the devices, so they can prepare."""
+    async def _resume_objects(self) -> None:
+        """Tell every `Pausable` object the plan is moving again.
+
+        Only on the way back in: not on abort, stop or halt.
+        """
         for obj in self._objs_seen:
             if isinstance(obj, Pausable):
                 await maybe_await(obj.resume())
```

</details>

<details><summary><code>_save</code></summary>

```diff
--- RunEngine._save
+++ PlanRunner._save
@@ -1,17 +1,13 @@
-    async def _save(self, msg):
+    async def _save(self, msg: Msg) -> typing.Any:
         """Save the event that is currently being bundled
 
         Expected message object is:
 
             Msg('save')
         """
         run_key = msg.run
-        if (
-            current_run := self._run_bundlers.get(run_key, key_absence_sentinel := object())
-        ) is key_absence_sentinel:
-            # sanity check -- this should be caught by 'create' which makes
-            # this code path impossible
-            ims_msg = "A 'save' message was sent but no run is open."
-            raise IllegalMessageSequence(ims_msg)
-        else:
-            await current_run.save(msg)
+        # sanity check -- this should be caught by 'create' which makes this
+        # code path impossible
+        ims_msg = "A 'save' message was sent but no run is open."
+        current_run = self._bundler_for(run_key, ims_msg)
+        await current_run.save(msg)
```

</details>

<details><summary><code>_stage</code></summary>

```diff
--- RunEngine._stage
+++ PlanRunner._stage
@@ -1,22 +1,22 @@
-    async def _stage(self, msg):
+    async def _stage(self, msg: Msg) -> typing.Any:
         """Instruct the RunEngine to stage the object
 
         Expected message object is:
 
             Msg('stage', object)
         """
         _, obj, args, kwargs, _ = msg
         # If an object has no 'stage' method, assume there is nothing to do.
         if not isinstance(obj, Stageable):
             return []
         group = kwargs.pop("group", None)
         ret = obj.stage()
         self._staged.add(obj)  # add first in case of failure below
-        await self._reset_checkpoint_state_coro()
+        self._reset_checkpoint_state()
 
         if not isinstance(ret, Status):
             return ret
 
         self._add_status_to_group(obj=obj, status_object=ret, group=group, action="stage")
 
         return ret
```

</details>

<details><summary><code>_subscribe</code></summary>

```diff
--- RunEngine._subscribe
+++ PlanRunner._subscribe
@@ -1,31 +1,30 @@
-    async def _subscribe(self, msg):
+    async def _subscribe(self, msg: Msg) -> typing.Any:
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
-        self.log.debug("Adding subscription %r", msg)
+        self._env.log.debug("Adding subscription %r", msg)
         _, obj, args, kwargs, _ = msg
-        token = self.subscribe(*args, **kwargs)
-        self._temp_callback_ids.add(token)
-        await self._reset_checkpoint_state_coro()
+        token = self._dispatcher.subscribe(*args, **kwargs)
+        self._reset_checkpoint_state()
         return token
```

</details>

<details><summary><code>_unmonitor</code></summary>

```diff
--- RunEngine._unmonitor
+++ PlanRunner._unmonitor
@@ -1,17 +1,13 @@
-    async def _unmonitor(self, msg):
+    async def _unmonitor(self, msg: Msg) -> typing.Any:
         """
         Stop monitoring; i.e., remove the callback emitting event documents.
 
         Expected message object is:
 
             Msg('unmonitor', obj)
         """
         run_key = msg.run
-        if (
-            current_run := self._run_bundlers.get(run_key, key_absence_sentinel := object())
-        ) is key_absence_sentinel:
-            ims_msg = "An 'unmonitor' message was sent but no run is open."
-            raise IllegalMessageSequence(ims_msg)
-        else:
-            await current_run.unmonitor(msg)
-        await self._reset_checkpoint_state_coro()
+        ims_msg = "An 'unmonitor' message was sent but no run is open."
+        current_run = self._bundler_for(run_key, ims_msg)
+        await current_run.unmonitor(msg)
+        self._reset_checkpoint_state()
```

</details>

<details><summary><code>_unstage</code></summary>

```diff
--- RunEngine._unstage
+++ PlanRunner._unstage
@@ -1,23 +1,23 @@
-    async def _unstage(self, msg):
+    async def _unstage(self, msg: Msg) -> typing.Any:
         """Instruct the RunEngine to unstage the object
 
         Expected message object is:
 
             Msg('unstage', object)
         """
         _, obj, args, kwargs, _ = msg
         # If an object has no 'unstage' method, assume there is nothing to do.
         if not isinstance(obj, Stageable):
             return []
         group = kwargs.pop("group", None)
         ret = obj.unstage()
         # use `discard()` to ignore objects that are not in the staged set.
         self._staged.discard(obj)
-        await self._reset_checkpoint_state_coro()
+        self._reset_checkpoint_state()
 
         if not isinstance(ret, Status):
             return ret
 
         self._add_status_to_group(obj=obj, status_object=ret, group=group, action="unstage")
 
         return ret
```

</details>

<details><summary><code>_unsubscribe</code></summary>

```diff
--- RunEngine._unsubscribe
+++ PlanRunner._unsubscribe
@@ -1,19 +1,18 @@
-    async def _unsubscribe(self, msg):
+    async def _unsubscribe(self, msg: Msg) -> typing.Any:
         """
         Remove a subscription during a call -- useful for a multi-run call
         where subscriptions are wanted for some runs but not others.
 
         Expected message object is:
 
             Msg('unsubscribe', None, TOKEN)
             Msg('unsubscribe', token=TOKEN)
 
         where ``TOKEN`` is the return value from ``RunEngine._subscribe()``
         """
-        self.log.debug("Removing subscription %r", msg)
+        self._env.log.debug("Removing subscription %r", msg)
         _, obj, arg, kwargs, _ = msg
         if (token := kwargs.get("token", key_absence_sentinel := object())) is key_absence_sentinel:
             (token,) = arg
-        self.unsubscribe(token)
-        self._temp_callback_ids.remove(token)
-        await self._reset_checkpoint_state_coro()
+        self._dispatcher.unsubscribe(token)
+        self._reset_checkpoint_state()
```

</details>

<details><summary><code>_wait</code></summary>

```diff
--- RunEngine._wait
+++ PlanRunner._wait
@@ -1,96 +1,96 @@
     @tracer.start_as_current_span(f"{_SPAN_NAME_PREFIX} wait")
     async def _wait(self, msg: Msg) -> bool:
         """Block progress until every object that was triggered or set
         with the keyword argument `group=<GROUP>` is done. Returns a boolean that is
         true when all triggered objects are done. When the keyword argument
         `error_on_timeout=<error_on_timeout>` is false, this method can return before all objects are done
         after a flush period given by the `timeout=<TIMEOUT>` keyword argument.
 
         Expected message object is:
 
             Msg('wait', group=<GROUP>, error_on_timeout=<ERROR_ON_TIMEOUT>)
 
         where ``<GROUP>`` is any hashable key and ``<ERROR_ON_TIMEOUT>`` is a boolean.
         """
         _set_span_msg_attributes(trace.get_current_span(), msg)
         done = False  # boolean that tracks whether waiting is complete
         if msg.args:
             (group,) = msg.args
         else:
             group = msg.kwargs["group"]
         error_on_timeout = msg.kwargs.get("error_on_timeout", True)
         watch = msg.kwargs.get("watch", ())
         watch_task: asyncio.Task | None = None
         if group:
             trace.get_current_span().set_attribute("group", group)
         else:
             trace.get_current_span().set_attribute("no_group_given", True)
         futs = self._groups.pop(group, set())
         if futs:
             status_objs = self._status_objs.pop(group)
             try:
                 if not error_on_timeout:
                     if group not in self._seen_wait_and_move_on_keys:
                         self._seen_wait_and_move_on_keys.add(group)
-                        self._call_waiting_hook(status_objs)
+                        self._hooks.waiting_on(status_objs)
                 else:  # if error_on_timeout False
                     # Notify the waiting_hook function that the RunEngine is
                     # waiting for these status_objs to complete. Users can use
                     # the information these encapsulate to create a progress
                     # bar.
-                    self._call_waiting_hook(status_objs)
+                    self._hooks.waiting_on(status_objs)
 
                 async def wait_for_first_exception(futures: set) -> list[asyncio.Future]:
                     return await self._wait_for(
                         Msg(
                             "wait_for",
                             None,
                             futures,
                             return_when=asyncio.FIRST_EXCEPTION,
                             timeout=msg.kwargs.get("timeout", None),
                         )
                     )
 
                 # Create the task waiting for the given group of statuses to complete
                 # or one of them to fail
                 status_task = asyncio.create_task(wait_for_first_exception(futs))
                 if watch:
                     # Create a task that waits for an exception on any watch group
                     # so we know whether to stop the wait early because of a watcher failure
                     watch_futs = set()
                     for w in watch:
                         watch_futs.update(self._groups.get(w, set()))
                     watch_task = asyncio.create_task(wait_for_first_exception(watch_futs))
 
-                    def cancel_status_task_if_error(fut: asyncio.Future[list[asyncio.Future]]):
+                    def cancel_status_task_if_error(fut: asyncio.Future[list[asyncio.Future]]) -> None:
                         # If _wait_for raised an exception, or if any of the status
                         # objects in the watch groups failed, cancel the status_task.
                         if fut.exception() or any(f.exception() for f in fut.result()):
                             status_task.cancel()
 
                     watch_task.add_done_callback(cancel_status_task_if_error)
                 await status_task
             except WaitForTimeoutError:
                 # We might wait to call wait again, so put the futures and status objects back in
                 self._groups[group] = futs
                 self._status_objs[group] = status_objs
                 if error_on_timeout:
                     raise
             finally:
                 if watch_task:
                     watch_task.cancel()
                 if error_on_timeout:
                     # Notify the waiting_hook function that we have moved on by
                     # sending it `None`. If all goes well, it could have
                     # inferred this from the status_obj, but there are edge
                     # cases.
-                    self._call_waiting_hook(None)
+                    self._hooks.waiting_on(None)
                     done = True
                 else:
                     done = all(obj.done for obj in status_objs)
                     if done:
-                        self._call_waiting_hook(None)
+                        self._hooks.waiting_on(None)
                         self._seen_wait_and_move_on_keys.remove(group)
         else:
             done = True
         return done
```

</details>

<details><summary><code>deferred_pause_requested</code></summary>

```diff
--- RunEngine.deferred_pause_requested
+++ PlanRunner.deferred_pause_requested
@@ -1,15 +1,4 @@
     @property
-    def deferred_pause_requested(self):
-        """
-        The property returns ``True`` if deferred pause was requested, but
-        not processed. The deferred pause is processed at the next checkpoint.
-        If the pause is requested past the last checkpoint, the plan runs
-        to completion and this property returns ``True`` until the next
-        plan is started. Starting the next plan clears deferred pause request.
-
-        Returns
-        -------
-        boolean
-            Indicates if deferred pause was requested, but not processed.
-        """
+    def deferred_pause_requested(self) -> bool:
+        """Whether a deferred pause is waiting for the next checkpoint."""
         return self._deferred_pause_requested
```

</details>

<details><summary><code>resumable</code></summary>

```diff
--- RunEngine.resumable
+++ PlanRunner.resumable
@@ -1,4 +1,4 @@
     @property
-    def resumable(self):
-        "i.e., can the plan in progress by rewound"
+    def resumable(self) -> bool:
+        "i.e., can the plan in progress be rewound"
         return self._msg_cache is not None
```

</details>

<details><summary><code>rewindable</code></summary>

```diff
--- RunEngine.rewindable
+++ PlanRunner.rewindable
@@ -1,9 +1,12 @@
     @property
-    def rewindable(self):
+    def rewindable(self) -> bool:
+        """Whether messages may be replayed on a rewind. Plans change it."""
         return self._rewindable_flag
     @rewindable.setter
-    def rewindable(self, v):
+    def rewindable(self, value: bool) -> None:
+        # A change drops the message cache. Both Msg('rewindable') and
+        # RunEngine.rewindable come through here.
         cur_state = self._rewindable_flag
-        self._rewindable_flag = bool(v)
+        self._rewindable_flag = bool(value)
         if self.resumable and self._rewindable_flag != cur_state:
             self._reset_checkpoint_state()
```

</details>

## Moved verbatim, or with type hints only

Identical: `_rewind`.

<details><summary><code>_clear_checkpoint</code></summary>

```diff
--- RunEngine._clear_checkpoint
+++ PlanRunner._clear_checkpoint
@@ -1,12 +1,12 @@
-    async def _clear_checkpoint(self, msg):
+    async def _clear_checkpoint(self, msg: Msg) -> typing.Any:
         """Clear a set checkpoint
 
         Expected message object is:
 
             Msg('clear_checkpoint')
         """
         # clear message cache
         self._msg_cache = None
         # clear stashed
         for current_run in self._run_bundlers.values():
             await current_run.clear_checkpoint(msg)
```

</details>

<details><summary><code>_complete</code></summary>

```diff
--- RunEngine._complete
+++ PlanRunner._complete
@@ -1,27 +1,27 @@
     @tracer.start_as_current_span(f"{_SPAN_NAME_PREFIX} complete")
-    async def _complete(self, msg):
+    async def _complete(self, msg: Msg) -> typing.Any:
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
         _set_span_msg_attributes(trace.get_current_span(), msg)
         kwargs = dict(msg.kwargs)
         group = kwargs.pop("group", None)
         obj = check_supports(msg.obj, Flyable)
         warn_if_msg_args_or_kwargs(msg, obj.complete, msg.args, kwargs)
         ret = obj.complete(*msg.args, **kwargs)
 
         self._add_status_to_group(obj=obj, status_object=ret, group=group, action="complete")
 
         return ret
```

</details>

<details><summary><code>_destroy_open_run_tracing_spans</code></summary>

```diff
--- RunEngine._destroy_open_run_tracing_spans
+++ PlanRunner._destroy_open_run_tracing_spans
@@ -1,5 +1,5 @@
-    def _destroy_open_run_tracing_spans(self):
+    def _destroy_open_run_tracing_spans(self) -> None:
         while len(self._run_tracing_spans):
             _span = self._run_tracing_spans.pop()
             _span.set_attribute("exit_status", "aborted")
             _span.end()
```

</details>

<details><summary><code>_locate</code></summary>

```diff
--- RunEngine._locate
+++ PlanRunner._locate
@@ -1,20 +1,20 @@
-    async def _locate(self, msg: Msg):
+    async def _locate(self, msg: Msg) -> typing.Any:
         """
         Locate some Movables and return their locations.
 
         Expected message object is:
 
             Msg('locate', obj1, ..., objn, squeeze=True)
 
         If a single obj is passed, obj.locate() is returned. If multiple objs
         are passed, obj.locate() is called in parallel for all objs and a list
         of the results returned. If squeeze is supplied and is False then it
         will always return a list of results even with a single object.
         """
         objs = [check_supports(obj, Locatable) for obj in (msg.obj,) + msg.args]
         # actually _locate_ the objects
         coros = [maybe_await(obj.locate()) for obj in objs]
         if len(coros) == 1 and msg.kwargs.get("squeeze", True):
             return await coros[0]
         else:
             return list(await asyncio.gather(*coros))
```

</details>

<details><summary><code>_null</code></summary>

```diff
--- RunEngine._null
+++ PlanRunner._null
@@ -1,5 +1,5 @@
-    async def _null(self, msg):
+    async def _null(self, msg: Msg) -> typing.Any:
         """
         A no-op message, mainly for debugging and testing.
         """
         pass
```

</details>

<details><summary><code>_prepare</code></summary>

```diff
--- RunEngine._prepare
+++ PlanRunner._prepare
@@ -1,20 +1,20 @@
-    async def _prepare(self, msg):
+    async def _prepare(self, msg: Msg) -> typing.Any:
         """Prepare a flyer for a flyscan
 
         Expected message object is:
 
         If `flyer_object` obeys the Preparable protocol, it should have a .prepare
         method that takes an argument to be set:
 
             Msg('prepare', flyer_object, value)
 
         Where value represents an initial state to move the flyer to.
         """
         obj = check_supports(msg.obj, Preparable)
         kwargs = dict(msg.kwargs)
         group = kwargs.pop("group", None)
         ret = obj.prepare(*msg.args, **kwargs)
 
         self._add_status_to_group(obj=obj, status_object=ret, group=group, action="prepare")
 
         return ret
```

</details>

<details><summary><code>_rewindable</code></summary>

```diff
--- RunEngine._rewindable
+++ PlanRunner._rewindable
@@ -1,13 +1,13 @@
-    async def _rewindable(self, msg):
+    async def _rewindable(self, msg: Msg) -> typing.Any:
         """Set rewindable state of RunEngine
 
         Expected message object is:
 
             Msg('rewindable', None, bool or None)
         """
 
         (rw_flag,) = msg.args
         if rw_flag is not None:
             self.rewindable = rw_flag
 
         return self.rewindable
```

</details>

<details><summary><code>_set</code></summary>

```diff
--- RunEngine._set
+++ PlanRunner._set
@@ -1,24 +1,24 @@
     @tracer.start_as_current_span(f"{_SPAN_NAME_PREFIX} set")
-    async def _set(self, msg):
+    async def _set(self, msg: Msg) -> typing.Any:
         """
         Set a device and cache the returned status object.
 
         Also, note that the device has been touched so it can be stopped upon
         exit.
 
         Expected message object is
 
             Msg('set', obj, *args, **kwargs)
 
         where arguments are passed through to `obj.set(*args, **kwargs)`.
         """
         _set_span_msg_attributes(trace.get_current_span(), msg)
         obj = check_supports(msg.obj, Movable)
         kwargs = dict(msg.kwargs)
         group = kwargs.pop("group", None)
         self._movable_objs_touched.add(obj)
         ret = obj.set(*msg.args, **kwargs)
 
         self._add_status_to_group(obj=obj, status_object=ret, group=group, action="set")
 
         return ret
```

</details>

<details><summary><code>_sleep</code></summary>

```diff
--- RunEngine._sleep
+++ PlanRunner._sleep
@@ -1,11 +1,11 @@
-    async def _sleep(self, msg):
+    async def _sleep(self, msg: Msg) -> typing.Any:
         """
         Sleep the event loop.
 
         Expected message object is:
 
             Msg('sleep', None, sleep_time)
 
         where `sleep_time` is in seconds
         """
         await asyncio.sleep(*msg.args)
```

</details>

<details><summary><code>_stop</code></summary>

```diff
--- RunEngine._stop
+++ PlanRunner._stop
@@ -1,10 +1,10 @@
-    async def _stop(self, msg):
+    async def _stop(self, msg: Msg) -> typing.Any:
         """
         Stop a device.
 
         Expected message object is:
 
             Msg('stop', obj)
         """
         obj = check_supports(msg.obj, Stoppable)
         return await maybe_await(obj.stop())
```

</details>

<details><summary><code>_stop_movable_objects</code></summary>

```diff
--- RunEngine._stop_movable_objects
+++ PlanRunner._stop_movable_objects
@@ -1,10 +1,10 @@
-    async def _stop_movable_objects(self, *, success=True):
+    async def _stop_movable_objects(self, *, success=True) -> None:
         "Call obj.stop() for all objects we have moved. Log any exceptions."
         for obj in self._movable_objs_touched:
             if isinstance(obj, Stoppable):
                 try:
                     await maybe_await(obj.stop(success=success))
                 except Exception:
-                    self.log.exception("Failed to stop %r.", obj)
+                    self._env.log.exception("Failed to stop %r.", obj)
             else:
-                self.log.debug("No 'stop' method available on %r", obj)
+                self._env.log.debug("No 'stop' method available on %r", obj)
```

</details>

<details><summary><code>_trigger</code></summary>

```diff
--- RunEngine._trigger
+++ PlanRunner._trigger
@@ -1,17 +1,17 @@
-    async def _trigger(self, msg):
+    async def _trigger(self, msg: Msg) -> typing.Any:
         """
         Trigger a device and cache the returned status object.
 
         Expected message object is:
 
             Msg('trigger', obj)
         """
         obj = check_supports(msg.obj, Triggerable)
         kwargs = dict(msg.kwargs)
         group = kwargs.pop("group", None)
         warn_if_msg_args_or_kwargs(msg, obj.trigger, msg.args, kwargs)
         ret = obj.trigger(*msg.args, **kwargs)
 
         self._add_status_to_group(obj=obj, status_object=ret, group=group, action="trigger")
 
         return ret
```

</details>

<details><summary><code>_wait_for</code></summary>

```diff
--- RunEngine._wait_for
+++ PlanRunner._wait_for
@@ -1,37 +1,37 @@
-    async def _wait_for(self, msg):
+    async def _wait_for(self, msg: Msg) -> typing.Any:
         """Instruct the RunEngine to wait for futures and return the resulting tasks.
 
         Expected message object is:
 
             Msg('wait_for', None, awaitable_factories, **kwargs)
 
         The keyword arguments will be passed through to `asyncio.wait`.
 
         The callables in awaitable_factories must have the signature ::
 
            def fut_fac() -> awaitable:
                'This must work multiple times'
 
         """
 
         (futs,) = msg.args
         futs = [asyncio.ensure_future(f()) for f in futs]
         # These tasks are ours: nothing else holds a reference with which to
         # cancel them. `asyncio.wait` does not cancel what it was waiting on
         # when it is itself cancelled, so a plan aborted while parked here left
         # them running on a loop about to be closed, and asyncio reported "Task
         # was destroyed but it is pending!" at some unrelated later moment.
         #
         # Only on the way out. A timeout leaves them alone deliberately: the
         # awaitables are built by factories that work more than once, and
         # waiting on the same group again after a timeout has to find whatever
         # it was waiting for still in flight.
         try:
             completed, pending = await asyncio.wait(futs, **msg.kwargs)
         except asyncio.CancelledError:
             for fut in futs:
                 fut.cancel()
             raise
         if pending:
             raise WaitForTimeoutError("Plan failed to complete in the specified time")
         return futs
```

</details>

## Rewritten

Nothing of the `RunEngine` method survives, so a diff would only mislead.

### `__init__`

```python
    def __init__(
        self,
        plan,
        env: PlanEnvironment,
        suspension: Suspension,
        hooks: PlanHooks,
        dispatcher: "Dispatcher",
        *,
        metadata: dict | None = None,
        subs=None,
        identity: typing.Any = None,
        commands: typing.Mapping[str, Callable] | None = None,
        without_commands: typing.Collection[str] = (),
        preprocessors: typing.Sequence[Callable] = (),
        initially_rewindable: bool = True,
    ) -> None:
        self._env = env
        self._hooks = hooks
        self._identity = identity if identity is not None else self

        # Create the state machine entry now, before any other thread can read it.
        _ = self._state

        self._run_tracing_spans: list[Span] = []  # open tracing spans

        # Cleared to pause; run() waits until it is set.
        self._run_permit = asyncio.Event()
        self._run_permit.set()

        # Read by tests through RunEngine._run_bundlers.
        self._run_bundlers: dict[typing.Any, RunBundler] = {}  # open run -> bundler
        self._metadata_per_call: dict[typing.Any, typing.Any] = {}  # md for every run
        self._deferred_pause_requested: bool = False  # pause at next 'checkpoint'

        # An exception instance or class, to be thrown into the plan.
        self._exception: typing.Any = None
        # The outcome, read by whoever ran the plan. `_exception` is consumed by
        # the run loop, so `exit_exception` keeps it.
        self.interrupted: bool = False
        self.exit_status: str = "success"
        self.exit_reason: str = ""
        self.exit_exception: BaseException | None = None

        self._staged: set[typing.Any] = set()  # staged, not yet unstaged
        self._objs_seen: set[typing.Any] = set()  # every object seen in a Msg
        self._movable_objs_touched: set[typing.Any] = set()  # everything we 'set'
        self.run_start_uids: list[typing.Any] = []

        # Suspenders installed by this plan, removed when it ends.
        self._plan_suspenders: set[SuspenderBase] = set()

        self._suspension = suspension
        # True from a hold or suspension opening until its wait ends.
        self._held_up = False
        # Watches the suspension while the plan runs.
        self._supervisor: asyncio.Task | None = None

        self._groups: defaultdict[str, set[Callable[[], asyncio.Future]]] = defaultdict(set)
        self._status_objs: defaultdict[typing.Any, set[typing.Any]] = defaultdict(set)
        # Read by test_flyer.py through RunEngine._seen_wait_and_move_on_keys.
        self._seen_wait_and_move_on_keys: set[typing.Any] = set()

        # Messages cached for a rewind; None once unrewindable (see `resumable`).
        self._msg_cache: deque[typing.Any] | None = deque()

        # Plans toggle this, so it lives on the runner, not the session.
        self._rewindable_flag: bool = initially_rewindable
        self._plan_stack: deque[typing.Any] = deque()  # generators to work off of
        self._response_stack: deque[typing.Any] = deque()  # responses to send into them

        # The task running the plan, created at the end of __init__. Tests
        # cancel it through RunEngine._task.
        self._task: asyncio.Task | None = None
        # Set once the plan is over, so its status objects stop reporting failures.
        self._pardon_failures = asyncio.Event()

        self._command_registry = self._build_command_registry(commands, without_commands)

        # Plan-scoped subscriptions, dropped with the runner.
        self._dispatcher = dispatcher
        for name, funcs in normalize_subs_input(subs).items():
            for func in funcs:
                self._dispatcher.subscribe(func, name)

        # Load the plan last, so preprocessors never see a half-built runner.
        self._plan = plan  # this ref is just used for metadata introspection
        if plan is None:
            # Idle runner: no plan and no task. A `RunEngine` keeps one between plans.
            return
        if metadata:
            self._metadata_per_call.update(metadata)
        gen = ensure_generator(plan)
        for wrapper_func in preprocessors:
            gen = wrapper_func(gen)
        self._push_plan(gen)

        # The plan cannot reach its first message before the loop next yields;
        # `PlanHooks.may_proceed` holds it there.
        self._task = asyncio.create_task(self._run())
```

## New

### `__await__`

```python
    def __await__(self) -> typing.Generator[typing.Any, None, typing.Any]:
        """Wait for the plan, and return what it returned.

        ::

            result = await runner

        Returns :data:`NO_PLAN_RETURN` if the plan did not complete. May be
        awaited more than once.
        """
        if self._task is None:
            raise RuntimeError(f"{self!r} was built with no plan, so there is nothing to wait for.")
        return self._task.__await__()
```

### `done`

```python
    def done(self) -> bool:
        """Whether the plan has finished, however it finished."""
        return self._task is not None and self._task.done()
```

### `_get_run_task`

```python
    def _get_run_task(self) -> asyncio.Task:
        """The task running the plan. RuntimeError for an idle runner."""
        if self._task is None:
            raise RuntimeError("No plan is running, so there is no task to interrupt.")
        return self._task
```

### `_loop`

```python
    @property
    def _loop(self) -> asyncio.AbstractEventLoop:
        """The event loop this plan is executed on."""
        return self._env.loop
```

### `_push_plan`

```python
    def _push_plan(self, plan) -> None:
        """Push a plan, with no response for it yet."""
        self._plan_stack.append(plan)
        self._response_stack.append(None)
```

### `_bundler_for`

```python
    def _bundler_for(self, run_key: typing.Any, complaint: str) -> RunBundler:
        """The bundler for ``run_key``'s open run; `IllegalMessageSequence` if none."""
        current_run = self._run_bundlers.get(run_key)
        if current_run is None:
            raise IllegalMessageSequence(complaint)
        return current_run
```

### `_end`

```python
    async def _end(self, *, success: bool, finalize: bool, reason: str = "") -> None:
        """End the plan. ``success`` closes its runs as a success; ``finalize`` lets it clean up."""
        if self.state.is_idle:
            raise TransitionError("RunEngine is already idle.")

        exception: type[BaseException]
        if success:
            state, exception = "stopping", RequestStop
        elif finalize:
            state, exception = "aborting", RequestAbort
        else:
            state, exception = "halting", PlanHalt
        self._hooks.stop_requested(success, finalize)

        self.interrupted = True
        self.exit_reason = reason
        if not success:
            # Set here: a plan that catches the exception and returns would
            # otherwise end as a success.
            self.exit_status = "abort"
            # A stop closes its spans the ordinary way.
            self._destroy_open_run_tracing_spans()

        was_paused = self.state == "paused"
        self._state = state
        if was_paused:
            # A paused plan must be released to run its cleanup. Record the
            # exception first: once released, the run loop clears `_exception`.
            self.exit_exception = exception()
            self._exception = exception
            self._run_permit.set()
        else:
            self._get_run_task().cancel()
```

### `_build_command_registry`

```python
    def _build_command_registry(
        self,
        commands: Mapping[str, Callable] | None,
        without_commands: typing.Collection[str],
    ) -> dict[str, Callable[[Msg], Awaitable[typing.Any]]]:
        """The commands this plan understands: built-ins, plus ``commands``, less ``without_commands``."""
        registry: dict[str, Callable[[Msg], Awaitable[typing.Any]]] = {
            name: fn.__get__(self) for name, fn in self._DEFAULT_COMMANDS.items()
        }
        registry.update(commands or {})
        for name in without_commands:
            registry.pop(name, None)
        return registry
```

### `_drop_plan_suspender`

```python
    def _drop_plan_suspender(self, suspender: SuspenderBase) -> None:
        """Uninstall one of this plan's plan-scoped suspenders."""
        self._plan_suspenders.discard(suspender)
        # `remove` clears the suspender's reason itself.
        suspender.remove()
```

### `_emit_async`

```python
    async def _emit_async(self, name, doc) -> None:
        """Dispatch a document, on the loop."""
        self._dispatcher.process(name, doc)
```

### `_on_state_change`

```python
    def _on_state_change(self, value, old_value) -> None:
        """Log a state change against the identity, and call the state hook."""
        announce_state_change(self._identity, self._hooks, old_value, value)
```

### `suspension_reasons`

```python
    @property
    def suspension_reasons(self) -> typing.Mapping[typing.Hashable, SuspensionReason]:
        """What is holding this plan up, including the session's reasons, by who tripped it."""
        return self._suspension.reasons
```
