The RunEngine run loop
======================

*Note: This is a technical document not optimized for user readability.*

In this document, we start with a simplified version of the bluesky RunEngine.
We add more complexity step by step, with commentary.

The heart of bluesky is the ``RunEngine._run`` co-routine which dispatches the
``Msg`` in the plan to functions that actually carry out the requested task.
The core operation is obscured by the layers of exception handling, state
management, and clean up the RunEngine is responsible for. (Some of this may
be refactored in the near future). This document is only going to discuss the
run loop, not Document generation or hardware clean up.

Minimal RunEngine
-----------------

A minimal (run-able) RunEngine is

.. code:: python

    from time import sleep
    import datetime
    now = datetime.datetime.now
    from bluesky import Msg

    function_map = {'print':
                    lambda msg: print('-- {!s:10.10s} : {: <25.25s} --'.format(now().time(), msg.obj)),
                    'sleep':
                    lambda msg: sleep(msg.args[0])}


    def RE_v0(plan):
        for msg in plan:
            func = function_map[msg.command]
            func(msg)

    welcome_plan = [Msg('print', 'hello'), Msg('sleep', None, 1), Msg('print', 'world!')]

    RE_v0(welcome_plan)

which captures one of the key abstractions of bluesky: A plan is
just an iterable of messages. This abstraction means that the to plan an
experiment you only need to generate a stream of ``Msg`` objects and the
RunEngine will take care of actually executing the code.

Adaptive Plans
--------------

Simply having a stream of commands is not quite enough, you may want to
have the code generating the stream of messages be aware of the return
value of a previous ``Msg`` to decide what to do next. This sort of
thing is supported in python using
`generators <https://docs.python.org/3.5/reference/expressions.html#generator-iterator-methods>`__
which 'suspend' their execution at a ``yield`` statement. When you
iterate over a generator, it runs until the next ``yield``
statement, suspends, and yields the value to the code which is iterating
over it.

Switching to generators requires we change our minimal RE to

.. code:: python

    from bluesky.utils import ensure_generator



    def RE_v1(plan):
        plan = ensure_generator(plan)
        last_result = None

        while True:
            try:
                msg = plan.send(last_result)
            except StopIteration:
                # generators indicate they are done by raising
                # StopIteration
                break
            func = function_map[msg.command]
            last_result = func(msg)


which still works with the ``welcome_plan``

.. code:: python

    RE_v1([Msg('print', 'hello'), Msg('sleep', None, 1), Msg('print', 'world!')])

but we can also do more sophisticated things like

.. code:: python

    function_map['sum'] = lambda msg: sum(msg.args)

    def adding_plan(a, b):
        yield Msg('print', '{} + {} = ??'.format(a, b))
        ret = yield Msg('sum', None, a, b)
        yield Msg('print', '{} + {} = {}'.format(a, b, ret))
        yield Msg('print', 'thanks for adding')

Which gives

.. code:: python

    RE_v1(adding_plan(1, 2))
    RE_v1(adding_plan(5, 2))

This is obviously overkill for simple addition, but enables this like an
adaptive dscan that changes the step size based on the local slope.

Exception Handling
------------------

In addition to ``generator.send`` (which inserts a value into the
generator) you can also use ``generator.throw`` which raises an
exception at the point where the generator is paused. If the generator
handles the exception (via a ``try...except`` block) then generator
runs until the next ``yield`` and ``throw`` returns the yielded
value. If the generator does not handle the exception (or raises a
different exception) then it is (re)raised by ``throw``.

We want to be able to capture any exceptions raised by the ``RE``
and pass those back to the plan.

.. code:: python


    def RE_v2(plan):
        plan = ensure_generator(plan)
        last_result = None
        _exception = None
        while True:
            try:
                if _exception is not None:
                    msg = plan.throw(_exception)
                    _exception = None
                else:
                    msg = plan.send(last_result)

            except StopIteration:
                break
            try:
                func = function_map[msg.command]
                last_result = func(msg)
            except Exception as e:
                _exception = e


We can now write plans that handle exception from the RE, in this case
reporting that the addition failed due to a ``TypeError``

.. code:: python

    def safe_adding_plan(a, b):
        yield Msg('print', '{} + {} = ??'.format(a, b))
        try:
            ret = yield Msg('sum', None, a, b)
        except TypeError:
            yield Msg('print', 'can not add {} + {}!'.format(a, b))
        else:
            yield Msg('print', '{} + {} = {}'.format(a, b, ret))
        finally:
            yield Msg('print', 'thanks for adding')

Compare the behavior of between ``adding_plan`` and ``addingplan`` in cases
where they succeed

.. code:: python

    RE_v2(safe_adding_plan(1, 2))
    RE_v2(adding_plan(1, 2))

and fail

.. code:: python

    RE_v2(safe_adding_plan('a', 2))
    RE_v2(adding_plan('a', 2))

Again, this is overkill for these simple cases, but this mechanism
allows us to write delta scans that always return the motors to their
original position, shut shutters, etc even if the plan fails or is
canceled.

Turn into a callable class
--------------------------

We are going to want to have access to the internal state of the
``_run`` loop very soon. An way to do this, while maintaining
the API we have above is to write a callable class instead of a
function.

.. code:: python

    class RunEngine_v3:
        def _sleep(self, msg):
            sleep(msg.args[0])

        def _print(self, msg):
            print('-- {!s:10.10s} : {: <25.25s} --'.format(now().time(), msg.obj)),

        def _sum(self, msg):
            return sum(msg.args)

        def __init__(self):
            self._command_registry = {
                'print': self._print,
                'sum': self._sum,
                'sleep': self._sleep}

        def __call__(self, plan):
            self._run(plan)

        def _run(self, plan):
            plan = ensure_generator(plan)
            last_result = None
            _exception = None
            while True:
                try:
                    if _exception is not None:
                        msg = plan.throw(_exception)
                        _exception = None
                    else:
                        msg = plan.send(last_result)

                except StopIteration:
                    break
                try:
                    func = self._command_registry[msg.command]
                    last_result = func(msg)
                except Exception as e:
                    _exception = e


    RE_v3 = RunEngine_v3()

In doing this we also pulled the function the commands dispatched to
into the class. While these methods are almost trivial, we will soon
have methods that alter the internal state of the ``RunEngine``.

``asyncio`` integration
-----------------------

So far all of these RE implementations have been synchronous functions,
that is they run straight through the plan. However, at a beamline we
need to be able to support asynchronous functionality and gracefully
interrupt the plan.

To enable this we are using ``asyncio`` from the python standard library
(new in 3.4) to provide the outer event loop. At this point we are
integrating together two event loops: the RE loop which is processing
the plan and the ``asyncio`` event loop which is managing multiple
frames of execution. The event loop may switch between execution frames
when a coroutine is suspended by a ``yield from`` statement. Thus we
change the methods we dispatch to and the main ``_run`` method to
co-routines by adding the ``@asyncio.coroutine`` decorator and calling
the dispatched functions via ``yield from`` rather than with a direct
function call.

We also added a ``msg_hook`` attribute to the ``RunEngine``
which is a super handy debugging tool to see exactly what messages are
being processed by the RunEngine. It can be set to any callable which
takes a single ``Msg`` as input (ex ``print``)

.. code:: python

    import asyncio


    class RunEngine_v4:
        def __init__(self, *, loop=None):
            # map messages to coro
            self._command_registry = {
                'print': self._print,
                'sum': self._sum,
                'sleep': self._sleep}

            # debugging hook
            self.msg_hook = None


            # bind RE to a specific loop
            if loop is None:
                loop = asyncio.get_event_loop()
            self.loop = loop

            # The RunEngine keeps track of a *lot* of state.
            # All flags and caches are defined here with a comment. Good luck.
            self._task = None  # asyncio.Task associated with call to self._run

        def __call__(self, plan):
            self._task = self.loop.create_task(self._run(plan))
            self.loop.run_until_complete(self._task)

            if self._task.done() and not self._task.cancelled():
                exc = self._task.exception()
                if exc is not None:
                    raise exc

        @asyncio.coroutine
        def _run(self, plan):
            plan = ensure_generator(plan)
            last_result = None
            _exception = None
            while True:
                try:
                    yield from asyncio.sleep(0.0001, loop=self.loop)
                    if _exception is not None:
                        msg = plan.throw(_exception)
                        _exception = None
                    else:
                        msg = plan.send(last_result)

                except StopIteration:
                    break

                if self.msg_hook:
                    self.msg_hook(msg)

                try:
                    func = self._command_registry[msg.command]
                    last_result = yield from func(msg)
                except Exception as e:
                    _exception = e

        @asyncio.coroutine
        def _sleep(self, msg):
            yield from asyncio.sleep(msg.args[0])

        @asyncio.coroutine
        def _print(self, msg):
            print('-- {!s:10.10s} : {: <25.25s} --'.format(now().time(), msg.obj)),

        @asyncio.coroutine
        def _sum(self, msg):
            return sum(msg.args)



    RE_v4 = RunEngine_v4()

Pausing, Resuming, and Rewinding
--------------------------------

Adding the ability to pause/resume/rewind a scan adds a fair amount of
complexity as now the ``RunEngine`` must keep track of a stack of plans
rather than a single plan, cache ``Msg`` as they go by and expose enough
API to control the behavior.

.. code:: python

    from collections import deque
    import asyncio

    import datetime
    import functools
    from bluesky.utils import (AsyncInput, FailedPause, InvalidCommand, Msg,
                               ensure_generator)
    from bluesky.run_engine import RunEngineStateMachine, PropertyMachine
    from super_state_machine.errors import TransitionError


    class RunEngine_v5:
        state = PropertyMachine(RunEngineStateMachine)
        _UNCACHEABLE_COMMANDS = ['pause', ]

        def __init__(self, *, loop=None):
            # map messages to coro
            self._command_registry = {
                'print': self._print,
                'sum': self._sum,
                # coros on real RE
                'sleep': self._sleep,
                'checkpoint': self._checkpoint,
                'clear_checkpoint': self._clear_checkpoint,
                'rewindable': self._rewindable,
                'pause': self._pause,
                'input': self._input,
                'null': self._null, }

            # debugging hook
            self.msg_hook = None

            # bind RE to a specific loop
            if loop is None:
                loop = asyncio.get_event_loop()
            self.loop = loop

            # The RunEngine keeps track of a *lot* of state.
            # All flags and caches are defined here with a comment. Good luck.
            self._task = None  # asyncio.Task associated with call to self._run

            self._deferred_pause_requested = False  # pause at next 'checkpoint'
            self._msg_cache = deque()  # history of processed msgs for rewinding
            self._rewindable_flag = True  # if the RE is allowed to replay msgs
            self._plan = None  # the scan plan instance from __call__
            self._plan_stack = deque()  # stack of generators to work off of
            self._response_stack = deque([None])  # resps to send into the plans
            self._interrupted = False  # True if paused, aborted, or failed

        def __call__(self, plan):
            # First thing's first: if we are in the wrong state, raise.
            if not self.state.is_idle:
                raise RuntimeError("The RunEngine is in a %s state" % self.state)

            self._clear_call_cache()

            self._plan = plan
            gen = ensure_generator(plan)

            self._plan_stack.append(gen)
            self._response_stack.append(None)

            self._task = self.loop.create_task(self._run())
            self.loop.run_forever()

            if self._task.done() and not self._task.cancelled():
                exc = self._task.exception()
                if exc is not None:
                    raise exc

        def _clear_call_cache(self):
            self._deferred_pause_requested = False
            self._plan_stack = deque()
            self._msg_cache = deque()
            self._response_stack = deque([None])
            self._exception = None
            self._task = None
            self._plan = None
            self._interrupted = False

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
        def resumable(self):
            "i.e., can the plan in progress by rewound"
            return self._msg_cache is not None

        @asyncio.coroutine
        def _run(self):
            pending_cancel_exception = None
            try:
                self.state = 'running'
                while True:
                    try:
                        yield from asyncio.sleep(0.0001, loop=self.loop)
                        # The case where we have a stashed exception
                        if self._exception is not None:
                            # throw the exception at the current plan
                            try:
                                msg = self._plan_stack[-1].throw(
                                    self._exception)
                            except Exception as e:
                                # The current plan did not handle it,
                                # maybe the next plan (if any) would like
                                # to try
                                self._plan_stack.pop()
                                if len(self._plan_stack):
                                    self._exception = e
                                    continue
                                # no plans left and still an unhandled exception
                                # re-raise to exit the infinite loop
                                else:
                                    raise
                            # clear the stashed exception, the top plan
                            # handled it.
                            else:
                                self._exception = None
                        # The normal case of clean operation
                        else:
                            resp = self._response_stack.pop()
                            try:
                                msg = self._plan_stack[-1].send(resp)
                            # We have exhausted the top generator
                            except StopIteration:
                                # pop the dead generator go back to the top
                                self._plan_stack.pop()
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
                                if len(self._plan_stack):
                                    self._exception = e
                                    continue
                                # or reraise to get out of the infinite loop
                                else:
                                    raise

                        if self.msg_hook:
                            self.msg_hook(msg)

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
                            self._exception = InvalidCommand(msg.command)
                            continue

                        # try to finally run the command the user asked for
                        try:
                            # this is one of two places that 'async'
                            # exceptions (coming in via throw) can be
                            # raised
                            response = yield from coro(msg)
                        # special case `CancelledError` and let the outer
                        # exception block deal with it.
                        except asyncio.CancelledError:
                            raise
                        # any other exception, stash it and go to the top of loop
                        except Exception as e:
                            self._exception = e
                            continue
                        # normal use, if it runs cleanly, stash the response and
                        # go to the top of the loop
                        else:
                            self._response_stack.append(response)
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
                    except asyncio.CancelledError as e:
                        # if we are handling this twice, raise and leave the plans
                        # alone
                        if self._exception is e:
                            raise e
                        # the case where FailedPause, RequestAbort or a coro
                        # raised error is not already stashed in _exception
                        if self._exception is None:
                            self._exception = e
                        pending_cancel_exception = e
            except StopIteration:
                pass
            finally:
                self.loop.stop()
                self.state = 'idle'
            # if the task was cancelled
            if pending_cancel_exception is not None:
                raise pending_cancel_exception
        @asyncio.coroutine
        def _sleep(self, msg):
            yield from asyncio.sleep(msg.args[0])

        @asyncio.coroutine
        def _print(self, msg):
            now = datetime.datetime.now
            print('-- {!s:10.10s} : {: <25.25s} --'.format(now().time(), msg.obj))

        @asyncio.coroutine
        def _sum(self, msg):
            return sum(msg.args)

        @asyncio.coroutine
        def _input(self, msg):
            """
            Process a 'input' Msg. Expected Msg:

                Msg('input', None)
                Msg('input', None, prompt='>')  # customize prompt
            """
            prompt = msg.kwargs.get('prompt', '')
            async_input = AsyncInput(self.loop)
            async_input = functools.partial(async_input, end='', flush=True)
            return (yield from async_input(prompt))

        @asyncio.coroutine
        def _pause(self, msg):
            """Request the run engine to pause

            Expected message object is:

                Msg('pause', defer=False, name=None, callback=None)

            See RunEngine.request_pause() docstring for explanation of the three
            keyword arguments in the `Msg` signature
            """
            self.request_pause(*msg.args, **msg.kwargs)

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
            if not self.resumable:
                # cannot resume, so we cannot pause.  Abort the scan
                print("No checkpoint; cannot pause.")
                print("Aborting: running cleanup and marking "
                      "exit_status as 'abort'...")
                self._exception = FailedPause()
                self._task.cancel()
                for task in self._failed_status_tasks:
                    task.cancel()
                return
            # stop accepting new tasks in the event loop (existing tasks will
            # still be processed)
            self.loop.stop()

        def resume(self):
            """Resume a paused plan from the last checkpoint.

            Returns
            -------
            uids : list
                list of Header uids (a.k.a RunStart uids) of run(s)
            """
            # The state machine does not capture the whole picture.
            if not self.state.is_paused:
                raise TransitionError("The RunEngine is the {0} state. "
                                      "You can only resume for the paused state."
                                      "".format(self.state))

            self._interrupted = False
            new_plan = self._rewind()
            self._plan_stack.append(new_plan)
            self._response_stack.append(None)

            self._resume_event_loop()
            return []

        def _rewind(self):
            '''Clean up in preparation for resuming from a pause or suspension.

            Returns
            -------
            new_plan : generator
                 A new plan made from the messages in the message cache

            '''
            new_plan = ensure_generator(list(self._msg_cache))
            self._msg_cache = deque()
            # This is needed to 'cancel' an open bundling (e.g. create) if
            # the pause happens after a 'checkpoint', after a 'create', but before
            # the paired 'save'.
            return new_plan

        def _resume_event_loop(self):
            # may be called by 'resume' or 'abort'
            self.state = 'running'
            self._last_sigint_time = None
            self._num_sigints_processed = 0

            if self._task.done():
                return
            self.loop.run_forever()
            if self._task.done() and not self._task.cancelled():
                exc = self._task.exception()
                if exc is not None:
                    raise exc

        @asyncio.coroutine
        def _checkpoint(self, msg):
            """Instruct the RunEngine to create a checkpoint so that we can rewind
            to this point if necessary

            Expected message object is:

                Msg('checkpoint')
            """
            yield from self._reset_checkpoint_state_coro()

            if self._deferred_pause_requested:
                # We are at a checkpoint; we are done deferring the pause.
                # Give the _check_for_signals coroutine time to look for
                # additional SIGINTs that would trigger an abort.
                yield from asyncio.sleep(0.5, loop=self.loop)
                self.request_pause(defer=False)

        def _reset_checkpoint_state(self):
            if self._msg_cache is None:
                return

            self._msg_cache = deque()

        _reset_checkpoint_state_coro = asyncio.coroutine(_reset_checkpoint_state)

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
        def _rewindable(self, msg):
            '''Set rewindable state of RunEngine

            Expected message object is:

                Msg('rewindable', None, bool or None)
            '''

            rw_flag, = msg.args
            if rw_flag is not None:
                self.rewindable = rw_flag

            return self.rewindable

        @asyncio.coroutine
        def _null(self, msg):
            """
            A no-op message, mainly for debugging and testing.
            """
            pass


    RE_v5 = RunEngine_v5()
    RE_v5.msg_hook = print


    def pausing_plan():
        yield Msg('null')
        yield Msg('null')
        yield Msg('pause')
        yield Msg('null')

Stop, Abort, Halt
-----------------

Suspending
----------

Object/hardware clean up
------------------------

Document creation and emission
------------------------------

SIGINT interception
-------------------
