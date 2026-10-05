.. _architecture:

Architecture: session, runner, suspension
===========================================

The objects behind the :class:`~bluesky.run_engine.RunEngine`, and which reaches which::

    RunEngine
    └── PlanSession               what outlives any one plan
        ├── Suspension            permission to run
        ├── Dispatcher            subscribers that outlive any plan
        ├── PlanHooks             where progress is watched from
        └╌╌ PlanRunner            one per plan, from start(plan); not held
            ├── Suspension        child of the session's
            ├── Dispatcher        child of the session's
            ├── PlanHooks         the session's, shared
            └── PlanEnvironment   frozen

The session builds one runner per plan and keeps no reference to it, so it can
run several plans at once. A :class:`~bluesky.run_engine.RunEngine` runs one at a time. See
:ref:`headless` for driving a session directly, and :ref:`plan_session_api`
for the classes.

- **Things chain upward, never downward.** A plan's suspension and dispatcher
  hold the session's; nothing holds the plan below it.
- **A plan's environment does not change under it.** What a plan reads is
  settled when it is launched.

The modules
-----------

=========================  =======================================================
Module                     Main classes, and what they are for
=========================  =======================================================
``bluesky.run_engine``     :class:`~bluesky.run_engine.RunEngine`: drives one plan at a time from a
                           terminal. Owns a session, blocks the caller, handles
                           Ctrl-C, prints what the hooks report.
``bluesky.plan_session``   :class:`~bluesky.plan_session.PlanSession`: metadata, subscribers,
                           session-scoped suspenders, hooks. :meth:`~bluesky.plan_session.PlanSession.start`
                           builds a runner.
``bluesky.plan_runner``    :class:`~bluesky.plan_runner.PlanRunner`: one plan's execution, awaitable.
                           :class:`~bluesky.plan_runner.PlanHooks`: what a runner reports.
                           :class:`~bluesky.plan_runner.PlanEnvironment`: the frozen settings a runner is
                           built with.
``bluesky.suspension``     :class:`~bluesky.suspension.Suspension`: tripped while any keyed
                           :class:`~bluesky.suspension.SuspensionReason` stands, here or in its parent.
``bluesky.suspenders``     ``SuspenderBase`` and its subclasses (:ref:`suspenders`): watch a signal,
                           trip or recover one reason on a suspension.
``bluesky.dispatcher``     :class:`~bluesky.dispatcher.Dispatcher`: hands each document to subscribers,
                           its parent's first.
``bluesky._loop``          Private. ``run_coro_on_loop``, ``call_soon_or_now``,
                           ``running_on``: the only ways onto the event loop.
=========================  =======================================================

What calls what
---------------

``RE`` is the :class:`~bluesky.run_engine.RunEngine` and ``runner`` its current
:class:`~bluesky.plan_runner.PlanRunner`. The runner reports through
:class:`~bluesky.plan_runner.PlanHooks`; ``RE`` sets most of them, to print and
to gate the plan. The gate is ``RE``'s: an event the runner awaits as
``hooks.may_proceed``.

**A plan runs.** :meth:`RE(plan) <bluesky.run_engine.RunEngine.__call__>`
crosses to the loop, shuts its gate and calls
:meth:`session.start(plan) <bluesky.plan_session.PlanSession.start>`. The
session snapshots its settings into a
:class:`~bluesky.plan_runner.PlanEnvironment` and builds a runner with a child
suspension and a child dispatcher; the runner starts its task, which waits at
the gate. Back on the main thread, ``RE`` enters its context managers
(installing the Ctrl-C handler) and opens the gate. The runner holds if its
suspension is tripped (``hooks.hold_began``), then processes messages, calling
``hooks.msg_received`` for each. ``RE`` blocks on a future carrying the
runner's result.

**A suspender trips.** A signal calls its suspender back, on whatever thread it
likes. The suspender crosses to the loop with ``call_soon_or_now``, decides,
and calls ``trip`` on its :class:`~bluesky.suspension.Suspension`. The
runner's supervisor task, parked on the suspension, wakes. Only while the plan
is running, it calls ``hooks.suspension_began`` (``RE`` prints) and pushes a
``_start_suspender`` message in front of the plan. That message stops what the
plan moved and pushes the suspension itself, a plan that runs the pre-plans,
waits for every reason to clear (calling ``hooks.suspender_joined`` as reasons
join, while the supervisor calls ``hooks.suspender_recovered`` as they
recover), calls ``hooks.suspension_ended``, runs the post-plans in reverse and
rewinds. ``RE`` is not involved beyond printing.

**A pause and a resume.** Ctrl-C reaches
:meth:`RE.request_pause() <bluesky.run_engine.RunEngine.request_pause>`, which
crosses to :meth:`runner.pause() <bluesky.plan_runner.PlanRunner.pause>`. The
runner calls ``hooks.pause_requested`` (``RE`` prints), interrupts its task,
puts its devices at rest and calls ``hooks.plan_paused``. ``RE``'s
``plan_paused`` shuts the gate and unblocks the main thread, so ``RE(plan)``
returns to the prompt. :meth:`RE.resume() <bluesky.run_engine.RunEngine.resume>`
enters the context managers, crosses to call
:meth:`runner.resume() <bluesky.plan_runner.PlanRunner.resume>` (which
rewinds, and holds again if the suspension is tripped), then opens the gate.
:meth:`RE.stop() <bluesky.run_engine.RunEngine.stop>`, ``RE.abort()`` and
``RE.halt()`` on a paused plan cross to
:meth:`runner.stop() <bluesky.plan_runner.PlanRunner.stop>`,
:meth:`runner.abort() <bluesky.plan_runner.PlanRunner.abort>` or
:meth:`runner.halt() <bluesky.plan_runner.PlanRunner.halt>`, which call
``hooks.stop_requested`` (``RE`` prints); ``RE`` then opens the gate as
``RE.resume()`` does, so the runner can clean up.

**A document.** A plan's message reaches the
:class:`~bluesky.bundlers.RunBundler`, which emits through the runner into the
plan's :class:`~bluesky.dispatcher.Dispatcher`. That calls the session's
subscribers, then the plan's own. ``RE`` is not involved: ``RE.subscribe``
subscribes to the session. A monitored signal's callback, on the device's
thread, hands its document to the loop with ``call_soon_or_now``, so
subscribers are only ever called on the loop. One arriving after its monitor
has ended is dropped, so none follows its run's stop document.

.. _which-thread:

Which thread
------------

The :class:`~bluesky.run_engine.RunEngine` is the object to call from any thread but the event loop's.
On the loop itself, where a plan body or a subscriber runs, its methods that wait for the loop raise;
``install_suspender``, ``remove_suspender`` and ``clear_suspenders`` run there directly. The session,
runner, suspensions and dispatchers are used on the event loop only, and hold no locks.

Threads outside bluesky's control reach the loop through ``bluesky._loop``:

* ``run_coro_on_loop`` crosses and waits, for a user calling the :class:`~bluesky.run_engine.RunEngine`
  from their own thread;
* ``call_soon_or_now`` crosses without waiting, for ophyd completing a status,
  a signal calling a suspender back, or a monitor producing a document.

``test_every_hop_onto_the_loop_is_one_of_the_few_we_mean`` pins these, plus the
loops ``bluesky.magics`` and ``bluesky.callbacks.zmq`` stop.

A trip is applied on the loop, so code that trips a signal and reads
``suspension_reasons`` in the next statement can see the old answer.
``suspension_reasons``, on the session and the runner, is an immutable mapping
replaced whole on each change, so it can be read from any thread.
