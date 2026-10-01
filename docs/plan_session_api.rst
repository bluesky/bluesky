.. _plan_session_api:

PlanSession and PlanRunner API
==============================

.. warning::

    Provisional. These classes are new, and may change in future releases in
    a way that is not backward-compatible.
    :class:`~bluesky.run_engine.RunEngine` is unaffected.

The classes behind the :class:`~bluesky.run_engine.RunEngine`, for running plans from code that already
has an event loop. :ref:`headless` shows them in use, and :ref:`architecture`
how they fit together.

Running plans
-------------

.. currentmodule:: bluesky.plan_session

A :class:`~bluesky.plan_session.PlanSession` holds what outlives a plan: metadata, subscribers,
session-scoped suspenders and hooks. ``start`` builds a runner for one plan.

.. autosummary::
   :nosignatures:
   :toctree: generated

   PlanSession
   PlanSession.start
   PlanSession.subscribe
   PlanSession.unsubscribe
   PlanSession.unsubscribe_all
   PlanSession.register_command
   PlanSession.unregister_command
   PlanSession.commands

Driving one plan
----------------

.. currentmodule:: bluesky.plan_runner

A :class:`~bluesky.plan_runner.PlanRunner` executes one plan. Await it for the plan's return value, or
``NO_PLAN_RETURN`` if it was stopped. These pause, resume and end it, and report
where it is.

.. autosummary::
   :nosignatures:
   :toctree: generated

   PlanRunner
   PlanRunner.pause
   PlanRunner.resume
   PlanRunner.stop
   PlanRunner.abort
   PlanRunner.halt
   PlanRunner.state
   PlanRunner.done
   PlanRunner.resumable
   PlanRunner.rewindable
   PlanRunner.deferred_pause_requested

A runner also has ``run_start_uids``, ``interrupted``, ``exit_status``,
``exit_reason`` and ``exit_exception``, filled in as the plan runs and ends;
see :class:`~bluesky.plan_runner.PlanRunner`.

Watching a plan
---------------

Nothing here prints. Each runner reports to the session's :class:`~bluesky.plan_runner.PlanHooks`; a
:class:`~bluesky.run_engine.RunEngine` prints what they report.

.. autosummary::
   :nosignatures:
   :toctree: generated

   PlanHooks

.. currentmodule:: bluesky.plan_session

.. autosummary::
   :nosignatures:
   :toctree: generated

   PlanSession.hooks

What a plan is given
--------------------

A runner is built with a frozen copy of the session's settings, so a change
to the session reaches the next plan, not the running one. The settings are
the session's attributes ``md``, ``preprocessors``, ``md_validator``,
``md_normalizer`` and ``scan_id_source``; see
:class:`~bluesky.plan_session.PlanSession`.

.. currentmodule:: bluesky.plan_runner

.. autosummary::
   :nosignatures:
   :toctree: generated

   PlanEnvironment

Suspending
----------

A plan runs only while its :class:`~bluesky.suspension.Suspension` has no reasons.
Suspenders trip it; the session's holds up every plan, a plan's only that
plan. See :ref:`suspenders` for the suspenders themselves.

.. currentmodule:: bluesky.plan_session

.. autosummary::
   :nosignatures:
   :toctree: generated

   PlanSession.install_suspender
   PlanSession.remove_suspender
   PlanSession.clear_suspenders
   PlanSession.suspenders
   PlanSession.suspension_reasons

.. currentmodule:: bluesky.plan_runner

.. autosummary::
   :nosignatures:
   :toctree: generated

   PlanRunner.suspenders
   PlanRunner.clear_suspenders
   PlanRunner.suspension_reasons

.. currentmodule:: bluesky.suspension

.. autosummary::
   :nosignatures:
   :toctree: generated

   Suspension
   SuspensionReason
   SuspensionChange

.. currentmodule:: bluesky.suspenders

.. autosummary::
   :nosignatures:
   :toctree: generated

   SuspenderBase

Documents
---------

A session and each runner have a :class:`~bluesky.dispatcher.Dispatcher`; a
runner's passes each document to the session's subscribers before its own.
:doc:`run_engine_api` documents its methods.
