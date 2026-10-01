.. _headless:

Running plans without a RunEngine
=================================

For code already inside an event loop, such as a data acquisition service.

.. warning::

   This API is provisional: names, signatures and behaviour may change without
   a deprecation period. :class:`~bluesky.run_engine.RunEngine` itself is not provisional.

A :class:`~bluesky.plan_session.PlanSession` holds what outlives any one plan:
metadata, settings, subscribers and suspenders. A
:class:`~bluesky.plan_runner.PlanRunner` executes one plan. A :class:`~bluesky.run_engine.RunEngine` uses
the same two.

Running one plan
----------------

.. doctest::

    >>> import asyncio
    >>> from bluesky import Msg
    >>> from bluesky.plan_session import PlanSession
    >>>
    >>> def plan():
    ...     yield Msg("null")
    ...     return "a value the plan returned"
    ...
    >>> async def main():
    ...     session = PlanSession()
    ...     runner = session.start(plan())
    ...     return await runner
    ...
    >>> asyncio.run(main())
    'a value the plan returned'

``start`` builds a runner from the session's current settings and starts
the plan. A malformed plan raises from ``start``. Awaiting the runner returns
what the plan returned; awaiting it again returns the same.

How the plan ended
------------------

.. doctest::

    >>> def scan():
    ...     yield Msg("open_run")
    ...     yield Msg("close_run")
    ...
    >>> async def main():
    ...     session = PlanSession()
    ...     runner = session.start(scan())
    ...     await runner
    ...     return runner.exit_status, runner.exit_reason, runner.interrupted
    ...
    >>> asyncio.run(main())
    ('success', '', False)

``exit_status`` is ``'success'``, ``'abort'`` or ``'fail'``, as recorded on
every RunStop. ``exit_reason`` says why. ``interrupted`` is True if the plan
was paused, stopped, aborted or halted. ``run_start_uids`` lists the runs it
opened. ``done()`` says whether the plan has finished; ``state`` is ``'idle'``
both before and after.

When a plan fails
-----------------

The runner cleans up, records how the plan ended, and the exception comes out
of the await:

.. doctest::

    >>> def falls_over():
    ...     yield Msg("open_run")
    ...     raise RuntimeError("the detector fell over")
    ...
    >>> async def main():
    ...     session = PlanSession()
    ...     runner = session.start(falls_over())
    ...     try:
    ...         await runner
    ...     except RuntimeError:
    ...         pass
    ...     return runner.exit_status, runner.exit_reason
    ...
    >>> asyncio.run(main())
    ('fail', 'the detector fell over')

What awaiting a runner does
---------------------------

==============================  ==========================  ===============================
The plan...                     ``await runner``            ``RE(plan)``
==============================  ==========================  ===============================
ran to its end                  returns its return value    returns
raised                          raises                      raises
was stopped, aborted or halted  returns ``NO_PLAN_RETURN``  raises ``RunEngineInterrupted``
is paused                       keeps waiting               raises ``RunEngineInterrupted``
==============================  ==========================  ===============================

.. doctest::

    >>> from bluesky.plan_runner import NO_PLAN_RETURN
    >>> def parks():
    ...     yield Msg("checkpoint")
    ...     yield Msg("pause")
    ...     yield Msg("null")
    ...     return "never reached"
    ...
    >>> async def main():
    ...     session = PlanSession()
    ...     runner = session.start(parks())
    ...     for _ in range(500):                    # let it reach the pause
    ...         if runner.state == "paused":
    ...             break
    ...         await asyncio.sleep(0.01)
    ...     await runner.stop()
    ...     result = await runner
    ...     return result is NO_PLAN_RETURN, runner.exit_status, runner.interrupted
    ...
    >>> asyncio.run(main())
    (True, 'success', True)

A paused plan's await completes only when another task resumes or ends it.
Watch ``hooks.state_changed`` to learn that it paused.

Documents
---------

Subscribe on the session to see every plan's documents:

.. doctest::

    >>> async def main():
    ...     session = PlanSession()
    ...     names = []
    ...     session.subscribe(lambda name, doc: names.append(name))
    ...     await session.start(scan())
    ...     await session.start(scan())
    ...     return names
    ...
    >>> asyncio.run(main())
    ['start', 'stop', 'start', 'stop']

Subscribe for one plan, and the subscription ends with its runner:

.. doctest::

    >>> async def main():
    ...     session = PlanSession()
    ...     durable, just_this_plan = [], []
    ...     session.subscribe(lambda name, doc: durable.append(name))
    ...     first = session.start(scan(), subs={"start": lambda n, d: just_this_plan.append(n)})
    ...     await first
    ...     await session.start(scan())
    ...     return durable, just_this_plan
    ...
    >>> asyncio.run(main())
    (['start', 'stop', 'start', 'stop'], ['start'])

The session's subscribers see each document before the plan's own.

Metadata
--------

Each plan gets a copy of ``session.md`` as it is launched, so a write takes
effect for the next plan:

.. doctest::

    >>> async def main():
    ...     session = PlanSession()
    ...     session.md["proposal"] = "p1234"
    ...     starts = []
    ...     session.subscribe(lambda name, doc: starts.append(doc), "start")
    ...     await session.start(scan())
    ...     return starts[0]["proposal"], starts[0]["scan_id"]
    ...
    >>> asyncio.run(main())
    ('p1234', 1)

``scan_id`` is allocated from the session as each run opens, so concurrent
plans never share one.

More than one plan at once
--------------------------

.. doctest::

    >>> async def main():
    ...     session = PlanSession()
    ...     first = session.start(scan())
    ...     second = session.start(scan())
    ...     await asyncio.gather(first, second)
    ...     return first.run_start_uids != second.run_start_uids, session.md["scan_id"]
    ...
    >>> asyncio.run(main())
    (True, 2)

Each plan's subscribers see only its documents, each run gets its own
``scan_id``, and one plan failing does not disturb the other. The session's
suspenders hold both.

.. warning::

    Two plans running at once must not touch the same hardware. Nothing stops
    them, and their messages will interleave. If two plans might reach the same
    device, run them one after another.

Suspending
----------

A suspender installed on the session holds up every plan it runs::

    session.install_suspender(suspender)

One installed from inside a plan, with ``Msg('install_suspender')``, holds up
only that plan, and ends with it.

Suspenders are the only way to raise a suspension, so that conditions compose:
two at once are one suspension, which ends when the last clears. To hold plans
on something else, such as a health endpoint, subclass
:class:`~bluesky.suspenders.SuspenderBase`; it needs only something that calls
it back when a value changes.

A plan started while a suspender is tripped waits before its first message.

``session.suspension_reasons`` maps whoever tripped each reason to it, and is
empty when nothing is holding plans up:

.. doctest::

    >>> async def main():
    ...     session = PlanSession()
    ...     return dict(session.suspension_reasons)
    ...
    >>> asyncio.run(main())
    {}

Driving a running plan
----------------------

The lifecycle verbs are coroutines::

    await runner.pause()                                # now
    await runner.pause(defer=True)                      # at the next checkpoint
    await runner.resume()

    await runner.stop()                                 # RunEngine.stop
    await runner.abort(reason="")                       # RunEngine.abort
    await runner.halt()                                 # RunEngine.halt

``stop`` closes runs as ``'success'``, ``abort`` and ``halt`` as ``'abort'``;
``halt`` does not let the plan run its own cleanup.

Hooks
-----

Nothing here prints. What a :class:`~bluesky.run_engine.RunEngine` prints goes to
``session.hooks``, shared by every runner and unset by default. See
:class:`~bluesky.plan_runner.PlanHooks` for each hook and how it is called.
