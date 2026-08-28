"""Tests for the seam between PlanSession, PlanExecutor and RunEngine.

These guard the properties that make the executor usable without a
RunEngine: that it is pure asyncio, and that it can be driven directly.
"""

import asyncio
import inspect
import threading

import pytest

from bluesky import Msg
from bluesky.run_engine import PlanExecutor, PlanSession

THREADING_PRIMITIVES = (
    threading.Event,
    threading.Lock().__class__,
    threading.RLock().__class__,
    threading.Condition,
    threading.Semaphore,
    threading.Barrier,
    threading.Thread,
)


def test_executor_holds_no_threading_primitives():
    """The executor is single threaded by construction.

    Everything it touches is reached from the event loop, so it needs no
    locks. If this fails, something that belongs to the RunEngine, which is
    the only thread-aware object of the three, has leaked down into it.
    """
    session = PlanSession(loop=asyncio.new_event_loop())
    executor = session.new_executor()

    offenders = {
        name: type(value).__name__
        for name, value in vars(executor).items()
        if isinstance(value, THREADING_PRIMITIVES)
    }
    assert offenders == {}


@pytest.mark.parametrize("cls", [PlanSession, PlanExecutor])
def test_source_takes_no_locks(cls):
    """Neither class ever blocks a thread, so neither may lock or join."""
    source = inspect.getsource(cls)
    for forbidden in ("threading.", "_state_lock", ".acquire(", ".join("):
        assert forbidden not in source, f"{cls.__name__} uses {forbidden}"


def test_session_holds_no_threading_primitives():
    """The session is reachable from the main thread and from the loop, but
    everything that writes to it runs on the loop, so it needs no locks."""
    session = PlanSession(loop=asyncio.new_event_loop())

    offenders = {
        name: type(value).__name__
        for name, value in vars(session).items()
        if isinstance(value, THREADING_PRIMITIVES)
    }
    assert offenders == {}


def test_run_a_plan_without_a_run_engine():
    """A PlanExecutor executes a plan with no RunEngine in the process."""
    collected = []

    async def main():
        session = PlanSession(md={"beamline": "test"})
        session.subscribe(lambda name, doc: collected.append(name))
        executor = session.new_executor()
        plan_return = await executor.run([Msg("open_run"), Msg("close_run")])
        return executor, plan_return

    executor, plan_return = asyncio.run(main())

    assert collected == ["start", "stop"]
    assert len(executor.run_start_uids) == 1
    assert executor.exit_status == "success"
    assert executor.state == "idle"
    assert not executor.interrupted
    assert plan_return is None


def test_plan_return_value_without_a_run_engine():
    async def main():
        executor = PlanSession().new_executor()

        def plan():
            yield Msg("null")
            return 42

        return await executor.run(plan())

    assert asyncio.run(main()) == 42


def test_result_describes_the_finished_plan():
    async def main():
        executor = PlanSession().new_executor()
        plan_return = await executor.run([Msg("open_run"), Msg("close_run")])
        return executor.result(plan_return)

    result = asyncio.run(main())
    assert result.exit_status == "success"
    assert not result.interrupted
    assert result.reason == ""
    assert len(result.run_start_uids) == 1


def test_session_outlives_its_executors():
    """One session, many plans in turn. Metadata and subscriptions persist."""
    names = []

    async def main():
        session = PlanSession(md={"beamline": "test"})
        session.subscribe(lambda name, doc: names.append(name))
        uids = []
        for _ in range(3):
            executor = session.new_executor()
            await executor.run([Msg("open_run"), Msg("close_run")])
            uids.extend(executor.run_start_uids)
        return session, uids

    session, uids = asyncio.run(main())
    assert len(uids) == len(set(uids)) == 3
    assert names == ["start", "stop"] * 3
    # scan_id is persistent metadata, so it counts up across plans
    assert session.md["scan_id"] == 3


def test_executor_starts_empty():
    """Building an executor is how the caches are cleared, so a new one must
    not carry anything over from the plan before it."""

    async def main():
        session = PlanSession()
        first = session.new_executor()
        await first.run([Msg("open_run"), Msg("close_run")])
        return first, session.new_executor()

    first, second = asyncio.run(main())
    assert first.run_start_uids and not second.run_start_uids
    assert second.exit_status == "success"
    assert second.exception is None
    # the caches themselves are private; this is the point of the class, so
    # reach in rather than let it go untested
    assert not second._plan_stack
    assert not second._msg_cache
    assert not second._objs_seen
    assert not second._run_bundlers


def test_run_engine_keeps_its_executor_after_the_plan(RE):
    """A finished plan can still be inspected through the RunEngine."""
    RE([Msg("open_run"), Msg("close_run")])
    assert len(RE._run_start_uids) == 1
    assert RE._executor.exit_status == "success"
    # ...and the next plan gets a fresh executor
    previous = RE._executor
    RE([Msg("open_run"), Msg("close_run")])
    assert RE._executor is not previous
    assert len(RE._run_start_uids) == 1


def test_registered_commands_survive_a_new_executor(RE):
    """register_command is remembered by the session, so it outlives the
    executor that happened to be current when it was called."""
    seen = []

    async def custom(msg):
        seen.append(msg.command)

    RE.register_command("custom-command", custom)
    for _ in range(2):
        RE([Msg("custom-command")])
    assert seen == ["custom-command"] * 2

    RE.unregister_command("custom-command")
    with pytest.raises(KeyError):
        RE([Msg("custom-command")])
