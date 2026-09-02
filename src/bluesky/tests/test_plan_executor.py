"""Tests for the seam between PlanSession, PlanExecutor and RunEngine.

These guard the properties that make the executor usable without a
RunEngine: that it is pure asyncio, and that it can be driven directly.
"""

import asyncio
import inspect
import threading

import pytest

from bluesky import Msg
from bluesky.plan_executor import PlanExecutor, PlanSession
from bluesky.tests import requires_ophyd
from bluesky.utils import RunEngineInterrupted

THREADING_PRIMITIVES = (
    threading.Event,
    threading.Lock().__class__,
    threading.RLock().__class__,
    threading.Condition,
    threading.Semaphore,
    threading.Barrier,
    threading.Thread,
)


@pytest.fixture
def idle_session():
    """A session on a loop of its own, for tests that never run a plan."""
    loop = asyncio.new_event_loop()
    try:
        yield PlanSession(loop=loop)
    finally:
        loop.close()


def test_the_old_import_location_still_works():
    """Both classes were defined in run_engine before they moved here, and it
    goes on re-exporting them for code written against that."""
    from bluesky import run_engine

    assert run_engine.PlanSession is PlanSession
    assert run_engine.PlanExecutor is PlanExecutor


def test_executor_holds_no_threading_primitives(idle_session):
    """The executor is single threaded by construction.

    Everything it touches is reached from the event loop, so it needs no
    locks. If this fails, something that belongs to the RunEngine, which is
    the only thread-aware object of the three, has leaked down into it.
    """
    executor = idle_session.new_executor()

    offenders = {
        name: type(value).__name__
        for name, value in vars(executor).items()
        if isinstance(value, THREADING_PRIMITIVES)
    }
    assert offenders == {}


@pytest.mark.parametrize("cls", [PlanSession, PlanExecutor])
def test_source_takes_no_locks(cls):
    """Neither class ever blocks a thread, so neither may lock or join.

    ``threading.get_ident()`` is allowed: PlanExecutor.emit reads it to
    decide whether it is already on the loop thread before optionally
    marshalling onto it. Reading the current thread's id blocks nothing, so
    it is exempted from the blanket "threading." ban below.
    """
    source = inspect.getsource(cls).replace("threading.get_ident(", "")
    for forbidden in ("threading.", "_state_lock", ".acquire(", ".join("):
        assert forbidden not in source, f"{cls.__name__} uses {forbidden}"


def test_session_holds_no_threading_primitives(idle_session):
    """The session is reachable from the main thread and from the loop, but
    everything that writes to it runs on the loop, so it needs no locks."""
    offenders = {
        name: type(value).__name__
        for name, value in vars(idle_session).items()
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


def test_request_pause_coro_survives_for_queueserver(RE):
    """bluesky-queueserver drives a non-blocking pause through this coroutine.

    Its worker cannot call the public ``request_pause``, which blocks and
    never returns if the loop is wedged, so it reaches for the private
    coroutine instead. There is no public equivalent yet, so this has to keep
    working.
    """

    def pause_from_another_thread():
        asyncio.run_coroutine_threadsafe(RE._request_pause_coro(False), loop=RE.loop).result()

    def plan():
        yield Msg("checkpoint")
        threading.Timer(0.1, pause_from_another_thread).start()
        yield Msg("sleep", None, 2)
        yield Msg("null")

    with pytest.raises(RunEngineInterrupted):
        RE(plan())
    assert RE.state == "paused"
    RE.stop()


# --- marshal_monitor_emission -----------------------------------------------
#
# A sync ophyd signal fires its monitor callback on whichever thread called
# .put(), and PlanExecutor.emit's default behaviour is to invoke subscribers
# on that same thread rather than the loop's. marshal_monitor_emission (a
# PlanSession kwarg, falling back to the BLUESKY_MARSHAL_MONITOR_EMISSION
# environment variable) turns that around. See bluesky#2050 for the tradeoffs;
# this is off by default so today's behaviour is unchanged unless someone
# opts in.


def test_marshal_monitor_emission_defaults_to_off(monkeypatch):
    """With no kwarg and no environment variable, the guard is off."""
    monkeypatch.delenv("BLUESKY_MARSHAL_MONITOR_EMISSION", raising=False)
    assert PlanSession().marshal_monitor_emission is False


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        ("True", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("False", False),
        ("no", False),
        ("off", False),
        ("", False),
    ],
)
def test_env_var_controls_marshal_monitor_emission(monkeypatch, value, expected):
    """The environment variable is the rollout knob for when no kwarg is given."""
    monkeypatch.setenv("BLUESKY_MARSHAL_MONITOR_EMISSION", value)
    assert PlanSession().marshal_monitor_emission is expected


def test_explicit_kwarg_overrides_the_env_var(monkeypatch):
    """The keyword argument wins over the environment variable either way."""
    monkeypatch.setenv("BLUESKY_MARSHAL_MONITOR_EMISSION", "1")
    assert PlanSession(marshal_monitor_emission=False).marshal_monitor_emission is False

    monkeypatch.delenv("BLUESKY_MARSHAL_MONITOR_EMISSION", raising=False)
    assert PlanSession(marshal_monitor_emission=True).marshal_monitor_emission is True


@requires_ophyd
def test_monitor_subscriber_runs_on_the_device_thread_by_default(RE):
    """Today's behaviour, pinned: a sync signal's monitor callback is invoked
    on whichever thread called .put(), not the RunEngine's loop thread."""
    import ophyd

    sig = ophyd.Signal(name="sig", value=0)
    seen_thread_ids = []
    monitoring = threading.Event()

    def collector(*args, **kwargs):
        seen_thread_ids.append(threading.get_ident())

    RE.subscribe(collector, "event")

    putter_thread_id = {}

    def putter():
        assert monitoring.wait(timeout=5)
        putter_thread_id["id"] = threading.get_ident()
        sig.put(1)

    thread = threading.Thread(target=putter)
    thread.start()

    def plan():
        yield Msg("open_run")
        yield Msg("monitor", sig)
        monitoring.set()
        yield Msg("sleep", None, 0.3)
        yield Msg("unmonitor", sig)
        yield Msg("close_run")

    RE(plan())
    thread.join(timeout=5)

    assert seen_thread_ids, "the monitor callback never fired"
    assert RE._th.ident != putter_thread_id["id"]
    assert seen_thread_ids == [putter_thread_id["id"]]


@requires_ophyd
def test_monitor_subscriber_runs_on_the_loop_thread_when_marshalled(RE):
    """With the guard on, the same subscriber runs on the loop thread instead."""
    import ophyd

    RE._session.marshal_monitor_emission = True

    sig = ophyd.Signal(name="sig", value=0)
    seen_thread_ids = []
    monitoring = threading.Event()

    def collector(*args, **kwargs):
        seen_thread_ids.append(threading.get_ident())

    RE.subscribe(collector, "event")

    putter_thread_id = {}

    def putter():
        assert monitoring.wait(timeout=5)
        putter_thread_id["id"] = threading.get_ident()
        sig.put(1)

    thread = threading.Thread(target=putter)
    thread.start()

    def plan():
        yield Msg("open_run")
        yield Msg("monitor", sig)
        monitoring.set()
        yield Msg("sleep", None, 0.3)
        yield Msg("unmonitor", sig)
        yield Msg("close_run")

    RE(plan())
    thread.join(timeout=5)

    assert seen_thread_ids, "the monitor callback never fired"
    assert putter_thread_id["id"] != RE._th.ident, "the put() itself should still be off the loop thread"
    assert seen_thread_ids == [RE._th.ident], "the callback should have been marshalled onto the loop thread"
