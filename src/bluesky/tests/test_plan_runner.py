"""Tests for PlanSession and PlanRunner without a RunEngine, and the RunEngine's use of them."""

import asyncio
import dataclasses
import inspect
import pathlib
import threading

import pytest
from ophyd.signal import Signal

import bluesky
from bluesky import Msg
from bluesky.plan_runner import PlanEnvironment, PlanRunner
from bluesky.plan_session import PlanSession
from bluesky.suspenders import SuspendBoolHigh
from bluesky.utils import InvalidCommand, RunEngineInterrupted

from .utils import _parked


class _RecordingSignal:
    """A Subscribable stand-in: enough for a suspender to be constructed."""

    name = "recording"

    def subscribe_reading(self, function):
        pass

    def clear_sub(self, function):
        pass


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


@pytest.mark.parametrize("cls", [PlanSession, PlanRunner])
def test_source_takes_no_locks(cls):
    source = inspect.getsource(cls)
    for forbidden in ("threading.", "_state_lock", ".acquire(", ".join("):
        assert forbidden not in source


def test_a_suspender_holds_no_threading_primitives():
    # Its state is only touched on the loop.
    from bluesky.suspenders import SuspendBoolHigh

    suspender = SuspendBoolHigh(_RecordingSignal())
    offenders = {
        name: type(value).__name__
        for name, value in vars(suspender).items()
        if isinstance(value, THREADING_PRIMITIVES)
    }
    assert offenders == {}


def test_the_runner_never_says_what_to_press():
    source = (pathlib.Path(bluesky.__file__).parent / "plan_runner.py").read_text()

    assert "Ctrl" not in source


def _look_at_a_fresh_runner(session, look, plan=()):
    """Build a runner on the session's loop, pass it to ``look``, and cancel its task."""

    async def main():
        runner = session.start(plan)
        try:
            return look(runner)
        finally:
            runner._task.cancel()

    return session._loop.run_until_complete(main())


def test_runner_holds_no_threading_primitives(idle_session):
    # Used only on the loop.

    def look(runner):
        return {
            name: type(value).__name__
            for name, value in vars(runner).items()
            if isinstance(value, THREADING_PRIMITIVES)
        }

    assert _look_at_a_fresh_runner(idle_session, look) == {}


def test_session_holds_no_threading_primitives(idle_session):
    # Written only on the loop.
    offenders = {
        name: type(value).__name__
        for name, value in vars(idle_session).items()
        if isinstance(value, THREADING_PRIMITIVES)
    }
    assert offenders == {}


def test_run_a_plan_without_a_run_engine():
    collected = []

    async def main():
        session = PlanSession(md={"beamline": "test"})
        session.subscribe(lambda name, doc: collected.append(name))
        runner = session.start([Msg("open_run"), Msg("close_run")])
        plan_return = await runner
        return runner, plan_return

    runner, plan_return = asyncio.run(main())

    assert collected == ["start", "stop"]
    assert len(runner.run_start_uids) == 1
    assert runner.exit_status == "success"
    assert runner.state == "idle"
    assert not runner.interrupted
    assert plan_return is None


def test_plan_return_value_without_a_run_engine():
    async def main():
        def plan():
            yield Msg("null")
            return 42

        runner = PlanSession().start(plan())
        return await runner

    assert asyncio.run(main()) == 42


def test_the_may_proceed_hook_holds_the_plan_before_its_first_message():
    seen = []

    async def main():
        session = PlanSession()
        held = asyncio.Event()

        async def hold():
            seen.append("hook")
            await held.wait()

        session.hooks.may_proceed = hold
        runner = session.start([Msg("null")])
        running = asyncio.ensure_future(runner)

        await _parked()
        state_while_held = str(runner.state)
        seen.append("released")
        held.set()
        await running
        return state_while_held, str(runner.state)

    state_while_held, state_after = asyncio.run(main())

    assert seen == ["hook", "released"]
    # The hook is awaited before the state leaves 'idle'.
    assert state_while_held == "idle"
    assert state_after == "idle"


def test_the_may_proceed_hook_holds_a_plan_released_from_a_pause():
    # Still 'paused' while it waits.
    seen = []

    async def main():
        session = PlanSession()
        permitted = asyncio.Event()
        permitted.set()
        paused = asyncio.Event()
        session.hooks.may_proceed = permitted.wait
        session.hooks.plan_paused = paused.set
        session.hooks.msg_received = lambda msg: seen.append(msg.command)
        runner = session.start([Msg("checkpoint"), Msg("pause"), Msg("null")])
        running = asyncio.ensure_future(runner)

        await paused.wait()
        permitted.clear()
        await runner.resume()
        await _parked()
        held = str(runner.state), list(seen)
        permitted.set()
        await running
        return held

    state_while_held, seen_while_held = asyncio.run(main())

    assert state_while_held == "paused"
    assert seen_while_held == ["checkpoint", "pause"]
    assert seen[-1] == "null"


def test_awaiting_twice_answers_the_same_thing_twice():

    async def main():
        def plan():
            yield Msg("null")
            return 42

        runner = PlanSession().start(plan())
        return await runner, await runner

    assert asyncio.run(main()) == (42, 42)


def test_awaiting_an_runner_with_no_plan_says_so(idle_session):

    async def main():
        with pytest.raises(RuntimeError, match="no plan"):
            await idle_session._idle_runner()

    idle_session._loop.run_until_complete(main())


def test_done_says_what_idle_cannot():
    # 'idle' means both "not started" and "finished".
    seen = {}

    async def main():
        session = PlanSession()
        held = asyncio.Event()
        session.hooks.may_proceed = held.wait
        runner = session.start([Msg("null")])

        await _parked()
        seen["held"] = (str(runner.state), runner.done())

        held.set()
        await runner
        seen["finished"] = (str(runner.state), runner.done())

    asyncio.run(main())

    assert seen["held"] == ("idle", False)
    assert seen["finished"] == ("idle", True)


def test_a_synchronous_proceed_hook_is_allowed():
    called = []

    async def main():
        session = PlanSession()
        session.hooks.may_proceed = lambda: called.append("proceed")
        await session.start([Msg("null")])

    asyncio.run(main())

    assert called == ["proceed"]


def test_the_runner_says_how_the_plan_finished():

    async def main():
        runner = PlanSession().start([Msg("open_run"), Msg("close_run")])
        await runner
        return runner

    runner = asyncio.run(main())
    assert runner.exit_status == "success"
    assert runner.exit_reason == ""
    assert not runner.interrupted
    assert len(runner.run_start_uids) == 1


def test_session_outlives_its_runners():
    # Metadata and subscriptions persist across plans.
    names = []

    async def main():
        session = PlanSession(md={"beamline": "test"})
        session.subscribe(lambda name, doc: names.append(name))
        uids = []
        for _ in range(3):
            runner = session.start([Msg("open_run"), Msg("close_run")])
            await runner
            uids.extend(runner.run_start_uids)
        return session, uids

    session, uids = asyncio.run(main())
    assert len(uids) == len(set(uids)) == 3
    assert names == ["start", "stop"] * 3
    # scan_id is persistent metadata, so it counts up across plans
    assert session.md["scan_id"] == 3


def test_two_plans_run_at_once_on_one_session():
    starts = []

    async def main():
        session = PlanSession(md={"beamline": "test"})
        session.subscribe(lambda name, doc: starts.append(doc) if name == "start" else None)
        plan = [Msg("open_run"), Msg("sleep", None, 0.05), Msg("close_run")]
        first = session.start(list(plan))
        second = session.start(list(plan))
        await asyncio.gather(first, second)
        return session, first, second

    session, first, second = asyncio.run(main())

    assert first.exit_status == second.exit_status == "success"
    assert first.run_start_uids != second.run_start_uids
    # Each run got its own scan id.
    assert sorted(doc["scan_id"] for doc in starts) == [1, 2]
    assert session.md["scan_id"] == 2


def test_a_session_suspender_holds_every_plan_running_under_it():
    finished: list[str] = []

    async def main():
        session = PlanSession()
        sig = Signal(value=0, name="beam")
        # Already bad, so both are held at their first message.
        sig.put(1)
        session.install_suspender(SuspendBoolHigh(sig))

        async def run(name, runner):
            await runner
            finished.append(name)

        first = session.start([Msg("checkpoint"), Msg("null")])
        second = session.start([Msg("checkpoint"), Msg("null")])
        runners = asyncio.gather(run("first", first), run("second", second))
        await _parked()
        held = list(finished)
        sig.put(0)
        await runners
        return held

    held = asyncio.run(main())
    assert held == []
    assert sorted(finished) == ["first", "second"]


def test_a_plans_own_suspender_holds_only_that_plan():
    finished: list[str] = []

    async def main():
        session = PlanSession()
        sig = Signal(value=0, name="mine")
        # Bad before install, so installing holds this plan.
        sig.put(1)
        susp = SuspendBoolHigh(sig)

        async def run(name, runner):
            await runner
            finished.append(name)

        held = session.start([Msg("checkpoint"), Msg("install_suspender", None, susp), Msg("null")])
        free = session.start([Msg("checkpoint"), Msg("null")])
        runners = asyncio.gather(run("held", held), run("free", free))
        await _parked()
        meanwhile = list(finished)
        sig.put(0)
        await runners
        return meanwhile

    meanwhile = asyncio.run(main())
    assert meanwhile == ["free"]
    assert finished == ["free", "held"]


def test_two_plans_documents_reach_the_session_and_only_their_own_subscribers():

    async def main():
        session = PlanSession()
        session_starts: list[str] = []
        first_starts: list[str] = []
        session.subscribe(lambda name, doc: session_starts.append(doc["uid"]) if name == "start" else None)

        plan = [Msg("open_run"), Msg("sleep", None, 0.05), Msg("close_run")]
        first = session.start(
            list(plan),
            subs={"start": [lambda name, doc: first_starts.append(doc["uid"])]},
        )
        second = session.start(list(plan))
        await asyncio.gather(first, second)
        return session_starts, first_starts, first, second

    session_starts, first_starts, first, second = asyncio.run(main())

    assert sorted(session_starts) == sorted([*first.run_start_uids, *second.run_start_uids])
    assert first_starts == list(first.run_start_uids)


def test_one_plan_failing_leaves_the_other_alone():

    async def main():
        session = PlanSession()
        good = session.start([Msg("open_run"), Msg("sleep", None, 0.1), Msg("close_run")])
        bad = session.start([Msg("open_run"), Msg("aardvark")])
        outcomes = await asyncio.gather(good, bad, return_exceptions=True)
        return good, bad, outcomes

    good, bad, outcomes = asyncio.run(main())

    assert good.exit_status == "success"
    assert isinstance(outcomes[1], InvalidCommand)
    assert bad.exit_status == "fail"


def test_a_setting_reaches_the_next_plan_and_not_the_running_one():

    async def main():
        session = PlanSession()
        assert not hasattr(session, "env") and not hasattr(session, "_env")

        already_built = session.start([Msg("null")])
        session.strict_pre_declare = True
        built_after = session.start([Msg("null")])

        assert already_built._env.strict_pre_declare is False
        assert built_after._env.strict_pre_declare is True
        assert already_built._env is not built_after._env

        await asyncio.gather(already_built, built_after)

    asyncio.run(main())


def test_metadata_is_snapshotted_and_the_session_keeps_its_own_store():

    async def main():
        session = PlanSession()
        store = {"from_the_store": True}
        session.md = store
        runner = session.start([Msg("null")])

        assert runner._env.md == {"from_the_store": True}
        assert runner._env.md is not store

        session.md["written_after_launch"] = True
        assert "written_after_launch" not in runner._env.md
        assert session.md is store

        await runner

    asyncio.run(main())


def test_the_environment_describes_only_what_a_running_plan_reads():
    # Preprocessors and the initial rewindable are arguments, not environment.
    fields = {f.name for f in dataclasses.fields(PlanEnvironment)}
    assert "preprocessors" not in fields
    assert "rewindable" not in fields


def test_a_plan_installs_a_suspender_for_itself_only():

    class _Susp:
        def __init__(self):
            self.installed_on = None
            self.removed = False

        def install(self, owner):
            self.installed_on = owner

        def remove(self):
            self.removed = True

    susp = _Susp()

    async def main():
        session = PlanSession()
        runner = session.start([Msg("install_suspender", None, susp)])
        await runner
        return session, runner

    session, runner = asyncio.run(main())

    assert susp.installed_on is runner._suspension
    assert susp.removed
    assert susp not in session.suspenders
    assert susp not in runner.suspenders


def test_a_session_scoped_suspender_outlives_the_plan_it_held():
    # It stays subscribed between plans.

    class _Susp:
        def __init__(self):
            self.installed_on = None
            self.removed = False

        def install(self, owner):
            self.installed_on = owner

        def remove(self):
            self.removed = True

    susp = _Susp()

    started: list[str] = []

    def next_plan():
        started.append("ran")
        yield Msg("null")

    async def main():
        session = PlanSession()
        session.install_suspender(susp)
        runner = session.start([Msg("null")])
        await runner
        # Trips after that plan ended, so the next plan is held.
        session._suspension.trip(susp, "beam is down")

        nxt = session.start(next_plan())
        task = asyncio.ensure_future(nxt)
        await _parked()
        held = not started
        session._suspension.recover(susp)
        await asyncio.wait_for(task, timeout=10)
        return session, runner, held

    session, runner, next_plan_was_held = asyncio.run(main())

    assert susp.installed_on is session._suspension
    assert not susp.removed
    assert susp in session.suspenders
    assert next_plan_was_held
    assert started == ["ran"]


def test_one_session_scoped_suspender_covers_every_running_plan():

    async def main():
        session = PlanSession()
        first = session.start([Msg("null")])
        second = session.start([Msg("null")])
        assert first._suspension.reasons or not session._suspension.reasons
        assert first._suspension is not second._suspension
        assert first._suspension is not second._suspension
        await asyncio.gather(first, second)

    asyncio.run(main())


def test_runner_starts_empty():

    async def main():
        session = PlanSession()
        first = session.start([Msg("open_run"), Msg("close_run")])
        await first
        # Read before yielding, as building the second runner started its plan.
        second = session.start([])
        try:
            return (
                first,
                second,
                {
                    name: getattr(second, name)
                    for name in ("run_start_uids", "exit_status", "_exception", "_msg_cache", "_objs_seen")
                },
                len(second._plan_stack),
                dict(second._run_bundlers),
            )
        finally:
            second._task.cancel()

    first, second, born, plan_stack_depth, run_bundlers = asyncio.run(main())
    assert first.run_start_uids and not born["run_start_uids"]
    assert born["exit_status"] == "success"
    assert born["_exception"] is None
    assert not born["_msg_cache"]
    assert not born["_objs_seen"]
    assert not run_bundlers
    # The plan stack holds this runner's own plan from construction.
    assert plan_stack_depth == 1


def test_the_old_import_location_still_works():
    from bluesky import run_engine

    assert run_engine.PlanSession is PlanSession
    assert run_engine.PlanRunner is PlanRunner


def _crossings(rel_path):
    """The innermost function around every hop onto the loop in a module."""
    import ast

    source = pathlib.Path(bluesky.__file__).parent / rel_path
    tree = ast.parse(source.read_text())
    crossing = {"call_soon_threadsafe", "run_coroutine_threadsafe"}
    found = set()

    def walk(node, enclosing):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, child.name)
            else:
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr in crossing
                ):
                    found.add(enclosing)
                walk(child, enclosing)

    walk(tree, None)
    return found


def test_every_hop_onto_the_loop_is_one_of_the_few_we_mean():
    # _loop.py holds the two ways in. magics.py and callbacks/zmq.py each stop a loop of
    # their own. A new hop belongs in _loop.py.
    package_dir = pathlib.Path(bluesky.__file__).parent
    hops = {}
    for path in sorted(package_dir.rglob("*.py")):
        rel = path.relative_to(package_dir)
        if rel.parts[0] in ("tests", "_vendor"):
            continue
        found = _crossings(str(rel))
        if found:
            hops[rel.as_posix()] = found

    assert hops == {
        "_loop.py": {"run_coro_on_loop", "call_soon_or_now"},
        "magics.py": {"_shutdown_magics_run_engine"},
        "callbacks/zmq.py": {"stop"},
    }


def test_a_malformed_plan_raises_on_the_calling_thread(RE):
    # The engine stays usable.
    with pytest.raises(TypeError):
        RE(42)

    assert RE._runner.state.is_idle
    RE([Msg("null")])


def test_run_engine_keeps_its_runner_after_the_plan(RE):
    RE([Msg("open_run"), Msg("close_run")])
    assert len(RE._run_start_uids) == 1
    assert RE._exit_status == "success"
    # ...and the next plan gets a fresh runner
    previous = RE._runner
    RE([Msg("open_run"), Msg("close_run")])
    assert RE._runner is not previous
    assert len(RE._run_start_uids) == 1


def test_registered_commands_survive_a_new_runner(RE):
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
    # Used by bluesky-queueserver.

    def pause_from_another_thread():
        asyncio.run_coroutine_threadsafe(RE._request_pause_coro(False), loop=RE.loop).result()

    def plan():
        yield Msg("checkpoint")
        threading.Thread(target=pause_from_another_thread).start()
        # Paused long before this ends.
        yield Msg("sleep", None, 10)
        yield Msg("null")

    with pytest.raises(RunEngineInterrupted):
        RE(plan())
    assert RE.state == "paused"
    RE.stop()


def test_session_subscribers_see_a_document_before_the_plan_s(RE):
    seen = []
    RE.subscribe(lambda name, doc: seen.append(("session", name)), "start")

    RE(
        [Msg("open_run"), Msg("close_run")],
        {"start": lambda name, doc: seen.append(("plan", name))},
    )

    assert [who for who, _ in seen] == ["session", "plan"]


def test_ignore_callback_exceptions_is_read_live_by_a_plan(RE):
    RE.ignore_callback_exceptions = True
    # Held at `hooks.may_proceed`, so the plan cannot run meanwhile.
    RE._new_runner([Msg("null")])
    assert RE._runner._dispatcher.ignore_exceptions is True

    RE.ignore_callback_exceptions = False
    assert RE._runner._dispatcher.ignore_exceptions is False

    # Replacing it with an idle runner discards it.
    RE._new_runner()


def test_re_class_answers_for_whoever_is_driving(RE):
    # The class of the runner's identity, else of the runner.
    seen = []

    def note():
        seen.append((yield Msg("RE_class")))

    RE(note())
    assert seen == [type(RE)]

    async def headless():
        seen.clear()
        await PlanSession().start(note())

    asyncio.run(headless())
    assert seen == [PlanRunner]


def test_a_plans_subscription_tokens_never_collide_with_its_sessions():
    from bluesky.dispatcher import Dispatcher, DocumentNames

    session = Dispatcher()
    plan = Dispatcher(session)
    seen = []
    session_token = session.subscribe(lambda name, doc: None)
    plan.subscribe(lambda name, doc: seen.append(name))

    # The session's token is not the plan's, so its own subscriber stays.
    plan.unsubscribe(session_token)
    plan.process(DocumentNames.start, {})
    assert seen == ["start"]
