import time

import pytest

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky import Msg
from bluesky.utils import IllegalMessageSequence, PlanProgress


def test_declare_progress_returns_status(RE):
    statuses = []

    def plan():
        status = yield Msg("declare_progress", name="scan")
        statuses.append(status)
        yield Msg("update_progress", name="scan", done=True)

    RE(plan())
    assert len(statuses) == 1
    assert isinstance(statuses[0], PlanProgress)
    assert statuses[0].name == "scan"
    assert statuses[0].done is True


def test_update_progress_notifies_watchers(RE):
    updates = []

    def plan():
        status = yield Msg("declare_progress", name="moving")
        status.watch(lambda **kw: updates.append(kw))
        yield Msg("update_progress", name="moving", current=3, initial=0, target=10, unit="mm")
        yield Msg("update_progress", name="moving", fraction=0.8)
        yield Msg("update_progress", name="moving", done=True)

    RE(plan())
    assert len(updates) == 3
    assert updates[0]["current"] == 3
    assert updates[0]["target"] == 10
    assert updates[0]["unit"] == "mm"
    assert updates[1]["fraction"] == 0.8
    # finish() replays the last state at 100% (current == target)
    assert updates[2]["current"] == updates[2]["target"] == 1


def test_update_progress_invalid_name(RE):
    def plan():
        yield Msg("update_progress", name="nonexistent", fraction=0.5)

    with pytest.raises(IllegalMessageSequence, match="No progress scope named 'nonexistent'"):
        RE(plan())


def test_declare_progress_duplicate_name(RE):
    def plan():
        yield Msg("declare_progress", name="scan")
        yield Msg("declare_progress", name="scan")

    with pytest.raises(IllegalMessageSequence, match="already open"):
        RE(plan())


def test_declare_progress_invalid_parent(RE):
    def plan():
        yield Msg("declare_progress", name="child", parent="nonexistent")

    with pytest.raises(IllegalMessageSequence, match="does not exist"):
        RE(plan())


def test_declare_progress_with_parent(RE):
    statuses = []

    def plan():
        yield Msg("declare_progress", name="outer")
        inner = yield Msg("declare_progress", name="inner", parent="outer")
        statuses.append(inner)
        yield Msg("update_progress", name="inner", done=True)
        yield Msg("update_progress", name="outer", done=True)

    RE(plan())
    assert statuses[0].parent is not None
    assert statuses[0].parent.name == "outer"


def test_progress_hook_called(RE):
    hook_calls = []
    RE.progress_hook = lambda x: hook_calls.append(x)

    def plan():
        yield Msg("declare_progress", name="scan")
        yield Msg("update_progress", name="scan", done=True)

    RE(plan())
    # declare: [status] (first, no prior to clear); done: None (clear), no active
    assert len(hook_calls) == 2
    assert isinstance(hook_calls[0], list)
    (status,) = hook_calls[0]
    assert isinstance(status, PlanProgress)
    assert hook_calls[1] is None


def test_progress_hook_orders_parents_before_children(RE):
    hook_calls = []
    RE.progress_hook = lambda x: hook_calls.append(x)

    def plan():
        yield Msg("declare_progress", name="outer")
        yield Msg("declare_progress", name="child_a", parent="outer")
        yield Msg("update_progress", name="child_a", done=True)
        yield Msg("declare_progress", name="child_b", parent="outer")
        yield Msg("update_progress", name="child_b", done=True)
        yield Msg("update_progress", name="outer", done=True)

    RE(plan())
    # Every non-None rebuild must list the parent before any of its children.
    for call in hook_calls:
        if call:
            names = [s.name for s in call]
            assert names[0] == "outer"
            for child in ("child_a", "child_b"):
                if child in names:
                    assert names.index("outer") < names.index(child)


def test_progress_auto_cleanup_on_run_end(RE):
    hook_calls = []
    RE.progress_hook = lambda x: hook_calls.append(x)

    def plan():
        yield Msg("declare_progress", name="scan")
        yield Msg("update_progress", name="scan", fraction=0.5)
        # Plan ends without done=True — cleanup should finish it

    RE(plan())
    # declare -> {status}, auto-cleanup -> None
    assert hook_calls[-1] is None


def test_reuse_name_after_done(RE):
    """After closing a progress scope, the same name can be reused."""

    def plan():
        yield Msg("declare_progress", name="step")
        yield Msg("update_progress", name="step", done=True)
        yield Msg("declare_progress", name="step")
        yield Msg("update_progress", name="step", done=True)

    RE(plan())  # should not raise


def test_plan_stub_progress_wrappers(RE):
    """The declare_progress/update_progress plan stubs emit the right messages."""
    updates = []

    def plan():
        status = yield from bps.declare_progress("scope")
        status.watch(lambda **kw: updates.append(kw))
        yield from bps.update_progress("scope", current=1, initial=0, target=2, unit="mm")
        yield from bps.update_progress("scope", done=True)

    RE(plan())
    assert len(updates) == 2
    assert updates[0]["current"] == 1
    assert updates[0]["target"] == 2
    assert updates[0]["unit"] == "mm"
    # finish() replays the last state at 100% (current == target)
    assert updates[1]["current"] == updates[1]["target"] == 2
    assert updates[1]["unit"] == "mm"


def test_scan_emits_progress_messages(RE, hw):
    """bp.scan threads progress_scope through scan_nd."""
    msgs = []
    RE.msg_hook = lambda msg: msgs.append(msg)
    RE(bp.scan([hw.det], hw.motor, -1, 1, 5, progress_scope="scan"))

    declares = [m for m in msgs if m.command == "declare_progress"]
    updates = [m for m in msgs if m.command == "update_progress"]
    assert len(declares) == 1
    assert declares[0].kwargs["name"] == "scan"
    assert declares[0].kwargs["parent"] is None

    steps = [m for m in updates if not m.kwargs.get("done")]
    assert len(steps) == 6  # initial 0% + 5 points
    assert steps[0].kwargs["current"] == 0
    assert steps[-1].kwargs["current"] == 5
    assert steps[-1].kwargs["target"] == 5
    assert updates[-1].kwargs["done"] is True


def test_grid_scan_emits_progress_messages(RE, hw):
    """bp.grid_scan threads progress_scope through scan_nd with a 2D cycler."""
    msgs = []
    RE.msg_hook = lambda msg: msgs.append(msg)
    RE(
        bp.grid_scan(
            [hw.det],
            hw.motor1,
            -1,
            1,
            2,
            hw.motor2,
            -1,
            1,
            3,
            progress_scope="grid",
        )
    )

    declares = [m for m in msgs if m.command == "declare_progress"]
    updates = [m for m in msgs if m.command == "update_progress"]
    assert len(declares) == 1
    assert declares[0].kwargs["name"] == "grid"

    steps = [m for m in updates if not m.kwargs.get("done")]
    assert len(steps) == 7  # initial 0% + (2 x 3 grid)
    assert steps[0].kwargs["current"] == 0
    assert steps[-1].kwargs["target"] == 6
    assert updates[-1].kwargs["done"] is True


def test_grid_scan_per_dim_progress_positions(RE, hw):
    """per_dim_progress reports each axis position zero-based-decoded from a 1-based count."""
    msgs = []
    RE.msg_hook = lambda msg: msgs.append(msg)
    RE(
        bp.grid_scan(
            [hw.det],
            hw.motor1,
            -1,
            1,
            2,
            hw.motor2,
            -1,
            1,
            3,
            progress_scope="grid",
            per_dim_progress=True,
        )
    )

    updates = [m for m in msgs if m.command == "update_progress" and not m.kwargs.get("done")]
    outer = [m.kwargs["current"] for m in updates if m.kwargs["name"] == f"grid/{hw.motor1.name}"]
    inner = [m.kwargs["current"] for m in updates if m.kwargs["name"] == f"grid/{hw.motor2.name}"]

    # Leading 0 is the initial 0% setup update for each axis.
    # Inner axis cycles 1,2,3 per outer step; outer axis ticks over after each
    # full inner sweep. A one-based decomposition would shift the inner sequence.
    assert inner == [0, 1, 2, 3, 1, 2, 3]
    assert outer == [0, 0, 0, 1, 1, 1, 2]


def test_count_emits_progress_messages(RE, hw):
    """bp.count threads progress_scope through repeat."""
    msgs = []
    RE.msg_hook = lambda msg: msgs.append(msg)
    RE(bp.count([hw.det], num=3, progress_scope="count"))

    declares = [m for m in msgs if m.command == "declare_progress"]
    updates = [m for m in msgs if m.command == "update_progress"]
    assert len(declares) == 1
    assert declares[0].kwargs["name"] == "count"

    assert len(updates) == 4  # initial 0% + 3 points
    assert updates[0].kwargs["current"] == 0
    assert updates[-1].kwargs["current"] == 3
    assert updates[-1].kwargs["target"] == 3
    assert updates[-1].kwargs["done"] is True


def test_tune_centroid_emits_progress_messages(RE, hw):
    msgs = []
    RE.msg_hook = lambda msg: msgs.append(msg)
    RE(bp.tune_centroid([hw.det], "det", hw.motor, -1.5, 1.5, 0.05, num=10, progress_scope="tune"))

    declares = [m for m in msgs if m.command == "declare_progress"]
    updates = [m for m in msgs if m.command == "update_progress"]
    assert len(declares) == 1
    assert declares[0].kwargs["name"] == "tune"
    assert declares[0].kwargs["parent"] is None

    steps = [m for m in updates if not m.kwargs.get("done")]
    # The adaptive plan has no fixed point count, so progress is a fraction.
    fractions = [m.kwargs["fraction"] for m in steps]
    assert fractions[0] == 0.0
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert all(b >= a for a, b in zip(fractions, fractions[1:]))
    assert fractions[-1] == pytest.approx(1.0)
    assert all(m.kwargs["unit"] == "step" for m in steps)

    dones = [m for m in updates if m.kwargs.get("done")]
    assert len(dones) == 1
    assert updates[-1].kwargs["done"] is True


def test_progress_elapsed_measured_on_status():
    """time_elapsed is measured on the status so rate/ETA survive bar rebuilds."""
    status = PlanProgress("scan")
    time.sleep(0.05)
    states = []
    status.watch(lambda **kw: states.append(kw))
    status._notify(current=1, initial=0, target=10, unit="step")

    # Elapsed reflects the status age, not a per-notification reset to ~0.
    assert states[-1]["time_elapsed"] is not None
    assert states[-1]["time_elapsed"] >= 0.05


def test_finish_preserves_scale_and_unit():
    """finish() completes the bar at the last-known scale/unit, not a bare fraction."""
    status = PlanProgress("scan")
    states = []
    status.watch(lambda **kw: states.append(kw))
    status._notify(current=3, initial=0, target=10, unit="step")
    status.finish()

    assert status.done is True
    assert states[-1]["current"] == states[-1]["target"] == 10
    assert states[-1]["unit"] == "step"
