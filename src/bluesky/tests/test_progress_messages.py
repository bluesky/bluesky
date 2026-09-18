import pytest

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
    # finish() sends a final notification with fraction=1.0
    assert updates[2]["fraction"] == 1.0


def test_update_progress_invalid_name(RE):
    def plan():
        yield Msg("update_progress", name="nonexistent", fraction=0.5)

    with pytest.raises(IllegalMessageSequence, match="No progress status named 'nonexistent'"):
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
    """After finishing a progress status, the same name can be reused."""

    def plan():
        yield Msg("declare_progress", name="step")
        yield Msg("update_progress", name="step", done=True)
        yield Msg("declare_progress", name="step")
        yield Msg("update_progress", name="step", done=True)

    RE(plan())  # should not raise
