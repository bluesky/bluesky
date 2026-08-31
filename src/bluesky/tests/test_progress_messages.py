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
    # First call: {status}, second call: None (cleanup)
    assert len(hook_calls) == 2
    assert isinstance(hook_calls[0], set)
    (status,) = hook_calls[0]
    assert isinstance(status, PlanProgress)
    assert hook_calls[1] is None


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
