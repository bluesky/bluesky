import io
import sys
import time

from bluesky.plan_stubs import mv
from bluesky.tests import requires_ophyd
from bluesky.utils import (
    PlanProgress,
    ProgressBar,
    ProgressBarManager,
    TerminalProgressBar,
    _BottomAnchorProxy,
)


@requires_ophyd
def test_status_without_watch():
    from ophyd.sim import NullStatus

    st = NullStatus()
    ProgressBar([st])


@requires_ophyd
def test_status_with_name(hw):
    from ophyd.status import DeviceStatus

    st = DeviceStatus(device=hw.det)
    pbar = ProgressBar([st])
    st._finished()

    st = DeviceStatus(device=hw.det)
    pbar = ProgressBar([st])
    assert pbar.delay_draw == 0.2
    time.sleep(0.25)
    st._finished()


def test_tuple_progress():
    class StatusPlaceholder:
        "Just enough to make ProgressBar happy. We will update manually."

        def __init__(self):
            self.done = False

        def watch(self, _): ...

    # where the status object computes the fraction
    st = StatusPlaceholder()
    pbar = ProgressBar([st])
    pbar.update(0, name="", current=(0, 0), initial=(0, 0), target=(1, 1), fraction=0)
    pbar.update(0, name="", current=(0.2, 0.2), initial=(0, 0), target=(1, 1), fraction=0.2)
    pbar.update(0, name="", current=(1, 1), initial=(0, 0), target=(1, 1), fraction=1)
    st.done = True
    pbar.update(0, name="", current=(1, 1), initial=(0, 0), target=(1, 1), fraction=1)

    # where the progress bar computes the fraction
    st = StatusPlaceholder()
    pbar = ProgressBar([st])
    pbar.update(0, name="", current=(0, 0), initial=(0, 0), target=(1, 1))
    pbar.update(0, name="", current=(0.2, 0.2), initial=(0, 0), target=(1, 1))
    pbar.update(0, name="", current=(1, 1), initial=(0, 0), target=(1, 1))
    st.done = True
    pbar.update(0, name="", current=(1, 1), initial=(0, 0), target=(1, 1))

    # minimal API
    st = StatusPlaceholder()
    pbar = ProgressBar([st])
    pbar.update(0)
    pbar.update(0)
    st.done = True
    pbar.update(0)

    # name only
    st = StatusPlaceholder()
    pbar = ProgressBar([st])
    pbar.update(0, name="foo")
    pbar.update(0, name="foo")
    st.done = True
    pbar.update(0, name="foo")


def test_mv_progress(RE, hw):
    motor1 = hw.motor1
    motor2 = hw.motor2

    RE.waiting_hook = ProgressBarManager()

    # moving time > delay_draw
    motor1.delay = 0.5
    motor2.delay = 0.5
    RE(mv(motor1, 0, motor2, 0))

    # moving time < delay_draw
    motor1.delay = 0.01
    motor2.delay = 0.01
    RE(mv(motor1, 0, motor2, 0))


def test_draw_before_update():
    class Status:
        done = False

        def watch(self, func): ...

    # Test that the default meter placeholder is valid to draw.
    pbar = ProgressBar([Status()])
    pbar.draw()


class _FakeStatus:
    def __init__(self):
        self.done = False

    def watch(self, func): ...


def test_progress_bar_manager_merges_streams():
    """A single manager driving two hook streams stacks both streams."""
    built = []

    def factory(statuses):
        built.append(list(statuses))

        class FakeBar:
            def clear(self): ...

        return FakeBar()

    manager = ProgressBarManager(pbar_factory=factory)
    waiting = manager.new_stream()
    progress = manager.new_stream()
    a, b = _FakeStatus(), _FakeStatus()

    waiting([a])
    progress([b])
    assert manager._statuses == [a, b]

    # Re-sending a stream's unchanged status does not rebuild.
    rebuilds = len(built)
    waiting([a])
    assert manager._statuses == [a, b]
    assert len(built) == rebuilds

    # Finished statuses are pruned from the merged display.
    a.done = True
    waiting([a])
    assert manager._statuses == [b]

    # None from a stream keeps the other stream's live statuses.
    progress_still = manager._statuses
    waiting(None)
    assert manager._statuses == progress_still == [b]

    # When every stream has ended the bar is torn down.
    progress(None)
    assert manager._statuses == []
    assert manager.pbar is None


def test_progress_bar_manager_none_clears_pending_stream():
    """A stream's None removes its still-pending statuses, not just the done ones."""
    manager = ProgressBarManager(pbar_factory=lambda statuses: None)
    waiting = manager.new_stream()
    progress = manager.new_stream()

    pending_wait = _FakeStatus()  # never marked done (e.g. a timed-out wait)
    plan_status = _FakeStatus()

    waiting([pending_wait])
    progress([plan_status])
    assert manager._statuses == [pending_wait, plan_status]

    # waiting_hook(None) on a timeout must drop the still-pending wait status
    # while leaving the progress stream untouched.
    waiting(None)
    assert pending_wait.done is False
    assert manager._statuses == [plan_status]


def test_bottom_anchor_proxy_reflows_bars(monkeypatch):
    """The proxy erases the bars, prints external text, then redraws them."""
    real = io.StringIO()
    manager = ProgressBarManager()
    pbar = TerminalProgressBar([_FakeStatus()], delay_draw=0)
    pbar.fp = real
    pbar.drawn = True
    pbar.done = False
    manager.pbar = pbar

    calls = []
    monkeypatch.setattr(pbar, "_erase", lambda: calls.append("erase"))
    monkeypatch.setattr(pbar, "draw", lambda: calls.append("draw"))

    proxy = _BottomAnchorProxy(real, manager)

    # A partial line is buffered; no reflow happens until the line completes.
    assert proxy.write("partial") == len("partial")
    assert calls == []
    assert "partial" not in real.getvalue()

    # Completing the line triggers erase -> write text -> draw.
    proxy.write(" line\n")
    assert calls == ["erase", "draw"]
    assert "partial line\n" in real.getvalue()


def test_bottom_anchor_proxy_passthrough_when_no_bar():
    """With no drawn terminal bar the proxy writes straight through."""
    real = io.StringIO()
    manager = ProgressBarManager()
    manager.pbar = None

    proxy = _BottomAnchorProxy(real, manager)
    assert proxy.write("hello") == len("hello")
    assert real.getvalue() == "hello"


def test_bottom_anchor_proxy_flush_emits_buffered_partial_line(monkeypatch):
    """flush() must drain a buffered partial line, not just flush the real stream."""
    real = io.StringIO()
    manager = ProgressBarManager()
    pbar = TerminalProgressBar([_FakeStatus()], delay_draw=0)
    pbar.fp = real
    pbar.drawn = True
    pbar.done = False
    manager.pbar = pbar

    calls = []
    monkeypatch.setattr(pbar, "_erase", lambda: calls.append("erase"))
    monkeypatch.setattr(pbar, "draw", lambda: calls.append("draw"))

    proxy = _BottomAnchorProxy(real, manager)

    # A newline-less write is buffered and would otherwise be lost on flush.
    proxy.write("message")
    assert "message" not in real.getvalue()

    proxy.flush()
    assert calls == ["erase", "draw"]
    assert "message" in real.getvalue()


def test_bottom_anchor_proxy_teardown_emits_buffered_partial_line():
    """Removing the proxy must flush buffered text instead of discarding it."""
    real = io.StringIO()
    manager = ProgressBarManager()

    pbar = TerminalProgressBar([_FakeStatus()], delay_draw=0)
    pbar.fp = real
    pbar.drawn = True
    pbar.done = False
    manager.pbar = pbar

    proxy = _BottomAnchorProxy(real, manager)
    manager._proxy = proxy
    saved_stdout = sys.stdout
    sys.stdout = proxy
    try:
        proxy.write("buffered")
        assert "buffered" not in real.getvalue()

        # Tearing down the proxy (as _rebuild does) must emit the buffered text.
        manager.pbar = None
        manager._remove_proxy()
        assert "buffered" in real.getvalue()
        assert manager._proxy is None
        assert sys.stdout is real
    finally:
        sys.stdout = saved_stdout


def test_rebuild_does_not_accumulate_watchers(monkeypatch):
    """Rebuilding the bar does not add watchers to long-lived statuses."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    scan = PlanProgress("scan")
    bars = []

    def factory(status_objs):
        bar = TerminalProgressBar(status_objs, delay_draw=0)
        bars.append(bar)
        return bar

    manager = ProgressBarManager(factory)
    progress_hook = manager.new_stream()
    waiting_hook = manager.new_stream()

    progress_hook([scan])
    for _ in range(10):
        # Each move status coming and going triggers a rebuild.
        waiting_hook([PlanProgress("move")])
        waiting_hook(None)

    assert len(bars) == 21
    assert len(scan._watchers) == 1
    # Bars receive wrappers that expose the real status.
    assert bars[-1].status_objs[0].status is scan

    scan._notify(current=1, initial=0, target=2)
    # Only the current bar is updated; it also got the replayed state on creation.
    assert "scan" in bars[-1].meters[0]
    assert all(bar.meters[0] == "" or bar.done for bar in bars[:-1])


def test_proxy_does_not_split_a_line_around_the_first_draw(monkeypatch):
    """A bar first drawn while a line is half-printed must not land inside that line."""
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    status = PlanProgress("scan")
    manager = ProgressBarManager(lambda objs: TerminalProgressBar(objs, delay_draw=10))
    manager([status])
    assert not manager.pbar.drawn

    sys.stdout.write("abc")
    # The bar's first draw arrives (e.g. from another thread) before the newline.
    manager.pbar._force_draw = True
    status._notify(current=1, initial=0, target=2)
    sys.stdout.write("\n")
    manager(None)

    text = out.getvalue()
    assert "abc\n" in text
    assert "abc\r" not in text
