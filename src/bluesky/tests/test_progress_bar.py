import io
import time

from bluesky.plan_stubs import mv
from bluesky.tests import requires_ophyd
from bluesky.utils import (
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
    """A single manager registered on two hooks merges both streams."""
    built = []

    def factory(statuses):
        built.append(list(statuses))

        class FakeBar:
            def clear(self): ...

        return FakeBar()

    manager = ProgressBarManager(pbar_factory=factory)
    a, b = _FakeStatus(), _FakeStatus()

    manager([a])  # e.g. waiting_hook stream
    manager([b])  # e.g. progress_hook stream
    assert manager._statuses == [a, b]

    # Re-sending a known status does not duplicate it or rebuild.
    rebuilds = len(built)
    manager([a])
    assert manager._statuses == [a, b]
    assert len(built) == rebuilds

    # Finished statuses are pruned from the merged display.
    a.done = True
    manager([a])
    assert manager._statuses == [b]

    # None from a stream keeps the remaining live statuses.
    manager(None)
    assert manager._statuses == [b]

    # When every status is done the bar is torn down.
    b.done = True
    manager(None)
    assert manager._statuses == []
    assert manager.pbar is None


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
