import signal
import time
from contextlib import contextmanager

import pytest

from bluesky.callbacks import CallbackBase
from bluesky.callbacks.buffer import BufferingWrapper


class SlowDummyCallback(CallbackBase):
    """Simulates a slow (blocking) callback for testing threaded wrappers."""

    def __init__(self, delay=0.1):
        self.delay = delay
        self.called = []

    def __call__(self, name, doc):
        time.sleep(self.delay)  # Simulate slow work
        self.called.append((name, doc))


@pytest.fixture
def fast_cb():
    yield SlowDummyCallback(delay=0)


@pytest.fixture
def slow_cb():
    yield SlowDummyCallback(delay=0.1)


@contextmanager
def wait_for_condition(condition, timeout=3, interval=0.01):
    """Wait for a condition to become True within a timeout period.

    Parameters
    ----------
        condition : callable
            A function that returns True when the condition is met.
        timeout : float
            Maximum time to wait for the condition to become True.
        interval : float, optional
            Time to wait between checks of the condition (default is 0.01 seconds).

    Usage
    -----
        with wait_for_condition(1, lambda: x > y):
            pass
    """
    start = time.time()
    while time.time() - start < timeout:
        if condition():
            yield
            return
        time.sleep(interval)
    raise TimeoutError("Condition not met within timeout")


@pytest.mark.parametrize("cb", ["fast_cb", "slow_cb"])
def test_calls_are_delegated(cb, request):
    cb = request.getfixturevalue(cb)
    buff_cb = BufferingWrapper(cb)

    buff_cb("start", {"x": 1})
    with wait_for_condition(lambda: ("start", {"x": 1}) in cb.called):
        assert len(cb.called) == 1

    buff_cb("stop", {"x": 2})
    with wait_for_condition(lambda: ("stop", {"x": 2}) in cb.called):
        assert len(cb.called) == 2


@pytest.mark.parametrize("cb", ["fast_cb", "slow_cb"])
def test_calls_are_delegated_and_finished(cb, request):
    cb = request.getfixturevalue(cb)
    buff_cb = BufferingWrapper(cb)

    assert buff_cb._thread.is_alive()
    assert len(cb.called) == 0

    buff_cb("start", {"x": 1})
    buff_cb("stop", {"x": 2})

    assert buff_cb._thread.is_alive()
    buff_cb.shutdown()
    assert not buff_cb._thread.is_alive()

    assert ("start", {"x": 1}) in cb.called
    assert ("stop", {"x": 2}) in cb.called


@pytest.mark.parametrize("cb", ["fast_cb", "slow_cb"])
def test_graceful_shutdown_blocks_queue(cb, request):
    cb = request.getfixturevalue(cb)
    buff_cb = BufferingWrapper(cb)

    buff_cb("event", {"data": 42})
    buff_cb.shutdown()

    with pytest.raises(RuntimeError):
        buff_cb("post-shutdown", {"fail": True})

    assert ("event", {"data": 42}) in cb.called
    assert ("post-shutdown", {"fail": True}) not in cb.called


@pytest.mark.parametrize("cb", ["fast_cb", "slow_cb"])
def test_double_shutdown_does_not_fail(cb, request):
    cb = request.getfixturevalue(cb)
    buff_cb = BufferingWrapper(cb)

    buff_cb("one", {})
    buff_cb.shutdown()
    buff_cb.shutdown()  # Second shutdown should be a no-op

    assert ("one", {}) in cb.called


@pytest.mark.parametrize("cb, expected_min_duration", [("fast_cb", 0.0), ("slow_cb", 0.5)])
def test_shutdown_waits_for_processing(cb, expected_min_duration, request):
    cb = request.getfixturevalue(cb)
    buff_cb = BufferingWrapper(cb)

    for i in range(5):
        buff_cb("event", {"val": i})

    t0 = time.time()
    time.sleep(0.02)  # Let the last document enter the queue before shutdown
    buff_cb.shutdown()
    duration = time.time() - t0
    assert duration >= expected_min_duration  # Ensure shutdown waited for processing

    # After shutdown, all should be processed
    assert len(cb.called) == 5
    for i in range(5):
        assert ("event", {"val": i}) in cb.called


@pytest.mark.parametrize("cb, expected_max_duration", [("fast_cb", 0.1), ("slow_cb", 0.1)])
def test_shutdown_without_wait(cb, expected_max_duration, request):
    cb = request.getfixturevalue(cb)
    buff_cb = BufferingWrapper(cb)

    for i in range(5):
        buff_cb("event", {"val": i})

    # Shutdown without waiting — thread may still be running briefly
    t0 = time.time()
    buff_cb.shutdown(wait=False)
    duration = time.time() - t0
    assert duration < expected_max_duration
    assert len(cb.called) <= 5


@pytest.mark.parametrize("cb", ["fast_cb", "slow_cb"])
def test_shutdown_stops_processing_new_items(cb, request):
    cb = request.getfixturevalue(cb)
    buff_cb = BufferingWrapper(cb)

    buff_cb("one", {})
    buff_cb.shutdown()

    with pytest.raises(RuntimeError):
        buff_cb("two", {})

    assert ("one", {}) in cb.called
    assert ("two", {}) not in cb.called


@pytest.mark.parametrize("cb", ["fast_cb", "slow_cb"])
def test_signal_handler_triggers_shutdown(cb, request, monkeypatch):
    cb = request.getfixturevalue(cb)
    buff_cb = BufferingWrapper(cb)

    # Monkeypatch to not actually exit the test runner
    monkeypatch.setattr(buff_cb, "shutdown", lambda: setattr(buff_cb, "_is_shutdown", True))
    with pytest.raises(SystemExit):
        buff_cb._signal_handler(signal.SIGINT, None)

    assert buff_cb._is_shutdown
