import threading
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


class CallbackWithException(CallbackBase):
    """A callback that raises an exception when called."""

    def __call__(self, name, doc):
        raise RuntimeError("This callback always raises an exception.")


@pytest.fixture
def make_wrapper():
    """Factory for ``BufferingWrapper`` instances that are shut down on teardown.

    Wrappers that are never explicitly shut down otherwise have their
    ``shutdown`` method run at interpreter exit via ``atexit``.  By that point
    the logging stream is closed, which produces noisy
    ``ValueError: I/O operation on closed file`` tracebacks.
    """
    created: list[BufferingWrapper] = []

    def factory(*args, **kwargs):
        wrapper = BufferingWrapper(*args, **kwargs)
        created.append(wrapper)
        return wrapper

    yield factory

    for wrapper in created:
        wrapper.shutdown()


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
def test_calls_are_delegated(cb, request, make_wrapper):
    cb = request.getfixturevalue(cb)
    buff_cb = make_wrapper(cb)

    buff_cb("start", {"x": 1})
    with wait_for_condition(lambda: ("start", {"x": 1}) in cb.called):
        assert len(cb.called) == 1

    buff_cb("stop", {"x": 2})
    with wait_for_condition(lambda: ("stop", {"x": 2}) in cb.called):
        assert len(cb.called) == 2


@pytest.mark.parametrize("cb", ["fast_cb", "slow_cb"])
def test_calls_are_delegated_and_finished(cb, request, make_wrapper):
    cb = request.getfixturevalue(cb)
    buff_cb = make_wrapper(cb)

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
def test_graceful_shutdown_blocks_queue(cb, request, make_wrapper):
    cb = request.getfixturevalue(cb)
    buff_cb = make_wrapper(cb)

    buff_cb("event", {"data": 42})
    buff_cb.shutdown()

    with pytest.raises(RuntimeError):
        buff_cb("post-shutdown", {"fail": True})

    assert ("event", {"data": 42}) in cb.called
    assert ("post-shutdown", {"fail": True}) not in cb.called


@pytest.mark.parametrize("cb", ["fast_cb", "slow_cb"])
def test_double_shutdown_does_not_fail(cb, request, make_wrapper):
    cb = request.getfixturevalue(cb)
    buff_cb = make_wrapper(cb)

    buff_cb("one", {})
    buff_cb.shutdown()
    buff_cb.shutdown()  # Second shutdown should be a no-op

    assert ("one", {}) in cb.called


@pytest.mark.parametrize("cb, expected_min_duration", [("fast_cb", 0.0), ("slow_cb", 0.5)])
def test_shutdown_waits_for_processing(cb, expected_min_duration, request, make_wrapper):
    cb = request.getfixturevalue(cb)
    buff_cb = make_wrapper(cb)

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
def test_shutdown_without_wait(cb, expected_max_duration, request, make_wrapper):
    cb = request.getfixturevalue(cb)
    buff_cb = make_wrapper(cb)

    for i in range(5):
        buff_cb("event", {"val": i})

    # Shutdown without waiting — thread may still be running briefly
    t0 = time.time()
    buff_cb.shutdown(wait=False)
    duration = time.time() - t0
    assert duration < expected_max_duration
    assert len(cb.called) <= 5


@pytest.mark.parametrize("cb", ["fast_cb", "slow_cb"])
def test_shutdown_stops_processing_new_items(cb, request, make_wrapper):
    cb = request.getfixturevalue(cb)
    buff_cb = make_wrapper(cb)

    buff_cb("one", {})
    buff_cb.shutdown()

    with pytest.raises(RuntimeError):
        buff_cb("two", {})

    assert ("one", {}) in cb.called
    assert ("two", {}) not in cb.called


def test_execption_in_callback(make_wrapper):
    """Test that exceptions in the callback are handled gracefully."""
    cb = CallbackWithException()
    buff_cb = make_wrapper(cb)

    with pytest.raises(RuntimeError, match="This callback always raises an exception."):
        cb("test", {"data": 123})

    # Ensure that the exception does not crash the buffering wrapper
    buff_cb("test", {"data": 123})
    assert buff_cb._thread.is_alive()  # The thread should still be running
    buff_cb("test", {"data": 123})


def test_callback_logging_exceptions(monkeypatch, make_wrapper):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    logger = SimpleNamespace(exception=MagicMock(), info=MagicMock())
    monkeypatch.setattr("bluesky.callbacks.buffer.logger", logger)

    cb = CallbackWithException()
    buff_cb = make_wrapper(cb)

    assert logger.exception.call_count == 0
    buff_cb("test", {"data": 123})
    with wait_for_condition(lambda: logger.exception.call_count == 1):
        assert True


class BlockingCallback(CallbackBase):
    """A callback that blocks until released, for exercising flush timeouts."""

    def __init__(self):
        self.release = threading.Event()
        self.called = []

    def __call__(self, name, doc):
        self.release.wait()
        self.called.append((name, doc))


def test_flush_waits_for_processing(make_wrapper):
    """flush() returns only after all enqueued documents are processed."""
    cb = SlowDummyCallback(delay=0.02)
    buff_cb = make_wrapper(cb)

    for i in range(10):
        buff_cb("event", {"seq_num": i})

    assert buff_cb.flush() is True
    assert len(cb.called) == 10  # all documents delivered, in order
    assert [doc["seq_num"] for _, doc in cb.called] == list(range(10))


def test_flush_empty_buffer_returns_immediately(make_wrapper):
    buff_cb = make_wrapper(SlowDummyCallback(delay=0))
    assert buff_cb.flush(timeout=1) is True


def test_flush_timeout(make_wrapper):
    """flush() returns False when the buffer can not drain within the timeout."""
    cb = BlockingCallback()
    buff_cb = make_wrapper(cb)

    buff_cb("event", {"seq_num": 0})
    assert buff_cb.flush(timeout=0.2) is False  # worker is blocked

    cb.release.set()
    assert buff_cb.flush(timeout=5) is True
    assert len(cb.called) == 1


def test_flush_with_raising_callback(make_wrapper):
    """Documents whose processing raises still count as processed: flush
    must not hang on them."""
    cb = CallbackWithException()
    buff_cb = make_wrapper(cb)

    for _ in range(5):
        buff_cb("event", {})
    assert buff_cb.flush(timeout=5) is True
    assert buff_cb._thread.is_alive()


def test_flush_after_shutdown(make_wrapper):
    """After a clean shutdown everything has been processed; flush returns True."""
    cb = SlowDummyCallback(delay=0.01)
    buff_cb = make_wrapper(cb)
    for i in range(5):
        buff_cb("event", {"seq_num": i})
    buff_cb.shutdown(wait=True)

    assert buff_cb.flush(timeout=1) is True
    assert len(cb.called) == 5


def test_read_after_write_pattern(make_wrapper):
    """The canonical use: write documents, flush, then read back."""
    store = {}

    def writer(name, doc):
        if name == "stop":
            store[doc["run_start"]] = "complete"

    buff_cb = make_wrapper(writer)
    buff_cb("start", {"uid": "abc"})
    buff_cb("stop", {"run_start": "abc"})

    assert buff_cb.flush(timeout=5) is True
    assert store.get("abc") == "complete"  # guaranteed visible after flush
