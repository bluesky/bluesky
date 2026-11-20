"""Exercise suspender bridge lifetimes through real RunEngine loops."""

import asyncio
import threading
import time
from contextlib import contextmanager

import pytest

from bluesky import Msg
from bluesky.run_engine import RunEngineInterrupted
from bluesky.suspenders import SuspendBoolHigh


@contextmanager
def _signal_driver(target):
    errors = []

    def wrapped():
        try:
            target()
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=wrapped)
    thread.start()
    try:
        yield
    finally:
        thread.join(timeout=5)
        assert not thread.is_alive(), "signal driver did not finish"
        assert not errors, f"signal callback failed in worker thread: {errors!r}"


def _run_one_cycle(RE, sig):
    checkpoint_seen = threading.Event()
    start_seen = threading.Event()
    resume_seen = threading.Event()
    messages = []

    def collect(message):
        messages.append(message)
        if message.command == "checkpoint":
            checkpoint_seen.set()
        elif message.command == "_start_suspender":
            start_seen.set()
        elif message.command == "_resume_from_suspender":
            resume_seen.set()

    RE.msg_hook = collect

    def drive():
        assert checkpoint_seen.wait(5)
        sig.put(1)
        assert start_seen.wait(5)
        sig.put(0)

    def plan():
        yield Msg("checkpoint")
        yield Msg("sleep", None, 0.5)

    with _signal_driver(drive):
        RE(plan())
    assert start_seen.is_set()
    assert resume_seen.is_set()
    return messages


def test_release_before_wait_completes_via_public_suspender_api(single_RE, hw):
    """A released Suspender wait remains completed before its first await."""
    RE = single_RE
    sig = hw.bool_sig
    sig.put(0)
    susp = SuspendBoolHigh(sig)
    RE.install_suspender(susp)

    # The RE is idle, so this creates the real suspender bridge without asking
    # the RE to start a suspension.  Keep the public wait callable.
    sig.put(1)
    futures, justification = susp.get_futures()
    assert justification
    assert len(futures) == 1
    wait = futures[0]

    # Release and let the RunEngine loop execute the queued release before
    # submitting the wait coroutine.  This tests set-before-await behavior
    # through SuspenderBase rather than through a raw asyncio.Event.
    sig.put(0)
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0.05), RE.loop).result(timeout=5)
    assert asyncio.run_coroutine_threadsafe(wait(), RE.loop).result(timeout=5) is True
    assert not susp.tripped


def test_blocked_loop_trip_and_release_is_safe(single_RE, hw):
    """A trip and release queued while the RE loop is blocked must not hang."""
    RE = single_RE
    sig = hw.bool_sig
    sig.put(0)
    callback_started = threading.Event()
    loop_blocked = threading.Event()
    driver_finished = threading.Event()
    messages = []

    class NotifyingSuspender(SuspendBoolHigh):
        def _should_suspend(self, value):
            callback_started.set()
            return super()._should_suspend(value)

    susp = NotifyingSuspender(sig)
    RE.install_suspender(susp)
    # install() invokes the callback once for the current value.  Only the
    # high transition from the driver should release this barrier.
    callback_started.clear()

    def drive():
        assert loop_blocked.wait(5)
        sig.put(1)
        assert callback_started.wait(5)
        # This happens before the blocking plan yields back to the RE loop.
        sig.put(0)
        driver_finished.set()

    def blocking_plan():
        loop_blocked.set()
        assert callback_started.wait(5)
        time.sleep(0.3)
        assert driver_finished.wait(5)
        yield Msg("checkpoint")
        yield Msg("sleep", None, 0.3)

    RE.msg_hook = messages.append
    with _signal_driver(drive):
        RE(blocking_plan())

    assert "_start_suspender" in [message.command for message in messages]
    assert not susp.tripped


def test_repeated_trip_release_cycles_on_one_suspender(single_RE, hw):
    """The same installed suspender can create and release two bridge events."""
    RE = single_RE
    sig = hw.bool_sig
    sig.put(0)
    susp = SuspendBoolHigh(sig)
    RE.install_suspender(susp)
    stage_one = threading.Event()
    stage_two = threading.Event()
    start_events = [threading.Event(), threading.Event()]
    resume_events = [threading.Event(), threading.Event()]
    messages = []
    counts = {"_start_suspender": 0, "_resume_from_suspender": 0}

    def collect(message):
        messages.append(message)
        if message.command == "_start_suspender":
            index = counts[message.command]
            counts[message.command] += 1
            if index < len(start_events):
                start_events[index].set()
        elif message.command == "_resume_from_suspender":
            index = counts[message.command]
            counts[message.command] += 1
            if index < len(resume_events):
                resume_events[index].set()

    RE.msg_hook = collect

    def drive():
        assert stage_one.wait(5)
        sig.put(1)
        assert start_events[0].wait(5)
        # Repeated notifications while already tripped/released must not
        # create a second suspension for either event.
        sig.put(1)
        sig.put(0)
        sig.put(0)
        assert resume_events[0].wait(5)
        assert stage_two.wait(5)
        sig.put(1)
        assert start_events[1].wait(5)
        sig.put(1)
        sig.put(0)
        sig.put(0)
        assert resume_events[1].wait(5)

    def plan():
        stage_one.set()
        yield Msg("checkpoint")
        yield Msg("sleep", None, 0.1)
        stage_two.set()
        yield Msg("checkpoint")
        yield Msg("sleep", None, 0.5)

    with _signal_driver(drive):
        RE(plan())
    assert counts == {"_start_suspender": 2, "_resume_from_suspender": 2}
    assert not susp.tripped


def test_cancelled_wait_then_release_does_not_poison_next_wait(single_RE, hw):
    """A paused wait can be released later and the next wait still works."""
    RE = single_RE
    sig = hw.bool_sig
    sig.put(1)
    susp = SuspendBoolHigh(sig)
    RE.install_suspender(susp)
    waiting = threading.Event()
    messages = []

    def collect(message):
        messages.append(message)
        if message[0] == "wait_for":
            waiting.set()

    def request_pause():
        assert waiting.wait(5)
        RE.request_pause()

    RE.msg_hook = collect
    with _signal_driver(request_pause), pytest.raises(RunEngineInterrupted):
        RE([Msg("checkpoint")])
    assert susp.tripped

    # The signal was intentionally left high through cancellation. Release it
    # only after the wait was cancelled, then let the interrupted plan finish.
    sig.put(0)
    RE.resume()
    assert not susp.tripped

    _run_one_cycle(RE, sig)
    assert not susp.tripped


def test_same_suspender_rebinds_to_a_fresh_run_engine_loop(make_RE, hw):
    """Removing from one RE and installing on a fresh RE loop remains usable."""
    first_RE = make_RE({}, call_returns_result=True)
    second_RE = make_RE({}, call_returns_result=True)
    sig = hw.bool_sig
    sig.put(0)
    susp = SuspendBoolHigh(sig)

    first_RE.install_suspender(susp)
    _run_one_cycle(first_RE, sig)
    first_RE.remove_suspender(susp)
    assert susp.RE is None

    second_RE.install_suspender(susp)
    _run_one_cycle(second_RE, sig)
    assert susp.RE is second_RE
    assert not susp.tripped
