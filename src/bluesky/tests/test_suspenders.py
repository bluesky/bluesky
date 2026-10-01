import asyncio
import concurrent.futures
import gc
import re
import threading
import time
import time as ttime
from collections.abc import Hashable, Mapping

import pytest
from ophyd.signal import Signal

from bluesky import Msg
from bluesky._loop import run_coro_on_loop
from bluesky.preprocessors import suspend_wrapper
from bluesky.protocols import Pausable
from bluesky.run_engine import RunEngineInterrupted
from bluesky.suspenders import (
    SuspendBoolHigh,
    SuspendBoolLow,
    SuspendCeil,
    SuspendFloor,
    SuspendInBand,
    SuspendOutBand,
    SuspendWhenChanged,
    SuspendWhenOutsideBand,
)
from bluesky.suspension import SuspensionReason, join_justifications
from bluesky.tests import ophyd_async, requires_ophyd_async
from bluesky.tests.utils import MsgCollector
from bluesky.utils import FailedPause

from .utils import CallbackSignal, _at_message, _fabricate_asycio_event, _when_parked

if ophyd_async:
    from ophyd_async.core import soft_signal_rw

parametrize_suspenders = pytest.mark.parametrize(
    "klass,sc_args,start_val,fail_val,resume_val,wait_time",
    [
        (SuspendBoolHigh, (), 0, 1, 0, 0.2),
        (SuspendBoolLow, (), 1, 0, 1, 0.2),
        (SuspendFloor, (0.5,), 1, 0, 1, 0.2),
        (SuspendCeil, (0.5,), 0, 1, 0, 0.2),
        (SuspendWhenOutsideBand, (0.5, 1.5), 1, 0, 1, 0.2),
        ((SuspendInBand, True), (0.5, 1.5), 1, 0, 1, 0.2),  # renamed to WhenOutsideBand
        ((SuspendOutBand, True), (0.5, 1.5), 0, 1, 0, 0.2),
    ],
)  # deprecated


def _check_suspender(klass, sc_args, sig, putter, start_val, fail_val, resume_val, wait_time, RE):
    try:
        klass, deprecated = klass
    except TypeError:
        deprecated = False
    if deprecated:
        with pytest.warns(UserWarning):
            my_suspender = klass(sig, *sc_args, sleep=wait_time)
    else:
        my_suspender = klass(sig, *sc_args, sleep=wait_time)
    RE.install_suspender(my_suspender)

    # make sure we start at good value!
    putter(start_val)
    # dumb scan
    scan = [Msg("checkpoint"), Msg("sleep", None, 0.2)]
    RE(scan)
    # paranoid
    assert RE.state == "idle"

    start = ttime.time()
    # queue up fail and resume conditions
    threading.Timer(0.1, putter, (fail_val,)).start()
    threading.Timer(0.5, putter, (resume_val,)).start()
    # start the scan
    RE(scan)
    stop = ttime.time()
    # assert we waited at least 2 seconds + the settle time
    delta = stop - start
    print(delta)
    # The suspension time is actually 0.5 - 0.1 = 0.4 seconds as timers run in parallel
    assert delta > 0.4 + wait_time + 0.2


@parametrize_suspenders
def test_suspender(klass, sc_args, start_val, fail_val, resume_val, wait_time, RE, hw):
    sig = hw.bool_sig

    def putter(val):
        sig.put(val)

    _check_suspender(klass, sc_args, sig, putter, start_val, fail_val, resume_val, wait_time, RE)


class _RecordingSignal:
    """A minimal Subscribable that records where it was called from."""

    name = "sig"

    def __init__(self):
        self.threads = {}

    def subscribe_reading(self, function):
        self.threads["subscribe_reading"] = threading.get_ident()
        function({self.name: {"value": 0, "timestamp": 0}})

    def clear_sub(self, function):
        self.threads["clear_sub"] = threading.get_ident()


@pytest.mark.parametrize("via_plan", [False, True])
def test_subscribes_on_run_engine_thread(RE, via_plan):
    "Subscriptions belong to the event loop that made them, e.g. CA monitors"
    sig = _RecordingSignal()
    susp = SuspendBoolHigh(sig)

    if via_plan:
        # install/remove are reached from the event loop thread this way
        RE([Msg("install_suspender", None, susp), Msg("remove_suspender", None, susp)])
    else:
        RE.install_suspender(susp)
        RE.remove_suspender(susp)

    loop_thread = _loop_thread_ident(RE)
    assert sig.threads == {"subscribe_reading": loop_thread, "clear_sub": loop_thread}


def test_remove_without_install_does_not_need_a_loop():
    sig = _RecordingSignal()

    SuspendBoolHigh(sig).remove()

    assert sig.threads == {}


def _loop_thread_ident(RE):
    """The ident of the thread the RunEngine's event loop runs in."""

    async def ident():
        return threading.get_ident()

    return asyncio.run_coroutine_threadsafe(ident(), RE.loop).result()


def _connected_soft_signal(RE, initial_value):
    """Make a soft signal connected on the RunEngine's event loop."""
    sig = soft_signal_rw(float, initial_value, "sig")
    asyncio.run_coroutine_threadsafe(sig.connect(), RE.loop).result()
    return sig


def _set_on_loop(RE, sig, value):
    """Set a signal from another thread.

    ``set`` makes an ``AsyncStatus`` as soon as it is called, so it has to be
    called on the event loop rather than merely awaited there.
    """

    async def set_it():
        await sig.set(value)

    asyncio.run_coroutine_threadsafe(set_it(), RE.loop).result()


@parametrize_suspenders
@requires_ophyd_async
def test_suspender_async_signal(klass, sc_args, start_val, fail_val, resume_val, wait_time, RE):
    sig = _connected_soft_signal(RE, start_val)

    def putter(val):
        _set_on_loop(RE, sig, val)

    _check_suspender(klass, sc_args, sig, putter, start_val, fail_val, resume_val, wait_time, RE)


@requires_ophyd_async
def test_pretripped_async_signal(RE):
    "Tests that install() sees the current value, as ophyd's subscribe(run=True) does"
    sig = _connected_soft_signal(RE, 1)
    susp = SuspendBoolHigh(sig)

    susp.install(RE)

    assert susp.tripped


@requires_ophyd_async
def test_suspender_plans_async_signal(RE, pause_if_stuck):
    "Tests that an async suspender can be installed and removed via Msg"
    sig = _connected_soft_signal(RE, 0)
    my_suspender = SuspendBoolHigh(sig)
    scan = [Msg("checkpoint"), Msg("sleep", None, 0.2)]
    commands = []

    def clear():
        commands.append("sig=0")
        sig.set(0)

    # installed from inside a plan, it suspends and resumes
    _at_message(RE, commands, sleep=lambda: sig.set(1), wait_for=lambda: _in_turn(RE, clear))
    RE([Msg("install_suspender", None, my_suspender)] + scan)
    assert commands.count("_start_suspender") == 1
    # Waited for the condition to clear, then replayed.
    assert commands[commands.index("wait_for") :] == [
        "wait_for",
        "sig=0",
        "_resume_from_suspender",
        "rewindable",
        "sleep",
    ]
    # and it ends with that plan; see test_suspender_installed_by_a_plan_ends_with_it.
    assert my_suspender not in RE.suspenders

    commands.clear()
    _at_message(RE, commands, sleep=lambda: sig.set(1))
    RE([Msg("remove_suspender", None, my_suspender)] + scan)
    _set_on_loop(RE, sig, 0)
    assert "_start_suspender" not in commands
    assert my_suspender not in RE.suspenders


@requires_ophyd_async
def test_event_type_is_rejected_for_a_subscribable_signal(RE):
    "A Subscribable signal has no event types, so asking for one must not be ignored"
    sig = _connected_soft_signal(RE, 0)
    susp = SuspendBoolHigh(sig)

    with pytest.raises(RuntimeError, match="event_type"):
        susp.install(RE, event_type="value")

    assert susp not in RE.suspenders
    # Nor does the suspender hold a suspension, or this would raise.
    RE.install_suspender(susp)
    RE.remove_suspender(susp)


@requires_ophyd_async
def test_suspend_when_changed_async_signal(RE):
    "expected_value cannot be read from a Subscribable signal until it is installed"
    sig = _connected_soft_signal(RE, 1)
    susp = SuspendWhenChanged(sig, allow_resume=True)
    assert susp.expected_value is None

    susp.install(RE)

    assert susp.expected_value == 1
    assert not susp.tripped

    _set_on_loop(RE, sig, 2)

    assert susp.tripped
    assert susp._get_justification() == 'Signal sig, got "2.0", expected "1.0"'


def test_pretripped(RE, hw):
    "Tests if suspender is tripped before __call__"
    sig = hw.bool_sig
    scan = [Msg("checkpoint")]
    msg_lst = []
    sig.put(1)

    def accum(msg):
        msg_lst.append(msg)

    susp = SuspendBoolHigh(sig)

    RE.install_suspender(susp)
    threading.Timer(1, sig.put, (0,)).start()
    RE.msg_hook = accum
    RE(scan)

    assert len(msg_lst) == 2
    assert ["wait_for", "checkpoint"] == [m[0] for m in msg_lst]


def test_a_pretripped_condition_is_announced_with_numbered_justifications(RE, capsys):
    beam = Signal(name="beam", value=0)
    shutter = Signal(name="shutter", value=0)
    # Put, not constructed high: an ophyd Signal reports only a put value on subscribe.
    beam.put(1)
    shutter.put(1)
    RE.install_suspender(SuspendBoolHigh(beam, tripped_message="no beam"))
    RE.install_suspender(SuspendBoolHigh(shutter, tripped_message="shutter shut"))

    def release(msg):
        # The plan is held ahead of its first message, so clear from inside it.
        if msg.command == "wait_for":
            beam.put(0)
            shutter.put(0)

    RE.msg_hook = release
    RE([Msg("checkpoint")])

    lines = capsys.readouterr().out.splitlines()
    start = lines.index(
        "At least one suspender has tripped. The plan will begin when all suspenders are ready. Justification:"
    )
    numbered = lines[start + 1 : start + 3]
    assert [line[:7] for line in numbered] == ["    1. ", "    2. "]
    assert {line[7:] for line in numbered} == {
        "Signal beam is high: no beam",
        "Signal shutter is high: shutter shut",
    }
    assert lines[start + 3 : start + 5] == ["", "Suspending... To get to the prompt, hit Ctrl-C twice to pause."]


def test_a_suspension_is_announced_with_when_it_occurred(RE, capsys):
    beam = Signal(name="beam", value=0)
    RE.install_suspender(SuspendBoolHigh(beam))
    seen = set()

    def drive(msg):
        # First time only: the suspension replays from the checkpoint.
        if msg.command in seen:
            return
        seen.add(msg.command)
        if msg.command == "null":
            beam.put(1)
        elif msg.command == "wait_for":
            beam.put(0)

    RE.msg_hook = drive
    RE([Msg("checkpoint"), Msg("null"), Msg("sleep", None, 0.2)])

    lines = capsys.readouterr().out.splitlines()
    occurred = [line for line in lines if "occurred at" in line]
    assert len(occurred) == 1
    assert re.fullmatch(r"Suspension occurred at \d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.", occurred[0])


def test_suspender_wrapper(RE, hw):

    wait_time = 0.2
    sleep_time = 0.2
    trigger_time = 0.5

    sig = hw.bool_sig
    scan = [Msg("checkpoint"), Msg("sleep", None, sleep_time)]
    sig.put(0)

    susp = SuspendBoolHigh(sig, sleep=wait_time)

    RE(suspend_wrapper(scan, susp))
    assert RE.state == "idle"

    sig.put(1)
    threading.Timer(trigger_time, sig.put, (0,)).start()

    start = ttime.time()

    RE(suspend_wrapper(scan, susp))
    stop = ttime.time()
    delta = stop - start
    assert delta > trigger_time + wait_time + sleep_time


@pytest.mark.parametrize(
    "pre_plan,post_plan,expected_list",
    [
        (
            [Msg("null")],
            None,
            [
                "checkpoint",
                "sleep",
                "_start_suspender",
                "rewindable",
                "null",
                "wait_for",
                "_resume_from_suspender",
                "rewindable",
                "sleep",
            ],
        ),
        (
            None,
            [Msg("null")],
            [
                "checkpoint",
                "sleep",
                "_start_suspender",
                "rewindable",
                "wait_for",
                "_resume_from_suspender",
                "null",
                "rewindable",
                "sleep",
            ],
        ),
        (
            [Msg("null")],
            [Msg("null")],
            [
                "checkpoint",
                "sleep",
                "_start_suspender",
                "rewindable",
                "null",
                "wait_for",
                "_resume_from_suspender",
                "null",
                "rewindable",
                "sleep",
            ],
        ),
        (
            lambda: [Msg("null")],
            lambda: [Msg("null")],
            [
                "checkpoint",
                "sleep",
                "_start_suspender",
                "rewindable",
                "null",
                "wait_for",
                "_resume_from_suspender",
                "null",
                "rewindable",
                "sleep",
            ],
        ),
    ],
)
def test_pre_suspend_plan(RE, pre_plan, post_plan, expected_list, hw, pause_if_stuck):
    sig = hw.bool_sig
    scan = [Msg("checkpoint"), Msg("sleep", None, 0.05)]
    msg_lst = []
    sig.put(0)

    susp = SuspendBoolHigh(sig, pre_plan=pre_plan, post_plan=post_plan)

    RE.install_suspender(susp)
    # Trip at the sleep, and clear at the wait.
    _at_message(RE, msg_lst, sleep=lambda: sig.put(1), wait_for=lambda: sig.put(0))
    RE(scan)

    assert len(msg_lst) == len(expected_list)
    assert expected_list == msg_lst

    RE.remove_suspender(susp)
    RE(scan)
    assert susp not in RE.suspenders

    RE.install_suspender(susp)
    RE.clear_suspenders()
    assert susp not in RE.suspenders
    assert not RE.suspenders


def test_pause_from_suspend(RE, hw):
    "Tests what happens when a pause is requested from a suspended state"
    sig = hw.bool_sig
    scan = [Msg("checkpoint")]
    msg_lst = []
    sig.put(1)

    def accum(msg):
        msg_lst.append(msg)

    susp = SuspendBoolHigh(sig)

    RE.install_suspender(susp)
    threading.Timer(1, RE.request_pause).start()
    threading.Timer(2, sig.put, (0,)).start()
    RE.msg_hook = accum
    with pytest.raises(RunEngineInterrupted):
        RE(scan)
    assert [m[0] for m in msg_lst] == ["wait_for"]
    RE.resume()
    assert ["wait_for", "wait_for", "checkpoint"] == [m[0] for m in msg_lst]


def test_suspend_when_changed_latches_expected_value_at_install(RE, hw):
    "The default latches on install, whichever protocol the signal implements"
    sig = hw.bool_sig
    sig.put(1)

    susp = SuspendWhenChanged(sig, allow_resume=True)
    # __init__ does not read the signal.
    assert susp.expected_value is None

    RE.install_suspender(susp)
    try:
        # Latched from the reading install calls back with.
        assert susp.expected_value == 1
        assert not susp.tripped
    finally:
        RE.remove_suspender(susp)


def test_suspend_when_changed_preserves_falsy_expected_value(hw):
    sig = hw.bool_sig
    sig.put(1)

    susp = SuspendWhenChanged(sig, expected_value=0)

    assert susp.expected_value == 0
    assert not susp._should_suspend(0)
    assert susp._should_suspend(1)


def test_deferred_pause_from_suspend(RE, hw):
    "Tests what happens when a soft pause is requested from a suspended state"
    sig = hw.bool_sig
    scan = [Msg("checkpoint"), Msg("null")]
    msg_lst = []
    deferred_pause_event = threading.Event()
    waiting_event = threading.Event()
    sig.put(1)

    def accum(msg):
        if msg[0] == "wait_for":
            waiting_event.set()
        msg_lst.append(msg)

    def wait_then_request_pause():
        waiting_event.wait(timeout=5)
        assert waiting_event.is_set()
        RE.request_pause(True)
        deferred_pause_event.set()

    def wait_then_put():
        deferred_pause_event.wait(timeout=5)
        assert deferred_pause_event.is_set()
        sig.put(0)

    susp = SuspendBoolHigh(sig)

    RE.install_suspender(susp)
    threading.Thread(target=wait_then_request_pause, daemon=True).start()
    threading.Thread(target=wait_then_put, daemon=True).start()
    RE.msg_hook = accum
    with pytest.raises(RunEngineInterrupted):
        RE(scan)
    assert [m[0] for m in msg_lst] == ["wait_for", "checkpoint"]
    RE.resume()
    assert ["wait_for", "checkpoint", "null"] == [m[0] for m in msg_lst]


def test_unresumable_suspend_fail(RE):
    "Tests what happens when a soft pause is requested from a suspended state"

    scan = [Msg("clear_checkpoint"), Msg("sleep", None, 2)]
    sig = CallbackSignal(name="unresumable_sig")
    RE.install_suspender(SuspendBoolHigh(sig))

    # At a message: a trip before the plan starts would hold, not abort.
    m_coll = MsgCollector(msg_hook=lambda msg: sig.put(1) if msg.command == "sleep" else None)
    RE.msg_hook = m_coll

    start = time.time()
    with pytest.raises(RunEngineInterrupted):
        RE(scan)
    stop = time.time()
    # Aborted at the trip, not after the sleep.
    assert [m.command for m in m_coll.msgs] == ["clear_checkpoint", "sleep"]
    assert stop - start < 1


def test_suspender_plans(RE, hw, pause_if_stuck):
    "Tests that the suspenders can be installed via Msg"
    sig = hw.bool_sig
    my_suspender = SuspendBoolHigh(sig)

    def putter(val):
        sig.put(val)

    putter(0)

    # A suspender a plan installs ends with the plan.
    seen = []

    def note_while_running():
        yield Msg("install_suspender", None, my_suspender)
        seen.append(my_suspender in RE.suspenders)
        yield Msg("remove_suspender", None, my_suspender)

    RE(note_while_running())
    assert seen == [True]
    assert my_suspender not in RE.suspenders
    RE([Msg("remove_suspender", None, my_suspender)])
    assert my_suspender not in RE.suspenders

    # Can we call both in a plan?
    RE([Msg("install_suspender", None, my_suspender), Msg("remove_suspender", None, my_suspender)])

    scan = [Msg("checkpoint"), Msg("sleep", None, 0.05)]

    # No suspend scan: does the wrapper error out?
    start = ttime.time()
    RE(suspend_wrapper(scan, my_suspender))
    stop = ttime.time()
    delta = stop - start
    assert delta < 0.9

    # Suspend scan
    commands = []
    _at_message(RE, commands, sleep=lambda: putter(1), wait_for=lambda: _in_turn(RE, _clear(commands, sig)))
    RE(suspend_wrapper(scan, my_suspender))
    assert commands.count("_start_suspender") == 1
    # Waited for the condition to clear, then replayed.
    assert commands[commands.index("wait_for") :] == [
        "wait_for",
        "bool_sig=0",
        "_resume_from_suspender",
        "rewindable",
        "sleep",
        "remove_suspender",
    ]

    # Did we clean up?
    commands.clear()
    _at_message(RE, commands, sleep=lambda: putter(1))
    RE(scan)
    putter(0)
    assert "_start_suspender" not in commands


def test_a_suspension_holds_only_devices_that_can_be_released(RE, hw):
    """A device is told a suspension began only if it can be told it ended.

    `bluesky.protocols.Pausable` requires ``resume`` as well as ``pause``, and
    the suspension path used to ask only for the attribute.
    """

    class PauseOnly:
        name = "pause_only"

        def __init__(self):
            self.paused = 0

        def pause(self):
            self.paused += 1

    class PauseAndResume(PauseOnly):
        name = "pause_and_resume"

        def __init__(self):
            super().__init__()
            self.resumed = 0

        def resume(self):
            self.resumed += 1

    pause_only = PauseOnly()
    pausable = PauseAndResume()
    assert not isinstance(pause_only, Pausable)
    assert isinstance(pausable, Pausable)

    sig = hw.bool_sig
    sig.put(0)
    RE.install_suspender(SuspendBoolHigh(sig, sleep=0.1))

    plan = [
        Msg("null", pause_only),
        Msg("null", pausable),
        Msg("checkpoint"),
        Msg("sleep", None, 0.2),
    ]
    threading.Timer(0.05, sig.put, (1,)).start()
    threading.Timer(0.3, sig.put, (0,)).start()
    RE(plan)

    # Told a hold began, and told it ended.
    assert pausable.paused == 1
    assert pausable.resumed == 1
    # Never told anything, because it could not have been told it was over.
    assert pause_only.paused == 0


def test_a_suspension_does_not_duplicate_a_monitored_signals_documents(RE, hw):
    """Resuming from a suspension must not re-subscribe a monitor.

    Monitors run throughout a suspension -- only a *pause* stops them -- so
    restoring them afterwards subscribed a second time, and every Event the
    signal produced after that was emitted twice.
    """

    class FakeMonitored:
        name = "fake_monitored"
        parent = None

        def __init__(self):
            self.cbs = []
            self.subscriptions = 0
            self.fires = 0

        def read(self):
            return {self.name: {"value": 1.0, "timestamp": 0.0}}

        def describe(self):
            return {self.name: {"source": "fake", "dtype": "number", "shape": []}}

        def read_configuration(self):
            return {}

        def describe_configuration(self):
            return {}

        def subscribe(self, cb, **kwargs):
            self.subscriptions += 1
            self.cbs.append(cb)

        def clear_sub(self, cb):
            self.cbs.remove(cb)

        def fire(self):
            self.fires += 1
            for cb in list(self.cbs):
                cb()

    dev = FakeMonitored()
    events = []
    RE.subscribe(lambda name, doc: events.append(doc), "event")

    sig = hw.bool_sig
    sig.put(0)
    RE.install_suspender(SuspendBoolHigh(sig, sleep=0.1))

    def plan():
        yield Msg("open_run")
        yield Msg("monitor", dev, name="mon")
        yield Msg("checkpoint")
        yield Msg("sleep", None, 0.2)
        dev.fire()
        yield Msg("close_run")

    threading.Timer(0.05, sig.put, (1,)).start()
    threading.Timer(0.3, sig.put, (0,)).start()
    RE(plan())

    # One subscription, made once and taken off at close_run.
    assert dev.subscriptions == 1
    assert dev.cbs == []
    # And so one Event per reading, not one per subscription per reading.
    assert dev.fires > 0
    assert len(events) == dev.fires


def test_a_second_condition_joins_the_open_suspension(RE, pause_if_stuck):
    # Each condition runs its own pre-plan.
    sig_a = Signal(value=0, name="sig_a")
    sig_b = Signal(value=0, name="sig_b")
    susp_a = SuspendBoolHigh(sig_a, pre_plan=[Msg("null")], post_plan=[Msg("null")])
    susp_b = SuspendBoolHigh(sig_b, pre_plan=[Msg("null")], post_plan=[Msg("null")])
    RE.install_suspender(susp_a)
    RE.install_suspender(susp_b)

    commands = []
    # Tied to messages, so sig_b joins sig_a's suspension rather than racing it.
    _at_message(
        RE,
        commands,
        sleep=lambda: sig_a.put(1),
        _start_suspender=lambda: sig_b.put(1),
        wait_for=lambda: _in_turn(RE, _clear(commands, sig_a), _clear(commands, sig_b)),
    )

    RE([Msg("checkpoint"), Msg("sleep", None, 0.05), Msg("null")])

    assert commands.count("_start_suspender") == 1
    # Released once.
    assert commands.count("_resume_from_suspender") == 1
    RE.clear_suspenders()


def test_a_trip_while_paused_runs_no_pre_plan_on_resume(RE, hw, pause_if_stuck):
    sig = hw.bool_sig
    sig.put(0)
    ran = []

    def note(tag):
        def plan():
            ran.append(tag)
            yield Msg("null")

        return plan

    susp = SuspendBoolHigh(sig, pre_plan=note("pre"), post_plan=note("post"))
    RE.install_suspender(susp)

    with pytest.raises(RunEngineInterrupted):
        RE([Msg("checkpoint"), Msg("pause"), Msg("null")])
    assert RE.state == "paused"

    sig.put(1)
    _settle(RE)
    assert susp.tripped
    assert ran == []

    commands = []
    _at_message(RE, commands, wait_for=lambda: _in_turn(RE, _clear(commands, sig)))
    RE.resume()

    # Resuming waited for the condition to clear.
    assert commands[:2] == ["wait_for", "bool_sig=0"]
    # Held, rather than suspended.
    assert "_start_suspender" not in commands
    assert ran == []
    RE.clear_suspenders()


def test_a_retrip_inside_the_settle_window_keeps_the_hold(RE, hw, pause_if_stuck):
    sig = CallbackSignal(name="retrip_sig")
    # Real time: the settle window is what this is about.
    RE.install_suspender(SuspendBoolHigh(sig, sleep=0.1))
    recovered = []

    def recover_for_good():
        recovered.append(ttime.time())
        sig.put(0)

    def goes_bad():
        sig.put(1)  # trips
        sig.put(0)  # recovers, scheduling a release 0.1s out
        sig.put(1)  # bad again, before that release can come due
        # After the stale release would have come due.
        RE.loop.call_later(0.2, recover_for_good)

    _at_message(RE, [], sleep=goes_bad)

    RE([Msg("checkpoint"), Msg("sleep", None, 0.1), Msg("null")])
    released = ttime.time()

    assert recovered
    assert released - recovered[0] > 0.1


def test_suspender_installed_by_a_plan_ends_with_it(RE, hw):
    sig = hw.bool_sig
    sig.put(0)
    susp = SuspendBoolHigh(sig)
    seen = []

    def note():
        yield Msg("install_suspender", None, susp)
        seen.append(("after install", susp in RE.suspenders))
        yield Msg("remove_suspender", None, susp)
        seen.append(("after remove", susp in RE.suspenders))
        yield Msg("install_suspender", None, susp)

    RE(note())

    assert seen == [("after install", True), ("after remove", False)]
    assert susp not in RE.suspenders


def test_clear_suspenders_while_paused_then_resume(RE, hw, pause_if_stuck):
    # At the prompt: RE(my_plan()), Ctrl-C, RE.clear_suspenders(), RE.resume().
    sig = hw.bool_sig
    sig.put(1)  # beam is already down
    susp = SuspendBoolHigh(sig)
    RE.install_suspender(susp)

    commands = []
    # Paused in the hold. request_pause waits on the loop, so call it from a thread.
    _at_message(RE, commands, wait_for=lambda: threading.Thread(target=RE.request_pause).start())
    with pytest.raises(RunEngineInterrupted):
        RE([Msg("checkpoint"), Msg("null")])
    assert RE.state == "paused"
    assert susp.tripped

    RE.clear_suspenders()
    assert RE.suspenders == ()

    RE.resume()
    assert RE.state == "idle"
    assert commands[-1] == "null"
    sig.put(0)


# A plan with a checkpoint to rewind to, and long enough to be interrupted.
SCAN = [Msg("checkpoint"), Msg("sleep", None, 0.05)]


def _in_turn(RE, *funcs):
    """From the loop, call each of ``funcs`` in turn, each once the plan has parked."""
    if funcs:
        first, *rest = funcs

        def call():
            first()
            _in_turn(RE, *rest)

        _when_parked(RE.loop, call)


def _clear(commands, sig):
    """A function putting ``sig`` to 0, noting it in ``commands`` as ``"<name>=0"``."""

    def clear():
        commands.append(f"{sig.name}=0")
        sig.put(0)

    return clear


def _soft_signal(RE, name):
    """A soft signal on the RunEngine's loop, reading false."""
    sig = soft_signal_rw(float, 0.0, name)
    asyncio.run_coroutine_threadsafe(sig.connect(), RE.loop).result()
    return sig


def _settle(RE):
    """Wait for the loop to apply a trip a signal callback scheduled."""
    done = concurrent.futures.Future()
    RE.loop.call_soon_threadsafe(lambda: done.set_result(None))
    done.result(timeout=10)


# --------------------------------------------------------------------------
# The suspension sequences.


def test_trips_while_a_plan_is_running(RE, hw, pause_if_stuck):
    # Rewinds to the checkpoint, waits, then replays.
    sig = hw.bool_sig
    sig.put(0)
    RE.install_suspender(SuspendBoolHigh(sig))
    commands = []

    _at_message(RE, commands, sleep=lambda: sig.put(1), wait_for=lambda: _in_turn(RE, _clear(commands, sig)))
    RE(SCAN)

    # Waited for the condition to clear, then replayed.
    assert commands[commands.index("wait_for") :] == [
        "wait_for",
        "bool_sig=0",
        "_resume_from_suspender",
        "rewindable",
        "sleep",
    ]
    assert commands.count("sleep") == 2
    assert commands.count("_start_suspender") == 1


def test_releases_while_a_plan_is_suspended(RE, hw, pause_if_stuck):
    # The settle-down sleep delays the release. Real time: the delay is what this is about.
    sig = hw.bool_sig
    sig.put(0)
    RE.install_suspender(SuspendBoolHigh(sig, sleep=0.1))
    cleared = []

    def clear():
        cleared.append(ttime.time())
        sig.put(0)

    _at_message(RE, [], sleep=lambda: sig.put(1), wait_for=lambda: _in_turn(RE, clear))
    RE(SCAN)
    delta = ttime.time() - cleared[0]

    # The 0.1 settle, plus the replayed sleep.
    assert delta > 0.1


def test_trips_while_no_plan_is_running(RE, hw, pause_if_stuck):
    # Nothing suspends, but the next plan waits before it starts.
    sig = hw.bool_sig
    sig.put(1)  # already bad before any plan exists
    RE.install_suspender(SuspendBoolHigh(sig))
    assert RE.state == "idle"

    commands = []
    _at_message(RE, commands, wait_for=lambda: _in_turn(RE, _clear(commands, sig)))
    RE(SCAN)

    # Waited for the condition to clear.
    assert commands[:2] == ["wait_for", "bool_sig=0"]
    # Held by a wait: no checkpoint yet, so no replay.
    assert commands.count("sleep") == 1
    assert "_start_suspender" not in commands


def test_releases_while_no_plan_is_running(RE, hw):
    # A condition that came and went holds nothing up.
    sig = hw.bool_sig
    sig.put(0)
    RE.install_suspender(SuspendBoolHigh(sig))
    sig.put(1)
    sig.put(0)

    start = ttime.time()
    RE(SCAN)
    delta = ttime.time() - start

    assert delta < 0.9


# --------------------------------------------------------------------------
# Several conditions, pauses, and edge cases.
@requires_ophyd_async
def test_two_conditions_in_one_turn_are_one_suspension(RE):
    # Both reasons are reported, and the plan rewinds once, not twice.
    beam, shutter = _soft_signal(RE, "beam_sig"), _soft_signal(RE, "shutter_sig")
    RE.install_suspender(SuspendBoolHigh(beam, tripped_message="beam"))
    RE.install_suspender(SuspendBoolHigh(shutter, tripped_message="shutter"))

    commands, seen = [], []

    def both_bad():
        beam.set(1)
        shutter.set(1)

    def look_then_release():
        seen.append(join_justifications(RE._session.suspension_reasons))
        beam.set(0)
        shutter.set(0)

    _at_message(RE, commands, sleep=both_bad, _start_suspender=look_then_release)
    RE(SCAN)

    assert len(seen) == 1
    assert "beam" in seen[0] and "shutter" in seen[0]
    # One rewind.
    assert commands.count("_start_suspender") == 1


def test_a_trip_while_paused_makes_resume_wait(RE, hw, pause_if_stuck):
    sig = hw.bool_sig
    sig.put(0)
    RE.install_suspender(SuspendBoolHigh(sig))

    with pytest.raises(RunEngineInterrupted):
        RE([Msg("checkpoint"), Msg("pause"), Msg("sleep", None, 0.2)])
    assert RE.state == "paused"

    sig.put(1)
    _settle(RE)
    assert RE._session.suspension_reasons

    commands = []
    _at_message(RE, commands, wait_for=lambda: _in_turn(RE, _clear(commands, sig)))
    RE.resume()
    # Resuming waited for the condition to clear.
    assert commands[:2] == ["wait_for", "bool_sig=0"]


def test_resume_announces_a_condition_that_tripped_while_paused(RE, capsys):
    beam = Signal(name="beam", value=0)
    RE.install_suspender(SuspendBoolHigh(beam, tripped_message="no beam"))

    with pytest.raises(RunEngineInterrupted):
        RE([Msg("checkpoint"), Msg("pause"), Msg("null")])
    beam.put(1)
    _settle(RE)
    capsys.readouterr()

    def release(msg):
        if msg.command == "wait_for":
            beam.put(0)

    RE.msg_hook = release
    RE.resume()

    lines = capsys.readouterr().out.splitlines()
    start = lines.index(
        "At least one suspender has tripped. The plan will continue when all suspenders are ready. Justification:"
    )
    assert lines[start + 1] == "    1. Signal beam is high: no beam"


def test_a_trip_as_the_plan_pauses_holds_the_resume(RE):
    # No pre- or post-plan runs.
    beam = Signal(name="beam", value=0)
    ran = []
    RE.install_suspender(
        SuspendBoolHigh(beam, pre_plan=lambda: ran.append("pre") or [], post_plan=lambda: ran.append("post") or [])
    )

    commands = []
    _at_message(RE, commands, pause=lambda: beam.put(1))
    with pytest.raises(RunEngineInterrupted):
        RE([Msg("checkpoint"), Msg("pause"), Msg("null")])
    assert commands == ["checkpoint", "pause"]

    seen = []

    def release(msg):
        seen.append((msg.command, beam.get()))
        if msg.command == "wait_for":
            beam.put(0)

    RE.msg_hook = release
    RE.resume()

    assert seen == [("wait_for", 1), ("null", 0)]
    assert ran == []


def test_a_pause_inside_a_suspension_resumes_into_it(RE):
    # Post-plans run once the condition clears.
    beam = Signal(name="beam", value=0)
    ran = []
    RE.install_suspender(
        SuspendBoolHigh(beam, pre_plan=lambda: ran.append("pre") or [], post_plan=lambda: ran.append("post") or [])
    )

    commands = []
    # request_pause waits on the loop, so call it from a thread.
    _at_message(
        RE,
        commands,
        null=lambda: beam.put(1),
        wait_for=lambda: threading.Thread(target=RE.request_pause).start(),
    )
    with pytest.raises(RunEngineInterrupted):
        RE([Msg("checkpoint"), Msg("null"), Msg("null")])
    assert "_start_suspender" in commands
    assert ran == ["pre"]

    seen = []

    def release(msg):
        seen.append((msg.command, beam.get(), list(ran)))
        if msg.command == "wait_for":
            beam.put(0)

    RE.msg_hook = release
    RE.resume()

    assert seen[:2] == [("wait_for", 1, ["pre"]), ("_resume_from_suspender", 0, ["pre"])]
    assert ran == ["pre", "post"]


def test_a_resume_from_a_pause_in_the_start_hold_says_it_will_continue(RE, capsys):
    beam = Signal(name="beam", value=0)
    beam.put(1)
    RE.install_suspender(SuspendBoolHigh(beam, tripped_message="no beam"))

    commands = []
    _at_message(RE, commands, wait_for=lambda: threading.Thread(target=RE.request_pause).start())
    with pytest.raises(RunEngineInterrupted):
        RE([Msg("checkpoint"), Msg("null")])
    assert commands == ["wait_for"]
    capsys.readouterr()

    seen = []

    def release(msg):
        seen.append((msg.command, beam.get()))
        if msg.command == "wait_for":
            beam.put(0)

    RE.msg_hook = release
    RE.resume()

    lines = capsys.readouterr().out.splitlines()
    start = lines.index(
        "At least one suspender has tripped. The plan will continue when all suspenders are ready. Justification:"
    )
    assert lines[start + 1] == "    1. Signal beam is high: no beam"
    assert seen == [("wait_for", 1), ("checkpoint", 0), ("null", 0)]


def test_a_trip_during_a_stops_cleanup_does_not_suspend(RE, pause_if_stuck):
    from bluesky.preprocessors import finalize_wrapper

    from .utils import _parked

    beam = CallbackSignal(name="beam")
    RE.install_suspender(SuspendBoolHigh(beam))
    ran = []

    def body():
        yield Msg("checkpoint")
        ran.append("body")
        yield Msg("pause")
        yield Msg("null")

    def cleanup():
        beam.put(1)
        # So a suspension that does open fails the test rather than hangs it.
        release = threading.Timer(2, beam.put, (0,))
        release.daemon = True
        release.start()
        # Until the loop is idle, by when a suspension would have opened.
        yield Msg("wait_for", None, [_parked])

    commands = []
    RE.msg_hook = lambda msg: commands.append(msg.command)
    with pytest.raises(RunEngineInterrupted):
        RE(finalize_wrapper(body(), cleanup()))
    RE.stop()

    assert "_start_suspender" not in commands
    assert ran == ["body"]
    assert RE.state == "idle"
    RE.clear_suspenders()


def test_a_pause_as_a_suspension_starts_holds_the_resume(RE, pause_if_stuck):
    # Paused while the suspension is still bringing devices to rest, so it never began.
    beam = CallbackSignal(name="beam")
    ran = []
    RE.install_suspender(SuspendBoolHigh(beam, pre_plan=lambda: ran.append("pre") or []))

    class SlowToPause:
        name = "slow"

        def __init__(self):
            self.pauses = 0

        async def pause(self):
            self.pauses += 1
            if self.pauses == 1:
                # request_pause waits on the loop, so call it from a thread.
                threading.Thread(target=RE.request_pause).start()
                # Cancelled by that pause.
                await asyncio.sleep(10)

        def resume(self):
            pass

    commands = []
    _at_message(RE, commands, sleep=lambda: beam.put(1))
    with pytest.raises(RunEngineInterrupted):
        RE([Msg("checkpoint"), Msg("null", SlowToPause()), Msg("sleep", None, 0.1), Msg("null")])
    assert commands[-1] == "_start_suspender"

    commands = []
    _at_message(RE, commands, wait_for=lambda: _in_turn(RE, _clear(commands, beam)))
    RE.resume()

    # Held, rather than suspended: nothing moves until the condition clears.
    assert commands[:2] == ["wait_for", "beam=0"]
    assert "_start_suspender" not in commands
    assert ran == []
    RE.clear_suspenders()


def test_resuming_into_a_suspension_says_it_still_waits(RE, capsys):
    beam = Signal(name="beam", value=0)
    RE.install_suspender(SuspendBoolHigh(beam, tripped_message="no beam"))

    commands = []
    # request_pause waits on the loop, so call it from a thread.
    _at_message(
        RE,
        commands,
        null=lambda: beam.put(1),
        wait_for=lambda: threading.Thread(target=RE.request_pause).start(),
    )
    with pytest.raises(RunEngineInterrupted):
        RE([Msg("checkpoint"), Msg("null"), Msg("null")])
    assert "_start_suspender" in commands
    capsys.readouterr()

    commands = []
    _at_message(RE, commands, wait_for=lambda: _in_turn(RE, _clear(commands, beam)))
    RE.resume()

    lines = capsys.readouterr().out.splitlines()
    start = lines.index(
        "At least one suspender has tripped. The plan will continue when all suspenders are ready. Justification:"
    )
    assert lines[start + 1] == "    1. Signal beam is high: no beam"
    assert commands[:3] == ["wait_for", "beam=0", "_resume_from_suspender"]


def test_a_pause_in_the_start_hold_waits_again_with_nothing_to_replay(RE):
    beam = Signal(name="beam", value=0)
    beam.put(1)
    RE.install_suspender(SuspendBoolHigh(beam, tripped_message="no beam"))
    # Nothing cached, so the resume replays no ``wait_for``.
    RE.rewindable = False

    commands = []
    _at_message(RE, commands, wait_for=lambda: threading.Thread(target=RE.request_pause).start())
    with pytest.raises(RunEngineInterrupted):
        RE([Msg("null")])
    assert commands == ["wait_for"]

    seen = []

    def release(msg):
        seen.append((msg.command, beam.get()))
        if msg.command == "wait_for":
            beam.put(0)

    RE.msg_hook = release
    RE.resume()

    assert seen == [("wait_for", 1), ("null", 0)]


def test_no_checkpoint_mid_plan_aborts(RE, hw):
    sig = hw.bool_sig
    sig.put(0)
    RE.install_suspender(SuspendBoolHigh(sig))
    commands = []

    _at_message(RE, commands, sleep=lambda: sig.put(1))
    with pytest.raises(RunEngineInterrupted):
        RE([Msg("clear_checkpoint"), Msg("sleep", None, 0.5)])
    assert RE.state == "idle"
    assert isinstance(RE._exception, FailedPause) or RE._exception is None


def test_no_checkpoint_abort_raises_nothing_into_the_loop(RE, hw):
    reported = []
    RE.loop.call_soon_threadsafe(RE.loop.set_exception_handler, lambda loop, ctx: reported.append(ctx))

    sig = hw.bool_sig
    sig.put(0)
    RE.install_suspender(SuspendBoolHigh(sig))
    commands = []

    _at_message(RE, commands, sleep=lambda: sig.put(1))
    with pytest.raises(RunEngineInterrupted):
        RE([Msg("clear_checkpoint"), Msg("sleep", None, 0.5)])
    assert RE.state == "idle"

    # The report is made from ``Task.__del__``, so collect before looking.
    gc.collect()
    # [ctx.get("message") for ctx in reported].
    assert not reported


def test_clear_suspenders_reaches_a_plans_own_from_the_prompt(RE, hw, pause_if_stuck):
    suspender = SuspendBoolHigh(hw.bool_sig)
    raised = []
    cleared = _fabricate_asycio_event(RE.loop)

    def clear_from_another_thread():
        try:
            RE.clear_suspenders()
        except BaseException as exc:  # noqa: BLE001
            raised.append(exc)
        finally:
            RE.loop.call_soon_threadsafe(cleared.set)

    def plan():
        yield Msg("install_suspender", None, suspender)
        yield Msg("checkpoint")
        yield Msg("wait_for", None, [cleared.wait])

    # From a thread, as the prompt is; a msg_hook would be on the loop.
    _at_message(RE, [], wait_for=lambda: threading.Thread(target=clear_from_another_thread).start())
    RE(plan())

    assert not raised
    assert RE.suspenders == ()


def test_the_suspender_methods_can_be_called_from_the_loop(RE, hw):
    # As from a plan body or a subscriber, as on main.
    sig = hw.bool_sig
    sig.put(0)
    susp = SuspendBoolHigh(sig)
    raised = []

    def install_from_the_loop():
        try:
            RE.install_suspender(susp)
        except BaseException as exc:  # noqa: BLE001
            raised.append(exc)

    commands = []
    # A msg_hook runs on the loop.
    _at_message(RE, commands, sleep=install_from_the_loop)
    RE([Msg("checkpoint"), Msg("sleep", None, 0.01)])

    assert raised == []
    assert susp in RE.suspenders
    RE.clear_suspenders()


def test_a_plan_cannot_remove_a_suspender_it_did_not_install(RE, hw):
    sig = hw.bool_sig
    sig.put(0)
    session_scoped = SuspendBoolHigh(sig)
    RE.install_suspender(session_scoped)

    with pytest.warns(UserWarning, match="can only remove a suspender it installed itself"):
        RE([Msg("remove_suspender", None, session_scoped)])

    assert session_scoped in RE.suspenders


def test_removing_a_suspender_settles_before_it_returns(RE, hw):
    sig = hw.bool_sig
    sig.put(1)  # bad, and it has emitted, so the subscription reports it
    suspender = SuspendBoolHigh(sig)

    RE.install_suspender(suspender)
    suspension = RE._session._suspension
    assert suspension.reasons

    RE.remove_suspender(suspender)
    assert not suspension.reasons


def test_installing_a_suspender_twice_is_an_error(RE, hw):
    # Allowed again once removed.
    suspender = SuspendBoolHigh(hw.bool_sig)
    RE.install_suspender(suspender)

    with pytest.raises(RuntimeError, match="already installed"):
        RE.install_suspender(suspender)

    RE.remove_suspender(suspender)
    RE.install_suspender(suspender)  # and removing frees it to be installed again


def test_a_pretripped_condition_runs_neither_of_its_plans(RE, hw, pause_if_stuck):
    # Held at its first message, so no pre-plan, and so no post-plan.
    sig = hw.bool_sig
    sig.put(1)  # already bad before anything is installed
    ran = []

    def note(tag):
        def plan():
            ran.append(tag)
            yield Msg("null")

        return plan

    RE.install_suspender(SuspendBoolHigh(sig, pre_plan=note("pre"), post_plan=note("post")))
    commands = []
    _at_message(RE, commands, wait_for=lambda: _in_turn(RE, _clear(commands, sig)))

    RE(SCAN)

    # Waited for the condition to clear.
    assert commands[:2] == ["wait_for", "bool_sig=0"]
    assert ran == []
    # Held, rather than suspended.
    assert commands.count("_start_suspender") == 0


def test_a_condition_joining_a_suspension_runs_its_pre_plan(RE, pause_if_stuck):
    first = Signal(value=0, name="first")
    second = Signal(value=0, name="second")
    first.put(0)
    second.put(0)
    finished = []

    def pre(name):
        def plan():
            yield Msg("null")
            finished.append(name)

        return plan

    RE.install_suspender(SuspendBoolHigh(first, pre_plan=pre("first")))
    RE.install_suspender(SuspendBoolHigh(second, pre_plan=pre("second")))

    commands = []
    # The second trips only once the first's suspension has opened.
    _at_message(
        RE,
        commands,
        sleep=lambda: first.put(1),
        _start_suspender=lambda: second.put(1),
        wait_for=lambda: _in_turn(RE, lambda: (first.put(0), second.put(0))),
    )
    RE(SCAN)

    assert finished == ["first", "second"]


def test_a_joining_pre_plan_that_raises_reaches_the_plan(RE, pause_if_stuck):
    first = Signal(value=0, name="first")
    second = Signal(value=0, name="second")

    def fine():
        yield Msg("null")

    def raises():
        yield Msg("null")
        raise RuntimeError("joiner pre-plan")

    RE.install_suspender(SuspendBoolHigh(first, pre_plan=fine))
    RE.install_suspender(SuspendBoolHigh(second, pre_plan=raises))

    commands = []
    # Neither clears, so a failure to propagate is caught by pause_if_stuck.
    _at_message(RE, commands, sleep=lambda: first.put(1), _start_suspender=lambda: second.put(1))

    with pytest.raises(RuntimeError, match="joiner pre-plan"):
        RE([Msg("checkpoint")] + [Msg("sleep", None, 0.2)] * 10)

    RE.clear_suspenders()


def test_a_suspension_arriving_after_the_plan_ends_does_nothing(RE):
    sig = CallbackSignal(name="too_late_sig")
    RE.install_suspender(SuspendBoolHigh(sig, tripped_message="too late"))

    RE([Msg("null")])
    assert RE._runner.state.is_idle

    sig.put(1)
    # Queued behind the trip, so the trip has landed when this returns.
    run_coro_on_loop(asyncio.sleep(0), RE._loop)

    assert RE._runner.state.is_idle
    assert "too late" in join_justifications(RE._session.suspension_reasons)


def test_pre_plans_run_in_fire_order_and_post_plans_in_reverse(RE, hw, pause_if_stuck):
    # With one rewind.
    from ophyd import Signal

    beam, shutter = hw.bool_sig, Signal(name="shutter_sig", value=0)
    beam.put(0)
    order = []

    def note(tag):
        def plan():
            order.append(tag)
            yield Msg("null")

        return plan

    RE.install_suspender(
        SuspendBoolHigh(beam, pre_plan=note("beam-pre"), post_plan=note("beam-post"), tripped_message="beam")
    )
    RE.install_suspender(
        SuspendBoolHigh(
            shutter, pre_plan=note("shutter-pre"), post_plan=note("shutter-post"), tripped_message="shutter"
        )
    )
    commands = []
    _at_message(
        RE,
        commands,
        sleep=lambda: beam.put(1),
        _start_suspender=lambda: shutter.put(1),
        wait_for=lambda: _in_turn(RE, lambda: (beam.put(0), shutter.put(0))),
    )
    RE(SCAN)

    assert order == ["beam-pre", "shutter-pre", "shutter-post", "beam-post"]
    assert commands.count("_start_suspender") == 1


@requires_ophyd_async
def test_two_conditions_tripping_in_one_turn_each_run_their_plans(RE):
    beam, shutter = _soft_signal(RE, "beam_sig"), _soft_signal(RE, "shutter_sig")
    order = []

    def note(tag):
        def plan():
            order.append(tag)
            yield Msg("null")

        return plan

    RE.install_suspender(SuspendBoolHigh(beam, pre_plan=note("beam-pre"), post_plan=note("beam-post")))
    RE.install_suspender(SuspendBoolHigh(shutter, pre_plan=note("shutter-pre"), post_plan=note("shutter-post")))
    commands = []

    def both_bad():
        beam.set(1)
        shutter.set(1)

    def both_good():
        beam.set(0)
        shutter.set(0)

    _at_message(RE, commands, sleep=both_bad, _start_suspender=both_good)
    RE(SCAN)

    assert order == ["beam-pre", "shutter-pre", "shutter-post", "beam-post"]
    assert commands.count("_start_suspender") == 1


def test_a_trip_just_after_the_plan_starts_still_suspends(RE, hw, pause_if_stuck):
    # Not taken for a pre-trip.
    sig = hw.bool_sig
    sig.put(0)
    RE.install_suspender(SuspendBoolHigh(sig))
    commands = []

    def trip_as_the_plan_starts(new_state, old_state):
        if (old_state, new_state) == ("idle", "running"):
            # The supervisor exists but has not had a turn yet.
            sig.put(1)

    RE.state_hook = trip_as_the_plan_starts
    _at_message(RE, commands, wait_for=lambda: _in_turn(RE, _clear(commands, sig)))
    RE(SCAN)

    # Waited for the condition to clear.
    assert commands[commands.index("wait_for") + 1] == "bool_sig=0"
    assert commands.count("_start_suspender") == 1


def test_installing_a_suspender_on_the_run_engine_still_works(RE, hw):
    # Deprecated, so it warns.
    sig = hw.bool_sig
    sig.put(0)
    susp = SuspendBoolHigh(sig)

    with pytest.warns(DeprecationWarning, match="takes the suspension"):
        susp.install(RE)

    assert susp in RE.suspenders
    sig.put(1)
    _settle(RE)
    assert RE._session.suspension_reasons
    sig.put(0)
    _settle(RE)
    assert not RE._session.suspension_reasons


def test_a_suspension_reaches_the_suspension_began_hook(RE, pause_if_stuck):
    suspensions: list[Mapping[Hashable, SuspensionReason]] = []
    RE._session.hooks.suspension_began = suspensions.append

    sig = Signal(value=0, name="s")
    sig.put(0)
    susp = SuspendBoolHigh(sig)
    RE.install_suspender(susp)

    commands = []
    _at_message(RE, commands, sleep=lambda: sig.put(1), wait_for=lambda: _in_turn(RE, _clear(commands, sig)))
    RE(SCAN)

    assert suspensions
    (reasons,) = suspensions
    assert list(reasons) == [susp]
    assert join_justifications(reasons) == "Signal s is high"


# --------------------------------------------------------------------------
# The suspension itself


def test_a_condition_tripping_while_paused_joins_the_open_suspension(RE, hw, pause_if_stuck):
    # Its pre-plan runs at the resume, its post-plan with the rest.
    beam, shutter = CallbackSignal(name="beam"), CallbackSignal(name="shutter")
    ran = []

    def note(tag):
        def plan():
            ran.append(tag)
            yield Msg("null")

        return plan

    RE.install_suspender(SuspendBoolHigh(beam, pre_plan=note("beam-pre"), post_plan=note("beam-post")))
    RE.install_suspender(SuspendBoolHigh(shutter, pre_plan=note("shutter-pre"), post_plan=note("shutter-post")))

    commands = []
    # Paused in the suspension. request_pause waits on the loop, so call it from a thread.
    _at_message(
        RE,
        commands,
        sleep=lambda: beam.put(1),
        wait_for=lambda: threading.Thread(target=RE.request_pause).start(),
    )
    with pytest.raises(RunEngineInterrupted):
        RE(SCAN)

    assert ran == ["beam-pre"]

    shutter.put(1)
    _settle(RE)
    # Nothing runs while paused.
    assert ran == ["beam-pre"]

    _at_message(RE, commands, wait_for=lambda: _in_turn(RE, _clear(commands, beam), _clear(commands, shutter)))
    RE.resume()

    assert ran == ["beam-pre", "shutter-pre", "shutter-post", "beam-post"]


def test_a_condition_that_comes_and_goes_while_paused_never_joins(RE, hw, pause_if_stuck):
    beam, shutter = CallbackSignal(name="beam"), CallbackSignal(name="shutter")
    ran = []

    def note(tag):
        def plan():
            ran.append(tag)
            yield Msg("null")

        return plan

    RE.install_suspender(SuspendBoolHigh(beam, pre_plan=note("beam-pre"), post_plan=note("beam-post")))
    RE.install_suspender(SuspendBoolHigh(shutter, pre_plan=note("shutter-pre"), post_plan=note("shutter-post")))

    commands = []
    # Paused in the suspension. request_pause waits on the loop, so call it from a thread.
    _at_message(
        RE,
        commands,
        sleep=lambda: beam.put(1),
        wait_for=lambda: threading.Thread(target=RE.request_pause).start(),
    )
    with pytest.raises(RunEngineInterrupted):
        RE(SCAN)

    shutter.put(1)
    shutter.put(0)
    _settle(RE)

    _at_message(RE, commands, wait_for=lambda: _in_turn(RE, _clear(commands, beam)))
    RE.resume()

    assert ran == ["beam-pre", "beam-post"]


# --------------------------------------------------------------------------
# A second condition, at each point in a suspension


def _two_conditions(RE, a_pre=None, a_post=None, **kwargs):
    """Suspenders on two signals, recording their plans in ``order``."""
    beam, shutter = CallbackSignal(name="beam"), CallbackSignal(name="shutter")
    order = []

    def note(tag, then=None):
        def plan():
            order.append(tag)
            if then is not None:
                then()
            yield Msg("null")

        return plan

    RE.install_suspender(
        SuspendBoolHigh(beam, pre_plan=note("A-pre", a_pre), post_plan=note("A-post", a_post), **kwargs)
    )
    RE.install_suspender(SuspendBoolHigh(shutter, pre_plan=note("B-pre"), post_plan=note("B-post")))
    return beam, shutter, order


def test_a_condition_tripping_during_a_pre_plan_joins(RE, pause_if_stuck):
    beam, shutter, order = _two_conditions(RE, a_pre=lambda: shutter.put(1))
    commands = []
    _at_message(
        RE,
        commands,
        sleep=lambda: beam.put(1),
        wait_for=lambda: _in_turn(RE, _clear(commands, beam), _clear(commands, shutter)),
    )
    RE(SCAN)

    assert order == ["A-pre", "B-pre", "B-post", "A-post"]
    assert commands.count("_start_suspender") == 1
    RE.clear_suspenders()


def test_a_condition_tripping_during_the_wait_joins(RE, pause_if_stuck):
    beam, shutter, order = _two_conditions(RE)
    commands = []
    _at_message(
        RE,
        commands,
        sleep=lambda: beam.put(1),
        wait_for=lambda: _in_turn(RE, lambda: shutter.put(1), _clear(commands, beam), _clear(commands, shutter)),
    )
    RE(SCAN)

    assert order == ["A-pre", "B-pre", "B-post", "A-post"]
    assert commands.count("_start_suspender") == 1
    RE.clear_suspenders()


def test_a_condition_tripping_in_the_first_ones_settle_window_joins(RE, pause_if_stuck):
    beam, shutter, order = _two_conditions(RE, sleep=0.1)
    commands = []

    def recover_then_second_trips():
        # Released at 0.1, unless something else still stands.
        beam.put(0)
        shutter.put(1)
        # Real time: the shutter still stands when the beam's release comes due.
        RE.loop.call_later(0.2, shutter.put, 0)

    _at_message(RE, commands, sleep=lambda: beam.put(1), wait_for=lambda: _in_turn(RE, recover_then_second_trips))
    RE(SCAN)

    assert order == ["A-pre", "B-pre", "B-post", "A-post"]
    assert commands.count("_start_suspender") == 1
    RE.clear_suspenders()


def test_a_condition_tripping_after_release_is_a_new_suspension(RE, pause_if_stuck):
    beam, shutter, order = _two_conditions(RE, a_post=lambda: shutter.put(1))
    commands = []
    _at_message(
        RE,
        commands,
        sleep=lambda: beam.put(1),
        wait_for=lambda: _in_turn(RE, _clear(commands, beam), _clear(commands, shutter)),
    )
    RE(SCAN)

    assert order == ["A-pre", "A-post", "B-pre", "B-post"]
    assert commands.count("_start_suspender") == 2
    RE.clear_suspenders()


def test_a_condition_tripping_as_the_plan_pauses_inside_a_suspension_joins_at_resume(RE, pause_if_stuck):
    beam, shutter, order = _two_conditions(RE)

    class Device:
        name = "device"

        def pause(self):
            if str(RE.state) == "pausing":
                shutter.put(1)

        def resume(self):
            pass

    commands = []
    # Paused in the suspension. request_pause waits on the loop, so call it from a thread.
    _at_message(
        RE,
        commands,
        sleep=lambda: beam.put(1),
        wait_for=lambda: threading.Thread(target=RE.request_pause).start(),
    )
    with pytest.raises(RunEngineInterrupted):
        RE([Msg("checkpoint"), Msg("null", Device()), Msg("sleep", None, 0.05)])
    assert order == ["A-pre"]

    _at_message(RE, commands, wait_for=lambda: _in_turn(RE, _clear(commands, beam), _clear(commands, shutter)))
    RE.resume()

    assert order == ["A-pre", "B-pre", "B-post", "A-post"]
    RE.clear_suspenders()


def test_a_joining_condition_is_announced(RE, capsys, pause_if_stuck):
    beam, shutter, _ = _two_conditions(RE)
    commands = []
    _at_message(
        RE,
        commands,
        sleep=lambda: beam.put(1),
        wait_for=lambda: _in_turn(RE, lambda: shutter.put(1), _clear(commands, beam), _clear(commands, shutter)),
    )
    capsys.readouterr()
    RE(SCAN)

    joined = [line for line in capsys.readouterr().out.splitlines() if "has also tripped" in line]
    assert len(joined) == 1
    assert "Signal shutter is high" in joined[0]
    RE.clear_suspenders()


def test_a_recovering_condition_says_when_the_plan_will_move(RE, capsys, pause_if_stuck):
    # Main's wording.
    beam = CallbackSignal(name="beam")
    # Real time: the settle is what is announced.
    RE.install_suspender(SuspendBoolHigh(beam, sleep=0.1))
    commands = []
    _at_message(RE, commands, sleep=lambda: beam.put(1), wait_for=lambda: _in_turn(RE, _clear(commands, beam)))
    capsys.readouterr()
    RE(SCAN)

    out = capsys.readouterr().out
    assert (
        "reports a return to nominal conditions. Will sleep for 0.1 seconds and then release suspension at" in out
    )
    RE.clear_suspenders()


@pytest.mark.parametrize("remove", [False, True])
def test_a_condition_dropped_at_once_still_says_so(RE, capsys, pause_if_stuck, remove):
    # Main announced a recovery with no sleep, and a removal, alike.
    beam = CallbackSignal(name="beam")
    suspender = SuspendBoolHigh(beam)
    RE.install_suspender(suspender)
    commands = []
    end = (lambda: RE.remove_suspender(suspender)) if remove else _clear(commands, beam)
    _at_message(RE, commands, sleep=lambda: beam.put(1), wait_for=lambda: _in_turn(RE, end))
    capsys.readouterr()
    RE(SCAN)

    out = capsys.readouterr().out
    assert "reports a return to nominal conditions. Will sleep for 0 seconds" in out
    RE.clear_suspenders()


def test_removing_a_suspender_off_the_loop_warns_and_still_removes_it(RE):
    sig = CallbackSignal(name="beam")
    susp = SuspendBoolHigh(sig)
    RE.install_suspender(susp)
    sig.put(1)
    _settle(RE)
    assert susp.tripped

    with pytest.warns(DeprecationWarning, match="remove_suspender"):
        susp.remove()

    assert not susp.tripped
    sig.put(1)
    _settle(RE)
    assert not susp.tripped
    sig.put(0)
    RE.clear_suspenders()
