import asyncio
import threading
import time
import time as ttime
from functools import partial

import pytest

from bluesky import Msg
from bluesky.preprocessors import suspend_wrapper
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
from bluesky.tests import ophyd_async, requires_ophyd_async
from bluesky.tests.utils import MsgCollector

from .utils import _fabricate_asycio_event

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
    my_suspender.install(RE)

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
        susp.install(RE)
        susp.remove()

    loop_thread = _loop_thread_ident(RE)
    assert sig.threads == {"subscribe_reading": loop_thread, "clear_sub": loop_thread}


def test_remove_without_install_does_not_need_a_loop():
    sig = _RecordingSignal()

    SuspendBoolHigh(sig).remove()

    assert sig.threads == {}


def test_RE_is_a_read_write_alias_of_session(RE):
    """``RE`` was a plain attribute before the session existed, so reading and
    writing it both have to keep working."""
    susp = SuspendBoolHigh(_RecordingSignal())
    assert susp.RE is susp.session is None

    RE.install_suspender(susp)
    assert susp.session is RE._session
    assert susp.RE is RE._session

    susp.RE = None
    assert susp.session is None

    RE.remove_suspender(susp)


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
def test_suspender_plans_async_signal(RE):
    "Tests that an async suspender can be installed and removed via Msg"
    sig = _connected_soft_signal(RE, 0)
    my_suspender = SuspendBoolHigh(sig, sleep=0.2)
    scan = [Msg("checkpoint"), Msg("sleep", None, 0.2)]

    def trip_then_clear():
        threading.Timer(0.1, _set_on_loop, (RE, sig, 1)).start()
        threading.Timer(0.5, _set_on_loop, (RE, sig, 0)).start()

    # installed from inside a plan, it suspends and resumes
    trip_then_clear()
    start = ttime.time()
    RE([Msg("install_suspender", None, my_suspender)] + scan)
    assert ttime.time() - start > 0.4 + 0.2 + 0.2
    assert my_suspender in RE.suspenders

    # removed from inside a plan, it no longer does
    trip_then_clear()
    start = ttime.time()
    RE([Msg("remove_suspender", None, my_suspender)] + scan)
    assert ttime.time() - start < 0.5
    assert my_suspender not in RE.suspenders


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

    # Waiting for an already-tripped suspender goes through the same
    # _start_suspender machinery as a mid-plan suspend (see
    # test_pre_suspend_plan), rather than a single bespoke 'wait_for', so the
    # msg_hook sees that machinery's messages ahead of the plan's own.
    assert [
        "_start_suspender",
        "rewindable",
        "wait_for",
        "_resume_from_suspender",
        "rewindable",
        "checkpoint",
    ] == [m[0] for m in msg_lst]
    assert RE.state == "idle"


def test_pretripped_plan_runs_to_completion_after_release(RE, hw):
    """A suspender already tripped when RE(plan) is called holds the plan back
    until it clears, then the plan's own messages run in order and the
    RunEngine ends idle -- the behaviour prologue used to provide."""
    sig = hw.bool_sig
    scan = [Msg("open_run"), Msg("close_run")]
    msg_lst = []
    sig.put(0)

    susp = SuspendBoolLow(sig, sleep=0)
    RE.install_suspender(susp)
    RE.msg_hook = lambda msg: msg_lst.append(msg.command)
    threading.Timer(0.5, sig.put, (1,)).start()

    start = ttime.time()
    RE(scan)
    stop = ttime.time()

    assert stop - start > 0.5
    assert [m for m in msg_lst if m in ("open_run", "close_run")] == ["open_run", "close_run"]
    assert RE.state == "idle"


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
def test_pre_suspend_plan(RE, pre_plan, post_plan, expected_list, hw):
    sig = hw.bool_sig
    scan = [Msg("checkpoint"), Msg("sleep", None, 0.2)]
    msg_lst = []
    sig.put(0)

    def accum(msg):
        msg_lst.append(msg)

    susp = SuspendBoolHigh(sig, pre_plan=pre_plan, post_plan=post_plan)

    RE.install_suspender(susp)
    threading.Timer(0.1, sig.put, (1,)).start()
    threading.Timer(1, sig.put, (0,)).start()
    RE.msg_hook = accum
    RE(scan)

    assert len(msg_lst) == len(expected_list)
    assert expected_list == [m[0] for m in msg_lst]

    RE.remove_suspender(susp)
    RE(scan)
    assert susp.RE is None

    RE.install_suspender(susp)
    RE.clear_suspenders()
    assert susp.RE is None
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
    # Waiting for the already-tripped suspender goes through _start_suspender,
    # same as a mid-plan suspend (see test_pretripped), so the pause catches
    # it still inside that machinery rather than at a bare 'wait_for'.
    assert [m[0] for m in msg_lst] == ["_start_suspender", "rewindable", "wait_for"]
    RE.resume()
    assert [
        "_start_suspender",
        "rewindable",
        "wait_for",
        "rewindable",
        "_resume_from_suspender",
        "rewindable",
        "checkpoint",
    ] == [m[0] for m in msg_lst]


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
    # See test_pretripped: waiting for the already-tripped suspender goes
    # through _start_suspender, same as a mid-plan suspend.
    assert [m[0] for m in msg_lst] == [
        "_start_suspender",
        "rewindable",
        "wait_for",
        "_resume_from_suspender",
        "rewindable",
        "checkpoint",
    ]
    RE.resume()
    assert [
        "_start_suspender",
        "rewindable",
        "wait_for",
        "_resume_from_suspender",
        "rewindable",
        "checkpoint",
        "null",
    ] == [m[0] for m in msg_lst]


def test_unresumable_suspend_fail(RE):
    "Tests what happens when a soft pause is requested from a suspended state"

    scan = [Msg("clear_checkpoint"), Msg("sleep", None, 2)]
    m_coll = MsgCollector()
    RE.msg_hook = m_coll

    ev = _fabricate_asycio_event(RE.loop)
    threading.Timer(0.1, partial(RE.request_suspend, fut=ev.wait)).start()
    threading.Timer(1, ev.set).start()
    start = time.time()
    with pytest.raises(RunEngineInterrupted):
        RE(scan)
    stop = time.time()
    assert 0.1 < stop - start < 1


def test_suspender_plans(RE, hw):
    "Tests that the suspenders can be installed via Msg"
    sig = hw.bool_sig
    my_suspender = SuspendBoolHigh(sig, sleep=0.2)

    def putter(val):
        sig.put(val)

    putter(0)

    # Do the messages work?
    RE([Msg("install_suspender", None, my_suspender)])
    assert my_suspender in RE.suspenders
    RE([Msg("remove_suspender", None, my_suspender)])
    assert my_suspender not in RE.suspenders

    # Can we call both in a plan?
    RE([Msg("install_suspender", None, my_suspender), Msg("remove_suspender", None, my_suspender)])

    scan = [Msg("checkpoint"), Msg("sleep", None, 0.2)]

    # No suspend scan: does the wrapper error out?
    start = ttime.time()
    RE(suspend_wrapper(scan, my_suspender))
    stop = ttime.time()
    delta = stop - start
    assert delta < 0.9

    # Suspend scan
    start = ttime.time()
    threading.Timer(0.1, putter, (1,)).start()
    threading.Timer(0.5, putter, (0,)).start()
    RE(suspend_wrapper(scan, my_suspender))
    stop = ttime.time()
    delta = stop - start
    assert delta > 0.9

    # Did we clean up?
    start = ttime.time()
    threading.Timer(0.1, putter, (1,)).start()
    threading.Timer(0.5, putter, (0,)).start()
    RE(scan)
    stop = ttime.time()
    delta = stop - start
    assert delta < 0.9
