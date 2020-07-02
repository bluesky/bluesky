import pytest
from functools import partial
from bluesky.preprocessors import suspend_wrapper
from bluesky.suspenders import (SuspendBoolHigh,
                                SuspendBoolLow,
                                SuspendFloor,
                                SuspendCeil,
                                SuspendWhenOutsideBand,
                                SuspendInBand,
                                SuspendOutBand)
from bluesky.tests.utils import MsgCollector
from bluesky import Msg
import time as ttime
from bluesky.run_engine import RunEngineInterrupted
import threading
import time
from .utils import _fabricate_asycio_event


@pytest.mark.parametrize(
    'klass,sc_args,start_val,fail_val,resume_val,wait_time',
    [(SuspendBoolHigh, (), 0, 1, 0, .2),
     (SuspendBoolLow, (), 1, 0, 1, .2),
     (SuspendFloor, (.5,), 1, 0, 1, .2),
     (SuspendCeil, (.5,), 0, 1, 0, .2),
     (SuspendWhenOutsideBand, (.5, 1.5), 1, 0, 1, .2),
     ((SuspendInBand, True), (.5, 1.5), 1, 0, 1, .2),  # renamed to WhenOutsideBand
     ((SuspendOutBand, True), (.5, 1.5), 0, 1, 0, .2)])  # deprecated
def test_suspender(klass, sc_args, start_val, fail_val,
                   resume_val, wait_time, RE, hw):
    sig = hw.bool_sig
    try:
        klass, deprecated = klass
    except TypeError:
        deprecated = False
    if deprecated:
        with pytest.warns(UserWarning):
            my_suspender = klass(sig,
                                 *sc_args, sleep=wait_time)
    else:
        my_suspender = klass(sig,
                             *sc_args, sleep=wait_time)
    my_suspender.install(RE)

    def putter(val):
        sig.put(val)

    # make sure we start at good value!
    putter(start_val)
    # dumb scan
    scan = [Msg('checkpoint'), Msg('sleep', None, .2)]
    RE(scan)
    # paranoid
    assert RE.state == 'idle'

    start = ttime.time()
    # queue up fail and resume conditions
    threading.Timer(.1, putter, (fail_val,)).start()
    threading.Timer(.5, putter, (resume_val,)).start()
    # start the scan
    RE(scan)
    stop = ttime.time()
    # assert we waited at least 2 seconds + the settle time
    delta = stop - start
    print(delta)
    assert delta > .5 + wait_time + .2


def test_pretripped(RE, hw):
    'Tests if suspender is tripped before __call__'
    sig = hw.bool_sig
    scan = [Msg('checkpoint')]
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
    assert ['wait_for', 'checkpoint'] == [m[0] for m in msg_lst]


@pytest.mark.parametrize('pre_plan,post_plan,expected_list',
                         [([Msg('null')], None,
                           ['checkpoint', 'sleep', 'rewindable', 'null',
                            'wait_for', 'resume', 'rewindable', 'sleep']),
                          (None, [Msg('null')],
                           ['checkpoint', 'sleep', 'rewindable',
                            'wait_for', 'resume', 'null', 'rewindable',
                            'sleep']),
                          ([Msg('null')], [Msg('null')],
                           ['checkpoint', 'sleep', 'rewindable', 'null',
                            'wait_for', 'resume', 'null', 'rewindable',
                            'sleep']),
                          (lambda: [Msg('null')], lambda: [Msg('null')],
                           ['checkpoint', 'sleep', 'rewindable', 'null',
                            'wait_for', 'resume', 'null', 'rewindable',
                            'sleep'])])
def test_pre_suspend_plan(RE, pre_plan, post_plan, expected_list, hw):
    sig = hw.bool_sig
    scan = [Msg('checkpoint'), Msg('sleep', None, .2)]
    msg_lst = []
    sig.put(0)

    def accum(msg):
        msg_lst.append(msg)

    susp = SuspendBoolHigh(sig, pre_plan=pre_plan,
                           post_plan=post_plan)

    RE.install_suspender(susp)
    threading.Timer(.1, sig.put, (1,)).start()
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
    'Tests what happens when a pause is requested from a suspended state'
    sig = hw.bool_sig
    scan = [Msg('checkpoint')]
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
    assert [m[0] for m in msg_lst] == ['wait_for']
    RE.resume()
    assert ['wait_for', 'wait_for', 'checkpoint'] == [m[0] for m in msg_lst]


def test_deferred_pause_from_suspend(RE, hw):
    'Tests what happens when a soft pause is requested from a suspended state'
    sig = hw.bool_sig
    scan = [Msg('checkpoint'), Msg('null')]
    msg_lst = []
    sig.put(1)

    def accum(msg):
        print(msg)
        msg_lst.append(msg)

    susp = SuspendBoolHigh(sig)

    RE.install_suspender(susp)
    threading.Timer(1, RE.request_pause, (True,)).start()
    threading.Timer(4, sig.put, (0,)).start()
    RE.msg_hook = accum
    with pytest.raises(RunEngineInterrupted):
        RE(scan)
    assert [m[0] for m in msg_lst] == ['wait_for', 'checkpoint']
    RE.resume()
    assert ['wait_for', 'checkpoint', 'null'] == [m[0] for m in msg_lst]


def test_unresumable_suspend_fail(RE):
    'Tests what happens when a soft pause is requested from a suspended state'

    scan = [Msg('clear_checkpoint'), Msg('sleep', None, 2)]
    m_coll = MsgCollector()
    RE.msg_hook = m_coll

    ev = _fabricate_asycio_event(RE.loop)
    threading.Timer(.1, partial(RE.request_suspend, fut=ev.wait)).start()
    threading.Timer(1, ev.set).start()
    start = time.time()
    with pytest.raises(RunEngineInterrupted):
        RE(scan)
    stop = time.time()
    assert .1 < stop - start < 1


def test_suspender_plans(RE, hw):
    'Tests that the suspenders can be installed via Msg'
    sig = hw.bool_sig
    my_suspender = SuspendBoolHigh(sig, sleep=0.2)

    def putter(val):
        sig.put(val)

    putter(0)

    # Do the messages work?
    RE([Msg('install_suspender', None, my_suspender)])
    assert my_suspender in RE.suspenders
    RE([Msg('remove_suspender', None, my_suspender)])
    assert my_suspender not in RE.suspenders

    # Can we call both in a plan?
    RE([Msg('install_suspender', None, my_suspender),
        Msg('remove_suspender', None, my_suspender)])

    scan = [Msg('checkpoint'), Msg('sleep', None, .2)]

    # No suspend scan: does the wrapper error out?
    start = ttime.time()
    RE(suspend_wrapper(scan, my_suspender))
    stop = ttime.time()
    delta = stop - start
    assert delta < .9

    # Suspend scan
    start = ttime.time()
    threading.Timer(.1, putter, (1,)).start()
    threading.Timer(.5, putter, (0,)).start()
    RE(suspend_wrapper(scan, my_suspender))
    stop = ttime.time()
    delta = stop - start
    assert delta > .9

    # Did we clean up?
    start = ttime.time()
    threading.Timer(.1, putter, (1,)).start()
    threading.Timer(.5, putter, (0,)).start()
    RE(scan)
    stop = ttime.time()
    delta = stop - start
    assert delta < .9
