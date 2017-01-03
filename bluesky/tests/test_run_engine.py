import asyncio
import threading
import os
import signal
import sys
from collections import defaultdict
import time as ttime
import pytest
from bluesky.run_engine import (RunEngineStateMachine,
                                TransitionError, IllegalMessageSequence,
                                NoReplayAllowed, FailedStatus)
from bluesky import Msg
from functools import partial
from bluesky.examples import det, Mover, TrivialFlyer, SynGauss, SimpleStatus
import bluesky.plans as bp
from bluesky.tests.utils import MsgCollector


def test_states():
    assert RunEngineStateMachine.States.states() == ['idle',
                                                     'running',
                                                     'paused']


def test_verbose(fresh_RE):
    fresh_RE.verbose = True
    assert fresh_RE.verbose
    # Emit all four kinds of document, exercising the logging.
    fresh_RE([Msg('open_run'), Msg('create'), Msg('read', det), Msg('save'),
              Msg('close_run')])


def test_reset(fresh_RE):
    fresh_RE([Msg('open_run'), Msg('pause')])
    assert fresh_RE._run_start_uid is not None
    fresh_RE.reset()
    assert fresh_RE._run_start_uid is None


def test_running_from_paused_state_raises(fresh_RE):
    fresh_RE([Msg('pause')])
    assert fresh_RE.state == 'paused'
    with pytest.raises(RuntimeError):
        fresh_RE([Msg('null')])
    fresh_RE.resume()
    assert fresh_RE.state == 'idle'
    fresh_RE([Msg('null')])


def test_resuming_from_idle_state_raises(fresh_RE):
    with pytest.raises(RuntimeError):
        fresh_RE.resume()
    fresh_RE([Msg('pause')])
    assert fresh_RE.state == 'paused'
    fresh_RE.resume()
    assert fresh_RE.state == 'idle'
    with pytest.raises(RuntimeError):
        fresh_RE.resume()


def test_stopping_from_idle_state_raises(fresh_RE):
    with pytest.raises(TransitionError):
        fresh_RE.stop()


def test_pausing_from_idle_state_raises(fresh_RE):
    with pytest.raises(TransitionError):
        fresh_RE.request_pause(defer=False)


def test_aborting_from_idle_state_raises(fresh_RE):
    with pytest.raises(TransitionError):
        fresh_RE.abort()


def test_register(fresh_RE):
    mutable = {}
    fresh_RE.verbose = True

    @asyncio.coroutine
    def func(msg):
        mutable['flag'] = True

    def plan():
        yield Msg('custom-command')

    fresh_RE.register_command('custom-command', func)
    fresh_RE(plan())
    assert 'flag' in mutable
    # Unregister command; now the Msg should not be recognized.
    fresh_RE.unregister_command('custom-command')
    with pytest.raises(KeyError):
        fresh_RE([Msg('custom-command')])


def test_stop_motors_and_log_any_errors(fresh_RE):
    # test that if stopping one motor raises an error, we can carry on
    stopped = {}

    class MoverWithFlag(Mover):
        def stop(self, *, success=False):
            stopped[self.name] = True

    class BrokenMoverWithFlag(Mover):
        def stop(self, *, success=False):
            stopped[self.name] = True
            raise Exception

    motor = MoverWithFlag('a', {'a': lambda x: x}, {'x': 0})
    broken_motor = BrokenMoverWithFlag('b', {'b': lambda x: x}, {'x': 0})

    fresh_RE([Msg('set', broken_motor, 1), Msg('set', motor, 1), Msg('pause')])
    assert 'a' in stopped
    assert 'b' in stopped
    fresh_RE.stop()

    fresh_RE([Msg('set', motor, 1), Msg('set', broken_motor, 1), Msg('pause')])
    assert 'a' in stopped
    assert 'b' in stopped
    fresh_RE.stop()


def test_collect_uncollected_and_log_any_errors(fresh_RE):
    # test that if stopping one motor raises an error, we can carry on
    collected = {}

    class DummyFlyerWithFlag(TrivialFlyer):
        def collect(self):
            collected[self.name] = True
            super().collect()

    class BrokenDummyFlyerWithFlag(DummyFlyerWithFlag):
        def collect(self):
            super().collect()
            raise Exception

    flyer1 = DummyFlyerWithFlag()
    flyer1.name = 'flyer1'
    flyer2 = DummyFlyerWithFlag()
    flyer2.name = 'flyer2'

    collected.clear()
    fresh_RE([Msg('open_run'), Msg('kickoff', flyer1), Msg('kickoff', flyer2)])
    assert 'flyer1' in collected
    assert 'flyer2' in collected

    collected.clear()
    fresh_RE([Msg('open_run'), Msg('kickoff', flyer2), Msg('kickoff', flyer1)])
    assert 'flyer1' in collected
    assert 'flyer2' in collected


def test_unstage_and_log_errors(fresh_RE):
    unstaged = {}

    class MoverWithFlag(Mover):

        def stage(self):
            return [self]

        def unstage(self):
            unstaged[self.name] = True
            return [self]

    class BrokenMoverWithFlag(Mover):

        def stage(self):
            return [self]

        def unstage(self):
            unstaged[self.name] = True
            return [self]

    a = MoverWithFlag('a', {'a': lambda x: x}, {'x': 0})
    b = BrokenMoverWithFlag('b', {'b': lambda x: x}, {'x': 0})

    unstaged.clear()
    fresh_RE([Msg('stage', a), Msg('stage', b)])
    assert 'a' in unstaged
    assert 'b' in unstaged

    unstaged.clear()
    fresh_RE([Msg('stage', b), Msg('stage', a)])
    assert 'a' in unstaged
    assert 'b' in unstaged


def test_open_run_twice_is_illegal(fresh_RE):
    with pytest.raises(IllegalMessageSequence):
        fresh_RE([Msg('open_run'), Msg('open_run')])


def test_saving_without_an_open_bundle_is_illegal(fresh_RE):
    with pytest.raises(IllegalMessageSequence):
        fresh_RE([Msg('open_run'), Msg('save')])


def test_opening_a_bundle_without_a_run_is_illegal(fresh_RE):
    with pytest.raises(IllegalMessageSequence):
        fresh_RE([Msg('create')])


def test_checkpoint_inside_a_bundle_is_illegal(fresh_RE):
    with pytest.raises(IllegalMessageSequence):
        fresh_RE([Msg('open_run'), Msg('create'), Msg('checkpoint')])


def test_redundant_monitors_are_illegal(fresh_RE):
    class Dummy:
        def __init__(self, name):
            self.name = name

        def read_configuration(self):
            return {}

        def describe_configuration(self):
            return {}

        def describe(self):
            return {}

        def subscribe(self, *args, **kwargs):
            pass

        def clear_sub(self, *args, **kwargs):
            pass

    dummy = Dummy('dummy')

    with pytest.raises(IllegalMessageSequence):
        fresh_RE([Msg('open_run'), Msg('monitor', dummy),
                  Msg('monitor', dummy)])

    # Monitoring, unmonitoring, and monitoring again is legal.
    fresh_RE([Msg('open_run'), Msg('monitor', dummy), Msg('unmonitor', dummy),
              Msg('monitor', dummy)])

    # Monitoring outside a run is illegal.
    with pytest.raises(IllegalMessageSequence):
        fresh_RE([Msg('monitor', dummy)])

    # Unmonitoring something that was never monitored is illegal.
    with pytest.raises(IllegalMessageSequence):
        fresh_RE([Msg('unmonitor', dummy)])


def test_flying_outside_a_run_is_illegal(fresh_RE):

    flyer = TrivialFlyer()

    # This is normal, legal usage.
    fresh_RE([Msg('open_run'), Msg('kickoff', flyer), Msg('collect', flyer),
              Msg('close_run')])

    # It is not legal to kickoff outside of a run.
    with pytest.raises(IllegalMessageSequence):
        fresh_RE([Msg('kickoff', flyer)])

    # And of course you can't collect a flyer you never kicked off.
    with pytest.raises(IllegalMessageSequence):
        fresh_RE([Msg('open_run'), Msg('collect', flyer), Msg('close_run')])


def test_empty_bundle(fresh_RE):
    mutable = {}

    def cb(name, doc):
        mutable['flag'] = True

    # In this case, an Event should be emitted.
    mutable.clear()
    fresh_RE([Msg('open_run'), Msg('create'), Msg('read', det), Msg('save')],
             subs={'event': cb})
    assert 'flag' in mutable

    # In this case, an Event should not be emitted because the bundle is
    # emtpy (i.e., there are no readings.)
    mutable.clear()
    fresh_RE([Msg('open_run'), Msg('create'), Msg('save')], subs={'event': cb})
    assert 'flag' not in mutable


def test_dispatcher_unsubscribe_all(fresh_RE):
    def count_callbacks(RE):
        return sum([len(cbs) for cbs in
                    RE.dispatcher.cb_registry.callbacks.values()])

    def cb(name, doc):
        pass

    fresh_RE.subscribe('all', cb)
    assert count_callbacks(fresh_RE) == 5
    fresh_RE.dispatcher.unsubscribe_all()
    assert count_callbacks(fresh_RE) == 0


def test_stage_and_unstage_are_optional_methods(fresh_RE):
    class Dummy:
        pass

    dummy = Dummy()

    fresh_RE([Msg('stage', dummy), Msg('unstage', dummy)])


def test_pause_resume_devices(fresh_RE):
    paused = {}
    resumed = {}

    class Dummy:
        def __init__(self, name):
            self.name = name

        def pause(self):
            paused[self.name] = True

        def resume(self):
            resumed[self.name] = True

    dummy = Dummy('dummy')

    fresh_RE([Msg('stage', dummy), Msg('pause')])
    fresh_RE.resume()
    assert 'dummy' in paused
    assert 'dummy' in resumed


def test_bad_call_args(fresh_RE):
    RE = fresh_RE
    with pytest.raises(Exception):
        RE(53)
    assert RE.state == 'idle'


def test_record_interruptions(fresh_RE):
    docs = defaultdict(lambda: [])

    def collect(name, doc):
        print("HI", name)
        docs[name].append(doc)
        print(docs)

    fresh_RE.subscribe('all', collect)
    fresh_RE.ignore_callback_exceptions = False
    fresh_RE.msg_hook = print

    # The 'pause' inside the run should generate an event iff
    # record_interruptions is True.
    plan = [Msg('pause'), Msg('open_run'), Msg('pause'), Msg('close_run')]

    assert not fresh_RE.record_interruptions
    fresh_RE(plan)
    fresh_RE.resume()
    fresh_RE.resume()
    assert len(docs['descriptor']) == 0
    assert len(docs['event']) == 0

    fresh_RE.record_interruptions = True
    fresh_RE(plan)
    fresh_RE.resume()
    fresh_RE.resume()
    assert len(docs['descriptor']) == 1
    assert len(docs['event']) == 2
    docs['event'][0]['data']['interruption'] == 'pause'
    docs['event'][1]['data']['interruption'] == 'resume'


def _make_unrewindable_marker():
    class UnReplayableSynGauss(SynGauss):
        def pause(self):
            raise NoReplayAllowed()

    motor = Mover('motor', {'motor': lambda x: x}, {'x': 0})

    def test_plan(motor, det):
        yield Msg('set', motor, 0)
        yield Msg('trigger', det)
        yield Msg('pause')
        yield Msg('set', motor, 1)
        yield Msg('trigger', det)

    inps = []
    inps.append((test_plan,
                 motor,
                 UnReplayableSynGauss('det', motor, 'motor', center=0, Imax=1),
                 ['set', 'trigger', 'pause', 'set', 'trigger']))

    inps.append((test_plan,
                 motor,
                 SynGauss('det', motor, 'motor', center=0, Imax=1),
                 ['set', 'trigger', 'pause',
                  'set', 'trigger', 'set', 'trigger']))

    return pytest.mark.parametrize('plan,motor,det,msg_seq', inps)


@_make_unrewindable_marker()
def test_unrewindable_det(fresh_RE, plan, motor, det, msg_seq):
    RE = fresh_RE
    msgs = []

    def collector(msg):
        msgs.append(msg)
    RE.msg_hook = collector
    RE(plan(motor, det))
    RE.resume()
    assert [m.command for m in msgs] == msg_seq


def _make_unrewindable_suspender_marker():
    class UnReplayableSynGauss(SynGauss):
        def pause(self):
            raise NoReplayAllowed()

    motor = Mover('motor', {'motor': lambda x: x}, {'x': 0})

    def test_plan(motor, det):
        yield Msg('set', motor, 0)
        yield Msg('trigger', det)
        yield Msg('sleep', None, 1)
        yield Msg('set', motor, 0)
        yield Msg('trigger', det)

    inps = []
    inps.append((test_plan,
                 motor,
                 UnReplayableSynGauss('det', motor, 'motor', center=0, Imax=1),
                 ['set', 'trigger', 'sleep', 'wait_for', 'set', 'trigger']))

    inps.append((test_plan,
                 motor,
                 SynGauss('det', motor, 'motor', center=0, Imax=1),
                 ['set', 'trigger', 'sleep', 'wait_for', 'set',
                  'trigger', 'sleep', 'set', 'trigger']))

    return pytest.mark.parametrize('plan,motor,det,msg_seq', inps)


@_make_unrewindable_suspender_marker()
def test_unrewindable_det_suspend(fresh_RE, plan, motor, det, msg_seq):
    RE = fresh_RE
    msgs = []

    def collector(msg):
        msgs.append(msg)
    RE.msg_hook = collector

    ev = asyncio.Event(loop=RE.loop)
    loop = RE.loop
    loop.call_later(.5, partial(RE.request_suspend, fut=ev.wait()))
    loop.call_later(1, ev.set)
    RE(plan(motor, det))
    assert [m.command for m in msgs] == msg_seq


@pytest.mark.parametrize('unpause_func', [lambda RE: RE.stop(),
                                          lambda RE: RE.abort(),
                                          lambda RE: RE.resume()])
def test_cleanup_after_pause(fresh_RE, unpause_func, motor_det):
    RE = fresh_RE

    motor, det = motor_det
    motor.set(1024)

    @bp.reset_positions_decorator()
    def simple_plan(motor):
        for j in range(15):
            yield Msg('set', motor, j)
        yield Msg('pause')
        for j in range(15):
            yield Msg('set', motor, -j)

    RE(simple_plan(motor))
    assert motor.position == 14
    unpause_func(RE)
    assert motor.position == 1024


def test_sigint_three_hits(fresh_RE, motor_det):
    motor, det = motor_det
    motor._fake_sleep = 0.3
    RE = fresh_RE

    pid = os.getpid()

    def sim_kill(n=1):
        for j in range(n):
            print('KILL')
            os.kill(pid, signal.SIGINT)

    lp = RE.loop
    motor.loop = lp
    lp.call_later(.02, sim_kill, 3)
    lp.call_later(.02, sim_kill, 3)
    lp.call_later(.02, sim_kill, 3)
    start_time = ttime.time()
    RE(bp.finalize_wrapper(bp.abs_set(motor, 1, wait=True),
                           bp.abs_set(motor, 0, wait=True)))
    end_time = ttime.time()
    assert end_time - start_time < 0.2  # not enough time for motor to cleanup
    RE.abort()  # now cleanup
    done_cleanup_time = ttime.time()
    assert done_cleanup_time - end_time > 0.3


@pytest.mark.skipif(sys.version_info < (3, 5),
                    reason="requires python3.5")
def test_sigint_many_hits_pln(fresh_RE):
    RE = fresh_RE
    pid = os.getpid()

    def sim_kill(n=1):
        for j in range(n):
            print('KILL', j)
            ttime.sleep(0.05)
            os.kill(pid, signal.SIGINT)

    def hanging_plan():
        "a plan that blocks the RunEngine's normal Ctrl+C handing with a sleep"
        ttime.sleep(10)
        yield Msg('null')

    start_time = ttime.time()
    timer = threading.Timer(0.2, sim_kill, (11,))
    timer.start()
    RE(hanging_plan())
    # Check that hammering SIGINT escaped from that 10-second sleep.
    assert ttime.time() - start_time < 2
    # The KeyboardInterrupt will have been converted to a hard pause.
    assert RE.state == 'idle'


@pytest.mark.skipif(sys.version_info < (3, 5),
                    reason="requires python3.5")
def test_sigint_many_hits_cb(fresh_RE):
    RE = fresh_RE
    pid = os.getpid()

    def sim_kill(n=1):
        for j in range(n):
            print('KILL')
            ttime.sleep(0.05)
            os.kill(pid, signal.SIGINT)

    @bp.run_decorator()
    def infinite_plan():
        while True:
            yield Msg('null')

    def hanging_callback(name, doc):
        ttime.sleep(10)

    start_time = ttime.time()
    timer = threading.Timer(0.2, sim_kill, (11,))
    timer.start()
    RE(infinite_plan(), {'start': hanging_callback})
    # Check that hammering SIGINT escaped from that 10-second sleep.
    assert ttime.time() - start_time < 2
    # The KeyboardInterrupt will have been converted to a hard pause.
    assert RE.state == 'idle'
    # Check that hammering SIGINT escaped from that 10-second sleep.
    assert ttime.time() - start_time < 2


def _make_plan_marker():
    @bp.reset_positions_decorator()
    def raiser(motor):
        for j in range(15):
            yield Msg('set', motor, j)
        raise RuntimeError()

    @bp.reset_positions_decorator()
    def pausing_raiser(motor):
        for j in range(15):
            yield Msg('set', motor, j)
        yield Msg('pause')
        raise RuntimeError()

    @bp.reset_positions_decorator()
    def bad_set(motor):
        for j in range(15):
            yield Msg('set', motor, j)
            yield Msg('set', None, j)

    @bp.reset_positions_decorator()
    def bad_msg(motor):
        for j in range(15):
            yield Msg('set', motor, j)
        yield Msg('aardvark')

    @bp.reset_positions_decorator()
    def cannot_pauser(motor):
        yield Msg('clear_checkpoint')
        for j in range(15):
            yield Msg('set', motor, j)
        yield Msg('pause')

    return pytest.mark.parametrize('plan', [raiser, bad_set, bad_msg,
                                            pausing_raiser, cannot_pauser])


@_make_plan_marker()
def test_cleanup_pathological_plans(fresh_RE, motor_det, plan):
    RE = fresh_RE

    motor, det = motor_det
    motor.set(1024)
    try:
        RE(plan(motor))
        if RE.state == 'paused':
            assert motor.position != 1024
            RE.resume()
    except Exception:
        pass
    assert motor.position == 1024


def test_finalizer_closeable():
    pre = (j for j in range(18))
    post = (j for j in range(18))

    plan = bp.finalize_wrapper(pre, post)

    for j in range(3):
        next(plan)
    plan.close()


def test_invalid_generator(fresh_RE, motor_det, capsys):

    RE = fresh_RE
    motor, det = motor_det

    # this is not a valid generator as it will try to yield if it
    # is throw a GeneratorExit
    def patho_finalize_wrapper(plan, post):
        try:
            yield from plan
        finally:
            yield from post

    def base_plan(motor):
        for j in range(5):
            yield Msg('set', motor, j * 2 + 1)
        yield Msg('pause')

    def post_plan(motor):
        yield Msg('set', motor, 5)

    def pre_suspend_plan():
        yield Msg('set', motor, 5)
        raise GeneratorExit('this one')

    def make_plan():
        return patho_finalize_wrapper(base_plan(motor),
                                      post_plan(motor))

    RE(make_plan())
    RE.request_suspend(None, pre_plan=pre_suspend_plan())
    capsys.readouterr()
    try:
        RE.resume()
    except GeneratorExit as sf:
        assert sf.args[0] == 'this one'

    actual_err, _ = capsys.readouterr()
    expected_prefix = 'The plan '
    expected_postfix = (' tried to yield a value on close.  '
                        'Please fix your plan.\n')[::-1]
    assert actual_err[:len(expected_prefix)] == expected_prefix
    assert actual_err[::-1][:len(expected_postfix)] == expected_postfix


def test_exception_cascade_REside(fresh_RE):
    RE = fresh_RE
    except_hit = False

    def pausing_plan():
        nonlocal except_hit
        for j in range(5):
            yield Msg('null')
        try:
            yield Msg('pause')
        except Exception:
            except_hit = True
            raise

    def pre_plan():
        yield Msg('aardvark')

    def post_plan():
        for j in range(5):
            yield Msg('null')

    RE(pausing_plan())
    ev = asyncio.Event(loop=RE.loop)
    ev.set()
    RE.request_suspend(ev.wait(), pre_plan=pre_plan())
    with pytest.raises(KeyError):
        RE.resume()
    assert except_hit


def test_exception_cascade_planside(fresh_RE):
    RE = fresh_RE
    except_hit = False

    def pausing_plan():
        nonlocal except_hit
        for j in range(5):
            yield Msg('null')
        try:
            yield Msg('pause')
        except Exception:
            except_hit = True
            raise

    def pre_plan():
        yield Msg('null')
        raise RuntimeError()

    def post_plan():
        for j in range(5):
            yield Msg('null')

    RE(pausing_plan())
    ev = asyncio.Event(loop=RE.loop)
    ev.set()
    RE.request_suspend(ev.wait(), pre_plan=pre_plan())
    with pytest.raises(RuntimeError):
        RE.resume()
    assert except_hit


def test_sideband_cancel(fresh_RE):
    RE = fresh_RE
    ev = asyncio.Event(loop=RE.loop)

    def done():
        print("Done")
        ev.set()

    def side_band_kill():
        RE._task.cancel()

    scan = [Msg('wait_for', None, [ev.wait(), ]), ]
    assert RE.state == 'idle'
    start = ttime.time()
    RE.loop.call_later(.5, side_band_kill)
    RE.loop.call_later(2, done)

    RE(scan)
    assert RE.state == 'idle'
    assert RE._task.cancelled()
    stop = ttime.time()

    assert .5 < (stop - start) < 2


def test_no_rewind(fresh_RE):
    RE = fresh_RE
    msg_lst = []

    def msg_collector(msg):
        msg_lst.append(msg)

    RE.rewindable = False

    plan = [Msg('null')] * 3 + [Msg('pause')] + [Msg('null')] * 3
    RE.msg_hook = msg_collector
    RE(plan)
    RE.resume()
    assert msg_lst == plan


def test_no_rewindable_msg(fresh_RE):
    RE = fresh_RE
    RE.rewindable = True

    msg_lst = []

    def msg_collector(msg):
        msg_lst.append(msg)

    plan = ([Msg('null')] * 3 +
            [Msg('pause'), Msg('rewindable', None, False)] +
            [Msg('null')] * 3)

    RE.msg_hook = msg_collector

    RE(plan)
    RE.resume()
    assert msg_lst[:4] == plan[:4]
    assert msg_lst[4:7] == plan[:3]
    assert msg_lst[7:] == plan[4:]


@pytest.mark.parametrize('start_state', [True, False])
def test_rewindable_state_retrival(fresh_RE, start_state):
    RE = fresh_RE
    RE.rewindable = start_state

    def rewind_plan(start_value):
        ret = yield Msg('rewindable', None, None)
        assert ret is start_state
        cache_state = ret
        ret = yield Msg('rewindable', None, start_state)
        assert ret is start_state

        ret = yield Msg('rewindable', None, not start_state)
        assert ret is (not start_state)

        ret = yield Msg('rewindable', None, cache_state)
        assert ret is start_state

    RE(rewind_plan(start_state))
    assert RE.rewindable is start_state


@pytest.mark.parametrize('start_state,msg_seq', ((True, ['open_run',
                                                         'rewindable',
                                                         'rewindable',
                                                         'trigger',
                                                         'trigger',
                                                         'wait',
                                                         'create',
                                                         'read',
                                                         'read',
                                                         'save',
                                                         'rewindable',
                                                         'close_run']),
                                                 (False, ['open_run',
                                                          'rewindable',
                                                          'trigger',
                                                          'trigger',
                                                          'wait',
                                                          'create',
                                                          'read',
                                                          'read',
                                                          'save',
                                                          'close_run'])))
def test_nonrewindable_detector(fresh_RE, motor_det, start_state, msg_seq):
    class FakeSig:
        def get(self):
            return False

    motor, det = motor_det
    det.rewindable = FakeSig()

    RE = fresh_RE
    RE.rewindable = start_state
    m_col = MsgCollector()
    RE.msg_hook = m_col

    RE(bp.run_wrapper(bp.trigger_and_read([motor, det])))

    assert [m.command for m in m_col.msgs] == msg_seq


@pytest.mark.parametrize('start_state,msg_seq', ((True,
                                                  ['rewindable',
                                                   'rewindable',
                                                   'aardvark',
                                                   'rewindable']),
                                                 (False, ['rewindable',
                                                          'aardvark'])))
def test_nonrewindable_finalizer(fresh_RE, motor_det, start_state, msg_seq):
    class FakeSig:
        def get(self):
            return False

    motor, det = motor_det
    det.rewindable = FakeSig()

    RE = fresh_RE
    RE.rewindable = start_state
    m_col = MsgCollector()
    RE.msg_hook = m_col

    def evil_plan():
        assert RE.rewindable is False
        yield Msg('aardvark')
    with pytest.raises(KeyError):
        RE(bp.rewindable_wrapper(evil_plan(), False))

    assert RE.rewindable is start_state

    assert [m.command for m in m_col.msgs] == msg_seq


def test_halt_from_pause(fresh_RE):
    RE = fresh_RE
    except_hit = False
    m_coll = MsgCollector()
    RE.msg_hook = m_coll

    def pausing_plan():
        nonlocal except_hit
        for j in range(5):
            yield Msg('null')
        try:
            yield Msg('pause')
        except Exception:
            yield Msg('null')
            except_hit = True
            raise

    RE(pausing_plan())
    RE.halt()
    assert not except_hit
    assert [m.command for m in m_coll.msgs] == ['null'] * 5 + ['pause']


def test_halt_async(fresh_RE):
    RE = fresh_RE
    except_hit = False
    m_coll = MsgCollector()
    RE.msg_hook = m_coll

    def sleeping_plan():
        nonlocal except_hit
        try:
            yield Msg('sleep', None, 50)
        except Exception:
            yield Msg('null')
            except_hit = True
            raise

    RE.loop.call_later(.1, RE.halt)
    start = ttime.time()
    RE(sleeping_plan())
    stop = ttime.time()
    assert .09 < stop - start < .2
    assert not except_hit
    assert [m.command for m in m_coll.msgs] == ['sleep']


@pytest.mark.parametrize('cancel_func',
                         [lambda RE: RE.stop(), lambda RE: RE.abort(),
                          lambda RE: RE.request_pause(defer=False)])
def test_prompt_stop(fresh_RE, cancel_func):
    RE = fresh_RE
    except_hit = False
    m_coll = MsgCollector()
    RE.msg_hook = m_coll

    def sleeping_plan():
        nonlocal except_hit
        try:
            yield Msg('sleep', None, 50)
        except Exception:
            yield Msg('null')
            except_hit = True
            raise

    RE.loop.call_later(.1, partial(cancel_func, RE))
    start = ttime.time()
    RE(sleeping_plan())
    stop = ttime.time()
    if RE.state != 'idle':
        RE.abort()
    assert 0.09 < stop - start < .2
    assert except_hit
    assert [m.command for m in m_coll.msgs] == ['sleep', 'null']


@pytest.mark.parametrize('change_func', [lambda RE: RE.stop(),
                                         lambda RE: RE.abort(),
                                         lambda RE: RE.halt(),
                                         lambda RE: RE.request_pause(),
                                         lambda RE: RE.resume()])
def test_bad_from_idle_transitions(fresh_RE, change_func):
    with pytest.raises(TransitionError):
        change_func(fresh_RE)


def test_empty_cache_pause(fresh_RE):
    RE = fresh_RE
    RE.rewindable = False
    pln = [Msg('open_run'),
           Msg('create'),
           Msg('pause'),
           Msg('save'),
           Msg('close_run')]
    RE(pln)
    RE.resume()


def test_state_hook(fresh_RE):
    RE = fresh_RE
    states = []

    def log_state(new, old):
        states.append((new, old))

    RE.state_hook = log_state
    RE([Msg('open_run'), Msg('pause'), Msg('close_run')])
    RE.resume()
    expected = [('running', 'idle'),
                ('paused', 'running'),
                ('running', 'paused'),
                ('idle', 'running')]
    assert states == expected


def test_max_depth(fresh_RE):
    RE = fresh_RE

    RE.max_depth is None
    RE([])  # should not raise

    # assume test framework needs less than 100 stacks... haha
    RE.max_depth = 100
    RE([])  # should not raise

    RE.max_depth = 0
    with pytest.raises(RuntimeError):
        RE([])


def test_preprocessors(fresh_RE):
    RE = fresh_RE

    def custom_cleanup(plan):
        yield from plan
        yield Msg('null', 'cleanup')  # just a sentinel

    def my_sub(name, doc):
        pass

    def custom_subs(plan):
        yield from bp.subs_wrapper(plan, my_sub)

    RE.preprocessors = [custom_cleanup, custom_subs]
    actual = []
    RE.msg_hook = lambda msg: actual.append(msg)
    RE([Msg('null')])
    print(actual)
    expected = [Msg('subscribe', None, 'all', my_sub),
                Msg('null'),
                Msg('null', 'cleanup'),
                Msg('unsubscribe', None, token=0)]
    assert actual == expected


def test_pardon_failures(fresh_RE):
    RE = fresh_RE
    st = SimpleStatus()

    class Dummy:
        name = 'dummy'
        def set(self, val):
            return st

    dummy = Dummy()

    RE([Msg('set', dummy, 1)])
    st._finished(success=False)
    RE([Msg('null')])


def test_failures_kill_run(fresh_RE):
    # just to make sure that 'pardon_failures' does not block *real* failures
    RE = fresh_RE

    class Dummy:
        name = 'dummy'
        def set(self, val):
            st = SimpleStatus()
            st._finished(success=False)
            return st

    dummy = Dummy()

    with pytest.raises(FailedStatus):
        RE([Msg('set', dummy, 1)])
