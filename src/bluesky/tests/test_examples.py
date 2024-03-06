import pytest
from bluesky.callbacks.mpl_plotting import LivePlot
from bluesky import (Msg, IllegalMessageSequence,
                     RunEngineInterrupted, FailedStatus)
import bluesky.plan_stubs as bps
import os
import signal
import time as ttime
import time
import threading
from functools import partial
from .utils import _fabricate_asycio_event

with pytest.warns(UserWarning):
    from bluesky.examples import (simple_scan, sleepy, wait_one,
                                  wait_multiple, conditional_pause,
                                  checkpoint_forever, simple_scan_saving,
                                  stepscan, fly_gen, conditional_break,
                                  )


def test_msgs(hw):
    m = Msg('set', hw.motor, {'motor': 5})
    assert m.command == 'set'
    assert m.obj is hw.motor
    assert m.args == ({'motor': 5},)
    assert m.kwargs == {}

    m = Msg('read', hw.motor)
    assert m.command == 'read'
    assert m.obj is hw.motor
    assert m.args == tuple()
    assert m.kwargs == {}

    m = Msg('create', name='primary')
    assert m.command == 'create'
    assert m.obj is None
    assert m.args == tuple()
    assert m.kwargs == {'name': 'primary'}

    m = Msg('sleep', None, 5)
    assert m.command == 'sleep'
    assert m.obj is None
    assert m.args == (5,)
    assert m.kwargs == {}


def run(RE, gen, *args, **kwargs):
    assert RE.state == 'idle'
    RE(gen(*args, **kwargs))
    assert RE.state == 'idle'


def test_simple(RE, hw):
    run(RE, simple_scan, hw.motor)


def test_conditional_break(RE, hw):
    run(RE, conditional_break, hw.det, hw.motor, 0.2)


def test_sleepy(RE, hw):
    run(RE, sleepy, hw.det, hw.motor)


def test_wait_one(RE, hw):
    run(RE, wait_one, hw.det, hw.motor)


def test_wait_multiple(RE, hw):
    run(RE, wait_multiple, hw.det, [hw.motor1, hw.motor2])


def test_hard_pause(RE, hw):
    assert RE.state == 'idle'
    with pytest.raises(RunEngineInterrupted):
        RE(conditional_pause(hw.det, hw.motor, False, True))
    assert RE.state == 'paused'
    with pytest.raises(RunEngineInterrupted):
        RE.resume()
    assert RE.state == 'paused'
    RE.abort()
    assert RE.state == 'idle'


def test_deferred_pause(RE):
    # deferred pause should be processed once and then clear
    # (future checkpoints should not trigger another pause)
    assert RE.state == 'idle'
    with pytest.raises(RunEngineInterrupted):
        RE([Msg('pause', defer=True), Msg('checkpoint'), Msg('checkpoint'),
            Msg('checkpoint')])
    assert RE.state == 'paused'
    RE.resume()
    assert RE.state == 'idle'


def test_deferred_pause1(RE):
    # deferred pause should never be processed, being superceded by a hard
    # pause
    with pytest.raises(RunEngineInterrupted):
        RE([Msg('pause', defer=True), Msg('pause', defer=False),
            Msg('checkpoint')])
    assert RE.state == 'paused'
    RE.resume()
    assert RE.state == 'idle'


def test_deferred_pause2(RE):
    with pytest.raises(RunEngineInterrupted):
        RE([Msg('pause', defer=True),
            Msg('checkpoint'),
            Msg('pause', defer=True),
            Msg('checkpoint')])
    assert RE.state == 'paused'
    with pytest.raises(RunEngineInterrupted):
        RE.resume()
    assert RE.state == 'paused'
    RE.resume()
    assert RE.state == 'idle'


def test_hard_pause_no_checkpoint(RE):
    assert RE.state == 'idle'
    with pytest.raises(RunEngineInterrupted):
        RE([Msg('clear_checkpoint'), Msg('pause', False)]),
    assert RE.state == 'idle'


def test_deferred_pause_no_checkpoint(RE):
    assert RE.state == 'idle'
    with pytest.raises(RunEngineInterrupted):
        RE([Msg('clear_checkpoint'), Msg('pause', True)])
    assert RE.state == 'idle'


def test_pause_from_outside(RE):
    assert RE.state == 'idle'

    def local_pause(delay):
        time.sleep(delay)
        RE.request_pause()

    th = threading.Thread(target=partial(local_pause, 1))
    th.start()
    with pytest.raises(RunEngineInterrupted):
        RE(checkpoint_forever())
    assert RE.state == 'paused'

    # Cue up a second pause requests in 2 seconds.
    th = threading.Thread(target=partial(local_pause, 2))
    th.start()
    with pytest.raises(RunEngineInterrupted):
        RE.resume()
    assert RE.state == 'paused'

    RE.abort()
    assert RE.state == 'idle'


def test_simple_scan_saving(RE, hw):
    run(RE, simple_scan_saving, hw.det, hw.motor)


def print_event_time(name, doc):
    print('===== EVENT TIME:', doc['time'], '=====')


def test_calltime_subscription(RE, hw):
    assert RE.state == 'idle'
    RE(simple_scan_saving(hw.det, hw.motor), {'event': print_event_time})
    assert RE.state == 'idle'


def test_stateful_subscription(RE, hw):
    assert RE.state == 'idle'
    token = RE.subscribe(print_event_time, 'event')
    RE(simple_scan_saving(hw.det, hw.motor))
    RE.unsubscribe(token)
    assert RE.state == 'idle'


def test_live_plotter(RE, hw):
    RE.ignore_callback_exceptions = False
    try:
        import matplotlib.pyplot as plt
        del plt
    except ImportError as ie:
        pytest.skip("Skipping live plot test because matplotlib is not installed."
                    "Error was: {}".format(ie))

    my_plotter = LivePlot('det', 'motor')
    assert RE.state == 'idle'
    RE(stepscan(hw.det, hw.motor), {'all': my_plotter})
    assert RE.state == 'idle'
    xlen = len(my_plotter.x_data)
    assert xlen > 0
    ylen = len(my_plotter.y_data)
    assert xlen == ylen
    RE.ignore_callback_exceptions = True


def test_sample_md_dict_requirement(RE, hw):
    # We avoid a json ValidationError and make a user-friendly ValueError.
    with pytest.raises(ValueError):
        RE(simple_scan(hw.motor), sample=1)
    RE(simple_scan(hw.motor), sample={'number': 1})  # should not raise
    RE(simple_scan(hw.motor), sample='label')  # should not raise


def test_md_dict(RE, hw):
    _md({}, RE, hw)


def test_md_historydict(RE, hw):
    try:
        import historydict
    except ImportError as ie:
        pytest.skip('Skipping test because historydict cannot be imported. '
                    'Error was {}'.format(ie))
    _md(historydict.HistoryDict(':memory:'), RE, hw)


def _md(md, RE, hw):
    RE.ignore_callback_exceptions = False

    # Check persistence.
    scan = simple_scan(hw.motor)
    RE(scan, project='sitting')
    # 'project' should not persist
    scan = simple_scan(hw.motor)
    RE(scan, {'start': [validate_dict_cb_opposite('project')]})
    # ...unless we add it to RE.md
    RE.md['project'] = 'sitting'
    scan = simple_scan(hw.motor)
    RE(scan, {'start': [validate_dict_cb('project', 'sitting')]})
    # new values to 'project' passed in the call override the value in md
    scan = simple_scan(hw.motor)
    RE(scan,
       {'start': [validate_dict_cb('project', 'standing')]},
       project='standing')
    # ...but they do not update the value in md
    assert RE.md['project'] == 'sitting'


def validate_dict_cb(key, val):
    def callback(name, doc):
        assert key in doc
        assert doc[key] == val
    return callback


def validate_dict_cb_opposite(key):
    def callback(name, doc):
        assert key not in doc
    return callback


def test_simple_fly(RE, hw):
    hw.flyer1.loop = RE.loop
    RE(fly_gen(hw.flyer1))


def test_list_of_msgs(RE, hw):
    # smoke tests checking that RunEngine accepts a plain list of Messages
    RE([Msg('open_run'), Msg('set', hw.motor, 5), Msg('close_run')])


def test_suspend(RE, hw):
    ev = _fabricate_asycio_event(RE.loop)

    test_list = [
        Msg('open_run'),
        Msg('checkpoint'),
        Msg('sleep', None, .2),
        Msg('set', hw.motor, 5),
        Msg('trigger', hw.det),
        Msg('declare_stream', None, hw.motor, hw.det, name='primary'),
        Msg('create', name='primary'),
        Msg('read', hw.motor),
        Msg('read', hw.det),
        Msg('save'),
        Msg('close_run'),
    ]
    assert RE.state == 'idle'

    def resume_cb():
        RE.loop.call_soon_threadsafe(ev.set)

    def local_suspend():
        RE.request_suspend(ev.wait)
        # wait a second and then resume
        threading.Timer(1, resume_cb).start()

    out = []

    def ev_cb(name, ev):
        out.append(ev)
    # trigger the suspend right after the check point
    threading.Timer(.1, local_suspend).start()
    # grab the start time
    start = ttime.time()
    # run, this will not return until it is done
    RE(test_list, {'event': ev_cb})
    # check to make sure it took long enough
    assert out[0]['time'] - start > 1.1

    assert RE.state == 'idle'


def test_pause_resume(RE):
    from bluesky.utils import ts_msg_hook
    RE.msg_hook = ts_msg_hook
    ev = _fabricate_asycio_event(RE.loop)

    def done():
        print("Done")
        RE.loop.call_soon_threadsafe(ev.set)

    pid = os.getpid()

    def sim_kill():
        os.kill(pid, signal.SIGINT)

    scan = [Msg('checkpoint'), Msg('wait_for', None, [ev.wait, ]), ]
    assert RE.state == 'idle'
    start = ttime.time()
    threading.Timer(1, sim_kill).start()
    threading.Timer(1.5, sim_kill).start()
    threading.Timer(2, done).start()

    with pytest.raises(RunEngineInterrupted):
        RE(scan)
    assert RE.state == 'paused'
    mid = ttime.time()
    RE.resume()
    assert RE.state == 'idle'
    stop = ttime.time()

    time.sleep(3)
    assert mid - start > 1
    assert stop - start > 2


def test_pause_abort(RE):
    ev = _fabricate_asycio_event(RE.loop)

    def done():
        print("Done")
        RE.loop.call_soon_threadsafe(ev.set)

    pid = os.getpid()

    def sim_kill():
        os.kill(pid, signal.SIGINT)

    scan = [Msg('checkpoint'), Msg('wait_for', None, [ev.wait, ]), ]
    assert RE.state == 'idle'
    start = ttime.time()
    threading.Timer(.1, sim_kill).start()
    threading.Timer(.2, sim_kill).start()
    threading.Timer(1, done).start()

    with pytest.raises(RunEngineInterrupted):
        RE(scan)
    assert RE.state == 'paused'
    mid = ttime.time()
    RE.abort()
    assert RE.state == 'idle'
    stop = ttime.time()

    assert mid - start > .1
    assert stop - start < 1


def test_abort(RE):
    ev = _fabricate_asycio_event(RE.loop)

    def done():
        print("Done")
        RE.loop.call_soon_threadsafe(ev.set)

    pid = os.getpid()

    def sim_kill():
        os.kill(pid, signal.SIGINT)

    scan = [Msg('checkpoint'), Msg('wait_for', None, [ev.wait, ]), ]
    assert RE.state == 'idle'
    start = ttime.time()
    threading.Timer(.1, sim_kill).start()
    threading.Timer(.2, sim_kill).start()
    threading.Timer(.4, done).start()
    with pytest.raises(RunEngineInterrupted):
        RE(scan)
    stop = ttime.time()

    assert RE.state == 'paused'
    assert stop - start < .4
    RE.abort()
    assert RE.state == 'idle'


def test_rogue_sigint(RE):

    def bad_scan():
        yield Msg('open_run')
        yield Msg('checkpoint')
        raise KeyboardInterrupt()

    with pytest.raises(RunEngineInterrupted):
        RE(bad_scan())
    assert RE.state == 'idle'


def test_seqnum_nonrepeated(RE, hw):

    def gen():
        yield Msg('open_run')
        yield Msg('declare_stream', None, hw.motor, name='primary')
        yield Msg('create', name='primary')
        yield Msg('set', hw.motor, 1)
        yield Msg('read', hw.motor)
        yield Msg('save')
        yield Msg('checkpoint')
        yield Msg('create', name='primary')
        yield Msg('set', hw.motor, 2)
        yield Msg('read', hw.motor)
        yield Msg('save')
        yield Msg('pause')
        yield Msg('create', name='primary')
        yield Msg('set', hw.motor, 3)
        yield Msg('read', hw.motor)
        yield Msg('save')
        yield Msg('close_run')

    seq_nums = []

    def f(name, doc):
        seq_nums.append(doc['seq_num'])

    RE.verbose = True

    with pytest.raises(RunEngineInterrupted):
        RE(gen(), {'event': f})

    print("RESUMING!!!!")
    RE.resume()
    assert seq_nums == [1, 2, 2, 3]


def test_duplicate_keys(RE, hw):
    # two detectors, same data keys

    def gen():
        yield (Msg('open_run'))
        yield Msg('declare_stream', None, hw.det, hw.identical_det, name='primary')
        yield (Msg('create', name='primary'))
        yield (Msg('trigger', hw.det))
        yield (Msg('trigger', hw.identical_det))
        yield (Msg('read', hw.det))
        yield (Msg('read', hw.identical_det))
        yield (Msg('save'))

    with pytest.raises(ValueError):
        RE(gen())


def test_illegal_sequences(RE, hw):
    def gen1():
        # two 'create' msgs in a row
        yield (Msg('open_run'))
        yield Msg('declare_stream', None, name='primary')
        yield (Msg('create', name='primary'))
        yield (Msg('create', name='primary'))
        yield (Msg('close_run'))

    with pytest.raises(IllegalMessageSequence):
        RE(gen1())

    def gen2():
        # two 'save' msgs in a row
        yield (Msg('open_run'))
        yield Msg('declare_stream', None, name='primary')
        yield (Msg('create', name='primary'))
        yield (Msg('save'))
        yield (Msg('save'))
        yield (Msg('close_run'))

    with pytest.raises(IllegalMessageSequence):
        RE(gen2())

    def gen3():
        # 'configure' after 'create', before 'save'
        yield (Msg('open_run'))
        yield Msg('declare_stream', None, name='primary')
        yield (Msg('create', name='primary'))
        yield (Msg('configure', hw.motor, {}))

    with pytest.raises(IllegalMessageSequence):
        RE(gen3())

    def gen4():
        # two 'drop' msgs in a row
        yield (Msg('open_run'))
        yield Msg('declare_stream', None, name='primary')
        yield (Msg('create', name='primary'))
        yield (Msg('drop'))
        yield (Msg('drop'))
        yield (Msg('close_run'))

    with pytest.raises(IllegalMessageSequence):
        RE(gen4())


def test_new_ev_desc(RE, hw):
    descs = []

    def collect_descs(name, doc):
        descs.append(doc)

    def gen1():
        # configure between two events -> two descs
        yield (Msg('open_run'))
        yield Msg('declare_stream', None, hw.motor, name='primary')
        yield (Msg('create', name='primary'))
        yield (Msg('read', hw.motor))
        yield (Msg('save'))
        yield (Msg('configure', hw.motor, {}))
        yield (Msg('create', name='primary'))
        yield (Msg('read', hw.motor))
        yield (Msg('save'))
        yield (Msg('close_run'))

    descs.clear()
    RE(gen1(), {'descriptor': collect_descs})
    assert len(descs) == 2

    def gen2():
        # configure between two events and explicitly before any events
        # -> two descs
        yield (Msg('open_run'))
        yield (Msg('configure', hw.motor, {}))
        yield Msg('declare_stream', None, hw.motor, name='primary')
        yield (Msg('create', name='primary'))
        yield (Msg('read', hw.motor))
        yield (Msg('save'))
        yield (Msg('configure', hw.motor, {}))
        yield (Msg('create', name='primary'))
        yield (Msg('read', hw.motor))
        yield (Msg('save'))
        yield (Msg('close_run'))

    descs.clear()
    RE(gen2(), {'descriptor': collect_descs})
    assert len(descs) == 2

    def gen3():
        # configure once before any events -> one desc
        yield (Msg('open_run'))
        yield (Msg('configure', hw.motor, {}))
        yield Msg('declare_stream', None, hw.motor, name='primary')
        yield (Msg('create', name='primary'))
        yield (Msg('read', hw.motor))
        yield (Msg('save'))
        yield (Msg('create', name='primary'))
        yield (Msg('read', hw.motor))
        yield (Msg('save'))
        yield (Msg('close_run'))

    descs.clear()
    RE(gen3(), {'descriptor': collect_descs})
    assert len(descs) == 1


def test_clear_checkpoint(RE):
    bad_plan = [Msg('checkpoint'),
                Msg('clear_checkpoint'),
                Msg('pause'),
                'lies']
    good_plan = [Msg('pause')]
    fine_plan = [Msg('clear_checkpoint')]

    with pytest.raises(RunEngineInterrupted):
        RE(good_plan)
    assert RE.state == 'paused'
    RE.stop()

    RE(fine_plan)
    assert RE.state == 'idle'

    # this should raise an attribute error if the last entry in the plan
    # is passed to the run engine (but that should not happen because it should
    # die when it hits the 'pause' and has no checkpoint)
    with pytest.raises(RunEngineInterrupted):
        RE(bad_plan)
    assert RE.state == 'idle'


def test_interruption_exception(RE):
    with pytest.raises(RunEngineInterrupted):
        RE([Msg('checkpoint'), Msg('pause')])
    RE.stop()


def test_failed_status_object(RE):
    try:
        from ophyd import StatusBase
    except ImportError:
        pytest.xfail('No ophyd')

    class failer:
        def set(self, inp):
            st = StatusBase()
            threading.Timer(1, st._finished,
                            kwargs=dict(success=False)).start()
            return st

        def trigger(self):
            st = StatusBase()
            threading.Timer(1, st._finished,
                            kwargs=dict(success=False)).start()
            return st

        def stop(self, *, success=False):
            pass

    ff = failer()
    with pytest.raises(FailedStatus):
        RE([Msg('set', ff, None, group='a'),
            Msg('wait', None, group='a')])

    with pytest.raises(FailedStatus):
        RE([Msg('trigger', ff, group='a'),
            Msg('wait', None, group='a')])


def test_rewindable_by_default(RE):
    class Sentinel(Exception):
        pass

    def plan():
        yield Msg('pause')
        raise Sentinel

    with pytest.raises(RunEngineInterrupted):
        RE(plan())
    assert RE.state == 'paused'
    with pytest.raises(Sentinel):
        RE.resume()

    def plan():
        yield Msg('clear_checkpoint')
        yield Msg('pause')

    with pytest.raises(RunEngineInterrupted):
        RE(plan())  # cannot pause
    assert RE.state == 'idle'


def test_sync_trigger_delay(hw):
    def _time_test(f, t, *args, **kwargs):
        start = ttime.time()
        st = f(*args, **kwargs)
        while not st.done:
            ttime.sleep(0.01)
        end = ttime.time()
        assert (end - start) > t

    hw.motor.delay = .5
    hw.det.exposure_time = .5
    assert hw.det.exposure_time == .5

    _time_test(hw.motor.set, .5, 1)
    _time_test(hw.det.trigger, .5)


def test_async_trigger_delay(RE, hw):
    hw.motor.loop = RE.loop
    hw.det.loop = RE.loop

    def _time_test(f, t, *args, **kwargs):
        start = ttime.time()
        RE(f(*args, **kwargs))
        end = ttime.time()
        assert (end - start) > t

    hw.motor.delay = .5
    hw.det.exposure_time = .5

    _time_test(bps.trigger, .5, hw.det, wait=True)
    _time_test(bps.abs_set, .5, hw.motor, 1, wait=True)
