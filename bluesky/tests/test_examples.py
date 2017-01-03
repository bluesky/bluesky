import pytest
from bluesky.examples import (motor, simple_scan, det, sleepy, wait_one,
                              wait_multiple, motor1, motor2, conditional_pause,
                              checkpoint_forever, simple_scan_saving,
                              stepscan, MockFlyer, fly_gen,
                              conditional_break, SynGauss, flyer1,
                              ReaderWithFileStore, ReaderWithFSHandler
                              )
from bluesky.callbacks import LivePlot
from bluesky import (Msg, IllegalMessageSequence,
                     RunEngineInterrupted, FailedStatus)
import bluesky.plans as bp
import os
import signal
import asyncio
import time as ttime
import numpy as np
from numpy.testing import assert_array_equal


def test_msgs(fresh_RE):
    m = Msg('set', motor, {'motor': 5})
    assert m.command == 'set'
    assert m.obj is motor
    assert m.args == ({'motor': 5},)
    assert m.kwargs == {}

    m = Msg('read', motor)
    assert m.command == 'read'
    assert m.obj is motor
    assert m.args == tuple()
    assert m.kwargs == {}

    m = Msg('create')
    assert m.command == 'create'
    assert m.obj is None
    assert m.args == tuple()
    assert m.kwargs == {}

    m = Msg('sleep', None, 5)
    assert m.command == 'sleep'
    assert m.obj is None
    assert m.args == (5,)
    assert m.kwargs == {}


def run(RE, gen, *args, **kwargs):
    assert RE.state == 'idle'
    RE(gen(*args, **kwargs))
    assert RE.state == 'idle'


def test_simple(fresh_RE):
    run(fresh_RE, simple_scan, motor)


def test_conditional_break(fresh_RE):
    run(fresh_RE, conditional_break, det, motor, 0.2)


def test_sleepy(fresh_RE):
    run(fresh_RE, sleepy, det, motor)


def test_wait_one(fresh_RE):
    run(fresh_RE, wait_one, det, motor)


def test_wait_multiple(fresh_RE):
    run(fresh_RE, wait_multiple, det, [motor1, motor2])


def test_hard_pause(fresh_RE):
    RE = fresh_RE
    assert RE.state == 'idle'
    RE(conditional_pause(det, motor, False, True))
    assert RE.state == 'paused'
    RE.resume()
    assert RE.state == 'paused'
    RE.abort()
    assert RE.state == 'idle'


def test_deferred_pause(fresh_RE):
    # deferred pause should be processed once and then clear
    # (future checkpoints should not trigger another pause)
    RE = fresh_RE
    assert RE.state == 'idle'
    RE([Msg('pause', defer=True), Msg('checkpoint'), Msg('checkpoint'),
        Msg('checkpoint')])
    assert RE.state == 'paused'
    RE.resume()
    assert RE.state == 'idle'


def test_deferred_pause1(fresh_RE):
    # deferred pause should never be processed, being superceded by a hard
    # pause
    RE = fresh_RE
    RE([Msg('pause', defer=True), Msg('pause', defer=False),
        Msg('checkpoint')])
    assert RE.state == 'paused'
    RE.resume()
    assert RE.state == 'idle'


def test_deferred_pause2(fresh_RE):
    RE = fresh_RE
    RE([Msg('pause', defer=True), Msg('checkpoint'), Msg('pause', defer=True),
        Msg('checkpoint')])
    assert RE.state == 'paused'
    RE.resume()
    assert RE.state == 'paused'
    RE.resume()
    assert RE.state == 'idle'


def test_hard_pause_no_checkpoint(fresh_RE):
    RE = fresh_RE
    assert RE.state == 'idle'
    RE([Msg('clear_checkpoint'), Msg('pause', False)]),
    assert RE.state == 'idle'


def test_deferred_pause_no_checkpoint(fresh_RE):
    RE = fresh_RE
    assert RE.state == 'idle'
    RE([Msg('clear_checkpoint'), Msg('pause', True)])
    assert RE.state == 'idle'


def test_pause_from_outside(fresh_RE):
    RE = fresh_RE
    assert RE.state == 'idle'

    def local_pause():
        RE.request_pause()

    RE.loop.call_later(1, local_pause)
    RE(checkpoint_forever())
    assert RE.state == 'paused'

    # Cue up a second pause requests in 2 seconds.
    RE.loop.call_later(2, local_pause)
    RE.resume()
    assert RE.state == 'paused'

    RE.abort()
    assert RE.state == 'idle'


def test_simple_scan_saving(fresh_RE):
    run(fresh_RE, simple_scan_saving, det, motor)


def print_event_time(name, doc):
    print('===== EVENT TIME:', doc['time'], '=====')


def test_calltime_subscription(fresh_RE):
    RE = fresh_RE
    assert RE.state == 'idle'
    RE(simple_scan_saving(det, motor), subs={'event': print_event_time})
    assert RE.state == 'idle'


def test_stateful_subscription(fresh_RE):
    RE = fresh_RE
    assert RE.state == 'idle'
    token = RE.subscribe('event', print_event_time)
    RE(simple_scan_saving(det, motor))
    RE.unsubscribe(token)
    assert RE.state == 'idle'


def test_live_plotter(fresh_RE):
    RE = fresh_RE
    RE.ignore_callback_exceptions = False
    try:
        import matplotlib.pyplot as plt
        del plt
    except ImportError as ie:
        pytest.skip("Skipping live plot test because matplotlib is not installed."
                    "Error was: {}".format(ie))

    my_plotter = LivePlot('det', 'motor')
    assert RE.state == 'idle'
    RE(stepscan(det, motor), subs={'all': my_plotter})
    assert RE.state == 'idle'
    xlen = len(my_plotter.x_data)
    assert xlen > 0
    ylen = len(my_plotter.y_data)
    assert xlen == ylen
    RE.ignore_callback_exceptions = True


def test_sample_md_dict_requirement(fresh_RE):
    RE = fresh_RE
    # We avoid a json ValidationError and make a user-friendly ValueError.
    with pytest.raises(ValueError):
        RE(simple_scan(motor), sample=1)
    RE(simple_scan(motor), sample={'number': 1})  # should not raise
    RE(simple_scan(motor), sample='label')  # should not raise


def test_md_dict(fresh_RE):
    _md({}, fresh_RE)


def test_md_historydict(fresh_RE):
    try:
        import historydict
    except ImportError as ie:
        pytest.skip('Skipping test because historydict cannot be imported. '
                    'Error was {}'.format(ie))
    _md(historydict.HistoryDict(':memory:'), fresh_RE)


def _md(md, RE):
    RE.ignore_callback_exceptions = False

    # Check persistence.
    scan = simple_scan(motor)
    RE(scan, project='sitting')
    # 'project' should not persist
    scan = simple_scan(motor)
    RE(scan, subs={'start': [validate_dict_cb_opposite('project')]})
    # ...unless we add it to RE.md
    RE.md['project'] = 'sitting'
    scan = simple_scan(motor)
    RE(scan, subs={'start': [validate_dict_cb('project', 'sitting')]})
    # new values to 'project' passed in the call override the value in md
    scan = simple_scan(motor)
    RE(scan, project='standing',
       subs={'start': [validate_dict_cb('project', 'standing')]})
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


def test_simple_fly(fresh_RE):
    RE = fresh_RE
    mm = MockFlyer('wheeeeee', det, motor, -1, 1, 15, RE.loop)
    RE(fly_gen(mm))
    assert mm._future.done()


def test_list_of_msgs(fresh_RE):
    RE = fresh_RE
    # smoke tests checking that RunEngine accepts a plain list of Messages
    RE([Msg('open_run'), Msg('set', motor, 5), Msg('close_run')])


def test_suspend(fresh_RE):
    RE = fresh_RE
    ev = asyncio.Event(loop=RE.loop)

    test_list = [
        Msg('open_run'),
        Msg('checkpoint'),
        Msg('sleep', None, .2),
        Msg('set', motor, 5),
        Msg('trigger', det),
        Msg('create'),
        Msg('read', motor),
        Msg('read', det),
        Msg('save'),
        Msg('close_run'),
    ]
    assert RE.state == 'idle'

    def local_suspend():
        RE.request_suspend(ev.wait())

    def resume_cb():
        ev.set()

    out = []

    def ev_cb(name, ev):
        out.append(ev)
    # trigger the suspend right after the check point
    RE.loop.call_later(.1, local_suspend)
    # wait a second and then resume
    RE.loop.call_later(1, resume_cb)
    # grab the start time
    start = ttime.time()
    # run, this will not return until it is done
    RE(test_list, subs={'event': ev_cb})
    # check to make sure it took long enough
    assert out[0]['time'] - start > 1.1

    assert RE.state == 'idle'


def test_pause_resume(fresh_RE):
    RE = fresh_RE
    ev = asyncio.Event(loop=RE.loop)

    def done():
        print("Done")
        ev.set()

    pid = os.getpid()

    def sim_kill():
        os.kill(pid, signal.SIGINT)

    scan = [Msg('checkpoint'), Msg('wait_for', None, [ev.wait(), ]), ]
    assert RE.state == 'idle'
    start = ttime.time()
    RE.loop.call_later(1, sim_kill)
    RE.loop.call_later(1.1, sim_kill)
    RE.loop.call_later(2, done)

    RE(scan)
    assert RE.state == 'paused'
    mid = ttime.time()
    RE.resume()
    assert RE.state == 'idle'
    stop = ttime.time()

    assert mid - start > 1
    assert stop - start > 2


def test_pause_abort(fresh_RE):
    RE = fresh_RE
    ev = asyncio.Event(loop=RE.loop)

    def done():
        print("Done")
        ev.set()

    pid = os.getpid()

    def sim_kill():
        os.kill(pid, signal.SIGINT)

    scan = [Msg('checkpoint'), Msg('wait_for', None, [ev.wait(), ]), ]
    assert RE.state == 'idle'
    start = ttime.time()
    RE.loop.call_later(.1, sim_kill)
    RE.loop.call_later(.2, sim_kill)
    RE.loop.call_later(1, done)

    RE(scan)
    assert RE.state == 'paused'
    mid = ttime.time()
    RE.abort()
    assert RE.state == 'idle'
    stop = ttime.time()

    RE.loop.run_until_complete(ev.wait())
    assert mid - start > .1
    assert stop - start < 1


def test_abort(fresh_RE):
    RE = fresh_RE
    ev = asyncio.Event(loop=RE.loop)

    def done():
        print("Done")
        ev.set()

    pid = os.getpid()

    def sim_kill():
        os.kill(pid, signal.SIGINT)

    scan = [Msg('checkpoint'), Msg('wait_for', None, [ev.wait(), ]), ]
    assert RE.state == 'idle'
    start = ttime.time()
    RE.loop.call_later(.1, sim_kill)
    RE.loop.call_later(.2, sim_kill)
    RE.loop.call_later(.3, done)
    RE(scan)
    stop = ttime.time()

    RE.loop.run_until_complete(ev.wait())
    assert RE.state == 'paused'
    assert stop - start < .3
    RE.abort()
    assert RE.state == 'idle'


def test_rogue_sigint(fresh_RE):
    RE = fresh_RE

    def bad_scan():
        yield Msg('open_run')
        yield Msg('checkpoint')
        raise KeyboardInterrupt()

    RE(bad_scan())
    assert RE.state == 'idle'


def test_seqnum_nonrepeated(fresh_RE):
    RE = fresh_RE

    def gen():
        yield Msg('open_run')
        yield Msg('create')
        yield Msg('set', motor, 1)
        yield Msg('read', motor)
        yield Msg('save')
        yield Msg('checkpoint')
        yield Msg('create')
        yield Msg('set', motor, 2)
        yield Msg('read', motor)
        yield Msg('save')
        yield Msg('pause')
        yield Msg('create')
        yield Msg('set', motor, 3)
        yield Msg('read', motor)
        yield Msg('save')
        yield Msg('close_run')

    seq_nums = []

    def f(name, doc):
        seq_nums.append(doc['seq_num'])

    RE.verbose = True
    RE(gen(), {'event': f})
    print("RESUMING!!!!")
    RE.resume()
    assert seq_nums == [1, 2, 2, 3]


def test_duplicate_keys(fresh_RE):
    RE = fresh_RE
    # two detectors, same data keys
    det1 = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)
    det2 = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)

    def gen():
        yield(Msg('open_run'))
        yield(Msg('create'))
        yield(Msg('trigger', det1))
        yield(Msg('trigger', det2))
        yield(Msg('read', det1))
        yield(Msg('read', det2))
        yield(Msg('save'))

    with pytest.raises(ValueError):
        RE(gen())


def test_illegal_sequences(fresh_RE):
    RE = fresh_RE

    def gen1():
        # two 'create' msgs in a row
        yield(Msg('open_run'))
        yield(Msg('create'))
        yield(Msg('create'))
        yield(Msg('close_run'))

    with pytest.raises(IllegalMessageSequence):
        RE(gen1())

    def gen2():
        # two 'save' msgs in a row
        yield(Msg('open_run'))
        yield(Msg('create'))
        yield(Msg('save'))
        yield(Msg('save'))
        yield(Msg('close_run'))

    with pytest.raises(IllegalMessageSequence):
        RE(gen1())

    def gen3():
        # 'configure' after 'create', before 'save'
        yield(Msg('open_run'))
        yield(Msg('create'))
        yield(Msg('configure', motor, {}))

    with pytest.raises(IllegalMessageSequence):
        RE(gen3())


def test_new_ev_desc(fresh_RE):
    RE = fresh_RE
    descs = []

    def collect_descs(name, doc):
        descs.append(doc)

    def gen1():
        # configure between two events -> two descs
        yield(Msg('open_run'))
        yield(Msg('create'))
        yield(Msg('read', motor))
        yield(Msg('save'))
        yield(Msg('configure', motor, {}))
        yield(Msg('create'))
        yield(Msg('read', motor))
        yield(Msg('save'))
        yield(Msg('close_run'))

    descs.clear()
    RE(gen1(), {'descriptor': collect_descs})
    assert len(descs) == 2

    def gen2():
        # configure between two events and explicitly before any events
        # -> two descs
        yield(Msg('open_run'))
        yield(Msg('configure', motor, {}))
        yield(Msg('create'))
        yield(Msg('read', motor))
        yield(Msg('save'))
        yield(Msg('configure', motor, {}))
        yield(Msg('create'))
        yield(Msg('read', motor))
        yield(Msg('save'))
        yield(Msg('close_run'))

    descs.clear()
    RE(gen2(), {'descriptor': collect_descs})
    assert len(descs) == 2

    def gen3():
        # configure once before any events -> one desc
        yield(Msg('open_run'))
        yield(Msg('configure', motor, {}))
        yield(Msg('create'))
        yield(Msg('read', motor))
        yield(Msg('save'))
        yield(Msg('create'))
        yield(Msg('read', motor))
        yield(Msg('save'))
        yield(Msg('close_run'))

    descs.clear()
    RE(gen3(), {'descriptor': collect_descs})
    assert len(descs) == 1


def test_clear_checkpoint(fresh_RE):
    RE = fresh_RE

    bad_plan = [Msg('checkpoint'),
                Msg('clear_checkpoint'),
                Msg('pause'),
                'lies']
    good_plan = [Msg('pause')]
    fine_plan = [Msg('clear_checkpoint')]

    RE(good_plan)
    assert RE.state == 'paused'
    RE.stop()

    RE(fine_plan)
    assert RE.state == 'idle'

    # this should raise an attribute error if the last entry in the plan
    # is passed to the run engine
    RE(bad_plan)
    assert RE.state == 'idle'


def test_interruption_exception(fresh_RE):
    RE = fresh_RE

    with pytest.raises(RunEngineInterrupted):
        RE([Msg('checkpoint'), Msg('pause')], raise_if_interrupted=True)
    RE.stop()


def test_failed_status_object(fresh_RE):
    try:
        from ophyd import StatusBase
    except ImportError:
        pytest.xfail('No ophyd')

    class failer:
        def set(self, inp):
            st = StatusBase()
            fresh_RE.loop.call_later(1, lambda: st._finished(success=False))
            return st

        def trigger(self):
            st = StatusBase()
            fresh_RE.loop.call_later(1, lambda: st._finished(success=False))
            return st

        def stop(self, *, success=False):
            pass

    ff = failer()
    with pytest.raises(FailedStatus):
        fresh_RE([Msg('set', ff, None, group='a'),
                  Msg('wait', None, group='a')])

    with pytest.raises(FailedStatus):
        fresh_RE([Msg('trigger', ff, group='a'),
                  Msg('wait', None, group='a')])


def test_rewindable_by_default(fresh_RE):
    RE = fresh_RE

    class Sentinel(Exception):
        pass

    def plan():
        yield Msg('pause')
        raise Sentinel

    RE(plan())
    assert RE.state == 'paused'
    with pytest.raises(Sentinel):
        RE.resume()

    def plan():
        yield Msg('clear_checkpoint')
        yield Msg('pause')

    RE(plan())  # cannot pause
    assert RE.state == 'idle'


def test_pickling_examples():
    try:
        import dill
    except ImportError:
        raise pytest.skip('requires dill')
    dill.loads(dill.dumps(det))
    dill.loads(dill.dumps(motor))
    dill.loads(dill.dumps(flyer1))


def test_sync_trigger_delay(motor_det):
    motor, det = motor_det

    def _time_test(f, t, *args, **kwargs):
        start = ttime.time()
        f(*args, **kwargs)
        end = ttime.time()
        assert (end - start) > t

    motor._fake_sleep = .5
    det.exposure_time = .5

    _time_test(motor.set, .5, 1)
    _time_test(det.trigger, .5)


def test_async_trigger_delay(motor_det, fresh_RE):
    RE = fresh_RE

    motor, det = motor_det
    motor.loop = RE.loop
    det.loop = RE.loop

    def _time_test(f, t, *args, **kwargs):
        start = ttime.time()
        RE(f(*args, **kwargs))
        end = ttime.time()
        assert (end - start) > t

    motor._fake_sleep = .5
    det.exposure_time = .5

    _time_test(bp.trigger, .5, det, wait=True)
    _time_test(bp.abs_set, .5, motor, 1, wait=True)


def test_fs_reader(db):
    fs = db.fs
    fs.register_handler('RWFS_NPY', ReaderWithFSHandler)
    det = ReaderWithFileStore('det',
                              {'img': lambda: np.array(np.ones((10, 10)))},
                              fs=fs)
    det.stage()
    det.trigger()
    reading = det.read().copy()
    det.unstage()
    datum_id = reading['img']['value']
    arr = fs.retrieve(datum_id)
    assert_array_equal(np.ones((10, 10)), arr)
