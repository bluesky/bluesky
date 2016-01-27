import pytest
from bluesky.examples import (motor, simple_scan, det, sleepy, wait_one,
                              wait_multiple, motor1, motor2, conditional_pause,
                              loop, checkpoint_forever, simple_scan_saving,
                              stepscan, MockFlyer, fly_gen, panic_timer,
                              conditional_break, SynGauss
                              )
from bluesky.callbacks import LivePlot
from bluesky import RunEngine, Msg, PanicError, IllegalMessageSequence
from bluesky.tests.utils import setup_test_run_engine
import os
import signal
import asyncio
import time as ttime

RE = setup_test_run_engine()


def test_msgs():
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

def run(gen, *args, **kwargs):
    assert RE.state == 'idle'
    RE(gen(*args, **kwargs))
    assert RE.state == 'idle'


def test_simple():
    run(simple_scan, motor)


def test_conditional_break():
    run(conditional_break, det, motor, 0.2)


def test_sleepy():
    run(sleepy, det, motor)

def test_wati_one():
    run(wait_one, det, motor)


def test_wait_multiple():
    run(wait_multiple, det, [motor1, motor2])


def test_hard_pause():
    assert RE.state == 'idle'
    RE(conditional_pause(det, motor, False, True))
    assert RE.state == 'paused'
    RE.resume()
    assert RE.state == 'paused'
    RE.abort()
    assert RE.state == 'idle'


def test_deferred_pause():
    assert RE.state == 'idle'
    RE(conditional_pause(det, motor, True, True))
    assert RE.state == 'paused'
    RE.resume()
    assert RE.state == 'paused'
    RE.abort()
    assert RE.state == 'idle'


def test_hard_pause_no_checkpoint():
    assert RE.state == 'idle'
    RE(conditional_pause(det, motor, False, False))
    assert RE.state == 'idle'


def test_deferred_pause_no_checkpoint():
    assert RE.state == 'idle'
    RE(conditional_pause(det, motor, True, False))
    assert RE.state == 'idle'


def test_pause_from_outside():
    assert RE.state == 'idle'

    def local_pause():
        RE.request_pause()

    loop.call_later(1, local_pause)
    RE(checkpoint_forever())
    assert RE.state == 'paused'

    # Cue up a second pause requests in 2 seconds.
    loop.call_later(2, local_pause)
    RE.resume()
    assert RE.state == 'paused'

    RE.abort()
    assert RE.state == 'idle'


def test_panic_during_pause():
    assert RE.state == 'idle'
    RE(conditional_pause(det, motor, False, True))
    RE.panic()
    assert RE._panic
    with pytest.raises(PanicError):
        RE.resume()
    # If we panic while paused, we can un-panic and resume.
    RE.all_is_well()
    assert RE.state == 'paused'
    RE.abort()
    assert RE.state == 'idle'


def test_panic_timer():
    assert RE.state == 'idle'
    panic_timer(RE, 1)  # panic in 1 second
    with pytest.raises(PanicError):
        RE(checkpoint_forever())
    # If we panic while runnning, we cannot resume. The run is aborted and we
    # land in 'idle'
    assert RE.state == 'idle'
    assert RE._panic
    RE.all_is_well()
    assert RE.state == 'idle'


def test_simple_scan_saving():
    run(simple_scan_saving, det, motor)


def print_event_time(name, doc):
    print('===== EVENT TIME:', doc['time'], '=====')


def test_calltime_subscription():
    assert RE.state == 'idle'
    RE(simple_scan_saving(det, motor), subs={'event': print_event_time})
    assert RE.state == 'idle'


def test_stateful_subscription():
    assert RE.state == 'idle'
    token = RE.subscribe('event', print_event_time)
    RE(simple_scan_saving(det, motor))
    RE.unsubscribe(token)
    assert RE.state == 'idle'

def test_live_plotter():
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


def test_sample_md_dict_requirement():
    # We avoid a json ValidationError and make a user-friendly ValueError.
    with pytest.raises(ValueError):
        RE(simple_scan(motor), sample=1)
    RE(simple_scan(motor), sample={'number': 1})  # should not raise


def test_md_dict():
    _md({})

def test_md_historydict():
    try:
        import historydict
    except ImportError as ie:
        pytest.skip('Skipping test because historydict cannot be imported. '
                    'Error was {}'.foramt(ie))
    _md(historydict.HistoryDict(':memory:'))


def _md(md):
    RE = RunEngine(md)
    RE.ignore_callback_exceptions = False
    scan = simple_scan(motor)
    with pytest.raises(KeyError):
        RE(scan)  # missing owner, beamline_id
    scan = simple_scan(motor)
    with pytest.raises(KeyError):
        RE(scan, owner='dan')
    scan = simple_scan(motor)
    RE(scan, owner='dan', beamline_id='his desk',
       group='some group', config={})  # this should work
    scan = simple_scan(motor)
    with pytest.raises(KeyError):
        RE(scan)  # this should fail; none was persisted
    RE.md['owner'] = 'dan'
    RE.md['group'] = 'some group'
    RE.md['config'] = {}
    RE.md['beamline_id'] = 'his desk'
    scan = simple_scan(motor)
    RE(scan)  # this should work
    RE.md.clear()
    scan = simple_scan(motor)
    with pytest.raises(KeyError):
        RE(scan)
    # We can prime the md directly.
    RE.md['owner'] = 'dan'
    RE.md['group'] = 'some group'
    RE.md['config'] = {}
    RE.md['beamline_id'] = 'his desk'
    scan = simple_scan(motor)
    RE(scan)

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


def test_simple_fly():
    mm = MockFlyer(det, motor)
    RE(fly_gen(mm, -1, 1, 15))
    assert mm._future.done()


def test_list_of_msgs():
    # smoke tests checking that RunEngine accepts a plain list of Messages
    RE([Msg('open_run'), Msg('set', motor, 5), Msg('close_run')])


def test_suspend():
    ev = asyncio.Event()

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
    loop.call_later(.1, local_suspend)
    # wait a second and then resume
    loop.call_later(1, resume_cb)
    # grab the start time
    start = ttime.time()
    # run, this will not return until it is done
    RE(test_list, subs={'event': ev_cb})
    # check to make sure it took long enough
    assert out[0]['time'] - start > 1.1

    assert RE.state == 'idle'


def test_pause_resume():
    ev = asyncio.Event()

    def done():
        print("Done")
        ev.set()

    pid = os.getpid()

    def sim_kill():
        os.kill(pid, signal.SIGINT)

    scan = [Msg('checkpoint'), Msg('wait_for', [ev.wait(), ]), ]
    assert RE.state == 'idle'
    start = ttime.time()
    loop.call_later(1, sim_kill)
    loop.call_later(2, done)

    RE(scan)
    assert RE.state == 'paused'
    mid = ttime.time()
    RE.resume()
    assert RE.state == 'idle'
    stop = ttime.time()

    assert mid - start > 1
    assert stop - start > 2


def test_pause_abort():
    ev = asyncio.Event()

    def done():
        print("Done")
        ev.set()

    pid = os.getpid()

    def sim_kill():
        os.kill(pid, signal.SIGINT)

    scan = [Msg('checkpoint'), Msg('wait_for', [ev.wait(), ]), ]
    assert RE.state == 'idle'
    start = ttime.time()
    loop.call_later(1, sim_kill)
    loop.call_later(2, done)

    RE(scan)
    assert RE.state == 'paused'
    mid = ttime.time()
    RE.abort()
    assert RE.state == 'idle'
    stop = ttime.time()

    assert mid - start > 1
    assert stop - start < 2


def test_abort():
    errmsg = ("Aborting is not successful from the test suite yet.  The new "
              "plan, as can be seen in this function is to subprocess the "
              "abort via two sequential SIGINT's because if we do it in the "
              "pytest process it kills pytest.")
    pytest.xfail(errmsg)
    import subprocess
    subprocess.check_call(['python', 'abort.py'], cwd = '.')


def test_rogue_sigint():
    def bad_scan():
        yield Msg('open_run')
        yield Msg('checkpoint')
        raise KeyboardInterrupt

    RE(bad_scan())
    assert RE.state == 'paused'
    RE.abort()


def test_seqnum_nonrepeated():
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


def test_duplicate_keys():
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


def test_illegal_sequences():

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


def test_new_ev_desc():

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


def test_bad_checkpoint():
    bad_plan = [Msg('open_run'), Msg('checkpoint'), Msg('close_run'),
                Msg('pause')]
    RE(bad_plan)
    # Resuming will cause us to write the same close_run twice.
    with pytest.raises(IllegalMessageSequence):
        RE.resume()