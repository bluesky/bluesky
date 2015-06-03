import nose
from nose.tools import assert_equal, assert_is, assert_is_none, assert_raises
from bluesky.examples import *
from bluesky import RunEngine, Mover, SynGauss, RunInterrupt, Msg


# global utility vars defined in setup()
RE = None
motor = None
motor1 = None
motor2 = None
motor3 = None
det = None


def setup():
    global RE, motor, motor1, motor2, motor3, det
    RE = RunEngine()
    motor = Mover('motor', ['pos'])
    motor1 = Mover('motor1', ['pos'])
    motor2 = Mover('motor2', ['pos'])
    motor3 = Mover('motor3', ['pos'])
    det = SynGauss('sg', motor, 'pos', center=0, Imax=1, sigma=1)

def test_msgs():
    m = Msg('set', motor, {'pos': 5})
    assert_equal(m.command, 'set')
    assert_is(m.obj, motor)
    assert_equal(m.args, ({'pos': 5},))
    assert_equal(m.kwargs, {})

    m = Msg('read', motor)
    assert_equal(m.command, 'read')
    assert_is(m.obj, motor)
    assert_equal(m.args, ())
    assert_equal(m.kwargs, {})

    m = Msg('create')
    assert_equal(m.command, 'create')
    assert_is_none(m.obj)
    assert_equal(m.args, ())
    assert_equal(m.kwargs, {})

    m = Msg('sleep', None, 5)
    assert_equal(m.command, 'sleep')
    assert_is_none(m.obj)
    assert_equal(m.args, (5,))
    assert_equal(m.kwargs, {})

def test_simple():
    RE.run(simple_scan(motor))

def test_conditional_break():
    RE.run(conditional_break(motor, det, threshold=0.2))

def test_sleepy():
    RE.run(sleepy(motor, det))

def test_wait_one():
    RE.run(wait_one(motor, det))

def test_wait_multiple():
    RE.run(wait_multiple([motor1, motor2], det))

#def test_conditional_hard_pause():
#    RE.run(conditional_hard_pause(motor, det))
#    assert_equal(RE.state, 'paused')
    #RE.resume()
    #assert_equal(RE.state, 'paused')
    #RE.abort()
    #assert_raises(Exception, f)

def test_simple_scan_saving():
    RE.run(simple_scan_saving(motor, det))

def print_event_time(doc):
    print('===== EVENT TIME:', doc['time'], '=====')

def test_calltime_subscription():
    RE.run(simple_scan_saving(motor, det), subscriptions={'event': print_event_time})

def test_stateful_subscription():
    token = RE.subscribe('event', print_event_time)
    RE.run(simple_scan_saving(motor, det), subscriptions={'event': print_event_time})
    RE.unsubscribe(token)

def test_stepscan():
    fig, ax = plt.subplots()
    my_plotter = live_scalar_plotter(ax, 'intensity', 'pos')
    RE.run(stepscan(motor, det), subscriptions={'event': my_plotter})
