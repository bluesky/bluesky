from nose.tools import assert_in, assert_equal, assert_raises
from bluesky.run_engine import RunEngine, Msg
from bluesky.examples import *
from bluesky.tests.utils import setup_test_run_engine
from nose.tools import raises


RE = setup_test_run_engine()


def exception_raiser(doc):
    raise Exception("Hey look it's an exception that better not kill the "
                    "scan!!")

def test_main_thread_callback_exceptions():

    RE(stepscan(motor, det), subs={'start': exception_raiser,
                                   'stop': exception_raiser,
                                   'event': exception_raiser,
                                   'descriptor': exception_raiser,
                                   'all': exception_raiser},
       beamline_id='testing', owner='tester')


def test_all():
    c = CallbackCounter()
    RE(stepscan(motor, det), subs={'all': c})
    assert_equal(c.value, 10 + 1 + 2)  # events, descriptor, start and stop

    c = CallbackCounter()
    token = RE.subscribe('all', c)
    RE(stepscan(motor, det))
    RE.unsubscribe(token)
    assert_equal(c.value, 10 + 1 + 2)


@raises(Exception)
def _raising_callbacks_helper(stream_name, callback):
    RE(stepscan(motor, det), subs={stream_name: callback})

    
def test_subscribe_msg():
    assert RE.state == 'idle'
    c = CallbackCounter()
    def counting_stepscan(motor, det):
        yield Msg('subscribe', None, 'start', c)
        yield from stepscan(motor, det)
    RE(counting_stepscan(motor, det))  # should advance c
    assert_equal(c.value, 1)
    RE(counting_stepscan(motor, det))  # should advance c
    assert_equal(c.value, 2)
    RE(stepscan(motor, det))  # should not
    assert_equal(c.value, 2)


def test_unknown_cb_raises():
    def f(doc):
        pass
    # Dispatches catches this case.
    assert_raises(KeyError, RE.subscribe, 'not a thing', f)
    # CallbackRegistry catches this case (different error).
    assert_raises(ValueError, RE._register_scan_callback, 'not a thing', f)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
