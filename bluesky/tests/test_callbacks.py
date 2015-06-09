from nose.tools import assert_in, assert_equal
from bluesky.run_engine import RunEngine
from bluesky.examples import *
from bluesky.tests.utils import setup_run_engine
from nose.tools import raises


RE = setup_run_engine()


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
    RE(stepscan(motor, det), subs={stream_name: callback},
       beamline_id='testing', owner='tester')

    
def test_callback_execution():
    # make main thread exceptions end the scan
    RE.dispatcher.cb_registry.halt_on_exception = True
    cb = exception_raiser
    for stream in ['all', 'start', 'event', 'stop', 'descriptor']:
        yield _raising_callbacks_helper, stream, cb


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
