from nose.tools import assert_in, assert_equal
from bluesky.run_engine import RunEngine
from bluesky.examples import *


RE = None


def setup():
    global RE
    RE = RunEngine()


def test_main_thread_callback_exceptions():
    def callbacker(doc):
        raise Exception("Hey look it's an exception that better not kill the "
                        "scan!!")

    RE(stepscan(motor, det), subs={'start': callbacker,
                                   'stop': callbacker,
                                   'event': callbacker,
                                   'descriptor': callbacker,
                                   'all': callbacker},
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


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
