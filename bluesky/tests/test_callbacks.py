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

    RE(stepscan(det, motor), subs={'start': exception_raiser,
                                   'stop': exception_raiser,
                                   'event': exception_raiser,
                                   'descriptor': exception_raiser,
                                   'all': exception_raiser},
       beamline_id='testing', owner='tester')


def test_all():
    c = CallbackCounter()
    RE(stepscan(det, motor), subs={'all': c})
    assert_equal(c.value, 10 + 1 + 2)  # events, descriptor, start and stop

    c = CallbackCounter()
    token = RE.subscribe('all', c)
    RE(stepscan(det, motor))
    RE.unsubscribe(token)
    assert_equal(c.value, 10 + 1 + 2)


@raises(Exception)
def _raising_callbacks_helper(stream_name, callback):
    RE(stepscan(det, motor), subs={stream_name: callback})


def test_subscribe_msg():
    assert RE.state == 'idle'
    c = CallbackCounter()

    def counting_stepscan(det, motor):
        yield Msg('subscribe', None, 'start', c)
        yield from stepscan(det, motor)

    RE(counting_stepscan(det, motor))  # should advance c
    assert_equal(c.value, 1)
    RE(counting_stepscan(det, motor))  # should advance c
    assert_equal(c.value, 2)
    RE(stepscan(det, motor))  # should not
    assert_equal(c.value, 2)


def test_unknown_cb_raises():
    def f(doc):
        pass
    # Dispatches catches this case.
    assert_raises(KeyError, RE.subscribe, 'not a thing', f)
    # CallbackRegistry catches this case (different error).
    assert_raises(ValueError, RE._register_scan_callback, 'not a thing', f)


@contextlib.contextmanager
def _print_redirect():
    old_stdout = sys.stdout
    try:
        fout = tempfile.tempfile()
        sys.stdout = fout
        yield fout
    finally:
        sys.stdout = old_stdout

def test_table():

    with _print_redirect as fout:
        table = LiveTable(['det', 'pos'])
        ad_scan = AdaptiveAscan([det], 'det', motor, -15, 5, .01, 1, .05, True)
        RE(ad_scan, subscriptions={'all': [table]})

    fout.rewind()


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
