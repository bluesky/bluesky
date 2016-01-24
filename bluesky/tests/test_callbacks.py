from nose.tools import assert_equal, assert_raises, assert_true
from nose import SkipTest
from bluesky.run_engine import Msg
from bluesky.examples import (motor, det, stepscan)
from bluesky.plans import AdaptiveAbsScanPlan, AbsScanPlan
from bluesky.callbacks import (CallbackCounter, LiveTable)
from bluesky.standard_config import mesh
from bluesky.tests.utils import setup_test_run_engine
from nose.tools import raises
import contextlib
import sys
import tempfile
import pytest

RE = setup_test_run_engine()


def exception_raiser(name, doc):
    raise Exception("it's an exception that better not kill the scan!!")


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
    assert c.value == 10 + 1 + 2  # events, descriptor, start and stop

    c = CallbackCounter()
    token = RE.subscribe('all', c)
    RE(stepscan(det, motor))
    RE.unsubscribe(token)
    assert c.value == 10 + 1 + 2


@pytest.raises(Exception)
def _raising_callbacks_helper(stream_name, callback):
    RE(stepscan(det, motor), subs={stream_name: callback})


def test_raising_ignored_or_not():
    assert RE.ignore_callback_exceptions
    def cb(name, doc):
        raise Exception
    RE(stepscan(det, motor), subs=cb)
    RE.ignore_callback_exceptions = False
    _raising_callbacks_helper('all', cb)


def test_subs_input():
    def cb1(name, doc):
        pass

    def cb2(name, doc):
        pass

    def cb3(name, doc):
        pass

    def cb_fact4(scan):
        def cb4(name, doc):
            pass
        return cb4

    def cb_fact5(scan):
        def cb5(name, doc):
            pass
        return cb5

    # Test input normalization on OO plans
    obj_ascan = AbsScanPlan([det], motor, 1, 5, 4)
    obj_ascan.subs = cb1
    assert obj_ascan.subs == {'all': [cb1]}
    obj_ascan.subs.update({'start': [cb2]})
    assert obj_ascan.subs == {'all': [cb1], 'start': [cb2]}
    obj_ascan.subs = [cb2, cb3]
    assert obj_ascan.subs == {'all': [cb2, cb3]}

    # Test input normalization on simple scans
    assert mesh.subs == mesh.default_subs
    mesh.subs.update({'start': [cb2]})
    expected = dict(mesh.default_subs)
    expected.update({'start': [cb2]})
    assert mesh.subs == expected
    mesh.subs = cb2
    assert mesh.subs == {'all': [cb2]}
    mesh.subs = [cb2, cb3]
    assert mesh.subs == {'all': [cb2, cb3]}
    mesh.subs.update({'start': [cb1]})
    assert mesh.subs == {'all': [cb2, cb3], 'start': [cb1]}

def test_subscribe_msg():
    assert RE.state == 'idle'
    c = CallbackCounter()

    def counting_stepscan(det, motor):
        yield Msg('subscribe', None, 'start', c)
        yield from stepscan(det, motor)

    RE(counting_stepscan(det, motor))  # should advance c
    assert c.value == 1
    RE(counting_stepscan(det, motor))  # should advance c
    assert c.value == 2
    RE(stepscan(det, motor))  # should not
    assert c.value == 2


def test_unknown_cb_raises():
    def f(name, doc):
        pass
    # Dispatches catches this case.
    with pytest.raises(KeyError):
        RE.subscripe('not a thing', f)
    # CallbackRegistry catches this case (different error).
    with pytest.raises(ValueError):
        RE._subscribe_lossless('not a thing', f)


@contextlib.contextmanager
def _print_redirect():
    old_stdout = sys.stdout
    try:
        fout = tempfile.TemporaryFile(mode='w+', encoding='utf-8')
        sys.stdout = fout
        yield fout
    finally:
        sys.stdout = old_stdout


def test_table():
    with _print_redirect() as fout:
        table = LiveTable(['det', 'motor'])
        ad_scan = AdaptiveAbsScanPlan([det], 'det', motor, -15, 5, .01, 1, .05, True)
        RE(ad_scan, subs={'all': [table]})

    fout.seek(0)

    for ln, kn in zip(fout, KNOWN_TABLE.split('\n')):
        # this is to strip the `\n` from the print output
        ln = ln.rstrip()

        if ln[0] == '+':
            # test the full line on the divider lines
            assert ln == kn
        else:
            # skip the 'time' column on data rows
            # this is easier than faking up times in the scan!
            assert np.all(ln[:16] == kn[:16])
            assert np.all(ln[26:] == kn[26:])


KNOWN_TABLE = """+------------+--------------+----------------+----------------+
|   seq_num  |        time  |           det  |         motor  |
+------------+--------------+----------------+----------------+
|         1  |  04:17:20.6  |      1.39e-49  |        -15.00  |
|         2  |  04:17:20.6  |      2.06e-46  |        -14.51  |
|         3  |  04:17:20.7  |      9.79e-43  |        -13.91  |
|         4  |  04:17:20.8  |      9.54e-39  |        -13.23  |
|         5  |  04:17:20.9  |      1.32e-34  |        -12.49  |
|         6  |  04:17:20.9  |      1.94e-30  |        -11.70  |
|         7  |  04:17:21.0  |      2.37e-26  |        -10.86  |
|         8  |  04:17:21.1  |      2.02e-22  |        -10.00  |
|         9  |  04:17:21.2  |      1.03e-18  |         -9.10  |
|        10  |  04:17:21.2  |      2.81e-15  |         -8.19  |
|        11  |  04:17:21.3  |      3.75e-12  |         -7.25  |
|        12  |  04:17:21.4  |      2.29e-09  |         -6.31  |
|        13  |  04:17:21.5  |      6.04e-07  |         -5.35  |
|        14  |  04:17:21.5  |      6.64e-05  |         -4.39  |
|        15  |  04:17:21.6  |      2.95e-03  |         -3.41  |
|        16  |  04:17:21.7  |          0.05  |         -2.44  |
|        17  |  04:17:21.8  |          0.35  |         -1.45  |
|        18  |  04:17:21.8  |          0.08  |         -2.27  |
|        19  |  04:17:21.9  |          0.12  |         -2.07  |
|        20  |  04:17:22.0  |          0.18  |         -1.86  |
|        21  |  04:17:22.1  |          0.25  |         -1.66  |
|        22  |  04:17:22.1  |          0.22  |         -1.73  |
|        23  |  04:17:22.2  |          0.28  |         -1.59  |
|        24  |  04:17:22.3  |          0.34  |         -1.46  |
|        25  |  04:17:22.4  |          0.33  |         -1.49  |
|        26  |  04:17:22.4  |          0.38  |         -1.38  |
|        27  |  04:17:22.5  |          0.44  |         -1.28  |
|        28  |  04:17:22.6  |          0.50  |         -1.18  |
|        29  |  04:17:22.7  |          0.56  |         -1.08  |
|        30  |  04:17:22.7  |          0.62  |         -0.98  |
|        31  |  04:17:22.8  |          0.67  |         -0.89  |
|        32  |  04:17:22.9  |          0.73  |         -0.80  |
|        33  |  04:17:23.0  |          0.78  |         -0.71  |
|        34  |  04:17:23.1  |          0.82  |         -0.62  |
|        35  |  04:17:23.1  |          0.87  |         -0.53  |
|        36  |  04:17:23.2  |          0.91  |         -0.44  |
|        37  |  04:17:23.3  |          0.94  |         -0.34  |
|        38  |  04:17:23.4  |          0.97  |         -0.23  |
|        39  |  04:17:23.5  |          0.99  |         -0.11  |
|        40  |  04:17:23.5  |          1.00  |          0.04  |
|        41  |  04:17:23.6  |          0.94  |          0.36  |
|        42  |  04:17:23.7  |          0.96  |          0.30  |
|        43  |  04:17:23.8  |          0.85  |          0.56  |
|        44  |  04:17:23.9  |          0.91  |          0.42  |
|        45  |  04:17:23.9  |          0.86  |          0.56  |
|        46  |  04:17:24.0  |          0.79  |          0.69  |
|        47  |  04:17:24.1  |          0.81  |          0.66  |
|        48  |  04:17:24.2  |          0.75  |          0.75  |
|        49  |  04:17:24.3  |          0.70  |          0.85  |
|        50  |  04:17:24.3  |          0.64  |          0.94  |
|        51  |  04:17:24.4  |          0.58  |          1.04  |
+------------+--------------+----------------+----------------+
|   seq_num  |        time  |           det  |         motor  |
+------------+--------------+----------------+----------------+
|        52  |  04:17:24.5  |          0.53  |          1.13  |
|        53  |  04:17:24.6  |          0.48  |          1.22  |
|        54  |  04:17:24.7  |          0.43  |          1.30  |
|        55  |  04:17:24.7  |          0.38  |          1.39  |
|        56  |  04:17:24.8  |          0.33  |          1.48  |
|        57  |  04:17:24.9  |          0.29  |          1.57  |
|        58  |  04:17:25.0  |          0.25  |          1.66  |
|        59  |  04:17:25.0  |          0.21  |          1.76  |
|        60  |  04:17:25.1  |          0.18  |          1.87  |
|        61  |  04:17:25.2  |          0.14  |          1.98  |
|        62  |  04:17:25.3  |          0.11  |          2.10  |
|        63  |  04:17:25.4  |          0.08  |          2.24  |
|        64  |  04:17:25.4  |          0.06  |          2.39  |
|        65  |  04:17:25.5  |          0.04  |          2.58  |
|        66  |  04:17:25.6  |          0.02  |          2.82  |
|        67  |  04:17:25.7  |      6.88e-03  |          3.16  |
|        68  |  04:17:25.8  |      1.42e-03  |          3.62  |
|        69  |  04:17:25.8  |      1.51e-04  |          4.20  |
|        70  |  04:17:25.9  |      7.67e-06  |          4.85  |
+------------+--------------+----------------+----------------+"""


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
