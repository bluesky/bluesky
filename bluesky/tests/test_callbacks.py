from nose.tools import assert_equal, assert_raises
from bluesky.run_engine import Msg
from bluesky.examples import (motor, det, stepscan)
from bluesky.scans import AdaptiveAbsScan
from bluesky.callbacks import (CallbackCounter, LiveTable)
from bluesky.tests.utils import setup_test_run_engine
from nose.tools import raises
import contextlib
import sys
import tempfile

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
        fout = tempfile.TemporaryFile(mode='w+', encoding='utf-8')
        sys.stdout = fout
        yield fout
    finally:
        sys.stdout = old_stdout


def test_table():
    with _print_redirect() as fout:
        table = LiveTable(['det', 'motor'])
        ad_scan = AdaptiveAbsScan([det], 'det', motor, -15, 5, .01, 1, .05, True)
        RE(ad_scan, subs={'all': [table]})

    fout.seek(0)

    for ln, kn in zip(fout, KNOWN_TABLE.split('\n')):
        # this is to strip the `\n` from the print output
        ln = ln.rstrip()

        if ln[0] == '+':
            # test the full line on the divider lines
            assert_equal(ln, kn)
        else:
            # skip the 'time' column on data rows
            # this is easier than faking up times in the scan!
            assert_equal(ln[:17], kn[:17])
            assert_equal(ln[31:], kn[31:])


KNOWN_TABLE = """+------------+-------------------+----------------+----------------+
|   seq_num  |             time  |           det  |         motor  |
+------------+-------------------+----------------+----------------+
|         1  |  04:17:20.620203  |      1.39e-49  |        -15.00  |
|         2  |  04:17:20.694218  |      2.06e-46  |        -14.51  |
|         3  |  04:17:20.767094  |      9.79e-43  |        -13.91  |
|         4  |  04:17:20.841348  |      9.54e-39  |        -13.23  |
|         5  |  04:17:20.914980  |      1.32e-34  |        -12.49  |
|         6  |  04:17:20.988519  |      1.94e-30  |        -11.70  |
|         7  |  04:17:21.062896  |      2.37e-26  |        -10.86  |
|         8  |  04:17:21.137501  |      2.02e-22  |        -10.00  |
|         9  |  04:17:21.210678  |      1.03e-18  |         -9.10  |
|        10  |  04:17:21.284372  |      2.81e-15  |         -8.19  |
|        11  |  04:17:21.358575  |      3.75e-12  |         -7.25  |
|        12  |  04:17:21.432629  |      2.29e-09  |         -6.31  |
|        13  |  04:17:21.507024  |      6.04e-07  |         -5.35  |
|        14  |  04:17:21.581299  |      6.64e-05  |         -4.39  |
|        15  |  04:17:21.655906  |      2.95e-03  |         -3.41  |
|        16  |  04:17:21.731028  |          0.05  |         -2.44  |
|        17  |  04:17:21.805566  |          0.35  |         -1.45  |
|        18  |  04:17:21.879456  |          0.08  |         -2.27  |
|        19  |  04:17:21.952599  |          0.12  |         -2.07  |
|        20  |  04:17:22.027966  |          0.18  |         -1.86  |
|        21  |  04:17:22.102809  |          0.25  |         -1.66  |
|        22  |  04:17:22.181095  |          0.22  |         -1.73  |
|        23  |  04:17:22.257095  |          0.28  |         -1.59  |
|        24  |  04:17:22.331499  |          0.34  |         -1.46  |
|        25  |  04:17:22.405178  |          0.33  |         -1.49  |
|        26  |  04:17:22.488368  |          0.38  |         -1.38  |
|        27  |  04:17:22.563106  |          0.44  |         -1.28  |
|        28  |  04:17:22.640744  |          0.50  |         -1.18  |
|        29  |  04:17:22.715858  |          0.56  |         -1.08  |
|        30  |  04:17:22.798048  |          0.62  |         -0.98  |
|        31  |  04:17:22.879203  |          0.67  |         -0.89  |
|        32  |  04:17:22.955878  |          0.73  |         -0.80  |
|        33  |  04:17:23.044266  |          0.78  |         -0.71  |
|        34  |  04:17:23.121108  |          0.82  |         -0.62  |
|        35  |  04:17:23.199288  |          0.87  |         -0.53  |
|        36  |  04:17:23.281522  |          0.91  |         -0.44  |
|        37  |  04:17:23.357659  |          0.94  |         -0.34  |
|        38  |  04:17:23.433860  |          0.97  |         -0.23  |
|        39  |  04:17:23.509150  |          0.99  |         -0.11  |
|        40  |  04:17:23.585533  |          1.00  |          0.04  |
|        41  |  04:17:23.669807  |          0.94  |          0.36  |
|        42  |  04:17:23.747575  |          0.96  |          0.30  |
|        43  |  04:17:23.824182  |          0.85  |          0.56  |
|        44  |  04:17:23.900877  |          0.91  |          0.42  |
|        45  |  04:17:23.979421  |          0.86  |          0.56  |
|        46  |  04:17:24.057623  |          0.79  |          0.69  |
|        47  |  04:17:24.139156  |          0.81  |          0.66  |
|        48  |  04:17:24.225662  |          0.75  |          0.75  |
|        49  |  04:17:24.301814  |          0.70  |          0.85  |
|        50  |  04:17:24.381336  |          0.64  |          0.94  |
|        51  |  04:17:24.463212  |          0.58  |          1.04  |
+------------+-------------------+----------------+----------------+
|   seq_num  |             time  |           det  |         motor  |
+------------+-------------------+----------------+----------------+
|        52  |  04:17:24.546003  |          0.53  |          1.13  |
|        53  |  04:17:24.622875  |          0.48  |          1.22  |
|        54  |  04:17:24.700469  |          0.43  |          1.30  |
|        55  |  04:17:24.777818  |          0.38  |          1.39  |
|        56  |  04:17:24.857511  |          0.33  |          1.48  |
|        57  |  04:17:24.936966  |          0.29  |          1.57  |
|        58  |  04:17:25.015495  |          0.25  |          1.66  |
|        59  |  04:17:25.093644  |          0.21  |          1.76  |
|        60  |  04:17:25.172166  |          0.18  |          1.87  |
|        61  |  04:17:25.258121  |          0.14  |          1.98  |
|        62  |  04:17:25.337046  |          0.11  |          2.10  |
|        63  |  04:17:25.415491  |          0.08  |          2.24  |
|        64  |  04:17:25.493488  |          0.06  |          2.39  |
|        65  |  04:17:25.572215  |          0.04  |          2.58  |
|        66  |  04:17:25.652222  |          0.02  |          2.82  |
|        67  |  04:17:25.729949  |      6.88e-03  |          3.16  |
|        68  |  04:17:25.807585  |      1.42e-03  |          3.62  |
|        69  |  04:17:25.897572  |      1.51e-04  |          4.20  |
|        70  |  04:17:25.979183  |      7.67e-06  |          4.85  |
+------------+-------------------+----------------+----------------+"""


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
