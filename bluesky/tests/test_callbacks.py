from bluesky.run_engine import Msg
from bluesky.examples import (motor, det, stepscan, motor1, motor2, det4)
from bluesky.plans import (AdaptiveAbsScanPlan, AbsScanPlan, scan,
                           outer_product_scan)
from bluesky.callbacks import (CallbackCounter, LiveTable, LiveFit,
                               LiveFitPlot, LivePlot)
from bluesky.callbacks.zmqpub import Publisher
from bluesky.callbacks.zmqsub import RemoteDispatcher
from bluesky.tests.utils import setup_test_run_engine
from bluesky.tests.utils import _print_redirect
import multiprocessing
import time
import pytest
import numpy as np
import matplotlib.pyplot as plt
RE = setup_test_run_engine()


def exception_raiser(name, doc):
    raise Exception("it's an exception that better not kill the scan!!")


def test_all():
    c = CallbackCounter()
    RE(stepscan(det, motor), subs={'all': c})
    assert c.value == 10 + 1 + 2  # events, descriptor, start and stop

    c = CallbackCounter()
    token = RE.subscribe('all', c)
    RE(stepscan(det, motor))
    RE.unsubscribe(token)
    assert c.value == 10 + 1 + 2


def _raising_callbacks_helper(stream_name, callback):
    with pytest.raises(Exception):
        RE(stepscan(det, motor), subs={stream_name: callback})


def test_raising_ignored_or_not():
    RE.ignore_callback_exceptions = True
    assert RE.ignore_callback_exceptions

    def cb(name, doc):
        raise Exception
    # by default (with ignore... = True) it warns
    with pytest.warns(UserWarning):
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
    assert obj_ascan.subs == {'all': [cb1], 'start': [], 'stop': [],
                              'descriptor': [], 'event': []}
    obj_ascan.subs.update({'start': [cb2]})
    assert obj_ascan.subs == {'all': [cb1], 'start': [cb2], 'stop': [],
                              'descriptor': [], 'event': []}
    obj_ascan.subs = [cb2, cb3]
    assert obj_ascan.subs == {'all': [cb2, cb3], 'start': [], 'stop': [],
                              'descriptor': [], 'event': []}


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
    with pytest.raises(KeyError):
        RE.subscribe('not a thing', f)
    # back-compat alias for subscribe
    with pytest.raises(KeyError):
        RE.subscribe_lossless('not a thing', f)
    with pytest.raises(KeyError):
        RE._subscribe_lossless('not a thing', f)


def test_table_warns():
    table = LiveTable(['field'])
    table('start', {})
    with pytest.warns(UserWarning):
        table('descriptor', {'uid': 'asdf', 'name': 'primary',
                             'data_keys': {'field': {'dtype': 'array'}}})


def test_table():
    with _print_redirect() as fout:
        det.precision = 2
        motor.precision = 2

        table = LiveTable(['det', 'motor'], min_width=16, extra_pad=2)
        ad_scan = AdaptiveAbsScanPlan([det], 'det', motor, -15, 5, .01, 1, .05,
                                      True)
        # use lossless sub here because rows can get dropped
        token = RE.subscribe_lossless('all', table)
        RE(ad_scan)
        RE.unsubscribe_lossless(token)

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
            assert ln[:16] == kn[:16]
            assert ln[26:] == kn[26:]


KNOWN_TABLE = """+------------+--------------+----------------+----------------+
|   seq_num  |        time  |           det  |         motor  |
+------------+--------------+----------------+----------------+
|         1  |  04:17:20.6  |          0.00  |        -15.00  |
|         2  |  04:17:20.6  |          0.00  |        -14.51  |
|         3  |  04:17:20.7  |          0.00  |        -13.91  |
|         4  |  04:17:20.8  |          0.00  |        -13.23  |
|         5  |  04:17:20.9  |          0.00  |        -12.49  |
|         6  |  04:17:20.9  |          0.00  |        -11.70  |
|         7  |  04:17:21.0  |          0.00  |        -10.86  |
|         8  |  04:17:21.1  |          0.00  |        -10.00  |
|         9  |  04:17:21.2  |          0.00  |         -9.10  |
|        10  |  04:17:21.2  |          0.00  |         -8.19  |
|        11  |  04:17:21.3  |          0.00  |         -7.25  |
|        12  |  04:17:21.4  |          0.00  |         -6.31  |
|        13  |  04:17:21.5  |          0.00  |         -5.35  |
|        14  |  04:17:21.5  |          0.00  |         -4.39  |
|        15  |  04:17:21.6  |          0.00  |         -3.41  |
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
+------------+--------------+----------------+----------------+
|   seq_num  |        time  |           det  |         motor  |
+------------+--------------+----------------+----------------+
|        50  |  04:17:24.3  |          0.64  |          0.94  |
|        51  |  04:17:24.4  |          0.58  |          1.04  |
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
|        67  |  04:17:25.7  |          0.01  |          3.16  |
|        68  |  04:17:25.8  |          0.00  |          3.62  |
|        69  |  04:17:25.8  |          0.00  |          4.20  |
|        70  |  04:17:25.9  |          0.00  |          4.85  |
+------------+--------------+----------------+----------------+"""


def test_zmq(fresh_RE):
    # COMPONENT 1
    # Run a forwarder device on a separate process.
    # This is a variant of the code in bluesky/examples/forwarder_device.py,
    # but with hard-coded config.

    def forwarder():
        import zmq

        def main(frontend_port, backend_port):
            print('Starting forwarder device...')
            try:
                context = zmq.Context(1)
                # Socket facing clients
                frontend = context.socket(zmq.SUB)
                frontend.bind("tcp://*:%d" % frontend_port)

                frontend.setsockopt_string(zmq.SUBSCRIBE, "")

                # Socket facing services
                backend = context.socket(zmq.PUB)
                backend.bind("tcp://*:%d" % backend_port)

                print('Receiving on %d; publishing on %d' % (frontend_port,
                                                             backend_port))
                zmq.device(zmq.FORWARDER, frontend, backend)
            except Exception as exc:
                print('Exception in forwarder device:', exc)
            finally:
                print('Closing forwarder device...')
                frontend.close()
                backend.close()
                context.term()

        main(5567, 5568)
    forwarder_proc = multiprocessing.Process(target=forwarder, daemon=True)
    forwarder_proc.start()
    time.sleep(5)  # Give this plenty of time to start up.

    # COMPONENT 2
    # Run a Publisher and a RunEngine in this main process.

    RE = fresh_RE
    p = Publisher(RE, '127.0.0.1', 5567)  # noqa

    # COMPONENT 3
    # Run a RemoteDispatcher on another separate process. Pass the documents
    # it receives over a Queue to this process, so we can count them for our
    # test.

    def make_and_start_dispatcher(queue):
        def put_in_queue(name, doc):
            print('putting ', name, 'in queue')
            queue.put((name, doc))
        d = RemoteDispatcher('127.0.0.1', 5568)
        d.subscribe('all', put_in_queue)
        print("REMOTE IS READY TO START")
        d._loop.call_later(9, d.stop)
        d.start()

    queue = multiprocessing.Queue()
    dispatcher_proc = multiprocessing.Process(target=make_and_start_dispatcher,
                                              daemon=True, args=(queue,))
    dispatcher_proc.start()
    time.sleep(5)  # As above, give this plenty of time to start.

    # Generate two documents. The Publisher will send them to the forwarder
    # device over 5567, and the forwarder will send them to the
    # RemoteDispatcher over 5568. The RemoteDispatcher will push them into
    # the queue, where we can verify that they round-tripped.

    local_accumulator = []

    def local_cb(name, doc):
        local_accumulator.append((name, doc))

    RE([Msg('open_run'), Msg('close_run')], local_cb)
    time.sleep(1)

    # Get the two documents from the queue (or timeout --- test will fail)
    remote_accumulator = []
    for i in range(2):
        remote_accumulator.append(queue.get(timeout=2))
    p.close()
    forwarder_proc.terminate()
    dispatcher_proc.terminate()
    forwarder_proc.join()
    dispatcher_proc.join()
    assert remote_accumulator == local_accumulator


def test_live_fit(fresh_RE, motor_det):
    RE = fresh_RE
    motor, det = motor_det
    try:
        import lmfit
    except ImportError:
        raise pytest.skip('requires lmfit')

    def gaussian(x, A, sigma, x0):
        return A*np.exp(-(x - x0)**2/(2 * sigma**2))

    model = lmfit.Model(gaussian)
    init_guess = {'A': 2,
                  'sigma': lmfit.Parameter('sigma', 3, min=0),
                  'x0': -0.2}
    cb = LiveFit(model, 'det', {'x': 'motor'}, init_guess,
                 update_every=50)
    RE(scan([det], motor, -1, 1, 50), cb)
    # results are in cb.result.values

    expected = {'A': 1, 'sigma': 1, 'x0': 0}
    for k, v in expected.items():
        assert np.allclose(cb.result.values[k], v, atol=1e-6)


def test_live_fit_multidim(fresh_RE):
    RE = fresh_RE

    try:
        import lmfit
    except ImportError:
        raise pytest.skip('requires lmfit')

    motor1._fake_sleep = 0
    motor2._fake_sleep = 0
    det4.exposure_time = 0

    def gaussian(x, y, A, sigma, x0, y0):
        return A*np.exp(-((x - x0)**2 + (y - y0)**2)/(2 * sigma**2))

    model = lmfit.Model(gaussian, ['x', 'y'])
    init_guess = {'A': 2,
                  'sigma': lmfit.Parameter('sigma', 3, min=0),
                  'x0': -0.2,
                  'y0': 0.3}
    cb = LiveFit(model, 'det4', {'x': 'motor1', 'y': 'motor2'}, init_guess,
                 update_every=50)
    RE(outer_product_scan([det4], motor1, -1, 1, 10, motor2, -1, 1, 10, False),
       cb)

    expected = {'A': 1, 'sigma': 1, 'x0': 0, 'y0': 0}
    for k, v in expected.items():
        assert np.allclose(cb.result.values[k], v, atol=1e-6)


def test_live_fit_plot(fresh_RE):
    RE = fresh_RE
    try:
        import lmfit
    except ImportError:
        raise pytest.skip('requires lmfit')

    def gaussian(x, A, sigma, x0):
        return A*np.exp(-(x - x0)**2/(2 * sigma**2))

    model = lmfit.Model(gaussian)
    init_guess = {'A': 2,
                  'sigma': lmfit.Parameter('sigma', 3, min=0),
                  'x0': -0.2}
    livefit = LiveFit(model, 'det', {'x': 'motor'}, init_guess,
                      update_every=50)
    lfplot = LiveFitPlot(livefit, color='r')
    lplot = LivePlot('det', 'motor', ax=plt.gca(), marker='o', ls='none')
    RE(scan([det], motor, -1, 1, 50), [lplot, lfplot])

    expected = {'A': 1, 'sigma': 1, 'x0': 0}
    for k, v in expected.items():
        assert np.allclose(livefit.result.values[k], v, atol=1e-6)
