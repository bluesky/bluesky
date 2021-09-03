from collections import defaultdict
from bluesky.run_engine import Msg, RunEngineInterrupted
from bluesky.plans import scan, grid_scan, count, inner_product_scan
from bluesky.object_plans import AbsScanPlan
from bluesky.preprocessors import run_wrapper, subs_wrapper
from bluesky.plan_stubs import pause
import bluesky.plans as bp
import bluesky.preprocessors as bpp
from bluesky.callbacks import CallbackCounter, LiveTable, LiveFit, CallbackBase
from bluesky.callbacks.core import make_callback_safe, make_class_safe
from bluesky.callbacks.mpl_plotting import (
    LiveScatter,
    LivePlot,
    LiveGrid,
    LiveFitPlot,
    LiveRaster,
    LiveMesh,
)
from bluesky.callbacks.broker import BrokerCallbackBase
from bluesky.tests.utils import _print_redirect, MsgCollector, DocCollector
from event_model import compose_run, DocumentNames
import pytest
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from unittest.mock import MagicMock
from itertools import permutations
import time


# copied from examples.py to avoid import
def stepscan(det, motor):
    yield Msg('open_run')
    for i in range(-5, 5):
        yield Msg('create', name='primary')
        yield Msg('set', motor, i)
        yield Msg('trigger', det)
        yield Msg('read', motor)
        yield Msg('read', det)
        yield Msg('save')
    yield Msg('close_run')


def exception_raiser(name, doc):
    raise Exception("it's an exception that better not kill the scan!!")


def test_all(RE, hw):
    c = CallbackCounter()
    RE(stepscan(hw.det, hw.motor), {'all': c})
    assert c.value == 10 + 1 + 2  # events, descriptor, start and stop

    c = CallbackCounter()
    token = RE.subscribe(c)
    RE(stepscan(hw.det, hw.motor))
    RE.unsubscribe(token)
    assert c.value == 10 + 1 + 2


def test_raising_ignored_or_not(RE, hw):
    RE.ignore_callback_exceptions = True
    assert RE.ignore_callback_exceptions

    def cb(name, doc):
        raise Exception

    # by default (with ignore... = True) it warns
    with pytest.warns(UserWarning):
        RE(stepscan(hw.det, hw.motor), cb)

    RE.ignore_callback_exceptions = False
    with pytest.raises(Exception):
        RE(stepscan(hw.det, hw.motor), cb)


def test_subs_input(hw):
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
    obj_ascan = AbsScanPlan([hw.det], hw.motor, 1, 5, 4)
    obj_ascan.subs = cb1
    assert obj_ascan.subs == {'all': [cb1], 'start': [], 'stop': [],
                              'descriptor': [], 'event': []}
    obj_ascan.subs.update({'start': [cb2]})
    assert obj_ascan.subs == {'all': [cb1], 'start': [cb2], 'stop': [],
                              'descriptor': [], 'event': []}
    obj_ascan.subs = [cb2, cb3]
    assert obj_ascan.subs == {'all': [cb2, cb3], 'start': [], 'stop': [],
                              'descriptor': [], 'event': []}


def test_subscribe_msg(RE, hw):
    assert RE.state == 'idle'
    c = CallbackCounter()

    def counting_stepscan(det, motor):
        yield Msg('subscribe', None, c, 'start')
        yield from stepscan(det, motor)

    RE(counting_stepscan(hw.det, hw.motor))  # should advance c
    assert c.value == 1
    RE(counting_stepscan(hw.det, hw.motor))  # should advance c
    assert c.value == 2
    RE(stepscan(hw.det, hw.motor))  # should not
    assert c.value == 2


def test_unknown_cb_raises(RE):

    def f(name, doc):
        pass

    with pytest.raises(KeyError):
        RE.subscribe(f, 'not a thing')
    # back-compat alias for subscribe
    with pytest.raises(KeyError):
        RE.subscribe_lossless(f, 'not a thing')
    with pytest.raises(KeyError):
        RE._subscribe_lossless(f, 'not a thing')


def test_table_warns():
    table = LiveTable(['field'])
    table('start', {})
    with pytest.warns(UserWarning):
        table('descriptor', {'uid': 'asdf', 'name': 'primary',
                             'data_keys': {'field': {'dtype': 'array'}}})


def test_table_external(RE, hw, db):
    RE.subscribe(db.insert)
    RE(count([hw.img]), LiveTable(['img']))


def _compare_tables(fout, known_table):
    for ln, kn in zip(fout, known_table.split('\n')):
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


def test_table(RE, hw):

    with _print_redirect() as fout:
        hw.det.precision = 2
        hw.motor.precision = 2
        hw.motor.setpoint.put(0.0)  # Make dtype 'number' not 'integer'.
        hw.det.trigger()
        assert hw.det.describe()['det']['precision'] == 2
        assert hw.motor.describe()['motor']['precision'] == 2
        assert hw.det.describe()['det']['dtype'] == 'number'
        assert hw.motor.describe()['motor']['dtype'] == 'number'

        table = LiveTable(['det', 'motor'], min_width=16, extra_pad=2, separator_lines=False)
        ad_scan = bp.adaptive_scan([hw.det], 'det', hw.motor,
                                   -15.0, 5., .01, 1, .05,
                                   True)
        # use lossless sub here because rows can get dropped
        token = RE.subscribe(table)
        RE(ad_scan)
        RE.unsubscribe_lossless(token)

    fout.seek(0)
    _compare_tables(fout, KNOWN_TABLE)


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


def test_evil_table_names(RE):
    from ophyd import Signal
    sigs = [
        Signal(value=0, name="a:b"),
        Signal(value=0, name="a,b"),
        Signal(value=0, name="a'b"),
        Signal(value=0, name="🐍"),
    ]
    table = LiveTable(
        [s.name for s in sigs],
        min_width=5, extra_pad=2, separator_lines=False
    )
    with _print_redirect() as fout:
        print()  # get a blank line in camptured output
        RE(bpp.subs_wrapper(bp.count(sigs, num=2), table))
    reference = """
+------------+--------------+--------+--------+--------+--------+
|   seq_num  |        time  |   a:b  |   a,b  |   a'b  |     🐍  |
+------------+--------------+--------+--------+--------+--------+
|         1  |  12:47:09.7  |     0  |     0  |     0  |     0  |
|         2  |  12:47:09.7  |     0  |     0  |     0  |     0  |
+------------+--------------+--------+--------+--------+--------+"""
    _compare_tables(fout, reference)


def test_live_fit(RE, hw):
    try:
        import lmfit
    except ImportError:
        raise pytest.skip('requires lmfit')

    def gaussian(x, A, sigma, x0):
        return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    model = lmfit.Model(gaussian)
    init_guess = {'A': 2,
                  'sigma': lmfit.Parameter('sigma', 3, min=0),
                  'x0': -0.2}
    cb = LiveFit(model, 'det', {'x': 'motor'}, init_guess,
                 update_every=50)
    RE(scan([hw.det], hw.motor, -1, 1, 50), cb)
    # results are in cb.result.values

    expected = {'A': 1, 'sigma': 1, 'x0': 0}
    for k, v in expected.items():
        assert np.allclose(cb.result.values[k], v, atol=1e-6)


def test_live_fit_multidim(RE, hw):

    try:
        import lmfit
    except ImportError:
        raise pytest.skip('requires lmfit')

    hw.motor1.delay = 0
    hw.motor2.delay = 0
    hw.det4.exposure_time = 0

    def gaussian(x, y, A, sigma, x0, y0):
        return A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    model = lmfit.Model(gaussian, ['x', 'y'])
    init_guess = {'A': 2,
                  'sigma': lmfit.Parameter('sigma', 3, min=0),
                  'x0': -0.2,
                  'y0': 0.3}
    cb = LiveFit(model, 'det4', {'x': 'motor1', 'y': 'motor2'}, init_guess,
                 update_every=50)
    RE(grid_scan([hw.det4],
                 hw.motor1, -1, 1, 10,
                 hw.motor2, -1, 1, 10, False),
       cb)

    expected = {'A': 1, 'sigma': 1, 'x0': 0, 'y0': 0}
    for k, v in expected.items():
        assert np.allclose(cb.result.values[k], v, atol=1e-6)


def test_live_plot_from_callbacks():
    import bluesky.callbacks.core
    # We don't want the shims in callbacks.core, see #1133 for discussion
    assert not hasattr(bluesky.callbacks.core, 'LivePlot')
    # We still want the shims in callbacks.__init__
    from bluesky.callbacks import LivePlot as LivePlotFromCallbacks
    assert LivePlotFromCallbacks is LivePlot

    # Make sure we can subclass it
    class FromCallbacks(LivePlotFromCallbacks):
        ...

    FromCallbacks('det', 'motor')


def test_live_fit_plot(RE, hw):
    try:
        import lmfit
    except ImportError:
        raise pytest.skip('requires lmfit')

    def gaussian(x, A, sigma, x0):
        return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    model = lmfit.Model(gaussian)
    init_guess = {'A': 2,
                  'sigma': lmfit.Parameter('sigma', 3, min=0),
                  'x0': -0.2}
    livefit = LiveFit(model, 'det', {'x': 'motor'}, init_guess,
                      update_every=50)
    lfplot = LiveFitPlot(livefit, color='r')
    lplot = LivePlot('det', 'motor', ax=plt.gca(), marker='o', ls='none')
    RE(scan([hw.det], hw.motor, -1, 1, 50), [lplot, lfplot])
    expected = {'A': 1, 'sigma': 1, 'x0': 0}
    for k, v in expected.items():
        assert np.allclose(livefit.result.values[k], v, atol=1e-6)


@pytest.mark.parametrize('int_meth, stop_num, msg_num',
                         [('stop', 1, 5),
                          ('abort', 1, 5),
                          ('halt', 1, 3)])
def test_interrupted_with_callbacks(RE, int_meth,
                                    stop_num, msg_num):

    docs = defaultdict(list)

    def collector_cb(name, doc):
        nonlocal docs
        docs[name].append(doc)

    RE.msg_hook = MsgCollector()

    with pytest.raises(RunEngineInterrupted):
        RE(subs_wrapper(run_wrapper(pause()),
                        {'all': collector_cb}))
    getattr(RE, int_meth)()

    assert len(docs['start']) == 1
    assert len(docs['event']) == 0
    assert len(docs['descriptor']) == 0
    assert len(docs['stop']) == stop_num
    assert len(RE.msg_hook.msgs) == msg_num


def test_live_grid(RE, hw):
    hw.motor1.delay = 0
    hw.motor2.delay = 0
    RE(grid_scan([hw.det4],
                 hw.motor1, -3, 3, 6,
                 hw.motor2, -5, 5, 10, False),
       LiveGrid((6, 10), 'det4'))

    # Test the deprecated name.
    with pytest.warns(UserWarning):
        RE(grid_scan([hw.det4], hw.motor1, -3, 3, 6, hw.motor2,
                     -5, 5, 10, False),
           LiveRaster((6, 10), 'det4'))


def test_live_scatter(RE, hw):
    RE(grid_scan([hw.det5],
                 hw.jittery_motor1, -3, 3, 6,
                 hw.jittery_motor2, -5, 5, 10, False),
       LiveScatter('jittery_motor1', 'jittery_motor2', 'det5',
                   xlim=(-3, 3), ylim=(-5, 5)))

    # Test the deprecated name.
    with pytest.warns(UserWarning):
        RE(grid_scan([hw.det5],
                     hw.jittery_motor1, -3, 3, 6,
                     hw.jittery_motor2, -5, 5, 10, False),
           LiveMesh('jittery_motor1', 'jittery_motor2', 'det5',
                    xlim=(-3, 3), ylim=(-5, 5)))


@pytest.mark.xfail(raises=NotImplementedError,
                   reason='This tests an API databroker has removed, and needs updating.')
def test_broker_base(RE, hw, db):
    class BrokerChecker(BrokerCallbackBase):
        def __init__(self, field, *, db=None):
            super().__init__(field, db=db)

        def event(self, doc):
            super().event(doc)
            assert isinstance(doc['data'][self.fields[0]], np.ndarray)

    RE.subscribe(db.insert)
    bc = BrokerChecker(('img',), db=db)
    RE.subscribe(bc)
    RE(count([hw.img]))


def test_broker_base_no_unpack(RE, hw, db):
    class BrokerChecker(BrokerCallbackBase):
        def __init__(self, field, *, db=None):
            super().__init__(field, db=db)

        def event(self, doc):
            super().event(doc)
            assert isinstance(doc['data'][self.fields[0]], np.ndarray)

    bc = BrokerChecker(('img',), db=db)
    RE.subscribe(bc)
    RE(count([hw.direct_img]))


def test_plotting_hints(RE, hw, db):
    ''' This tests the run and checks that the correct hints are created.
        Hints are mainly created to help the BestEffortCallback in plotting the
        data.
        Use a callback to do the checking.
    '''
    dc = DocCollector()
    RE.subscribe(dc.insert)

    # check that the inner product hints are passed correctly
    hint = {'dimensions': [([hw.motor1.name, hw.motor2.name, hw.motor3.name],
                            'primary')]}
    RE(inner_product_scan([hw.det], 20, hw.motor1, -1, 1, hw.motor2, -1, 1,
                          hw.motor3, -2, 0))
    assert dc.start[-1]['hints'] == hint

    # check that the outer product (grid_scan) hints are passed correctly
    hint = {'dimensions': [(['motor1'], 'primary'),
                           (['motor2'], 'primary'),
                           (['motor3'], 'primary')]}
    # grid_scan passes "rectilinear" gridding as well
    # make sure this is also passed
    output_hint = hint.copy()
    output_hint['gridding'] = 'rectilinear'
    RE(grid_scan([hw.det], hw.motor1, -1, 1, 2, hw.motor2, -1, 1, 2,
                 True, hw.motor3, -2, 0, 2, True))

    assert dc.start[-1]['hints'] == output_hint

    # check that if gridding is supplied, it's not overwritten by grid_scan
    # check that the outer product (grid_scan) hints are passed correctly
    hint = {'dimensions': [(['motor1'], 'primary'),
                           (['motor2'], 'primary'),
                           (['motor3'], 'primary')],
            'gridding': 'rectilinear'}
    RE(grid_scan([hw.det], hw.motor1, -1, 1, 2, hw.motor2, -1, 1, 2,
                 True, hw.motor3, -2, 0, 2, True))
    assert dc.start[-1]['hints'] == hint


def test_broken_table():
    start_doc, descriptor_factory, *_ = compose_run()
    desc, compose_event, _ = descriptor_factory(
        name="primary",
        data_keys={
            "x": {"dtype": "integer", "source": "", "shape": []},
            "y": {"dtype": "number", "source": "", "shape": []},
        },
    )

    ev1 = compose_event(
        data={"x": 1, "y": 2.0}, timestamps={k: time.time() for k in ("x", "y")}
    )
    ev2 = compose_event(
        data={"x": 1, "y": 2}, timestamps={k: time.time() for k in ("x", "y")}
    )
    ev3 = compose_event(
        data={"x": 1.0, "y": 2.0}, timestamps={k: time.time() for k in ("x", "y")}
    )
    ev4 = compose_event(
        data={"x": 1.0, "y": "aardvark"},
        timestamps={k: time.time() for k in ("x", "y")},
    )

    sio = StringIO()
    LT = LiveTable(["x", "y"], out=lambda s: sio.write(s + "\n"))
    LT("start", start_doc)
    LT("descriptor", desc)
    LT("event", ev1)
    LT("event", ev2)
    LT("event", ev3)
    LT("event", ev4)

    sio.seek(0)
    lines = sio.readlines()

    # The instance of LiveTable will include two empty separator lines by default
    assert len(lines) == 9
    for ln in lines[-2:]:
        assert ln.strip() == "failed to format row"


def test_callback_safe():
    @make_callback_safe
    def test_function(to_fail):
        if to_fail:
            raise RuntimeError
        return to_fail

    assert test_function(True) is None
    assert test_function(False) is False


def test_callback_safe_logger():
    from unittest.mock import MagicMock
    from types import SimpleNamespace

    logger = SimpleNamespace(exception=MagicMock())

    @make_callback_safe(logger=logger)
    def test_function(to_fail):
        if to_fail:
            raise RuntimeError
        return to_fail

    assert test_function(True) is None
    assert logger.exception.call_count == 1
    assert test_function(False) is False
    assert logger.exception.call_count == 1


@pytest.fixture
def EvilBaseClass(request):
    class MyError(RuntimeError):
        ...

    class EvilCallback(CallbackBase):
        my_excepttion_type = MyError

        def event(self, doc):
            raise MyError

        def bulk_events(self, doc):
            raise MyError

        def resource(self, doc):
            raise MyError

        def datum(self, doc):
            raise MyError

        def bulk_datum(self, doc):
            raise MyError

        def descriptor(self, doc):
            raise MyError

        def start(self, doc):
            raise MyError

        def stop(self, doc):
            raise MyError

        def event_page(self, doc):
            raise MyError

        def datum_page(self, doc):
            raise MyError

    return EvilCallback


def test_callbackclass(EvilBaseClass):
    ecb = EvilBaseClass()
    for n in DocumentNames:
        with pytest.raises(EvilBaseClass.my_excepttion_type):
            ecb(n.name, {})


def test_callbackclass_safe(EvilBaseClass):
    @make_class_safe
    class SafeEvilBaseClass(EvilBaseClass):
        ...

    scb = SafeEvilBaseClass()
    for n in DocumentNames:
        scb(n.name, {})


def test_callbackclass_safe_logger(EvilBaseClass):
    logger = MagicMock()

    @make_class_safe(logger=logger)
    class SafeEvilBaseClass2(EvilBaseClass):
        ...

    scb = SafeEvilBaseClass2()
    for n in DocumentNames:
        scb(n.name, {})

    assert logger.exception.call_count == len(DocumentNames)


@pytest.mark.parametrize("strict", ["1", None])
@pytest.mark.parametrize(
    "documents",
    (
        list(
            set(
                tuple(sorted(x, key=lambda x: x.name))
                for x in permutations(DocumentNames, 1)
            )
        )
        + list(
            set(
                tuple(sorted(x, key=lambda x: x.name))
                for x in permutations(DocumentNames, 2)
            )
        )
        + list(
            set(
                tuple(sorted(x, key=lambda x: x.name))
                for x in permutations(DocumentNames, 3)
            )
        )
        + [list(DocumentNames)]
    ),
)
def test_callbackclass_safe_filtered(EvilBaseClass, documents, monkeypatch, strict):
    if strict is not None:
        monkeypatch.setenv("BLUESKY_DEBUG_CALLBACKS", strict)
    else:
        monkeypatch.delenv("BLUESKY_DEBUG_CALLBACKS", raising=False)
    logger = MagicMock()

    @make_class_safe(logger=logger, to_wrap=tuple(x.name for x in documents))
    class SafeEvilBaseClass2(EvilBaseClass):
        ...

    scb = SafeEvilBaseClass2()
    for n in documents:
        if strict:
            with pytest.raises(EvilBaseClass.my_excepttion_type):
                scb(n.name, {})
        else:
            scb(n.name, {})

    for n in set(DocumentNames) - set(documents):
        with pytest.raises(EvilBaseClass.my_excepttion_type):
            scb(n.name, {})

    assert logger.exception.call_count == len(documents)


def test_in_plan_qt_callback(RE, hw):
    from bluesky.callbacks.mpl_plotting import _get_teleporter

    _get_teleporter()

    def my_plan():
        motor = hw.motor
        det = hw.det

        motor.delay = 1

        plan = bp.scan([det], motor, -5, 5, 25)
        plan = subs_wrapper(
            bp.scan([det], motor, -5, 5, 25), LivePlot(det.name, motor.name)
        )
        return (yield from plan)

    RE(my_plan())
