from functools import partial
import sys
import asyncio
import time as ttime
import pytest
from multiprocessing import Process
import signal
from bluesky import Msg
from bluesky.tests.utils import setup_test_run_engine
import os

RE = setup_test_run_engine()
loop = asyncio.get_event_loop()
pcaspy_process = None

@pytest.fixture("module")
def ensure_epics(request):
    try:
        from pcaspy import Driver, SimpleServer
    except ImportError as ie:
        pytest.skip("pcaspy is not available. Skipping all suspenders test."
                    "ImportError: {}".format(ie))
    def teardown():
        nonlocal pcaspy_process
        os.kill(pcaspy_process.pid, signal.SIGINT)
        pcaspy_process.join()

    request.addfinalizer(teardown)

    def to_subproc():

        prefix = 'BSTEST:'
        pvdb = {'VAL': {'prec': 3}}

        class myDriver(Driver):
            def __init__(self):
                super(myDriver, self).__init__()

        server = SimpleServer()
        server.createPV(prefix, pvdb)
        driver = myDriver()

        # process CA transactions
        while True:
            try:
                server.process(0.1)
            except KeyboardInterrupt:
                break

    pcaspy_process = Process(target=to_subproc)
    pcaspy_process.start()


def test_epics_smoke(ensure_epics):
    # pytest.xfail("Epics integration testing is broken.")

    try:
        import epics
    except ImportError as ie:
        pytest.skip("epics is not installed. Skipping epics smoke test."
                    "ImportError: {}".format(ie))
    pv = epics.PV('BSTEST:VAL')
    pv.value = 123456
    print(pv)
    print(pv.value)
    print(pv.connect())
    assert pv.connect()
    for j in range(1, 15):
        pv.put(j, wait=True)
        ret = pv.get(use_monitor=False)
        assert ret == j


def _test_suspender(suspender_class, sc_args, start_val, fail_val,
                    resume_val, wait_time):

    # pytest.xfail("Epics integration testing is broken.")
    try:
        import epics
    except ImportError as ie:
        pytest.skip("epics is not installed. Skipping suspenders test"
                    "ImportError: {}".format(ie))
    if sys.platform == 'darwin':
        pytest.xfail('OSX event loop is different; resolve this later')
    my_suspender = suspender_class('BSTEST:VAL', *sc_args, sleep=wait_time)
    my_suspender.install(RE)
    print(my_suspender._lock)
    pv = epics.PV('BSTEST:VAL')
    putter = partial(pv.put, wait=True)
    # make sure we start at good value!
    putter(start_val)
    # dumb scan
    scan = [Msg('checkpoint'), Msg('sleep', None, .2)]
    RE(scan)
    # paranoid
    assert RE.state == 'idle'

    start = ttime.time()
    # queue up fail and resume conditions
    loop.call_later(.1, putter, fail_val)
    loop.call_later(1, putter, resume_val)
    # start the scan
    RE(scan)
    stop = ttime.time()
    # paranoid clean up of pv call back
    my_suspender._pv.disconnect()
    # assert we waited at least 2 seconds + the settle time
    delta = stop - start
    print(delta)
    assert delta > 1 + wait_time + .2


def test_suspending(ensure_epics):
    # pytest.xfail("Epics integration testing is unreliable.")
    try:
        from bluesky.suspenders import (PVSuspendBoolHigh,
                                        PVSuspendBoolLow,
                                        PVSuspendFloor,
                                        PVSuspendCeil,
                                        PVSuspendInBand,
                                        PVSuspendOutBand)
    except ImportError as ie:
        pytest.skip('bluesky suspenders not available. ImportError: {}'.format(ie))

    _test_suspender(PVSuspendBoolHigh, (), 0, 1, 0, .5)
    _test_suspender(PVSuspendBoolLow, (), 1, 0, 1, .5)
    _test_suspender(PVSuspendFloor, (.5,), 1, 0, 1, .5)
    _test_suspender(PVSuspendCeil, (.5,), 0, 1, 0, .5)
    _test_suspender(PVSuspendInBand, (.5, 1.5), 1, 0, 1, .5)
    _test_suspender(PVSuspendOutBand, (.5, 1.5), 0, 1, 0, .5)
