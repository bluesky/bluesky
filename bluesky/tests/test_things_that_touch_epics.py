import asyncio
import pytest
from multiprocessing import Process
import signal

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
