from multiprocessing import Process
from pcaspy import Driver, SimpleServer
from nose.tools import assert_equal
from nose import SkipTest

import epics


def to_subproc():

    prefix = 'BSTEST:'
    pvdb = {
        'VAL': {
            'prec': 3,
        },
    }

    class myDriver(Driver):
        def __init__(self):
            super(myDriver, self).__init__()

    if __name__ == '__main__':
        server = SimpleServer()
        server.createPV(prefix, pvdb)
        driver = myDriver()

        # process CA transactions
        while True:
            server.process(0.1)

p = None


def setup():
    print("STARTUP")
    global p
    p = Process(target=to_subproc)
    p.start()


def teardown():
    if p is not None:
        p.terminate()


def test_epics_smoke():
    raise SkipTest()

    pv = epics.PV('BSTEST:VAL')
    pv.value = 123456
    print(pv)
    print(pv.value)
    assert pv.connect()
    for j in range(1, 15):
        pv.value = j
        assert_equal(pv.value, j)
