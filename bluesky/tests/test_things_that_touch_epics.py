from functools import partial
import sys
from nose.tools import assert_equal, assert_greater
from nose import SkipTest
import asyncio
import time as ttime

from bluesky import Msg
from bluesky.tests.utils import setup_test_run_engine
from bluesky.testing.noseclasses import KnownFailureTest
RE = setup_test_run_engine()
loop = asyncio.get_event_loop()


def test_epics_smoke():

    try:
        import epics
    except ImportError:
        raise SkipTest
    pv = epics.PV('BSTEST:VAL')
    pv.value = 123456
    print(pv)
    print(pv.value)
    print(pv.connect())
    assert pv.connect()
    for j in range(1, 15):
        pv.put(j, wait=True)
        ret = pv.get(use_monitor=False)
        assert_equal(ret, j)


def _test_suspender(suspender_class, sc_args, start_val, fail_val,
                    resume_val, wait_time):

    try:
        import epics
    except ImportError:
        raise SkipTest
    if sys.platform == 'darwin':
        # OSX event loop is different; resolve this later
        raise KnownFailureTest()
    my_suspender = suspender_class(RE, 'BSTEST:VAL', *sc_args, sleep=wait_time)
    print(my_suspender._lock)
    pv = epics.PV('BSTEST:VAL')
    putter = partial(pv.put, wait=True)
    # make sure we start at good value!
    putter(start_val)
    # dumb scan
    scan = [Msg('checkpoint'), Msg('sleep', None, .2)]
    # paranoid
    assert_equal(RE.state, 'idle')

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
    print(stop - start)
    assert_greater(stop - start, 1 + wait_time + .2)


def test_suspending():
    try:
        from bluesky.suspenders import (PVSuspendBoolHigh,
                                        PVSuspendBoolLow,
                                        PVSuspendFloor,
                                        PVSuspendCeil,
                                        PVSuspendInBand,
                                        PVSuspendOutBand)
    except ImportError:
        raise SkipTest

    yield _test_suspender, PVSuspendBoolHigh, (), 0, 1, 0, .5
    yield _test_suspender, PVSuspendBoolLow, (), 1, 0, 1, .5
    yield _test_suspender, PVSuspendFloor, (.5,), 1, 0, 1, .5
    yield _test_suspender, PVSuspendCeil, (.5,), 0, 1, 0, .5
    yield _test_suspender, PVSuspendInBand, (.5, 1.5), 1, 0, 1, .5
    yield _test_suspender, PVSuspendOutBand, (.5, 1.5), 0, 1, 0, .5
