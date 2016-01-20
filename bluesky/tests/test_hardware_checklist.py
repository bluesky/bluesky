import uuid
import nose
from bluesky.testing.noseclasses import KnownFailureTest
from nose.tools import assert_raises
import time
from bluesky.hardware_checklist import *


def test_check_storage():
    check_storage('/', 1)
    assert_raises(RuntimeError, check_storage, '/', 10000000000000000000)


def test_connect_channelarchiver():
    # Just test failure, not success.
    assert_raises(RuntimeError, connect_channelarchiver,
                  'http://bnl.gov/asfoijewapfoia')


def test_connect_pv():
    try:
        import epics
    except ImportError:
        raise nose.SkipTest()
    pv_name = 'BSTEST:VAL'
    connect_pv(pv_name)
    epics.caput(pv_name, 5, wait=True)
    assert_pv_equal(pv_name, 5)
    assert_pv_greater(pv_name, 4)
    assert_pv_less(pv_name, 6)
    assert_pv_in_band(pv_name, 4, 6)
    assert_raises(AssertionError, assert_pv_in_band, pv_name, 2, 4)
    assert_pv_out_of_band(pv_name, 2, 4)
