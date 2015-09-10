import uuid
import nose
from bluesky.testing.noseclasses import KnownFailureTest
from nose.tools import assert_raises
import time
from bluesky.hardware_checklist import *


def test_connect_mds_mongodb():
    try:
        from metadatastore.utils.testing import mds_setup, mds_teardown
        import metadatastore.commands as mdsc
    except ImportError:
        raise nose.SkipTest

    try:
        mds_setup()
        # Until we insert something, the db is not actually created.

        mdsc.insert_runstart(scan_id=3022013,
                             beamline_id='testbed',
                             owner='tester',
                             group='awesome-devs',
                             project='Nikea',
                             time=time.time(),
                             uid=str(uuid.uuid4()))
        connect_mds_mongodb()

    except:
        raise
    finally:
        mds_teardown()


def test_connect_fs_mongodb():
    raise KnownFailureTest
    # FS does not have fancy connection_config that MDS does
    # Once it is caught up, this will be pass.
    try:
        import filestore
    except ImportError:
        raise nose.SkipTest
    from filestore.utils.testing import fs_setup, fs_teardown
    try:
        fs_setup()
        connect_fs_mongodb()
    except:
        raise
    finally:
        fs_teardown()


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
