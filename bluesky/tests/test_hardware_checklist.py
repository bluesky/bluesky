import uuid
from bluesky.hardware_checklist import (
    assert_pv_equal, connect_pv, connect_channelarchiver, check_storage,
    assert_pv_greater, assert_pv_less, assert_pv_in_band, assert_pv_out_of_band
)
import pytest


def test_check_storage():
    check_storage('/', 1)
    with pytest.raises(RuntimeError):
        check_storage('/', 10000000000000000000)


def test_connect_channelarchiver():
    try:
        import requests
    except ImportError as ie:
        pytest.skip('requests is required to test channelarchiver connection.'
                    'ImportError: {}'.format(ie))
        requests = None

    # Just test failure, not success.
    with pytest.raises(RuntimeError):
        connect_channelarchiver('http://bnl.gov/asfoijewapfoia')
