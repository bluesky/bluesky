from nose.tools import assert_raises, raises
from bluesky.scans import *
from bluesky.tests.utils import setup_test_run_engine
from bluesky.utils import ScanValidator
from bluesky.examples import *

RE = setup_test_run_engine()


def bad_checkpoint_scan():
    yield Msg('create')
    yield Msg('checkpoint')
    yield Msg('save')


def bad_subscription_scan():
    yield Msg('subscribe', None, 'start')


def bad_save_scan():
    yield Msg('save')


def bad_save_scan2():
    yield Msg('create')
    yield Msg('save')
    yield Msg('save')


def bad_configure_scan():
    yield Msg('configure', det)
    yield Msg('configure', det)


def bad_deconfigure_scan():
    yield Msg('deconfigure', det)


def bad_deconfigure_scan2():
    yield Msg('configure', det)
    yield Msg('deconfigure', det)
    yield Msg('deconfigure', det)


def good_checkpoint_scan():
    yield Msg('checkpoint')
    yield Msg('create')
    yield Msg('save')


def good_subscription_scan():
    yield Msg('subscribe', None, 'stop')
    yield Msg('subscribe', None, 'descriptor')
    yield Msg('subscribe', None, 'event')


def good_deconfigure_scan():
    for i in range(2):
        yield Msg('configure', det)
        yield Msg('deconfigure', det)


def _validator_helper(scan, run_engine, raises):
    sv = ScanValidator(scan, run_engine)
    if raises:
        assert_raises(raises, sv.validate)
    else:
        sv.validate()


def validator_tester():
    global RE
    bad_scans = [
        bad_checkpoint_scan(),
        bad_save_scan(),
        bad_save_scan2(),
        bad_subscription_scan(),
        bad_configure_scan(),
        bad_deconfigure_scan(),
        bad_deconfigure_scan2(),
    ]
    good_scans = [
        good_checkpoint_scan(),
        good_subscription_scan(),
        good_deconfigure_scan(),
    ]

    for scan in bad_scans:
        yield _validator_helper, scan, RE, ValueError

    for scan in good_scans:
        yield _validator_helper, scan, RE, None


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
