"""
This module creates an instance of the RunEngine and configures it. None of
this is *essential* but it is extremely useful and generally recommended.
"""
import os
import logging
import history
from bluesky.run_engine import RunEngine
from bluesky.register_mds import register_mds
from bluesky.hardware_checklist import (connect_mds_mongodb,
                                        connect_fs_mongodb, connect_olog,
                                        connect_channelarchiver,
                                        check_storage, connect_pv,
                                        assert_pv_equal, assert_pv_greater,
                                        assert_pv_less, assert_pv_in_band,
                                        assert_pv_out_of_band)
import bluesky.shortcuts as sc

logger = logging.getLogger(__name__)


### Set up a History object to handle peristence (scan ID, etc.)

SEARCH_PATH = []
ENV_VAR = 'BLUESKY_HISTORY_LOCATION'
if ENV_VAR in os.environ:
    SEARCH_PATH.append(os.environ[ENV_VAR])
SEARCH_PATH.extend([os.path.expanduser('~/.config/bluesky/bluesky_history.db'),
                    '/etc/bluesky/bluesky_history.db'])


def get_history():
    target_path = os.path.join(os.path.expanduser('~'), '.bluesky',
                               'metadata_history.db')
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        if os.path.isfile(target_path):
            print('Found metadata history in existing file.')
        else:
            print('Storing metadata history in a new file.')
        return history.History(target_path)
    except IOError as exc:
        print(exc)
        print('Storing History in memory; it will not persist.')
        return history.History(':memory:')


RE = RunEngine(get_history())
try:
    RE.md['owner']
except KeyError:
    from getpass import getuser
    RE.md['owner'] = getuser()
register_mds(RE)  # subscribes to MDS-related callbacks


# Instantiate shortcuts.
ct = count = sc.Count(RE)
ascan = sc.AbsoluteScan(RE)
mesh = sc.OuterProductAbsoluteScan(RE)
a2scan = a3scan = sc.InnerProductAbsoluteScan(RE)
dscan = lup = sc.DeltaScan(RE)
d2scan = d3scan = sc.InnerProductDeltaScan(RE)
th2th = sc.ThetaTwoThetaScan(RE)
hscan = sc.HScan(RE)
kscan = sc.KScan(RE)
lscan = sc.LScan(RE)
tscan = sc.AbsoluteTemperatureScan(RE)
dtscan = sc.DeltaTemperatureScan(RE)
hklscan = sc.OuterProductHKLScan(RE)
hklmesh = sc.InnerProductHKLScan(RE)

# Provide global aliases to RunEngine methods.
resume = RE.resume
stop = RE.stop
abort = RE.abort
panic = RE.panic
all_is_well = RE.all_is_well


def olog_wrapper(logbook, logbooks):
    """Wrap an olog logbook for use with RunEngine

    The admittedly confusing parameter names reflect our understanding of Olog
    conventions.

    Parameters
    ----------
    logbook : pyolog.logbook
        logbook object
    logbooks : list of strings
        names of logbooks to write entries to

    Returns
    -------
    callable
       callable with the right signature for use with RunEngine.logbook
    """
    def _logbook_log(msg, d):
        msg = msg.format(**d)
        d = {k: repr(v) for k, v in d.items()}
        logbook.log(msg,
                    # TODO Figure out why this returns 500 from olog.
                    # properties={'Bluesky': d},
                    ensure=True,
                    logbooks=logbooks)

    return _logbook_log


def show_debug_logs():
    logging.basicConfig()
    logging.getLogger('bluesky').setLevel(logging.DEBUG)


def basic_checklist(ca_url=None, disk_storage=None, pv_names=None,
                    pv_conditions=None):
    """
    Run checklist of functions that ensure the setup is working.

    Not all of these items are required for bluesky to run and collect data.
    Nonetheless, if you ignore failures you ignore them at your own risk!

    - Check that storage disks have free space available.
    - Check connection to metadatastore and filestore databases.
    - Check connection to the olog.
    - Check connection to the channel archiver.
    - Check that certain PVs are responsive.
    - Check that readings from certain PVs have a reasonable value.

    Parameters
    ----------
    ca_url : string, optional
        url to channel archiver
    disk_storage : list of tuples, optional
        List of pairs, giving a path to a disk and bytes of free space needed.
        For example, [('path/to/disk', minimum_required_bytes_free),]
    pv_names : list, optional
        List of PVs to run 'caget' on and require a timely response.
        For example, ['PV:CAT', 'PV:DOG']
    pv_conditions : list of tuples, optional
        List of conditions about the values of certain PVs.
        For example, [(pv_name, message, func, *args),]
        The func should be one of the following functions in
        bluesky.hardware_checklist: assert_equal, assert_greater,
        assert_less, assert_in_band, assert_out_of_band.

    Examples
    --------
    >>> checklist('http://xf23id-ca.cs.nsls2.local:4800',
                  [('/home', 1000000), ('/GPFS', 10000000)],
                  ['PV:RED', 'PV:BLUE'],
                  [('PV:CAT', 'cat is 5', assert_pv_equal, 5),
                   ('PV:DOG', 'dog is at least 4', assert_pv_greater, 4),
                   ('PV:BEAR', 'bear is 4-6', assert_pv_in_band, 4, 6)])
    """
    print("  Attempting to connect to metadatastore mongodb...")
    _try_and_print(connect_mds_mongodb)
    print("  Attempting to connect to filestore mongodb...")
    _try_and_print(connect_fs_mongodb)
    print("  Attempting to connect to the olog...")
    _try_and_print(connect_olog)

    if ca_url is None:
        print("- Skipping channel arciver check; no URL was provided.")
    else:
        print("  Attempting to connect to the channel archiver...")
        _try_and_print(connect_channelarchiver, ca_url)
    if disk_storage is None:
        print("- Skipping storage disk check; no disks were specified.")
    else:
        for disk, required_free in disk_storage:
            print("  Checking that %s has at least %d bytes free..." %
                  (disk, required_free))
            _try_and_print(check_storage, disk, required_free)

    if pv_names is None:
        print("- Skipping PV responsiveness checks; no PV names were given.")
    else:
        for pv_name in pv_names:
            print("  Checking that the PV '%s' is responsive..." % pv_name)
            _try_and_print(connect_pv, pv_name)

    if pv_conditions is None:
        print("- Skipping PV conditions; none were specified.")
    else:
        for pv_name, msg, func, *args in pv_conditions:
            print("  Checking that %s..." % msg)
            _try_and_print(func, pv_name, *args)


def _try_and_print(func, *args):
    "Gratuitous function to print a checkmark or X on the preceding line"
    try:
        func(*args)
    except:
        print('\x1b[1A\u2717')
        raise
    else:
        print('\x1b[1A\u2713')
