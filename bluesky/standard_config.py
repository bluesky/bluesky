"""
None of this is essential, but it is useful and generally recommended.
"""
import os
import asyncio
from getpass import getuser
import logging
import warnings
from bluesky.run_engine import RunEngine
from bluesky.register_mds import register_mds
from bluesky.hardware_checklist import (connect_olog,
                                        connect_channelarchiver,
                                        check_storage, connect_pv,
                                        assert_pv_equal, assert_pv_greater,
                                        assert_pv_less, assert_pv_in_band,
                                        assert_pv_out_of_band)
from bluesky.global_state import gs, abort, stop, resume, panic, all_is_well
from bluesky.spec_api import *
from bluesky.callbacks import LiveTable, LivePlot, LiveMesh, print_metadata
from databroker import DataBroker as db, get_events, get_images, get_table

# pylab-esque imports
from time import sleep
import numpy as np

try:
    import historydict

    def get_history():
        target_path = os.path.join(os.path.expanduser('~'), '.bluesky',
                                   'metadata_history.db')
        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            if os.path.isfile(target_path):
                print('Found metadata history in existing file.')
            else:
                print('Storing metadata history in a new file.')
            return historydict.HistoryDict(target_path)
        except IOError as exc:
            print(exc)
            print('Storing HistoryDict in memory; it will not persist.')
            return historydict.HistoryDict(':memory:')

except ImportError:
    warnings.warn("You do not have historydict installed, your metadata will not"
                  "be persistent or have any history of the values.")

    def get_history():
        return dict()

logger = logging.getLogger(__name__)


loop = asyncio.get_event_loop()
loop.set_debug(False)



### Set up a History object to handle peristence (scan ID, etc.)

SEARCH_PATH = []
ENV_VAR = 'BLUESKY_HISTORY_LOCATION'
if ENV_VAR in os.environ:
    SEARCH_PATH.append(os.environ[ENV_VAR])
SEARCH_PATH.extend([os.path.expanduser('~/.config/bluesky/bluesky_history.db'),
                    '/etc/bluesky/bluesky_history.db'])



gs.RE.md = get_history()
gs.RE.md['owner'] = getuser()
register_mds(gs.RE)  # subscribes to MDS-related callbacks


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
                    pv_conditions=None, swallow_errors=False):
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
    print("  Attempting to connect to the olog...")
    _try_and_print(connect_olog, swallow_errors=swallow_errors)

    if ca_url is None:
        print("- Skipping channel arciver check; no URL was provided.")
    else:
        print("  Attempting to connect to the channel archiver...")
        _try_and_print(connect_channelarchiver, ca_url,
                       swallow_errors=swallow_errors)
    if disk_storage is None:
        print("- Skipping storage disk check; no disks were specified.")
    else:
        for disk, required_free in disk_storage:
            print("  Checking that %s has at least %d bytes free..." %
                  (disk, required_free))
            _try_and_print(check_storage, disk, required_free,
                           swallow_errors=swallow_errors)

    if pv_names is None:
        print("- Skipping PV responsiveness checks; no PV names were given.")
    else:
        for pv_name in pv_names:
            print("  Checking that the PV '%s' is responsive..." % pv_name)
            _try_and_print(connect_pv, pv_name, swallow_errors=swallow_errors)

    if pv_conditions is None:
        print("- Skipping PV conditions; none were specified.")
    else:
        for pv_name, msg, func, *args in pv_conditions:
            print("  Checking that %s..." % msg)
            _try_and_print(func, pv_name, *args, swallow_errors=swallow_errors)


def _try_and_print(func, *args, swallow_errors=False):
    "Gratuitous function to print a checkmark or X on the preceding line"
    try:
        func(*args)
    except:
        print('\x1b[1A\u2717')
        if not swallow_errors:
            raise
    else:
        print('\x1b[1A\u2713')
