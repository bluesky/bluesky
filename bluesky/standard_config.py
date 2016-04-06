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


def show_debug_logs():
    logging.basicConfig()
    logging.getLogger('bluesky').setLevel(logging.DEBUG)
