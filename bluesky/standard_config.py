"""
None of this is essential, but it is useful and generally recommended.
"""
from getpass import getuser
import logging
from bluesky.register_mds import register_mds
from bluesky.global_state import gs, abort, stop, resume, panic, all_is_well
from bluesky.spec_api import *
from bluesky.callbacks import LiveTable, LivePlot, LiveMesh, print_metadata
from bluesky.utils import get_history
from databroker import DataBroker as db, get_events, get_images, get_table

# pylab-esque imports
from time import sleep
import numpy as np


logger = logging.getLogger(__name__)


### Set up a History object to handle peristence (scan ID, etc.)
gs.RE.md['owner'] = getuser()
register_mds(gs.RE)  # subscribes to MDS-related callbacks


def show_debug_logs():
    logging.basicConfig()
    logging.getLogger('bluesky').setLevel(logging.DEBUG)
