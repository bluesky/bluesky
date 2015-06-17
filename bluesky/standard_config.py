"""
This module creates an instance of the RunEngine and configures it. None of
this is *essential* but it is extremely useful and generally recommended.
"""
import os
import logging
import history
from .run_engine import RunEngine
from .legacy_scans import LegacyAscan, LegacyDscan, LegacyCount
from .register_mds import register_mds
from .callbacks import print_metadata
from matplotlib import pyplot as plt


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
RE.subscribe('start', print_metadata)

# Instantiate legacy API objects.
ascan = LegacyAscan(RE)
dscan = LegacyDscan(RE)
ct = LegacyCount(RE)


def rollcall():
    """Return a list of objects that look like hardware.

    This is crude -- it just looks for objects that have the methods 'read' and
    'describe'.
    """
    objs = []
    for obj in globals():
        if hasattr(obj, 'read') and hasattr(obj, 'describe'):
            objs.append(obj)
    return objs


def olog_wrapper(logbook, logbooks):
    """Wrap a olog logbook for use with RunEngine

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
