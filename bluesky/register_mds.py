import metadatastore.api as mds
from metadatastore.commands import bulk_insert_events
import copy
import time as ttime
from bluesky.run_engine import DocumentNames


__all__ = ['register_mds']


# For why this function is necessary, see
# http://stackoverflow.com/a/13355291/1221924
def _make_insert_func(func):
    def inserter(name, doc):
        return func(**doc)
    return inserter


known_run_start_keys = ['time', 'scan_id', 'beamline_id', 'uid', 'owner',
                        'group', 'project']


def _insert_run_start(name, doc):
    """Add a beamline config that, for now, only knows the time."""
    # Move dynamic keys into 'custom' for MDS API.
    # We should change this in MDS to save the time of copying here:
    doc = copy.deepcopy(doc)
    for key in list(doc):
        if key not in known_run_start_keys:
            try:
                doc['custom']
            except KeyError:
                doc['custom'] = {}
            doc['custom'][key] = doc.pop(key)
    return mds.insert_run_start(**doc)


def _insert_bulk_events(name, doc):
    "Add a beamline config that, for now, only knows the time."
    for desc_uid, events in doc.items():
        bulk_insert_events(desc_uid, events)


insert_funcs = {DocumentNames.event: _make_insert_func(mds.insert_event),
                DocumentNames.bulk_events: _insert_bulk_events,
                DocumentNames.descriptor: _make_insert_func(
                    mds.insert_descriptor),
                DocumentNames.start: _insert_run_start,  # see above
                DocumentNames.stop: _make_insert_func(mds.insert_run_stop)}


def register_mds(runengine):
    """
    Register metadatastore insert_* functions to consume documents from scan.

    Parameters
    ----------
    scan : ophyd.scans.Scan
    """
    for name in insert_funcs.keys():
        runengine._register_scan_callback(name, insert_funcs[name])
