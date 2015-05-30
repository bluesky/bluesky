import metadatastore.commands as mdscmd
import time as ttime


def _make_blc():
    return mdscmd.insert_beamline_config({}, time=ttime.time())


def insert_run_start(doc):
    "Add a beamline config that, for now, only knows the time."
    doc['beamline_config'] = _make_blc()
    mdscmd.insert_run_start(**doc)


def insert_event(doc):
    mdscmd.insert_event(**doc)

def insert_event_descriptor(doc):
    mdscmd.insert_event_descriptor(**doc)

def insert_run_stop(doc):
    mdscmd.insert_run_stop(**doc)


insert_funcs = {'event': insert_event,
                'descriptor': insert_event_descriptor,
                'start': insert_run_start,
                'stop': insert_run_stop}


def register_mds(runengine):
    """
    Register metadatastore insert_* functions to consume documents from scan.

    Parameters
    ----------
    scan : ophyd.scans.Scan
    """
    for name in insert_funcs.keys():
        runengine._register_scan_callback(name, insert_funcs[name])
