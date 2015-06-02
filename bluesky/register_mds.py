import metadatastore.commands as mdscmd
import time as ttime


def _make_blc():
    return mdscmd.insert_beamline_config({}, time=ttime.time())


# For why this function is necessary, see
# http://stackoverflow.com/a/13355291/1221924
def make_insert_func(func):
    return lambda doc: func(**doc)


def insert_run_start(doc):
    "Add a beamline config that, for now, only knows the time."
    doc['beamline_config'] = _make_blc()
    return make_insert_func(mdscmd.insert_run_start)(doc)


insert_funcs = {'event': make_insert_func(mdscmd.insert_event),
                'descriptor': make_insert_func(mdscmd.insert_event_descriptor),
                'start': insert_run_start,  # special case; see above
                'stop': make_insert_func(mdscmd.insert_run_stop)}


def register_mds(runengine):
    """
    Register metadatastore insert_* functions to consume documents from scan.

    Parameters
    ----------
    scan : ophyd.scans.Scan
    """
    for name in insert_funcs.keys():
        runengine._register_scan_callback(name, insert_funcs[name])
