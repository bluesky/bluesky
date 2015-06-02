import metadatastore.api as mds
import time as ttime

__all__ = ['register_mds']


def _make_blc():
    return mds.insert_beamline_config({}, time=ttime.time())


# For why this function is necessary, see
# http://stackoverflow.com/a/13355291/1221924
def _make_insert_func(func):
    return lambda doc: func(**doc)


def _insert_run_start(doc):
    "Add a beamline config that, for now, only knows the time."
    doc['beamline_config'] = _make_blc()
    return _make_insert_func(mds.insert_run_start)(doc)


insert_funcs = {'event': _make_insert_func(mds.insert_event),
                'descriptor': _make_insert_func(mds.insert_event_descriptor),
                'start': _insert_run_start,  # special case; see above
                'stop': _make_insert_func(mds.insert_run_stop)}


def register_mds(runengine):
    """
    Register metadatastore insert_* functions to consume documents from scan.

    Parameters
    ----------
    scan : ophyd.scans.Scan
    """
    for name in insert_funcs.keys():
        runengine._register_scan_callback(name, insert_funcs[name])
