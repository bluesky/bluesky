import metadataclient.api as mdc
from bluesky.run_engine import DocumentNames


__all__ = ['register_mdc']


# For why this function is necessary, see
# http://stackoverflow.com/a/13355291/1221924
def _make_insert_func(func):
    def inserter(name, doc):
        return func(**doc)
    return inserter


def _insert_bulk_events(name, doc):
    """Bulk insert each event stream in doc."""
    for desc_uid, events in doc.items():
        if events:
            mdc.bulk_insert_events(desc_uid, events)


insert_funcs = {DocumentNames.event: _make_insert_func(mdc.insert_event),
                DocumentNames.bulk_events: _insert_bulk_events,
                DocumentNames.descriptor:
                    _make_insert_func(mdc.insert_descriptor),
                DocumentNames.start: _make_insert_func(mdc.insert_run_start),
                DocumentNames.stop: _make_insert_func(mdc.insert_run_stop)}


def register_mdc(runengine):
    """
    Register metadatastore insert_* functions to consume documents from scan.

    Parameters
    ----------
    scan : ophyd.plans.Scan
    """
    for name in insert_funcs.keys():
        runengine._subscribe_lossless(name, insert_funcs[name])
