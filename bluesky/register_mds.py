import uuid
from bluesky.run_engine import DocumentNames


__all__ = ['register_mds', 'register_mdc']


_GLOBAL_IF_CACHE = {}


def register_mdc(runengine):
    """Register metadataclient insert_* functions to consume documents from scan.

    Parameters
    ----------
    runengine : RunEngine
    """
    import metadataclient.commands as mds
    _register_md(runengine, mds)


def register_mds(runengine):
    """Register metadatastore insert_* functions to consume documents from scan.

    Parameters
    ----------
    runengine : RunEngine
    """
    import metadatastore.commands as mds
    _register_md(runengine, mds)


def _register_md(runengine, mds):
    # For why this function is necessary, see
    # http://stackoverflow.com/a/13355291/1221924
    def _make_insert_func(func):
        # we need these inner function do discarded the name
        def inserter(name, doc):
            return func(**doc)
        return inserter

    # we need this function to un-pack the nested bulk event documents
    def _insert_bulk_events(name, doc):
        """Bulk insert each event stream in doc."""
        for desc_uid, events in doc.items():
            # if events is empty, mongo chokes
            if events:
                mds.bulk_insert_events(desc_uid, events)

    insert_funcs = {DocumentNames.event: _make_insert_func(mds.insert_event),
                    DocumentNames.bulk_events: _insert_bulk_events,
                    DocumentNames.descriptor:
                        _make_insert_func(mds.insert_descriptor),
                    DocumentNames.start:
                        _make_insert_func(mds.insert_run_start),
                    DocumentNames.stop: _make_insert_func(mds.insert_run_stop)}

    # need to do this so we keep a ref to functions around so
    # Dispatcher does not drop them (as it only keeps a weakref)
    _GLOBAL_IF_CACHE[str(uuid.uuid4())] = insert_funcs

    # actually attach the callbacks to the RunEngine
    for name in insert_funcs.keys():
        runengine._subscribe_lossless(name, insert_funcs[name])
