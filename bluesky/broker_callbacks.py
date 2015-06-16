
"""
See also bluesky/callbacks.py

Callbacks that require DataBroker are in this separate file to isolate the
dependency.
"""

from dataportal import DataBroker as db
from filestore.api import retrieve
from metadatastore.commands import (find_event_descriptors, find_run_starts,
                                    find_events)


def post_run(callback):
    """
    Trigger a callback to process all the Documents from a run at the end.

    This function does not receive the Document stream during collection.
    It retrieves the complete set of Documents from the DataBroker after
    collection is complete.

    Parameters
    ----------
    callback : callable
        a function that accepts all four Documents

    Returns
    -------
    func : function
        a function that acepts a RunStop Document

    Examples
    --------
    Print a table with full (lossless) result set at the end of a run.

    >>> s = Ascan(motor, [det1], [1,2,3])
    >>> table = LiveTable(['det1', 'motor'])
    >>> RE(s, {'stop': post_run(table)})
    +------------+-------------------+----------------+----------------+
    |   seq_num  |             time  |          det1  |         motor  |
    +------------+-------------------+----------------+----------------+
    |         3  |  14:02:32.218348  |          5.00  |          3.00  |
    |         2  |  14:02:32.158503  |          5.00  |          2.00  |
    |         1  |  14:02:32.099807  |          5.00  |          1.00  |
    +------------+-------------------+----------------+----------------+
    """
    def f(stop_doc):
        uid = stop_doc['run_start']
        start, = find_run_starts(uid=uid)
        descriptors = find_event_descriptors(run_start=uid)
        # For convenience, I'll rely on the broker to get Events.
        header = db[uid]
        events = db.fetch_events(header)
        callback.start(start)
        for d in descriptors:
            callback.descriptor(d)
        for e in events:
            callback.event(e)
        callback.stop(stop_doc)
    return f


def filestore_validation(run_stop):
    uid = run_stop['run_start_uid']
    hdr = db[uid]
    events = db.fetch_events(hdr)
    # if there are filestore references in the events then this will attempt
    # to dereference all of them and if it cannot find an entry, should raise
    # from filestore.
    list(events)
