"""
See also bluesky/callbacks.py

Callbacks that require DataBroker are in this separate file to isolate the
dependency.
"""

from dataportal import DataBroker as db


def filestore_validation(run_stop):
    uid = run_stop['run_start_uid']
    hdr = db[uid]
    events = db.fetch_events(hdr)
    # if there are filestore references in the events then this will attempt
    # to dereference all of them and if it cannot find an entry, should raise
    # from filestore.
    list(events)
