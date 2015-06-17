"""
See also bluesky/callbacks.py

Callbacks that require DataBroker are in this separate file to isolate the
dependency.
"""
import sys
import traceback
from dataportal import DataBroker as db


def validate_all_events(run_stop):
    uid = run_stop['run_start_uid']
    hdr = db[uid]
    events = db.fetch_events(hdr)
    # if there are filestore references in the events then this will attempt
    # to dereference all of them and if it cannot find an entry, should raise
    # from filestore.
    while True:
        try:
            e = next(events)
        except Exception as e:
            print("Accessing event (%d) %s raised the following Exception" %
                  (e.seq_num, e.uid))
            tb_list = traceback.extract_tb(sys.exc_info()[2])
            for item in traceback._format_list_iter(tb_list):
                print('\t%s' % item)
