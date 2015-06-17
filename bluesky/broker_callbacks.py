
"""
See also bluesky/callbacks.py

Callbacks that require DataBroker are in this separate file to isolate the
dependency.
"""
import sys
import traceback
from dataportal import DataBroker as db
from filestore.retrieve import get_spec_handler
from filestore.odm_templates import Datum
from metadatastore.api import (find_event_descriptors, find_events,
                               find_run_stops, find_run_starts)


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


def validate_all_events(run_stop):
    """Dereference all filestore references in the events

    CAUTION: MIGHT BE VERY SLOW!!

    Parameters
    ----------
    run_stop : uid or RunStop document
    """
    try:
        run_start_uid = run_stop['run_start']
    except KeyError:
        # run_stop must be the run_start_uid OR THE USER DID NOT RESPECT THE
        # DOCSTRING!!! ERMAGHERD!!!one1!!
        run_stop, = find_run_stops(uid=run_stop)
        run_start_uid = run_stop.run_start_uid
    hdr = db[run_stop.run_start.uid]
    events = db.fetch_events(hdr)
    # if there are filestore references in the events then this will attempt
    # to dereference all of them and if it cannot find an entry, should raise
    # from filestore.
    while True:
        try:
            ev = next(events)
        except StopIteration:
            break
        except Exception as e:
            print("Accessing event (%d) %s raised an Exception. Printing "
                  "the exception and continuing" % (ev.seq_num, ev.uid))
            print("Exception raised: %s" % e)
            tb_list = traceback.extract_tb(sys.exc_info()[2])
            for item in traceback._format_list_iter(tb_list):
                print('\t%s' % item)


def _uses_filestore(data_key):
    if 'external' in data_key:
        if data_key['external'].contains("FILESTORE"):
            return True
    return False


def validate_all_resources(run_stop):
    """Open up all resources that this run_stop corresponds to

    Should be no slower than validate_all_events and in some cases this will
    be much much faster

    Developer Warning: This touches touch the guts of filestore...

    Parameters
    ----------
    run_stop : uid or RunStop document
    """
    try:
        run_start_uid = run_stop['run_start']
    except KeyError:
        # run_stop must be the run_start_uid OR THE USER DIDNT RESPECT THE
        # DOCSTRING!!! ERMAGHERD!!!one1!!
        run_stop, = find_run_stops(uid=run_stop)
        run_start_uid = run_stop.run_start_uid

    descriptors = find_event_descriptors(run_start_uid=run_start_uid)
    resource_cache = set()
    for desc in descriptors:
        # Find which data keys, if any, use filestore.
        keys = [key for key in desc['data_keys'] if _uses_filestore(key)]
        if keys:
            # Loop through Events for this descriptor and try retrieving
            # the data for each of the keys.
            events = find_events(descriptor_uid=desc.uid)
            for event in events:
                for key in keys:
                    datum = Datum.objects(__raw__={'uid': event[key]})[0]
                    resource = datum.resource
                    if resource.uid not in resource_cache:
                        resource_cache.add(resource.uid)
                        # open the resource!
                        get_spec_handler(resource)
