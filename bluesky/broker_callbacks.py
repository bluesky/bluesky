"""
See also bluesky/callbacks.py

Callbacks that require DataBroker are in this separate file to isolate the
dependency.
"""

from dataportal import DataBroker as db
from metadatastore.commands import find_event_descriptors
from filestore.api import retrieve


def filestore_validation(run_stop):
    uid = run_stop['run_start_uid']
    descriptors = find_event_descriptors(run_start_uid=uid)
    for desc in descriptors:
        # Find which data keys, if any, use filestore.
        keys = [key for key in desc['data_keys'] if uses_filestore(key)]
        if keys:
            # Loop through Events for this descriptor and try retrieving
            # the data for each of the keys.
            events = find_events(descriptor_uid=desc.uid)
            for event in events:
                for key in keys:
                    retrieve(event[key])


def _uses_filestore(data_key):
    if 'external' in data_key:
        if data_key['external'].contains("FILESTORE"):
            return True 
    return False
