import logging
import time as ttime
from collections import Iterable, ChainMap

import streamz
import jsonschema
import numpy as np
from event_model import DocumentNames, schemas

from .core import CallbackBase
from ..run_engine import Dispatcher
from ..utils import new_uid

logger = logging.getLogger(__name__)


class LiveStream(CallbackBase):
    """Create a secondary event stream of processed data"""
    def __init__(self, in_node, out_node=None):
        self.in_node = in_node
        # Use the same node as an output if None is given
        self.out_node = out_node or self.in_node
        # The output should propagate the modified Event document down the
        # pipeline
        self.event = self.in_node.emit
        self.out_node.sink(self.send_event)
        # Public dispatcher for callbacks
        self.dispatcher = Dispatcher()
        # Local caches for internal use
        self.seq_count = 0  # Maintain our own sequence count for this stream
        self._stream_start_uid = None  # Generated start doc uid
        self._raw_descriptors = dict()  # List of past raw event descriptors
        self._described = list()  # List of raw event desc. we have translated
        self._last_descriptor = None  # List of sent descriptors

    def start(self, doc):
        """Receive a stop document, re-emit it for this event stream"""
        self._stream_start_uid = new_uid()
        # Create a new start document with a new uid, start time, and the uid
        # of the original start document. Preserve the rest of the metadata
        # that we retrieved from the start document
        md = ChainMap({'uid': self._stream_start_uid,
                       'original_run_uid': doc['uid'],
                       'time': ttime.time()},
                      doc)
        # Emit the start document for anyone subscribed to our Dispatcher
        self.emit(DocumentNames.start, dict(md))

    def descriptor(self, doc):
        """Receive a descriptor and store for later reference"""
        self._raw_descriptors[doc['uid']] = doc
        super().descriptor(doc)

    def send_event(self, doc):
        """
        Receive an event document

        If we have not released a description yet, we can view the event
        document and the last description we received from the raw RunEngine
        and create a new updated description document. The event document is
        also modified to have a new sequence number and description source so
        that it is not confused with the raw data stream being emitted from the
        RunEngine
        """
        # If we haven't described this configuration
        # Send a new document to our subscribers
        if doc['descriptor'] not in self._described:
            prior_desc = self._raw_descriptors[doc['descriptor']]
            # Create a new description document for the output of the stream
            data_keys = dict()
            # Parse the event document creating a new description. If the key
            # existed in the original source description, just assume that it
            # is the same type, units and shape. Otherwise do some
            # investigation
            for key, val in doc['data'].items():
                # Described priorly
                if key in prior_desc['data_keys']:
                    key_desc = prior_desc['data_keys'][key]
                # String key
                elif isinstance(val, str):
                    key_desc = {'dtype': 'string',
                                'shape': []}
                # Iterable
                elif isinstance(val, Iterable):
                    key_desc = {'dtype': 'array',
                                'shape': np.shape(val)}
                # Number
                else:
                    key_desc = {'dtype': 'number',
                                'shape': []}
                # Modify the source
                key_desc['source'] = 'Stream: {}'.format(self.out_node.name
                                                         or 'Unnamed Stream')
                # Store in our new descriptor
                data_keys[key] = key_desc
            # Create our complete description document
            desc = ChainMap({'uid': new_uid(), 'time': ttime.time(),
                             'run_start': self._stream_start_uid,
                             'object_keys': {'stream':
                                             list(data_keys.keys())}},
                            prior_desc)
            # Store information about our descriptors
            self._last_descriptor = dict(desc)
            self._described.append(doc['descriptor'])
            # Emit the document to all subscribers
            self.emit(DocumentNames.descriptor, self._last_descriptor)

        # Clean the Event document produced by graph network. The data is left
        # untouched, but the relevant uids, timestamps, seq_num are modified so
        # that this event is not confused with the raw data stream
        self.seq_count += 1
        current_time = ttime.time()
        evt = ChainMap({'uid': new_uid(),
                        'descriptor': self._last_descriptor['uid'],
                        'timestamps': dict((key, current_time)
                                           for key in doc['data'].keys()),
                        'seq_num': self.seq_count, 'time': current_time},
                       doc)
        # Emit the event document
        self.emit(DocumentNames.event, dict(evt))

    def stop(self, doc):
        """Receive a stop document, re-emit it for this event stream"""
        # Create a new stop document with a new_uid, pointing to the correct
        # start document uid, and tally the number of events we have emitted.
        # The rest of the stop information is passed on to the next callback
        md = ChainMap(dict(run_start=self._stream_start_uid,
                           time=ttime.time(), uid=new_uid(),
                           num_events=self.seq_count),
                      doc)
        self.emit(DocumentNames.stop, dict(md))
        # Clear the local caches for the run
        self._raw_descriptors.clear()
        self._described.clear()
        self.seq_count = 0
        self._last_descriptor = None
        self._stream_start_uid = None

    def emit(self, name, doc):
        """Emit a document, first checking the schema"""
        jsonschema.validate(doc, schemas[name])
        self.dispatcher.process(name, doc)

    def subscribe(self, func, name='all'):
        """Convenience function for dispatcher subscription"""
        return self.dispatcher.subscribe(func, name)

    def unsubscribe(self, token):
        """Convenience function for dispatcher un-subscription"""
        self.dispatcher.unsubscribe(token)


class AverageStream(LiveStream):
    """Stream that averages data points together"""
    def __init__(self, n):
        self.n = n
        # Define our nodes
        in_node = streamz.Source(stream_name='Input')
        self._averager = in_node.partition(n)

        def average_events(cache):
            average_evt = dict()
            desc_id = cache[0]['descriptor']
            # Check that all of our events came from the same configuration
            if not all([desc_id == evt['descriptor'] for evt in cache]):
                raise Exception('The events in this bundle are from '
                                'different configurations!')
            # Use the last descriptor to avoid strings and objects
            data_keys = self._raw_descriptors[desc_id]['data_keys']
            for key, info in data_keys.items():
                # Information from non-number fields is dropped
                if info['dtype'] in ('number', 'array'):
                    # Average together
                    average_evt[key] = np.mean([evt['data'][key]
                                                for evt in cache], axis=0)
            return {'data': average_evt, 'descriptor': desc_id}

        out_node = self._averager.map(average_events)
        out_node.name = 'Average %s'.format(self.n)
        # Initialize the Stream
        super().__init__(in_node=in_node, out_node=out_node)
