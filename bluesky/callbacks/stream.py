import time as ttime
from collections.abc import Iterable
from collections import ChainMap

import numpy as np
from event_model import DocumentNames, schema_validators

from .core import CallbackBase
from ..run_engine import Dispatcher
from ..utils import new_uid


class LiveDispatcher(CallbackBase):
    """
    A secondary event stream of processed data

    The LiveDipatcher base implementation does not change any of the data
    emitted, this task is left to sub-classes, but instead handles
    reimplementing a secondary event stream that fits the same schema demanded
    by the RunEngine itself. In order to reduce the work done by these
    processed data pipelines, the LiveDispatcher handles the nitty-gritty
    details of formatting the event documents. This includes creating new uids,
    numbering events and creating descriptors.

    The LiveDispatcher can be subscribed to using the same syntax as the
    RunEngine, effectively creating a small chain of callbacks

    .. code::

        # Create our dispatcher
        ld = LiveDispatcher()
        # Subscribe it to receive events from the RunEgine
        RE.subscribe(ld)
        # Subscribe any callbacks we desire to second stream
        ld.subscribe(LivePlot('det', x='motor'))
    """
    def __init__(self):
        # Public dispatcher for callbacks
        self.dispatcher = Dispatcher()
        # Local caches for internal use
        self.seq_count = 0  # Maintain our own sequence count for this stream
        self.raw_descriptors = dict()  # Store raw descriptors for use later
        self._stream_start_uid = None  # Generated start doc uid
        self._descriptors = dict()  # Dictionary of sent descriptors

    def start(self, doc, _md=None):
        """Receive a raw start document, re-emit it for the modified stream"""
        self._stream_start_uid = new_uid()
        _md = _md or dict()
        # Create a new start document with a new uid, start time, and the uid
        # of the original start document. Preserve the rest of the metadata
        # that we retrieved from the start document
        md = ChainMap({'uid': self._stream_start_uid,
                       'original_run_uid': doc['uid'],
                       'time': ttime.time()},
                      _md, doc)
        # Dispatch the start document for anyone subscribed to our Dispatcher
        self.emit(DocumentNames.start, dict(md))
        super().start(doc)

    def descriptor(self, doc):
        """Store a descriptor"""
        self.raw_descriptors[doc['uid']] = doc
        super().descriptor(doc)

    def event(self, doc, **kwargs):
        """
        Receive an event document from the raw stream.

        This should be reimplemented by a subclass.

        Parameters
        ----------
        doc : event

        kwargs:
            All keyword arguments are passed to :meth:`.process_event`
        """
        self.process_event(doc, **kwargs)
        return super().event(doc)

    def process_event(self, doc, stream_name='primary',
                      id_args=None, config=None):
        """
        Process a modified event document then emit it for the modified stream

        This will pass an Event document to the dispatcher. If we have received
        a new event descriptor from the original stream, or we have recieved a
        new set of `id_args` or `descriptor_id` , a new descriptor document is
        first issued and passed through to the dispatcher.  When issuing a new
        event, the new descriptor is given a new source field.

        Parameters
        ----------
        doc : event

        stream_name : str, optional
            String identifier for a particular stream

        id_args : tuple, optional
            Additional tuple of hashable objects to identify the stream

        config: dict, optional
            Additional configuration information to be included in the event
            descriptor

        Notes
        -----
        Any callback subscribed to the `Dispatcher` will receive these event
        streams.  If nothing is subscribed, these documents will not go
        anywhere.
        """
        id_args = id_args or (doc['descriptor'],)
        config = config or dict()
        # Determine the descriptor id
        desc_id = frozenset((tuple(doc['data'].keys()), stream_name, id_args))
        # If we haven't described this configuration
        # Send a new document to our subscribers
        if (stream_name not in self._descriptors or
                desc_id not in self._descriptors[stream_name]):
            # Create a new description document for the output of the stream
            data_keys = dict()
            # Parse the event document creating a new description. If the key
            # existed in the original source description, just assume that it
            # is the same type, units and shape. Otherwise do some
            # investigation
            raw_desc = self.raw_descriptors.get(doc['descriptor'], {})
            for key, val in doc['data'].items():
                # Described priorly
                if key in raw_desc['data_keys']:
                    key_desc = raw_desc['data_keys'][key]
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
                key_desc['source'] = 'Stream'
                # Store in our new descriptor
                data_keys[key] = key_desc
            # Create our complete description document
            desc = ChainMap({'uid': new_uid(), 'time': ttime.time(),
                             'run_start': self._stream_start_uid,
                             'data_keys': data_keys, 'configuration': config,
                             'object_keys': {'stream':
                                             list(data_keys.keys())}},
                            raw_desc)
            # Store information about our descriptors
            desc = dict(desc)
            if stream_name not in self._descriptors:
                self._descriptors[stream_name] = dict()
            self._descriptors[stream_name][desc_id] = desc
            # Emit the document to all subscribers
            self.emit(DocumentNames.descriptor, desc)

        # Clean the Event document produced by graph network. The data is left
        # untouched, but the relevant uids, timestamps, seq_num are modified so
        # that this event is not confused with the raw data stream
        self.seq_count += 1
        desc_uid = self._descriptors[stream_name][desc_id]['uid']
        current_time = ttime.time()
        evt = ChainMap({'uid': new_uid(), 'descriptor': desc_uid,
                        'timestamps': dict((key, current_time)
                                           for key in doc['data'].keys()),
                        'seq_num': self.seq_count, 'time': current_time},
                       doc)
        # Emit the event document
        self.emit(DocumentNames.event, dict(evt))

    def stop(self, doc, _md=None):
        """Receive a raw stop document, re-emit it for the modified stream"""
        # Create a new stop document with a new_uid, pointing to the correct
        # start document uid, and tally the number of events we have emitted.
        # The rest of the stop information is passed on to the next callback
        _md = _md or dict()
        num_events = dict((stream, len(self._descriptors[stream]))
                          for stream in self._descriptors.keys())
        md = ChainMap(dict(run_start=self._stream_start_uid,
                           time=ttime.time(), uid=new_uid(),
                           num_events=num_events),
                      doc)
        self.emit(DocumentNames.stop, dict(md))
        # Clear the local caches for the run
        self.seq_count = 0
        self.raw_descriptors.clear()
        self._descriptors.clear()
        self._stream_start_uid = None
        super().stop(doc)

    def emit(self, name, doc):
        """Check the document schema and send to the dispatcher"""
        schema_validators[name].validate(doc)
        self.dispatcher.process(name, doc)

    def subscribe(self, func, name='all'):
        """Convenience function for dispatcher subscription"""
        return self.dispatcher.subscribe(func, name)

    def unsubscribe(self, token):
        """Convenience function for dispatcher un-subscription"""
        self.dispatcher.unsubscribe(token)
