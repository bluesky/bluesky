import math
import pytest

import numpy as np

from bluesky.callbacks import CallbackCounter
from bluesky.examples import stepscan
from bluesky.tests.utils import DocCollector
from bluesky.callbacks.stream import LiveDispatcher

# Do not run these test if streamz is not installed
try:
    import streamz
    has_streamz = True
except ImportError:
    has_streamz = False

requires_streamz = pytest.mark.skipif(not has_streamz,
                                      reason='Missing streamz library')


class NegativeStream(LiveDispatcher):
    """Stream that only adds metadata to start document"""
    def start(self, doc):
        doc.update({"stream_level": "boring"})
        super().start(doc)

    def event(self, doc):
        modified = dict()
        for key, val in doc['data'].items():
            modified['modified_{}'.format(key)] = -math.fabs(val)
        doc['data'] = modified
        return super().event(doc)


class AverageStream(LiveDispatcher):
    """Stream that averages data points together"""
    def __init__(self, n=None):
        self.n = n
        self.in_node = None
        self.out_node = None
        self.averager = None
        super().__init__()

    def start(self, doc):
        """
        Create the stream after seeing the start document

        The callback looks for the 'average' key in the start document to
        configure itself.
        """
        # Grab the average key
        self.n = doc.get('average', self.n)
        # Define our nodes
        if not self.in_node:
            self.in_node = streamz.Source(stream_name='Input')

        self.averager = self.in_node.partition(self.n)

        def average_events(cache):
            average_evt = dict()
            desc_id = cache[0]['descriptor']
            # Check that all of our events came from the same configuration
            if not all([desc_id == evt['descriptor'] for evt in cache]):
                raise Exception('The events in this bundle are from '
                                'different configurations!')
            # Use the last descriptor to avoid strings and objects
            data_keys = self.raw_descriptors[desc_id]['data_keys']
            for key, info in data_keys.items():
                # Information from non-number fields is dropped
                if info['dtype'] in ('number', 'array', 'integer'):
                    # Average together
                    average_evt[key] = np.mean([evt['data'][key]
                                                for evt in cache], axis=0)
            return {'data': average_evt, 'descriptor': desc_id}

        self.out_node = self.averager.map(average_events)
        self.out_node.sink(super().event)
        super().start(doc)

    def event(self, doc):
        """Send an Event through the stream"""
        self.in_node.emit(doc)

    def stop(self, doc):
        """Delete the stream when run stops"""
        self.in_node = None
        self.out_node = None
        self.averager = None
        super().stop(doc)


def test_straight_through_stream(RE, hw):
    # Just a stream that sinks the events it receives
    ss = NegativeStream()
    # Create callback chain
    c = CallbackCounter()
    d = DocCollector()
    ss.subscribe(c)
    ss.subscribe(d.insert)
    # Run a basic plan
    RE(stepscan(hw.det, hw.motor), {'all': ss})
    # Check that our metadata is there
    assert c.value == 10 + 1 + 2  # events, descriptor, start and stop
    assert d.start[0]['stream_level'] == 'boring'
    desc = d.descriptor[d.start[0]['uid']][0]
    events = d.event[desc['uid']]
    print(desc)
    print([evt['data'] for evt in events])
    tmp_valid = all([evt['data'][key] <= 0
                     for evt in events
                     for key in evt['data'].keys()])
    assert tmp_valid
    tmp_valid = all([key in desc['data_keys']
                     for key in events[0]['data'].keys()])
    assert tmp_valid


@requires_streamz
def test_average_stream(RE, hw):
    # Create callback chain
    avg = AverageStream(10)
    c = CallbackCounter()
    d = DocCollector()
    avg.subscribe(c)
    avg.subscribe(d.insert)
    # Run a basic plan
    RE(stepscan(hw.det, hw.motor), {'all': avg})
    assert c.value == 1 + 1 + 2  # events, descriptor, start and stop
    # See that we made sensible descriptor
    start_uid = d.start[0]['uid']
    assert start_uid in d.descriptor
    desc_uid = d.descriptor[start_uid][0]['uid']
    assert desc_uid in d.event
    evt = d.event[desc_uid][0]
    assert evt['seq_num'] == 1
    assert all([key in d.descriptor[start_uid][0]['data_keys']
                for key in evt['data'].keys()])
    # See that we returned the correct average
    assert evt['data']['motor'] == -0.5  # mean of range(-5, 5)
    assert evt['data']['motor_setpoint'] == -0.5  # mean of range(-5, 5)
    assert start_uid in d.stop
    assert d.stop[start_uid]['num_events'] == {'primary': 1}
