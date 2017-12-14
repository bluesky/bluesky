import pytest

from bluesky.callbacks import CallbackCounter
from bluesky.examples import stepscan
from bluesky.tests.utils import DocCollector
from bluesky.callbacks import LiveTable

# Do not run these test if streamz is not installed
try:
    import streamz
    has_streamz = True
except ImportError:
    has_streamz = False
else:
    from bluesky.callbacks.stream import LiveStream, AverageStream


requires_streamz = pytest.mark.skipif(not has_streamz,
                                      reason='Missing streamz library')


@requires_streamz
def test_straight_through_stream(RE, hw):
    # Just a stream that sinks the events it receives
    s = streamz.Source()
    # Create callback chain
    ls = LiveStream(s)
    c = CallbackCounter()
    ls.subscribe(c)
    # Run a basic plan
    RE(stepscan(hw.det, hw.motor), {'all': ls})
    assert c.value == 10 + 1 + 2  # events, descriptor, start and stop


@requires_streamz
def test_average_stream(RE, hw):
    # Create callback chain
    ls = AverageStream(10)
    c = CallbackCounter()
    d = DocCollector()
    ls.subscribe(c)
    ls.subscribe(d.insert)
    # Run a basic plan
    RE.subscribe(LiveTable(('motor', 'det')))
    RE(stepscan(hw.det, hw.motor), {'all': ls})
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
