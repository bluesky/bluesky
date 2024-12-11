import math

from bluesky.callbacks import CallbackCounter
from bluesky.callbacks.stream import LiveDispatcher
from bluesky.examples import stepscan
from bluesky.tests.utils import DocCollector


class NegativeStream(LiveDispatcher):
    """Stream that only adds metadata to start document"""

    def start(self, doc):
        doc.update({"stream_level": "boring"})
        super().start(doc)

    def event(self, doc):
        modified = dict()  # noqa: C408
        for key, val in doc["data"].items():
            modified[f"modified_{key}"] = -math.fabs(val)
        doc["data"] = modified
        return super().event(doc)


def test_straight_through_stream(RE, hw):
    # Just a stream that sinks the events it receives
    ss = NegativeStream()
    # Create callback chain
    c = CallbackCounter()
    d = DocCollector()
    ss.subscribe(c)
    ss.subscribe(d.insert)
    # Run a basic plan
    RE(stepscan(hw.det, hw.motor), {"all": ss})
    # Check that our metadata is there
    assert c.value == 10 + 1 + 2  # events, descriptor, start and stop
    assert d.start[0]["stream_level"] == "boring"
    desc = d.descriptor[d.start[0]["uid"]][0]
    events = d.event[desc["uid"]]
    print(desc)
    print([evt["data"] for evt in events])
    tmp_valid = all([evt["data"][key] <= 0 for evt in events for key in evt["data"].keys()])  # noqa: C419
    assert tmp_valid
    tmp_valid = all([key in desc["data_keys"] for key in events[0]["data"].keys()])  # noqa: C419
    assert tmp_valid
