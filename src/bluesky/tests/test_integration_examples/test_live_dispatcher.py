"""Testing external integrations provided in examples for LiveDispatcher with data streams
Examples are found in: docs/callbacks.rst
"""

import numpy as np
import pytest
from ophyd.sim import Cpt, SynGauss, SynSignal

from bluesky.callbacks import CallbackCounter
from bluesky.callbacks.stream import LiveDispatcher
from bluesky.examples import stepscan
from bluesky.tests.utils import DocCollector

reactivex = pytest.importorskip("reactivex")
pytest.importorskip("reactivex.operators")


class AverageStream(LiveDispatcher):
    """ReactiveX Implementation of AverageStream"""

    def __init__(self, n=None):
        self.n = n
        self.source = reactivex.subject.ReplaySubject()
        self.out_stream = None
        super().__init__()

    def start(self, doc):
        """
        Create the stream after seeing the start document

        The callback looks for the 'average' key in the start document to
        configure itself.
        """
        # Grab the average key
        self.n = doc.get("average", self.n)
        # Define our nodes
        if self.source is None:
            self.source = reactivex.subject.ReplaySubject()
        if not self.out_stream:
            self.out_stream = self.source.pipe(
                reactivex.operators.buffer_with_count(self.n), reactivex.operators.map(self._average_events)
            )
            self.out_stream.subscribe(super().event)
        super().start(doc)

    def _average_events(self, events):
        average_evt = {}
        desc_id = events[0]["descriptor"]
        if not all(desc_id == event["descriptor"] for event in events):
            raise Exception("Events are from different configurations!")
        data_keys = self.raw_descriptors[desc_id]["data_keys"]
        for key, info in data_keys.items():
            # Information from non-number fields is dropped
            if info["dtype"] in ("number", "array", "integer"):
                # Average together
                average_evt[key] = np.mean([evt["data"][key] for evt in events], axis=0)
            else:
                raise TypeError(f"Data Key {key} has invalid data type: {info['dtype']}")
        return {"data": average_evt, "descriptor": desc_id}

    def event(self, doc, **kwargs):
        self.source.on_next(doc)

    def stop(self, doc):
        """Delete the stream when run stops"""
        self.source.on_completed()
        self.out_stream = None
        self.source = None
        super().stop(doc)


def test_average_stream(RE, hw):
    # Create callback chain
    avg = AverageStream(10)
    c = CallbackCounter()
    d = DocCollector()
    avg.subscribe(c)
    avg.subscribe(d.insert)
    # Run a basic plan
    RE(stepscan(hw.det, hw.motor), {"all": avg})
    assert c.value == 1 + 1 + 2  # events, descriptor, start and stop
    # See that we made sensible descriptor
    start_uid = d.start[0]["uid"]
    assert start_uid in d.descriptor
    desc_uid = d.descriptor[start_uid][0]["uid"]
    assert desc_uid in d.event
    evt = d.event[desc_uid][0]
    assert evt["seq_num"] == 1
    assert all([key in d.descriptor[start_uid][0]["data_keys"] for key in evt["data"].keys()])  # noqa: C419
    # See that we returned the correct average
    assert evt["data"]["motor"] == -0.5  # mean of range(-5, 5)
    assert evt["data"]["motor_setpoint"] == -0.5  # mean of range(-5, 5)
    assert start_uid in d.stop
    assert d.stop[start_uid]["num_events"] == {"primary": 1}
    # Repeat for array data
    RE(stepscan(hw.direct_img, hw.motor), {"all": avg})
    assert c.value == 1 + 1 + 2 + 1 + 1 + 2  # events, descriptor, start and stop
    start_uid = d.start[1]["uid"]
    desc_uid = d.descriptor[start_uid][0]["uid"]
    evt = d.event[desc_uid][0]
    assert evt["data"]["img"].ndim == 2


class TestDet(SynGauss):
    string_cpt = Cpt(SynSignal, func=lambda: "string", kind="config")


def test_average_stream_errors(RE, hw):
    noisy_det = TestDet(
        "noisy_det",
        hw.motor,
        "motor",
        center=0,
        Imax=1,
        noise="uniform",
        sigma=1,
        noise_multiplier=0.1,
        labels={"detectors"},
    )

    # Create callback chain
    avg = AverageStream(10)
    c = CallbackCounter()
    d = DocCollector()
    avg.subscribe(c)
    avg.subscribe(d.insert)
    RE(stepscan(noisy_det, hw.motor), {"all": avg})
    assert c.value == 1 + 1 + 2  # events, descriptor, start and stop
    # Will riase if the string becomes hinted
    noisy_det.string_cpt.kind = "hinted"
    with pytest.raises(TypeError):
        RE(stepscan(noisy_det, hw.motor), {"all": avg})
