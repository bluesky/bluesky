import functools
from collections import defaultdict

import pytest
from event_model import EventModelValueError
from event_model.documents.event_descriptor import DataKey
from event_model.documents.stream_datum import StreamDatum
from event_model.documents.stream_resource import StreamResource
from ophyd.sim import StatusBase

from bluesky.plans import fly


def call_counter(func):
    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        self.call_counts[func.__name__] += 1
        return ret

    return inner


class AlwaysSucceedsStatus(StatusBase):
    @property
    def success(self):
        return True


class StreamDatumFlyer:
    """A flyer writing StreamDatum that may optionally emit ``extra`` more indices than requested."""

    parent = None

    def __init__(self, name, extra=0):
        self.name = name
        self.extra = extra
        self.counter = 0
        self.call_counts = defaultdict(int)
        self.complete_timeout = 0.5

    def kickoff(self):
        return AlwaysSucceedsStatus(timeout=self.complete_timeout)

    @call_counter
    def complete(self):
        return AlwaysSucceedsStatus(timeout=self.complete_timeout)

    def get_index(self):
        return self.call_counts["collect_asset_docs"] + 1

    def describe(self):
        return {f"{self.name}-sd": DataKey(source="file", dtype="number", shape=[], external="STREAM:")}

    describe_collect = describe

    def read(self):
        return {}

    @call_counter
    def collect_asset_docs(self, index=None):
        index = (index or self.get_index()) + self.extra
        data_key = f"{self.name}-sd"
        uid = f"{data_key}-uid"
        if self.counter == 0:
            yield (
                "stream_resource",
                StreamResource(  # type: ignore[typeddict-item]
                    resource_kwargs={"dataset": f"/{data_key}/data"},
                    data_key=data_key,
                    root="/root",
                    resource_path="/path.h5",
                    spec="AD_HDF5_SWMR_STREAM",
                    uid=uid,
                ),
            )
        yield (
            "stream_datum",
            StreamDatum(
                stream_resource=uid,
                descriptor="",
                uid=f"{uid}/{self.counter}",
                indices={"start": self.counter, "stop": self.counter + index},
                seq_nums={"start": 0, "stop": 0},
            ),
        )
        self.counter += index


def test_fly_plan_no_flush_period(RE):
    slow_flyer = StreamDatumFlyer(name="slow_flyer")

    RE(fly([slow_flyer], stream_name="stream_name"))

    assert slow_flyer.call_counts["complete"] == 1
    assert slow_flyer.call_counts["collect_asset_docs"] == 1


def test_fly_plan_with_flush_period_set(RE):
    slow_flyer = StreamDatumFlyer(name="slow_flyer")

    RE(fly([slow_flyer], collect_flush_period=0.1, stream_name="stream_name"))

    assert slow_flyer.call_counts["complete"] == 1
    assert slow_flyer.call_counts["collect_asset_docs"] == pytest.approx(slow_flyer.complete_timeout / 0.1, rel=2)


def test_fly_disparate_event_counts_into_one_stream_fails(RE):
    """Flying two detectors with a stream name creates one stream, so detectors
    producing different numbers of StreamDatum indices cannot be forced into it."""
    det1 = StreamDatumFlyer(name="det1", extra=0)
    det2 = StreamDatumFlyer(name="det2", extra=2)  # det2 produces 2 more indices than det1
    with pytest.raises(
        EventModelValueError,
        match=r"are of a different width `\d+` than other detectors in the same collect\(\) or save\(\)",
    ):
        RE(fly([det1, det2], stream_name="main"))
