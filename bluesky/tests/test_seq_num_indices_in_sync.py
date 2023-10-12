from random import randint, sample
from typing import Dict, Iterator, List, Optional

import pytest
from event_model import ComposeStreamResource, EventModelValueError
from event_model.documents.event_descriptor import DataKey
from event_model.documents.event_page import PartialEventPage
from event_model.documents.resource import Resource
from event_model.documents.stream_datum import StreamRange

from bluesky import Msg
from bluesky.protocols import (Asset, EventPageCollectable, Flyable, Pausable,
                               Readable, Reading, SyncOrAsync,
                               WritesExternalAssets)
from bluesky.run_engine import RunEngineInterrupted


class ExternalAssetDevice:
    sequence_counter_at_chunks = None
    current_chunk = 0

    def __init__(
        self,
        number_of_chunks: int,
        number_of_frames: int,
        detectors: Optional[List[str]] = None,
        stream_datum_contains_one_index: bool = False
    ):
        self.detectors = detectors or ["det1", "det2", "det3"]
        self.compose_stream_resource = ComposeStreamResource()
        self.stream_resource_compose_datum_pairs = tuple(
            self.compose_stream_resource("", "", f"non_existent_{det}.hdf5", det, {}) for det in self.detectors
        )
        # Number of collect calls that will be made
        self.number_of_chunks = number_of_chunks

        # Number of frames in a detector in the run
        self.number_of_frames = number_of_frames

        if not stream_datum_contains_one_index:
            # To simulate jitter
            self.sequence_counter_at_chunks = sorted(
                sample(range(1, self.number_of_frames), self.number_of_chunks - 1)
            )

        else:
            # If there's no jitter and every index is {start:n, stop:n+1}
            self.sequence_counter_at_chunks = range(1, self.number_of_chunks)

        self.new_random_width = self.sequence_counter_at_chunks[0]

    def collect_resources(self) -> Iterator[Resource]:
        for stream_resource, _ in self.stream_resource_compose_datum_pairs:
            yield ("stream_resource", stream_resource)

    def collect_stream_datum(self) -> Iterator[Asset]:
        if self.current_chunk >= self.number_of_chunks:  # No more collect()
            return

        elif self.current_chunk == 0:  # First collect()
            yield from self.collect_resources()

        elif self.current_chunk == self.number_of_chunks - 1:  # Last collect()
            self.new_random_width = (
                self.number_of_frames
                - self.sequence_counter_at_chunks[self.number_of_chunks - 2]
            )
        else:
            self.new_random_width = (
                self.sequence_counter_at_chunks[self.current_chunk]
                - self.sequence_counter_at_chunks[self.current_chunk - 1]
            )

        self.current_chunk += 1

        for _, compose_stream_datum in self.stream_resource_compose_datum_pairs:
            indices_start = randint(0, 250)
            yield (
                "stream_datum",
                compose_stream_datum(
                    indices=StreamRange(start=indices_start, stop=indices_start + self.new_random_width)
                )
            )

    def collect_stream_datum_mismatched_indices(self):
        if self.current_chunk < self.number_of_chunks:
            yield from self.collect_stream_datum()

        indices_start = randint(0, 250)
        yield from (
            (
                "stream_datum",
                compose_stream_datum(
                    indices=StreamRange(
                        start=indices_start,
                        stop=indices_start + i
                    )
                )
            )
            for i, (_, compose_stream_datum)
            in enumerate(self.stream_resource_compose_datum_pairs)
        )

    def collect_stream_datum_repeat_stream_resource(self) -> Iterator[Asset]:
        if self.current_chunk == int(self.number_of_chunks/2):
            # New stream_resource half way through the run
            self.stream_resource_compose_datum_pairs = tuple(
                self.compose_stream_resource(
                    "", "", f"non_existent_{det}.hdf5", det, {}
                ) for det in self.detectors
            )
            yield from self.collect_resources()

        yield from self.collect_stream_datum()

    def collect_Pageable_with_name(self) -> SyncOrAsync[Iterator[PartialEventPage]]:
        def timestamps():
            return {"det4": [1] * self.new_random_width}

        def data():
            return {"det4": ["a"] * self.new_random_width}

        return [
            PartialEventPage(
                timestamps=timestamps(), data=data()
            )
        ]


def kickoff(self, *_, **__):
    class X:
        def add_callback(cls, *_, **__):
            ...
    return X


def complete(self, *_, **__):
    class X:
        def add_callback(cls, *_, **__):
            ...
    return X


def pause(self, *_, **__):
    ...


def resume(self, *_, **__):
    ...


def read_Readable(self) -> Dict[str, Reading]:
    return dict(det3=dict(value=1.2, timestamp=0.0))


def describe_Readable(self) -> Dict[str, DataKey]:
    return dict(
        det1=dict(source="hw1", dtype="number", shape=[], external="STREAM:"),
        det2=dict(source="hw1", dtype="number", shape=[], external="STREAM:"),
        det3=dict(source="hw2", dtype="number", shape=[])
    )


def describe_collect_with_name(self) -> SyncOrAsync[Dict[str, DataKey]]:
    description = {
        str(det): DataKey(shape=[], source="stream1", dtype="string", external="STREAM:")
        for det in ["det1", "det2", "det3"]
    }
    description.update({"det4": DataKey(shape=[], source="stream1", dtype="string")})
    return description


def test_flyscan_with_stream_datum_pause(RE):
    external_asset_device = ExternalAssetDevice(10, 100)

    class DummyDeviceFlyableEventPageCollectablePausable(
        Flyable, EventPageCollectable, Pausable
    ):
        kickoff = kickoff
        complete = complete
        pause = pause
        resume = resume
        collect_pages = external_asset_device.collect_Pageable_with_name
        collect_asset_docs = external_asset_device.collect_stream_datum
        describe_collect = describe_collect_with_name
        name = "x"

    x = DummyDeviceFlyableEventPageCollectablePausable()
    collector = []

    with pytest.raises(RunEngineInterrupted):
        RE(
            [
                Msg("open_run", x),
                Msg("kickoff", x),
                Msg("pause", x),
                *[Msg("collect", x, name="primary")] * 10,
                Msg("complete", x),
                Msg("close_run", x),
            ],
            lambda *args: collector.append(args)
        )

    RE.resume()

    stream_datum_collected_docs = [doc[1] for doc in collector if doc[0] == "stream_datum"]
    assert len(stream_datum_collected_docs) == 30

    collector_iter = iter(collector)
    stream_resource_collected = []
    # Check for contiguous stream_resource
    try:
        while collector_iter:
            next_collected_doc = next(collector_iter)
            if next_collected_doc[0] == "stream_resource":
                for _ in range(3):
                    assert next_collected_doc[0] == "stream_resource"
                    stream_resource_collected.append(next_collected_doc)
                    next_collected_doc = next(collector_iter)
    except StopIteration:
        ...

    assert stream_resource_collected
    assert len(stream_resource_collected) == 3

    # There are 30 stream datum all together, every 3 will have the same seq_nums
    for i in range(0, 30, 3):
        stream_datum_chunk = stream_datum_collected_docs[i:i + 3]
        seq_nums = stream_datum_chunk[0]["seq_nums"]
        for stream_datum in stream_datum_chunk[-2:]:
            assert stream_datum["seq_nums"] == seq_nums

    assert stream_datum_collected_docs[-1]["seq_nums"]["stop"] == 101


def test_flyscan_with_mismatched_indices(RE):

    external_asset_device = ExternalAssetDevice(9, 100)

    class DummyDeviceFlyableEventPageCollectablePausableMismatchedStreamDatumIndices(
        Flyable, EventPageCollectable, Pausable
    ):
        kickoff = kickoff
        complete = complete
        pause = pause
        resume = resume
        collect_pages = external_asset_device.collect_Pageable_with_name
        collect_asset_docs = external_asset_device.collect_stream_datum_mismatched_indices
        describe_collect = describe_collect_with_name
        name = "x"

    x = DummyDeviceFlyableEventPageCollectablePausableMismatchedStreamDatumIndices()
    collector = []
    plan = [
        Msg("open_run", x),
        Msg("kickoff", x),
        Msg("pause", x),
        *[Msg("collect", x, name="stream1")] * 10,
        Msg("complete", x),
        Msg("close_run", x),
    ]

    with pytest.raises(RunEngineInterrupted):
        RE(
            plan,
            lambda *args: collector.append(args)
        )

    with pytest.raises(EventModelValueError):
        RE.resume()


def test_rd_desc_with_declare_stream(RE):
    external_asset_device = ExternalAssetDevice(
        10, 10, detectors=["det1", "det2"], stream_datum_contains_one_index=True
    )

    class DummyDeviceReadablePausableExternalAssets(
        Readable, Pausable, WritesExternalAssets
    ):
        read = read_Readable
        describe = describe_Readable
        pause = pause
        resume = resume
        collect_asset_docs = external_asset_device.collect_stream_datum
        name = "x"

    x = DummyDeviceReadablePausableExternalAssets()
    collector = []
    plan = [
        Msg("open_run", x),
        *[
            Msg("create", name="hw1"),
            Msg("read", x),
            Msg("save", x),
        ] * 10,
        Msg("close_run", x),
    ]

    RE(
        plan,
        lambda *args: collector.append(args)
    )

    stream_datums_in_collector = [doc[1] for doc in collector if doc[0] == "stream_datum"]
    assert len(stream_datums_in_collector) == 20
    assert stream_datums_in_collector[-1]["seq_nums"]["stop"] == 11


def test_changing_stream_resource_after_stream_datum_emitted(RE):
    external_asset_device = ExternalAssetDevice(10, 100)

    class DummyDeviceFlyableEventPageCollectable(
        Flyable, EventPageCollectable
    ):
        kickoff = kickoff
        complete = complete
        collect_pages = external_asset_device.collect_Pageable_with_name
        collect_asset_docs = external_asset_device.collect_stream_datum_repeat_stream_resource
        describe_collect = describe_collect_with_name
        name = "x"

    x = DummyDeviceFlyableEventPageCollectable()
    collector = []
    plan = [
        Msg("open_run", x),
        Msg("kickoff", x),
        *[Msg("collect", x, name="primary")] * 10,
        Msg("complete", x),
        Msg("close_run", x),
    ]

    RE(
        plan,
        lambda *args: collector.append(args)
    )

    stream_datum_collected_docs = [doc[1] for doc in collector if doc[0] == "stream_datum"]
    assert len(stream_datum_collected_docs) == 30

    collector_iter = iter(collector)
    stream_resource_collected = []
    # Check for contiguous stream_resource
    try:
        while collector_iter:
            next_collected_doc = next(collector_iter)
            if next_collected_doc[0] == "stream_resource":
                for _ in range(3):
                    assert next_collected_doc[0] == "stream_resource"
                    stream_resource_collected.append(next_collected_doc)
                    next_collected_doc = next(collector_iter)
    except StopIteration:
        ...

    assert stream_resource_collected
    assert len(stream_resource_collected) == 6

    # There are 30 stream datum all together, every 3 will have the same seq_nums
    for i in range(0, 30, 3):
        stream_datum_chunk = stream_datum_collected_docs[i:i + 3]
        seq_nums = stream_datum_chunk[0]["seq_nums"]
        for stream_datum in stream_datum_chunk[-2:]:
            assert stream_datum["seq_nums"] == seq_nums

    assert stream_datum_collected_docs[-1]["seq_nums"]["stop"] == 101
