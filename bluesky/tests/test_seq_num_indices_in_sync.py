from itertools import chain
from random import randint, sample
from typing import Dict, Iterator

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
    previous_random_width = 0
    new_random_width = 0

    def __init__(self, detectors=None):

        self.detectors = detectors or ["det1", "det2", "det3"]
        self.compose_stream_resource = ComposeStreamResource()
        self.stream_resource_compose_datum_pairs = tuple(
            self.compose_stream_resource("", "", f"non_existent_{det}.hdf5", det, {}) for det in self.detectors
        )

    def collect_resources(self) -> Iterator[Resource]:
        for stream_resource, _ in self.stream_resource_compose_datum_pairs:
            yield ("stream_resource", stream_resource)

    def collect_stream_datum(self, number_of_chunks: int, number_of_frames: int) -> Iterator[Asset]:
        # To simulate jitter
        sequence_counter_at_chunks = sorted(sample(range(1, number_of_frames), number_of_chunks - 1))

        for i in range(number_of_chunks - 1):
            if i == 0:
                self.new_random_width = sequence_counter_at_chunks[i]
            else:
                self.new_random_width = sequence_counter_at_chunks[i] - sequence_counter_at_chunks[i-1]

            for _, compose_stream_datum in self.stream_resource_compose_datum_pairs:
                indices_start = randint(0, 250)
                yield (
                    "stream_datum",
                    compose_stream_datum(
                        indices=StreamRange(start=indices_start, stop=indices_start + self.new_random_width)
                    )
                )

        # The last chunk finishes at seq_num: stop: 100
        self.new_random_width = number_of_frames - sequence_counter_at_chunks[number_of_chunks - 2]
        for _, compose_stream_datum in self.stream_resource_compose_datum_pairs:
            indices_start = randint(0, 250)
            yield (
                "stream_datum",
                compose_stream_datum(
                    indices=StreamRange(start=indices_start, stop=indices_start + self.new_random_width)
                )
            )


def collect_external(self):
    external_asset_device = ExternalAssetDevice()
    yield from external_asset_device.collect_resources()
    yield from external_asset_device.collect_stream_datum(10, 100)


def collect_external_one_detector(self):
    external_asset_device = ExternalAssetDevice(detectors=["det2"])
    yield from external_asset_device.collect_resources()
    yield from external_asset_device.collect_stream_datum(10, 100)


def collect_external_three_detectors_mismatched_indices(self):
    external_asset_device = ExternalAssetDevice()
    yield from external_asset_device.collect_resources()

    yield from external_asset_device.collect_stream_datum(9, 100)

    # Do the same as the above test, but our detectors give mismatched indices in the last
    # chunk of stream_datums
    yield from (
        (
            "stream_datum",
            compose_stream_datum(
                indices=StreamRange(
                    start=0,
                    stop=external_asset_device.new_random_width + i
                )
            )
        )
        for i, (_, compose_stream_datum) in enumerate(external_asset_device.stream_resource_compose_datum_pairs)
    )


def collect_Pageable_without_name(self) -> SyncOrAsync[Iterator[PartialEventPage]]:

    def timestamps():
        return {det: [1.0] for det in ["det1", "det2", "det3"]}

    def data(name):
        return {det: [name] for det in ["det1", "det2", "det3"]}

    partial_event_pages = [
        PartialEventPage(
            timestamps=timestamps(), data=data(name)
        )
        for name in chain(["primary"], range(16))
    ]

    return partial_event_pages


def collect_Pageable_with_name(self) -> SyncOrAsync[Iterator[PartialEventPage]]:
    def timestamps():
        return {"det1": [1.1, 1.2], "det2": [2.2, 2.3], "det3": [3.3, 3.4]}

    def data():
        return {"det1": [4321, 5432], "det2": [1234, 2345], "det3": [0, 1]}

    partial_event_pages = [
        PartialEventPage(
            timestamps=timestamps(), data=data()
        )
        for x in range(7)
    ]

    return partial_event_pages


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
    return dict(det2=dict(value=1.2, timestamp=0.0))


def describe_Readable(self) -> Dict[str, DataKey]:
    return dict(
        det1=dict(source="hw1", dtype="number", shape=[], external="STREAM:"),
        det2=dict(source="hw2", dtype="number", shape=[])
    )


def describe_with_name(self) -> SyncOrAsync[Dict[str, DataKey]]:
    data_keys = {
        str(det): DataKey(shape=[], source="stream1", dtype="string")
        for det in ["det1", "det2", "det3"]
    }

    return data_keys


def test_flyscan_with_stream_datum_pause(RE):
    class DummyDeviceFlyableEventPageCollectablePausable(
        Flyable, EventPageCollectable, Pausable
    ):
        kickoff = kickoff
        complete = complete
        pause = pause
        resume = resume
        collect_pages = collect_Pageable_with_name
        collect_asset_docs = collect_external
        describe_collect = describe_with_name
        name = "x"

    x = DummyDeviceFlyableEventPageCollectablePausable()
    collector = []

    with pytest.raises(RunEngineInterrupted):
        RE(
            [
                Msg("open_run", x),
                Msg("kickoff", x),
                Msg("pause", x),
                Msg("collect", x, name="primary"),
                Msg("complete", x),
                Msg("close_run", x),
            ],
            lambda *args: collector.append(args)
        )

    RE.resume()

    collector_iter = iter(collector)
    stream_resource_collected = []
    while not stream_resource_collected and collector_iter:
        next_collected_doc = next(collector_iter)
        if next_collected_doc[0] == "stream_resource":
            for _ in range(3):
                stream_resource_collected.append(next_collected_doc)
                next_collected_doc = next(collector_iter)

    assert stream_resource_collected
    for doc in stream_resource_collected:
        assert doc[0] == "stream_resource"

    stream_datum_collected_docs = [doc[1] for doc in collector if doc[0] == "stream_datum"]
    assert len(stream_datum_collected_docs) == 30

    # There are 30 stream datum all together, every 3 will have the same seq_nums
    for i in range(0, 30, 3):
        stream_datum_chunk = stream_datum_collected_docs[i:i + 3]
        seq_nums = stream_datum_chunk[0]["seq_nums"]
        for stream_datum in stream_datum_chunk[-2:]:
            assert stream_datum["seq_nums"] == seq_nums

    assert stream_datum_collected_docs[-1]["seq_nums"]["stop"] == 101


def test_flyscan_with_mismatched_indices(RE):
    class DummyDeviceFlyableEventPageCollectablePausableMismatchedStreamDatumIndices(
        Flyable, EventPageCollectable, Pausable
    ):
        kickoff = kickoff
        complete = complete
        pause = pause
        resume = resume
        collect_pages = collect_Pageable_with_name
        collect_asset_docs = collect_external_three_detectors_mismatched_indices
        describe_collect = describe_with_name
        name = "x"

    x = DummyDeviceFlyableEventPageCollectablePausableMismatchedStreamDatumIndices()
    collector = []

    with pytest.raises(RunEngineInterrupted):
        RE(
            [
                Msg("open_run", x),
                Msg("kickoff", x),
                Msg("pause", x),
                Msg("collect", x, name="primary"),
                Msg("complete", x),
                Msg("close_run", x),
            ],
            lambda *args: collector.append(args)
        )

    with pytest.raises(EventModelValueError):
        RE.resume()


def test_rd_desc_with_declare_stream(RE):
    class DummyDeviceReadableEvenPageCollectablePausableExternalAssets(
        Readable, EventPageCollectable, Pausable, WritesExternalAssets
    ):
        read = read_Readable
        describe = describe_Readable
        collect_pages = collect_Pageable_with_name
        pause = pause
        resume = resume
        collect_asset_docs = collect_external
        describe_collect = describe_with_name
        name = "x"

    x = DummyDeviceReadableEvenPageCollectablePausableExternalAssets()
    collector = []

    with pytest.raises(RunEngineInterrupted):
        RE(
            [
                Msg("open_run", x),
                Msg("create", name="x"),
                Msg("declare_stream", None, x, collect=True, name="primary"),
                Msg("read", x),
                Msg("pause", x),
                Msg("save", x),
                Msg("close_run", x),
            ],
            lambda *args: collector.append(args)
        )
    RE.resume()

    stream_datums_in_collector = [doc[1] for doc in collector if doc[0] == "stream_datum"]
    assert len(stream_datums_in_collector) == 30
    assert stream_datums_in_collector[-1]["seq_nums"]["stop"] == 101
