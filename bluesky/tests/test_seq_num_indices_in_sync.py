from collections import deque
from itertools import chain
from random import randint
from typing import Dict, Iterator

import pytest
from event_model import ComposeStreamResource, EventModelValueError
from event_model.documents.event_descriptor import DataKey
from event_model.documents.event_page import PartialEventPage
from event_model.documents.stream_datum import StreamRange

from bluesky import Msg
from bluesky.protocols import (Asset, EventPageCollectable, Flyable, Pausable,
                               Readable, Reading, SyncOrAsync,
                               WritesExternalAssets)
from bluesky.run_engine import RunEngineInterrupted


class ExternalAssetDevice:
    DETECTORS = ["det1", "det2", "det3"]
    MAX_NO_OF_FRAMES_TO_COLLECT = 1000
    current_indices_stop = 0
    previous_random_width = 0

    compose_stream_resource = ComposeStreamResource()
    stream_resource, compose_stream_datum = compose_stream_resource("", "", "non_existent.hdf5", {})

    def __init__(self, max_no_of_frames_to_collect=None, detectors=None):
        self.MAX_NO_OF_FRAMES_TO_COLLECT = max_no_of_frames_to_collect or self.MAX_NO_OF_FRAMES_TO_COLLECT
        self.DETECTORS = detectors or self.DETECTORS

    def collect_from_each_detector(self) -> Iterator[Asset]:
        # To simulate jitter
        new_random_width = 0
        while new_random_width == self.previous_random_width:
            new_random_width = randint(1, 4)
        self.previous_random_width = new_random_width

        new_indices_start = self.current_indices_stop + 1
        new_indices_stop = new_indices_start + new_random_width
        self.current_indices_stop = new_indices_stop
        for detector in self.DETECTORS:
            yield (
                "stream_datum",
                self.compose_stream_datum(
                    data_keys=[detector],
                    indices=StreamRange(start=new_indices_start, stop=new_indices_stop)
                )
            )


def collect_stream_datum_one_detector(self):
    external_asset_device = ExternalAssetDevice(detectors=["det1"], max_no_of_frames_to_collect=100)
    while (
        external_asset_device.current_indices_stop < external_asset_device.MAX_NO_OF_FRAMES_TO_COLLECT - 10
    ):
        yield from external_asset_device.collect_from_each_detector()

    # Collect whatever frames are leftover, we know the total number of frames in the
    # flyscan but the jitter will be variable, the device keeps track and sends the same
    # seq_nums
    yield from (
        (
            "stream_datum",
            external_asset_device.compose_stream_datum(
                data_keys=[detector],
                indices=StreamRange(
                    start=external_asset_device.current_indices_stop + 1,
                    stop=external_asset_device.MAX_NO_OF_FRAMES_TO_COLLECT
                )
            )
        )
        for detector in external_asset_device.DETECTORS
    )


def collect_stream_datum_three_detectors(self):
    # Collect from each detector
    external_asset_device = ExternalAssetDevice()
    while (
        external_asset_device.current_indices_stop < external_asset_device.MAX_NO_OF_FRAMES_TO_COLLECT - 10
    ):
        yield from external_asset_device.collect_from_each_detector()

    # Collect whatever frames are leftover, we know the total number of frames in the
    # flyscan but the jitter will be variable, the device keeps track and sends the same
    # seq_nums
    yield from (
        (
            "stream_datum",
            external_asset_device.compose_stream_datum(
                data_keys=[detector],
                indices=StreamRange(
                    start=external_asset_device.current_indices_stop + 1,
                    stop=external_asset_device.MAX_NO_OF_FRAMES_TO_COLLECT
                )
            )
        )
        for detector in external_asset_device.DETECTORS
    )


def collect_stream_datum_three_detectors_mismatched_indices(self):
    external_asset_device = ExternalAssetDevice()
    while (
        external_asset_device.current_indices_stop < external_asset_device.MAX_NO_OF_FRAMES_TO_COLLECT - 15
    ):
        yield from external_asset_device.collect_from_each_detector()

    # Do the same as the above test, but insert a mismatched index in one of the detectors,
    # the last two frames are of different widths across the detectors

    yield from (
        (
            "stream_datum",
            external_asset_device.compose_stream_datum(
                data_keys=[detector],
                indices=StreamRange(
                    start=external_asset_device.current_indices_stop + 1,
                    stop=external_asset_device.MAX_NO_OF_FRAMES_TO_COLLECT - idx - 1
                )
            )
        )
        for idx, detector in enumerate(external_asset_device.DETECTORS)
    )
    yield from (
        (
            "stream_datum",
            external_asset_device.compose_stream_datum(
                data_keys=[detector],
                indices=StreamRange(
                    start=external_asset_device.MAX_NO_OF_FRAMES_TO_COLLECT - idx,
                    stop=external_asset_device.MAX_NO_OF_FRAMES_TO_COLLECT
                )
            )
        )
        for idx, detector in enumerate(external_asset_device.DETECTORS)
    )


def collect_stream_datum_three_detectors_too_many_indices_from_one_detector(self):
    external_asset_device = ExternalAssetDevice()
    while (
        external_asset_device.current_indices_stop < external_asset_device.MAX_NO_OF_FRAMES_TO_COLLECT - 10
    ):
        yield from external_asset_device.collect_from_each_detector()

    # Do the same as the above test, but insert a mismatched index in one of the detectors
    yield from (
        (
            "stream_datum",
            external_asset_device.compose_stream_datum(
                data_keys=[detector],
                indices=StreamRange(
                    start=external_asset_device.current_indices_stop + 1,
                    stop=external_asset_device.MAX_NO_OF_FRAMES_TO_COLLECT
                )
            )
        )
        for detector in ["det1, det2"]
    )

    # Create two extra seq_nums in det3 that aren't represented in det1 and det2
    yield from (
        (
            "stream_datum",
            external_asset_device.compose_stream_datum(
                data_keys=["det3"],
                indices=StreamRange(
                    start=start,
                    stop=stop
                )
            )
        )
        for start, stop in [
            (
                external_asset_device.current_indices_stop + 1,
                external_asset_device.current_indices_stop + 2,
            ),
            (
                external_asset_device.current_indices_stop + 3,
                external_asset_device.MAX_NO_OF_FRAMES_TO_COLLECT
            )
        ]
    )


def describe_without_name(self) -> SyncOrAsync[Dict[str, Dict[str, DataKey]]]:
    data_keys = {
        str(name): {
            str(det): DataKey(shape=[], source="stream1", dtype="string") for det in ["det1", "det2", "det3"]
        }
        for name in chain(["primary"], range(16))
    }
    return data_keys


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
        collect_pages = collect_Pageable_without_name
        collect_asset_docs = collect_stream_datum_three_detectors
        describe_collect = describe_without_name
        name = "x"

    x = DummyDeviceFlyableEventPageCollectablePausable()
    collector = []

    with pytest.raises(RunEngineInterrupted):
        RE(
            [
                Msg("open_run", x),
                Msg("kickoff", x),
                Msg("pause", x),
                Msg("collect", x),
                Msg("complete", x),
                Msg("close_run", x),
            ],
            lambda *args: collector.append(args)
        )

    RE.resume()

    stream_datums_in_collector = [doc[1] for doc in collector if doc[0] == "stream_datum"]

    det1_seq_nums, det2_seq_nums, det3_seq_nums = (
        deque([
            stream_datum["seq_nums"] for stream_datum in stream_datums_in_collector
            if stream_datum["data_keys"] == [det]
        ])
        for det in ["det1", "det2", "det3"]
    )

    # We know there'll be 1000 in total
    assert det1_seq_nums[-1]["stop"] == det2_seq_nums[-1]["stop"] == det3_seq_nums[-1]["stop"] == 999
    assert det1_seq_nums == det2_seq_nums == det3_seq_nums

    # Check that there is a difference between the seq_nums
    old_seq_nums = det1_seq_nums.popleft()
    old_seq_num_difference = old_seq_nums["stop"] - old_seq_nums["start"]
    seq_nums_are_different_widths = False
    while det1_seq_nums:
        new_seq_nums = det1_seq_nums.popleft()
        new_seq_num_difference = new_seq_nums["stop"] - new_seq_nums["start"]
        if old_seq_num_difference != new_seq_num_difference:
            seq_nums_are_different_widths = True
            break
        old_seq_num_difference = new_seq_num_difference

    assert seq_nums_are_different_widths


def test_flyscan_with_too_many_indices(RE):
    class DummyDeviceFlyableEventPageCollectablePausableTooManyIndices(
        Flyable, EventPageCollectable, Pausable
    ):
        kickoff = kickoff
        complete = complete
        pause = pause
        resume = resume
        collect_pages = collect_Pageable_without_name
        collect_asset_docs = collect_stream_datum_three_detectors_too_many_indices_from_one_detector
        describe_collect = describe_without_name
        name = "x"

    x = DummyDeviceFlyableEventPageCollectablePausableTooManyIndices()
    collector = []

    with pytest.raises(RunEngineInterrupted):
        RE(
            [
                Msg("open_run", x),
                Msg("kickoff", x),
                Msg("pause", x),
                Msg("collect", x),
                Msg("complete", x),
                Msg("close_run", x),
            ],
            lambda *args: collector.append(args)
        )

    with pytest.raises(EventModelValueError):
        RE.resume()


def test_flyscan_with_mismatched_indices(RE):
    class DummyDeviceFlyableEventPageCollectablePausableMismatchedStreamDatumIndices(
        Flyable, EventPageCollectable, Pausable
    ):
        kickoff = kickoff
        complete = complete
        pause = pause
        resume = resume
        collect_pages = collect_Pageable_without_name
        collect_asset_docs = collect_stream_datum_three_detectors_too_many_indices_from_one_detector
        describe_collect = describe_without_name
        name = "x"

    x = DummyDeviceFlyableEventPageCollectablePausableMismatchedStreamDatumIndices()
    collector = []

    with pytest.raises(RunEngineInterrupted):
        RE(
            [
                Msg("open_run", x),
                Msg("kickoff", x),
                Msg("pause", x),
                Msg("collect", x),
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
        collect_asset_docs = collect_stream_datum_one_detector
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
    from pprint import pprint
    pprint(stream_datums_in_collector)
