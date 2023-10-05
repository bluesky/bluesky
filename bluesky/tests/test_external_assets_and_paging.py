from typing import Dict, Iterator
from bluesky import Msg
from event_model.documents.event_descriptor import DataKey
from event_model.documents.event_page import PartialEventPage
from event_model.documents.event import PartialEvent
from event_model.documents.resource import PartialResource
from event_model.documents.datum import Datum
from event_model.documents.stream_datum import StreamDatum
from event_model.documents.stream_resource import StreamResource
from bluesky.utils import new_uid, IllegalMessageSequence
from itertools import chain
import pytest
from bluesky.protocols import (
    Asset, Readable,
    Reading, SyncOrAsync, WritesExternalAssets, Flyable, EventCollectable, EventPageCollectable, Collectable
)


def read_Readable(self) -> Dict[str, Reading]:
    return dict(det2=dict(value=1.2, timestamp=0.0))


def describe_Readable(self) -> Dict[str, DataKey]:
    return dict(
        det1=dict(source="hw1", dtype="number", shape=[], external="STREAM:"),
        det2=dict(source="hw2", dtype="number", shape=[])
    )


def collect_asset_docs_Datum(self) -> Iterator[Asset]:
    resource_uid = new_uid()
    resource = PartialResource(
        resource_kwargs={"some_argument": 1337},
        root="awesome",
        spec=".dat",
        resource_path="no/really,/awesome",
        uid=resource_uid
    )

    yield "resource", resource
    datum = Datum(
        datum_id="some_resource_name/123123",
        resource=resource_uid,
        datum_kwargs={"some_argument": 1234},
    )
    yield "datum", datum


def collect_asset_docs_StreamDatum_one_detector(self) -> Iterator[Asset]:
    resource_uid = new_uid()

    stream_resource = StreamResource(
        resource_kwargs={"argument": 1},
        data_key="det1",
        resource_path="An/awesome/path",
        root="some_detail",
        spec=".hdf5",
        uid=resource_uid,
    )
    yield "stream_resource", stream_resource

    stream_datum = StreamDatum(
        stream_resource=resource_uid,
        descriptor="",
        uid=new_uid(),
        seq_nums={"start": 0, "stop": 0},
        indices={"start": 0, "stop": 1},
    )
    yield "stream_datum", stream_datum


def collect_asset_docs_StreamDatum(self) -> Iterator[Asset]:
    detectors = ["det1", "det2"]
    resource_uid = {det: new_uid() for det in detectors}
    for det in detectors:
        stream_resource = StreamResource(
            resource_kwargs={"argument": 1},
            data_key=det,
            resource_path="An/awesome/path",
            root="some_detail",
            spec=".hdf5",
            uid=resource_uid[det],
        )
        yield "stream_resource", stream_resource

    for det in detectors * 2:
        stream_datum = StreamDatum(
            stream_resource=resource_uid[det],
            descriptor="",
            uid=new_uid(),
            seq_nums={"start": 0, "stop": 0},
            indices={"start": 0, "stop": 1},
        )
        yield "stream_datum", stream_datum


@pytest.mark.parametrize(
    'asset_type,resource_type,collect_asset_docs_fun',
    [
        ("datum", "resource", collect_asset_docs_Datum),
        ("stream_datum", "stream_resource", collect_asset_docs_StreamDatum_one_detector),
        # Resources are received before collecting the datum
    ]
)
def test_rd_desc_different_asset_types(RE, asset_type, resource_type, collect_asset_docs_fun):
    class X(Readable, WritesExternalAssets):
        read = read_Readable
        describe = describe_Readable
        collect_asset_docs = collect_asset_docs_fun
        name = "x"

    x = X()
    collector = []
    RE(
        [
            Msg("open_run", x),
            Msg("create", name="primary"),
            Msg("read", x),
            Msg("save", x),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
    )

    assert len([x for x in collector if x[0] == "event"]) == 1
    assert len([x for x in collector if x[0] == "event_page"]) == 0
    datums = [x for x in collector if x[0] == asset_type]
    resources = [x for x in collector if x[0] == resource_type]
    assert len(datums) == 1
    assert len(resources) == 1


def describe_with_name_stream_detectors(self) -> SyncOrAsync[Dict[str, DataKey]]:
    data_keys = {
        str(det): DataKey(shape=[], source="stream1", dtype="string", external="STREAM:")
        for det in ["det1", "det2"]
    }
    data_keys.update({"det3": DataKey(shape=[], source="stream1", dtype="string")})

    return data_keys


def collect_Pageable_with_name(self) -> SyncOrAsync[Iterator[PartialEventPage]]:
    def timestamps():
        return {"det3": [3.3, 3.4]}

    def data():
        return {"det3": [0, 1]}

    partial_event_pages = [
        PartialEventPage(
            timestamps=timestamps(), data=data()
        )
        for x in range(7)
    ]

    return partial_event_pages


def describe_with_name(self) -> SyncOrAsync[Dict[str, DataKey]]:
    data_keys = {
        str(det): DataKey(shape=[], source="stream1", dtype="string")
        for det in ["det1", "det2", "det3"]
    }
    return data_keys


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


def collect_events_all_detectors(self) -> SyncOrAsync[Iterator[PartialEvent]]:
    def timestamps(x):
        return {"det1": x + 0.1, "det2": x + 0.2, "det3": x + 0.3}

    def data(x):
        return {"det1": x, "det2": x*2, "det3": x*3}

    partial_events = [
        PartialEvent(
            timestamps=timestamps(x), data=data(x), time=float(x)
        )
        for x in range(16)
    ]
    return partial_events


def kickoff_dummy_callback(self, *_, **__):
    class X:
        def add_callback(cls, *_, **__):
            ...
    return X


def complete_dummy_callback(self, *_, **__):
    class X:
        def add_callback(cls, *_, **__):
            ...
    return X


def test_flyscan_with_pages_with_no_name(RE):
    class X(Flyable, EventPageCollectable):
        kickoff = kickoff_dummy_callback
        complete = complete_dummy_callback
        collect_pages = collect_Pageable_without_name
        describe_collect = describe_without_name
        name = "x"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("kickoff", x),
            Msg("collect", x),
            Msg("complete", x),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )

    event_pages = [x for x in collector if x[0] == "event_page"]
    assert len(event_pages) == 17
    assert len(event_pages[0][1]["timestamps"]["det1"]) == 1


def test_flyscan_with_pages_passed_in_name(RE):
    class X(Flyable, EventPageCollectable):
        kickoff = kickoff_dummy_callback
        complete = complete_dummy_callback
        collect_pages = collect_Pageable_with_name
        describe_collect = describe_with_name_stream_detectors
        name = "x"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("collect", x, name="primary"),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )


@pytest.mark.parametrize(
    'asset_type,collect_asset_docs_fun',
    [
        ("datum", collect_asset_docs_Datum),
        ("stream_datum", collect_asset_docs_StreamDatum),
        # StreamResource is receieved in the collect of stream_datum
    ]
)
def test_flyscan_with_pages_with_no_name_and_external_assets(RE, asset_type, collect_asset_docs_fun):
    class X(Flyable, EventPageCollectable, WritesExternalAssets):
        kickoff = kickoff_dummy_callback
        complete = complete_dummy_callback
        collect_pages = collect_Pageable_with_name
        describe_collect = describe_with_name_stream_detectors
        collect_asset_docs = collect_asset_docs_fun
        name = "x"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("kickoff", x),
            Msg("collect", x, name="primary"),
            Msg("complete", x),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )

    event_pages_in_collector = [x[1] for x in collector if x[0] == "event_page"]
    assert len(event_pages_in_collector) == 7
    # One detector is collected in an event_page
    assert len(event_pages_in_collector[0]["data"]) == 1
    assert len(event_pages_in_collector[0]["data"]["det3"]) == 2

    if asset_type == "stream_datum":
        # Two detector are collected in stream_datum, each with two stream_datum
        datum_in_collector = [x[1] for x in collector if x[0] == asset_type]
        assert len(datum_in_collector) == 4


@pytest.mark.parametrize(
    'asset_type,collect_asset_docs_fun',
    [
        ("datum", collect_asset_docs_Datum),
        ("stream_datum", collect_asset_docs_StreamDatum),
        # StreamResource is receieved in the collect of stream_datum
    ]
)
def test_flyscan_with_pages_passed_in_name_and_external_assets(RE, asset_type, collect_asset_docs_fun):
    class X(Flyable, EventPageCollectable, WritesExternalAssets):
        kickoff = kickoff_dummy_callback
        complete = complete_dummy_callback
        collect_pages = collect_Pageable_with_name
        describe_collect = describe_with_name_stream_detectors
        collect_asset_docs = collect_asset_docs_fun
        name = "x"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("kickoff", x),
            Msg("complete", x),
            Msg("collect", x, name="1"),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )


def test_rd_desc_with_declare_stream(RE):
    class X(Readable, EventPageCollectable):
        read = read_Readable
        collect_pages = collect_Pageable_with_name
        describe = describe_Readable
        describe_collect = describe_with_name_stream_detectors
        name = "name"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("create", name="x"),
            Msg("declare_stream", None, x, collect=True, name="primary"),
            Msg("read", x),
            Msg("save", x),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )


def test_flyscan_without_pageing_or_name(RE):
    class X(Flyable, EventCollectable):
        kickoff = kickoff_dummy_callback
        complete = complete_dummy_callback
        collect = collect_events_all_detectors
        describe_collect = describe_without_name
        name = "x"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("kickoff", x),
            Msg("complete", x),
            Msg("collect", x),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )


def test_flyscan_without_paging_with_name(RE):

    class X(Flyable, EventCollectable):
        kickoff = kickoff_dummy_callback
        complete = complete_dummy_callback
        collect = collect_events_all_detectors
        describe_collect = describe_with_name
        name = "x"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("kickoff", x),
            Msg("complete", x),
            Msg("collect", x, name="primary"),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )


def test_describe_collect_with_stream(RE):
    class X(EventCollectable):
        collect = collect_events_all_detectors
        describe_collect = describe_without_name
        name = "x"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("collect", x, stream=True),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )


def test_describe_collect_paging_with_stream(RE):
    class X(EventPageCollectable):
        collect_pages = collect_Pageable_without_name
        describe_collect = describe_without_name
        name = "x"

    x = X()
    collector = []
    with pytest.raises(IllegalMessageSequence):
        RE([
                Msg("open_run", x),
                Msg("collect", x, stream=True),
                Msg("close_run", x),
            ],
            lambda *args: collector.append(args)
           )


def test_describe_collect_pre_declare_stream(RE):
    class X(Flyable, EventPageCollectable):
        collect_pages = collect_Pageable_with_name
        describe_collect = describe_with_name
        kickoff = kickoff_dummy_callback
        complete = complete_dummy_callback
        name = "x"

    x = X()
    collector = []
    with pytest.raises(IllegalMessageSequence):
        RE([
                Msg("open_run", x),
                Msg("declare_stream", None, x, collect=True, name="primary"),
                Msg("kickoff", x),
                Msg("complete", x),
                Msg("collect", x, stream=True),
                Msg("close_run", x),
            ],
            lambda *args: collector.append(args)
           )


def test_describe_with_stream_datum_no_events(RE):

    class X(Collectable, WritesExternalAssets):
        describe_collect = describe_with_name_stream_detectors
        collect_asset_docs = collect_asset_docs_StreamDatum
        name = "x"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("collect", x, name="primary"),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )
