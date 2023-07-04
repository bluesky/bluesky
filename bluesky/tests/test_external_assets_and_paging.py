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
import pytest
from bluesky.protocols import (
    Asset, Readable,
    Reading, SyncOrAsync, WritesExternalAssets, Flyable, EventCollectable, EventPageCollectable, Collectable
)


def read_Readable(self) -> Dict[str, Reading]:
    return dict(x=dict(value=1.2, timestamp=0.0))


def describe_Readable(self) -> Dict[str, DataKey]:
    return dict(
        y=dict(source="dummy", dtype="number", shape=[], external="STREAM:"),
        x=dict(source="dummy", dtype="number", shape=[])
    )


def collect_asset_docs_Resource(self) -> Iterator[Asset]:
    resource = PartialResource(
        resource_kwargs={"some_argument": 1337},
        root="awesome",
        spec=".dat",
        resource_path="no/really,/awesome",
        uid=new_uid(),
    )

    return iter([("resource", resource)])


def collect_asset_docs_Datum(self) -> Iterator[Asset]:
    datum = Datum(
        datum_id="some_resource_name/123123",
        resource="uid_of_that_resource",
        datum_kwargs={"some_argument": 1234},
    )
    return iter([("datum", datum)])


def collect_asset_docs_StreamResource(self) -> Iterator[Asset]:
    stream_resource = StreamResource(
        resource_kwargs={"argument": 1},
        resource_path="An/awesome/path",
        root="some_detail",
        spec=".hdf5",
        uid=new_uid(),
    )
    return iter([("stream_resource", stream_resource)])


def collect_asset_docs_StreamDatum(self) -> Iterator[Asset]:
    stream_datum = StreamDatum(
        block_idx=32,
        event_count=1233,
        event_offset=1,
        stream_resource=new_uid(),
        uid=new_uid(),
        data_keys=["y"],
        seq_nums={"a": 12},
        indices={"b", 1},
    )
    return iter([("stream_datum", stream_datum)])


@pytest.mark.parametrize(
    'asset_type,collect_asset_docs_fun',
    [
        ("resource", collect_asset_docs_Resource),
        ("datum", collect_asset_docs_Datum),
        ("stream_resource", collect_asset_docs_StreamResource),
        ("stream_datum", collect_asset_docs_StreamDatum),
    ]
)
def test_rd_desc_different_asset_types(RE, asset_type, collect_asset_docs_fun):
    class X(Readable, WritesExternalAssets):
        read = read_Readable
        describe = describe_Readable
        collect_asset_docs = collect_asset_docs_fun
        name = "x"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("create", name="x"),
            Msg("read", x),
            Msg("save", x),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )
    assert len(collector) == 5
    assert collector[2][0] == asset_type


def describe_with_name(self) -> SyncOrAsync[Dict[str, DataKey]]:
    data_keys = {
        str(x): DataKey(shape=[1], source="a", dtype="string")
        for x in ["x", "y", "z"]
    }

    return data_keys


def collect_Pageable_with_name(self) -> SyncOrAsync[Iterator[PartialEventPage]]:
    def timestamps():
        return {str(x): [x] for x in ["x", "y", "z"]}

    partial_event_pages = [
        PartialEventPage(
            timestamps=timestamps(), data=timestamps()
        )
        for x in range(7)
    ]

    return partial_event_pages


def describe_without_name(self) -> SyncOrAsync[Dict[str, Dict[str, DataKey]]]:
    data_keys = {
        str(y): {str(x): DataKey(shape=[1], source="a", dtype="string") for x in ["x", "y", "z"]}
        for y in range(16)
    }
    return data_keys


def collect_Pageable_without_name(self) -> SyncOrAsync[Iterator[PartialEventPage]]:
    def timestamps(x):
        return {str(y): [x] for y in ["x", "y", "z"]}

    partial_event_pages = [
        PartialEventPage(
            timestamps=timestamps(x), data=timestamps(x)
        )
        for x in range(16)
    ]

    return partial_event_pages


def collect(self) -> SyncOrAsync[Iterator[PartialEvent]]:
    def timestamps(x):
        return {"x": x + 0.1, "y": x + 0.2, "z": x + 0.3}

    def data(x):
        return {"x": x, "y": x*2, "z": x*3}

    partial_events = [
        PartialEvent(
            timestamps=timestamps(x), data=data(x), time=float(x)
        )
        for x in range(16)
    ]
    return partial_events


def test_flyscan_with_pages_with_no_name(RE):
    class X(Flyable, EventPageCollectable):
        def kickoff(self, *_, **__):
            ...

        def complete(self, *_, **__):
            ...

        collect_pages = collect_Pageable_without_name
        describe_collect = describe_without_name
        name = "x"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("collect", x),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )


def test_flyscan_with_pages_passed_in_name(RE):
    class X(Flyable, EventPageCollectable):
        def kickoff(self, *_, **__):
            ...

        def complete(self, *_, **__):
            ...

        collect_pages = collect_Pageable_with_name
        describe_collect = describe_with_name
        name = "x"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("collect", x, name="1"),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )


@pytest.mark.parametrize(
    'asset_type,collect_asset_docs_fun',
    [
        ("resource", collect_asset_docs_Resource),
        ("datum", collect_asset_docs_Datum),
        ("stream_resource", collect_asset_docs_StreamResource),
        ("stream_datum", collect_asset_docs_StreamDatum),
    ]
)
def test_flyscan_with_pages_with_no_name_and_external_assets(RE, asset_type, collect_asset_docs_fun):
    class X(Flyable, EventPageCollectable, WritesExternalAssets):
        def kickoff(self, *_, **__):
            ...

        def complete(self, *_, **__):
            ...

        collect_pages = collect_Pageable_without_name
        describe_collect = describe_without_name
        collect_asset_docs = collect_asset_docs_fun
        name = "x"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("collect", x),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )


@pytest.mark.parametrize(
    'asset_type,collect_asset_docs_fun',
    [
        ("resource", collect_asset_docs_Resource),
        ("datum", collect_asset_docs_Datum),
        ("stream_resource", collect_asset_docs_StreamResource),
        ("stream_datum", collect_asset_docs_StreamDatum),
    ]
)
def test_flyscan_with_pages_passed_in_name_and_external_assets(RE, asset_type, collect_asset_docs_fun):
    class X(Flyable, EventPageCollectable, WritesExternalAssets):
        def kickoff(self, *_, **__):
            ...

        def complete(self, *_, **__):
            ...

        collect_pages = collect_Pageable_with_name
        describe_collect = describe_with_name
        collect_asset_docs = collect_asset_docs_fun
        name = "x"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("collect", x, name="1"),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )


def test_flyscan_without_pageing_or_name(RE):
    class X(Flyable, EventCollectable):
        def kickoff(self, *_, **__):
            ...

        def complete(self, *_, **__):
            ...

        collect = collect
        describe_collect = describe_without_name
        name = "x"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("collect", x),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )


def test_flyscan_without_paging_with_name(RE):
    class X(Flyable, EventCollectable):
        def kickoff(self, *_, **__):
            ...

        def complete(self, *_, **__):
            ...

        collect = collect
        describe_collect = describe_with_name
        name = "x"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("collect", x, name="1"),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )


def test_describe_collect_with_stream(RE):
    class X(EventCollectable):
        collect = collect
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
    from pprint import pprint
    pprint(collector)


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
    from pprint import pprint
    pprint(collector)


@pytest.mark.parametrize(
    'asset_type,collect_asset_docs_fun',
    [
        ("resource", collect_asset_docs_Resource),
        ("datum", collect_asset_docs_Datum),
        ("stream_resource", collect_asset_docs_StreamResource),
        ("stream_datum", collect_asset_docs_StreamDatum),
    ]
)
def test_describe_with_external_assets_no_collect(RE, asset_type, collect_asset_docs_fun):

    class X(Collectable, WritesExternalAssets):
        describe_collect = describe_without_name
        collect_asset_docs = collect_asset_docs_fun
        name = "x"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("collect", x),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )
