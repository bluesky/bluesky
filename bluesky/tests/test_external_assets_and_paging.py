from typing import Dict, Iterator
from bluesky import Msg
from event_model.documents.event_descriptor import DataKey
from event_model.documents.event_page import PartialEventPage
from event_model.documents.resource import PartialResource
from event_model.documents.datum import Datum
from event_model.documents.stream_datum import StreamDatum
from event_model.documents.stream_resource import StreamResource
from bluesky.utils import new_uid
import pytest
from bluesky.protocols import (
    Asset, Readable,
    Reading, SyncOrAsync, WritesExternalAssets, Flyable, EventCollectable, EventPageCollectable
)

def read_Readable(self) -> Dict[str, Reading]:
    return dict(x=dict(value=1.2, timestamp=0.0))


def describe_Readable(self) -> Dict[str, DataKey]:
    return dict(x=dict(source="dummy", dtype="number", shape=[]))


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
        data_keys=["a", "b", "c"],
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


def describe_Pageable_with_name(self) -> SyncOrAsync[Dict[str, DataKey]]:
    data_keys = {
        str(x): DataKey(shape=[1], source="a", dtype="string")
        for x in range(7)
    }

    return data_keys


def collect_Pageable_with_name(self) -> SyncOrAsync[Iterator[PartialEventPage]]:
    def timestamps(x):
        return {str(x): [x] for x in range(7)}

    partial_event_pages = [
        PartialEventPage(
            timestamps=timestamps(x), data=timestamps(x)
        )
        for x in range(7)
    ]

    return partial_event_pages


def describe_Pageable_without_name(self) -> SyncOrAsync[Dict[str, Dict[str, DataKey]]]:
    data_keys = {
        str(y): {str(x): DataKey(shape=[1], source="a", dtype="string") for x in [7, 8]}
        for y in range(16)
    }
    return data_keys


def collect_Pageable_without_name(self) -> SyncOrAsync[Iterator[PartialEventPage]]:
    def timestamps(x):
        return {str(y): [x] for y in [7, 8]}

    partial_event_pages = [
        PartialEventPage(
            timestamps=timestamps(x), data=timestamps(x)
        )
        for x in range(16)
    ]

    return partial_event_pages


def test_flyscan_with_pages_with_no_name(RE):
    class X(Flyable, EventPageCollectable):
        def kickoff(self, *_, **__):
            ...

        def complete(self, *_, **__):
            ...

        collect_pages = collect_Pageable_without_name
        describe_collect = describe_Pageable_without_name
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
    from pprint import pprint
    pprint(collector)


def test_flyscan_with_pages_passed_in_name(RE):
    class X(Flyable, EventPageCollectable):
        def kickoff(self, *_, **__):
            ...

        def complete(self, *_, **__):
            ...

        collect_pages = collect_Pageable_with_name
        describe_collect = describe_Pageable_with_name
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
    from pprint import pprint
    pprint(collector)


def test_desc_coll(RE, hw):
    """Returns a partial event"""
    ...


def test_desc_coll_res(RE, hw):
    """Returns a partial event + resource"""
    ...


def test_desc_page_coll_page(RE, hw):
    """Returns a partial event page"""
    ...


def test_desc_page_coll_page_sr(RE, hw):
    """Returns a partial event page + stream resource"""
    ...
