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
    Reading, Pageable, SyncOrAsync, WritesExternalAssets, Flyable
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


def describe_Pageable(self) -> SyncOrAsync[Dict[str, DataKey]]:
    data_keys = {
        str(x): DataKey(dtype="string", shape=[], source="a_source")
        for x in range(100)
    }
    return data_keys


def collect_Pageable(self) -> SyncOrAsync[Iterator[PartialEventPage]]:
    def timestamps(x):
        return {str(y): range(x) for y in range(x)}

    return iter(
        [
            PartialEventPage(
                timestamps=timestamps(x), data=timestamps(x)
            )
            for x in range(6)
        ]
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


def test_flyscan_with_pages(RE):
    class X(Flyable, Pageable):
        def kickoff(self, *_, **__):
            ...

        def complete(self, *_, **__):
            ...

        describe_pages = describe_Pageable
        collect_pages = collect_Pageable
        name = "x"

    x = X()
    collector = []
    RE([
            Msg("open_run", x),
            Msg("close_run", x),
        ],
        lambda *args: collector.append(args)
       )


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
