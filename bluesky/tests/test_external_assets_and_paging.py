from typing import Dict, Iterator, Optional
from event_model.documents.event_descriptor import DataKey
from event_model.documents.event import PartialEvent
from event_model.documents.resource import PartialResource
from event_model.documents.datum import Datum
from event_model.documents.stream_datum import StreamDatum
from event_model.documents.stream_resource import StreamResource
from bluesky.utils import new_uid
import bluesky.plans as bp
import bluesky.plan_stubs as bps
import pytest
from bluesky.protocols import (
    Asset,
    Readable,
    Reading,
    HasName,
    WritesExternalAssets,
    EventCollectable,
    EventPageCollectable,
    Collectable,
    WritesStreamAssets,
    StreamAsset,
)


class DocHolder(dict):
    def append(self, name, doc):
        self.setdefault(name, []).append(doc)

    def assert_emitted(self, **numbers: int):
        assert list(self) == list(numbers)
        assert {name: len(d) for name, d in self.items()} == numbers


def merge_dicts(*functions):
    def wrapper(self):
        ret = {}
        for f in functions:
            ret.update(f(self))
        return ret

    return wrapper


def merge_iterators(*iterators):
    def wrapper(self):
        for it in iterators:
            yield from it(self)

    return wrapper


def collect_plan(*objs, pre_declare: bool, stream=False):
    yield from bps.open_run()
    if pre_declare:
        yield from bps.declare_stream(*objs, collect=True)
    yield from bps.collect(*objs, stream=stream)
    yield from bps.close_run()


class Named(HasName):
    name: str = ""
    parent = None

    def __init__(self, name: str) -> None:
        self.name = name


def describe_datum(self: Named) -> Dict[str, DataKey]:
    """Describe a single data key backed with Resource"""
    return {
        f"{self.name}-datum": DataKey(
            source="file", dtype="number", shape=[1000, 1500], external="OLD:"
        )
    }


def collect_asset_docs_datum(self) -> Iterator[Asset]:
    """Produce Resource and Datum for a single frame in a file"""
    resource = PartialResource(
        resource_kwargs={"resource_arg": 42},
        root="/root",
        spec="ADHDF5_SWMR",
        resource_path="/path.h5",
        uid="RESOURCEUID",
    )
    yield "resource", resource
    datum = Datum(
        datum_id="RESOURCEUID/1",
        resource="RESOURCEUID",
        datum_kwargs={"datum_arg": 43},
    )
    yield "datum", datum


def read_datum(self: Named) -> Dict[str, Reading]:
    """Read a single reference to a Datum"""
    return {f"{self.name}-datum": {"value": "RESOURCEUID/1", "timestamp": 456}}


class DatumReadable(Named, Readable, WritesExternalAssets):
    """A readable that produces a single Event backed by a Datum"""

    describe = describe_datum
    read = read_datum
    collect_asset_docs = collect_asset_docs_datum


def describe_stream_datum(self: Named) -> Dict[str, DataKey]:
    """Describe 2 datasets which will be backed by StreamResources"""
    return {
        f"{self.name}-sd1": DataKey(
            source="file", dtype="number", shape=[1000, 1500], external="STREAM:"
        ),
        f"{self.name}-sd2": DataKey(
            source="file", dtype="number", shape=[], external="STREAM:"
        ),
    }


def get_index(self) -> int:
    """Report how many frames were written"""
    return 10


def collect_asset_docs_stream_datum(
    self: Named, index: Optional[int] = None
) -> Iterator[StreamAsset]:
    """Produce a StreamResource and StreamDatum for 2 data keys for 0:index"""
    index = index or 1
    for data_key in [f"{self.name}-sd1", f"{self.name}-sd2"]:
        stream_resource = StreamResource(
            resource_kwargs={"dataset": f"/{data_key}/data"},
            data_key=data_key,
            root="/root",
            resource_path="/path.h5",
            spec="ADHDF5_SWMR_STREAM",
            uid=new_uid(),
        )
        yield "stream_resource", stream_resource

        stream_datum = StreamDatum(
            stream_resource=stream_resource["uid"],
            descriptor="",
            uid=f'{stream_resource["uid"]}/1',
            indices={"start": 0, "stop": index},
            seq_nums={"start": 0, "stop": 0},
        )
        yield "stream_datum", stream_datum


def read_empty(self) -> Dict[str, Reading]:
    """Produce an empty event"""
    return {}


class StreamDatumReadable(Named, Readable, WritesStreamAssets):
    """A readable that produces a single frame from 2 StreamResources"""

    describe = describe_stream_datum
    read = read_empty
    collect_asset_docs = collect_asset_docs_stream_datum
    get_index = get_index


def describe_pv(self: Named) -> Dict[str, DataKey]:
    """Describe a single data_key backed by a PV value"""
    return {f"{self.name}-pv": DataKey(source="pv", dtype="number", shape=[])}


def read_pv(self: Named) -> Dict[str, Reading]:
    """Read a single data_key from a PV"""
    return {f"{self.name}-pv": Reading(value=5.8, timestamp=123)}


class PvAndDatumReadable(Named, Readable, WritesExternalAssets):
    """An odd device that produces a single event with one datakey from a pv, and one backed by a Datum"""

    describe = merge_dicts(describe_pv, describe_datum)
    read = merge_dicts(read_pv, read_datum)
    collect_asset_docs = collect_asset_docs_datum


class PvAndStreamDatumReadable(Named, Readable, WritesExternalAssets):
    """An odd device that produces a single event with one datakey from a pv, and one backed by a StreamDatum"""

    describe = merge_dicts(describe_pv, describe_stream_datum)
    read = merge_dicts(read_pv, read_empty)
    collect_asset_docs = collect_asset_docs_stream_datum
    get_index = get_index


def describe_collect_old_pv(self) -> Dict[str, Dict[str, DataKey]]:
    """Doubly nested old describe format with 2 pvs in each stream"""
    return {
        stream: {
            f"{stream}-{pv}": DataKey(source="pv", dtype="number", shape=[])
            for pv in ["pv1", "pv2"]
        }
        for stream in ["stream1", "stream2"]
    }


def collect_old_pv(self) -> Iterator[PartialEvent]:
    """Produce 2 events in each of the 2 streams"""
    for i in range(2):
        for stream in ["stream1", "stream2"]:
            yield PartialEvent(
                data={f"{stream}-{pv}": i for pv in ["pv1", "pv2"]},
                timestamps={f"{stream}-{pv}": 100 + i for pv in ["pv1", "pv2"]},
            )


class OldPvCollectable(Named, EventCollectable):
    """Produce events in 2 streams backed by PVs"""

    describe_collect = describe_collect_old_pv
    collect = collect_old_pv


def describe_collect_old_datum(self: Named):
    """Produces a single data key in a stream backed by a Datum"""
    return {
        "stream3": {
            f"{self.name}-datum": DataKey(
                source="file", dtype="number", shape=[1000, 1500], external="OLD:"
            )
        }
    }


def collect_old_datum(self):
    """Produces a single event backed by a Datum"""
    yield PartialEvent(
        data={f"{self.name}-datum": "RESOURCEUID/1"},
        timestamps={f"{self.name}-datum": 456},
        filled={f"{self.name}-datum": False},
    )


class OldDatumCollectable(Named, EventCollectable, WritesExternalAssets):
    """Produces a single stream with a single event in it backed by a Datum"""

    describe_collect = describe_collect_old_datum
    collect = collect_old_datum
    collect_asset_docs = collect_asset_docs_datum


def describe_collect_two_streams_same_names(self):
    """Buggy old describe output with the same data key in 2 streams"""
    return {
        stream: {"pv": DataKey(source="pv", dtype="number", shape=[])}
        for stream in ["stream1", "stream2"]
    }


def collect_one_pv(self):
    """Event producing value of a single pv"""
    yield PartialEvent(data={"pv": 0}, timestamps={"pv": 100})


class MultiKeyOldCollectable(Named, EventCollectable):
    """Buggy Device that should fail describe_collect as it makes 2 streams with the same data_key"""

    describe_collect = describe_collect_two_streams_same_names
    collect = collect_one_pv


class OldPvAndDatumCollectable(Named, EventCollectable, WritesExternalAssets):
    """Old style Device that produces 2 streams with PVs and 1 stream with Datum"""

    describe_collect = merge_dicts(describe_collect_old_pv, describe_collect_old_datum)
    collect = merge_iterators(collect_old_pv, collect_old_datum)
    collect_asset_docs = collect_asset_docs_datum


def describe_collect_pv(self: Named) -> Dict[str, Dict[str, DataKey]]:
    """New style describe collect with 2 PVs"""
    return {
        f"{self.name}-{pv}": DataKey(source="pv", dtype="number", shape=[])
        for pv in ["pv1", "pv2"]
    }


def collect_pv(self: Named) -> Iterator[PartialEvent]:
    """Collect for the new style describe collect of 2 data keys from PVs"""
    for i in range(2):
        yield PartialEvent(
            data={f"{self.name}-{pv}": i for pv in ["pv1", "pv2"]},
            timestamps={f"{self.name}-{pv}": 100 + i for pv in ["pv1", "pv2"]},
        )


def collect_pages_pv(self: Named) -> Iterator[PartialEvent]:
    """Same as collect_pv but in EventPage form"""
    yield PartialEvent(
        data={f"{self.name}-{pv}": [0, 1] for pv in ["pv1", "pv2"]},
        timestamps={f"{self.name}-{pv}": [100, 101] for pv in ["pv1", "pv2"]},
    )


class PvCollectable(Named, EventCollectable):
    """Produces events with 2 data keys backed by PVs"""
    describe_collect = describe_collect_pv
    collect = collect_pv


class PvPageCollectable(Named, EventPageCollectable):
    """Produces event pages with 2 data keys backed by PVs"""
    describe_collect = describe_collect_pv
    collect_pages = collect_pages_pv


class StreamDatumCollectable(Named, Collectable, WritesStreamAssets):
    """Produces no events, but only StreamResources for 2 data keys"""
    describe_collect = describe_stream_datum
    collect_asset_docs = collect_asset_docs_stream_datum
    get_index = get_index


def test_datum_readable_counts(RE):
    det = DatumReadable(name="det")
    docs = DocHolder()
    RE(bp.count([det]), docs.append)
    docs.assert_emitted(start=1, descriptor=1, resource=1, datum=1, event=1, stop=1)
    assert docs["descriptor"][0]["name"] == "primary"
    assert list(docs["descriptor"][0]["data_keys"]) == ["det-datum"]
    assert docs["event"][0]["descriptor"] == docs["descriptor"][0]["uid"]
    assert docs["event"][0]["data"] == {"det-datum": "RESOURCEUID/1"}
    assert docs["event"][0]["timestamps"] == {"det-datum": 456}
    assert docs["event"][0]["filled"] == {"det-datum": False}


def test_stream_datum_readable_counts(RE):
    det = StreamDatumReadable(name="det")
    docs = DocHolder()
    RE(bp.count([det]), docs.append)
    docs.assert_emitted(
        start=1, descriptor=1, stream_resource=2, stream_datum=2, event=1, stop=1
    )
    assert docs["descriptor"][0]["name"] == "primary"
    assert list(docs["descriptor"][0]["data_keys"]) == ["det-sd1", "det-sd2"]
    assert all(
        sd["descriptor"] == docs["descriptor"][0]["uid"] for sd in docs["stream_datum"]
    )
    assert docs["event"][0]["data"] == {}
    assert docs["event"][0]["filled"] == {}


def test_combinations_counts(RE):
    det1 = PvAndDatumReadable(name="det1")
    det2 = PvAndStreamDatumReadable(name="det2")
    docs = DocHolder()
    RE(bp.count([det1, det2]), docs.append)
    docs.assert_emitted(
        start=1,
        descriptor=1,
        resource=1,
        datum=1,
        stream_resource=2,
        stream_datum=2,
        event=1,
        stop=1,
    )
    assert docs["descriptor"][0]["name"] == "primary"
    # TODO: only works with a set at the moment
    assert list(docs["descriptor"][0]["data_keys"]) == [
        "det1-pv",
        "det1-datum",
        "det2-pv",
        "det2-sd1",
        "det2-sd2",
    ]
    assert docs["event"][0]["descriptor"] == docs["descriptor"][0]["uid"]
    assert docs["event"][0]["data"] == {
        "det1-datum": "RESOURCEUID/1",
        "det1-pv": 5.8,
        "det2-pv": 5.8,
    }
    assert docs["event"][0]["timestamps"] == {
        "det1-datum": 456,
        "det1-pv": 123,
        "det2-pv": 123,
    }
    assert docs["event"][0]["filled"] == {"det1-datum": False}


def test_collect_stream_true_raises(RE):
    with pytest.raises(
        RuntimeError,
        match="Collect now emits EventPages (stream=False), so emitting Events (stream=True) is no longer supported",
    ):
        RE(collect_plan(OldPvCollectable("det"), pre_declare=False, stream=True))


def test_old_describe_fails_predeclare(RE):
    with pytest.raises(
        RuntimeError,
        match="Old style describe_collect output with stream name not supported in declare_stream",
    ):
        RE(collect_plan(OldPvCollectable(name="det"), pre_declare=True))


def test_same_key_in_multiple_streams_fails(RE):
    with pytest.raises(
        RuntimeError,
        match="Multiple streams ['stream1', 'stream2'] would emit the same data_keys ['pv']",
    ):
        RE(collect_plan(MultiKeyOldCollectable(name="det"), pre_declare=False))


def test_old_pv_collectable(RE):
    det = OldPvCollectable(name="det")
    docs = DocHolder()
    RE(collect_plan(det, pre_declare=False), docs.append)
    docs.assert_emitted(start=1, descriptor=2, event_page=2, stop=1)
    assert [d["name"] for d in docs["descriptor"]] == ["stream1", "stream2"]
    assert list(docs["descriptor"][0]["data_keys"]) == ["stream1-pv1", "stream1-pv2"]
    assert list(docs["descriptor"][1]["data_keys"]) == ["stream2-pv1", "stream2-pv2"]
    assert docs["event_page"][0]["data"] == {
        "stream1-pv1": [0, 1],
        "stream1-pv2": [0, 1],
    }
    assert docs["event_page"][0]["timestamps"] == {
        "stream1-pv1": [100, 101],
        "stream1-pv2": [100, 101],
    }
    assert docs["event_page"][1]["data"] == {
        "stream2-pv1": [0, 1],
        "stream2-pv2": [0, 1],
    }


def test_old_datum_collectable(RE):
    det = OldDatumCollectable(name="det")
    docs = DocHolder()
    RE(collect_plan(det, pre_declare=False), docs.append)
    docs.assert_emitted(
        start=1, descriptor=1, resource=1, datum=1, event_page=1, stop=1
    )
    assert docs["descriptor"][0]["name"] == "stream3"
    assert list(docs["descriptor"][0]["data_keys"]) == ["det-datum"]
    assert docs["event_page"][0]["descriptor"] == docs["descriptor"][0]["uid"]
    assert docs["event_page"][0]["data"] == {"det-datum": ["RESOURCEUID/1"]}
    assert docs["event_page"][0]["timestamps"] == {"det-datum": [456]}
    assert docs["event_page"][0]["filled"] == {"det-datum": [False]}


def test_old_datum_and_pv_collectable(RE):
    det = OldPvAndDatumCollectable(name="det")
    docs = DocHolder()
    RE(collect_plan(det, pre_declare=False), docs.append)
    docs.assert_emitted(
        start=1, descriptor=3, resource=1, datum=1, event_page=3, stop=1
    )
    assert [d["name"] for d in docs["descriptor"]] == ["stream1", "stream2", "stream3"]
    assert list(docs["descriptor"][0]["data_keys"]) == ["stream1-pv1", "stream1-pv2"]
    assert list(docs["descriptor"][2]["data_keys"]) == ["det-datum"]
    assert docs["event_page"][0]["data"] == {
        "stream1-pv1": [0, 1],
        "stream1-pv2": [0, 1],
    }
    assert docs["event_page"][1]["data"] == {
        "stream2-pv1": [0, 1],
        "stream2-pv2": [0, 1],
    }
    assert docs["event_page"][2]["data"] == {"det-datum": ["RESOURCEUID/1"]}


def test_new_collect_needs_predeclare(RE):
    with pytest.raises(
        RuntimeError,
        match="New style describe_collect output requires declare_stream before collect",
    ):
        RE(collect_plan(PvCollectable(name="det"), pre_declare=False))


@pytest.mark.parametrize("cls", [PvCollectable, PvPageCollectable])
def test_pv_collectable(RE, cls):
    det = cls(name="det")
    docs = DocHolder()
    RE(collect_plan(det, pre_declare=True), docs.append)
    docs.assert_emitted(start=1, descriptor=1, event_page=1, stop=1)
    assert docs["descriptor"][0]["name"] == "primary"
    assert list(docs["descriptor"][0]["data_keys"]) == ["det-pv1", "det-pv2"]
    assert docs["event_page"][0]["data"] == {
        "det-pv1": [0, 1],
        "det-pv2": [0, 1],
    }
    assert docs["event_page"][0]["timestamps"] == {
        "det-pv1": [100, 101],
        "det-pv2": [100, 101],
    }


def test_stream_datum_collectable(RE):
    det = StreamDatumCollectable(name="det")
    docs = DocHolder()
    RE(collect_plan(det, pre_declare=True), docs.append)
    docs.assert_emitted(
        start=1, descriptor=1, stream_resource=2, stream_datum=2, stop=1
    )
    assert docs["descriptor"][0]["name"] == "primary"
    assert list(docs["descriptor"][0]["data_keys"]) == ["det-sd1", "det-sd2"]
    assert docs["stream_resource"][0]["data_key"] == "det-sd1"
    assert docs["stream_resource"][1]["data_key"] == "det-sd2"
    assert docs["stream_datum"][0]["descriptor"] == docs["descriptor"][0]["uid"]
    assert docs["stream_datum"][1]["descriptor"] == docs["descriptor"][0]["uid"]


@pytest.mark.parametrize("cls1", [PvCollectable, PvPageCollectable])
@pytest.mark.parametrize(
    "cls2", [PvCollectable, PvPageCollectable, StreamDatumCollectable]
)
def test_many_collectables_fails(RE, cls1, cls2):
    det1, det2 = cls1(name="det1"), cls2(name="det2")
    with pytest.raises(
        RuntimeError,
        match="Cannot collect the output of multiple collect() or collect_pages() into the same stream",
    ):
        RE(collect_plan(det1, det2, pre_declare=False))


def test_many_stream_datum_collectables(RE):
    det1, det2 = (
        StreamDatumCollectable(name="det1"),
        StreamDatumCollectable(name="det2"),
    )
    docs = DocHolder()
    RE(collect_plan(det1, det2, pre_declare=True), docs.append)
    docs.assert_emitted(
        start=1, descriptor=1, stream_resource=4, stream_datum=4, stop=1
    )
    data_keys = ["det1-sd1", "det1-sd2", "det2-sd1", "det2-sd2"]
    assert docs["descriptor"][0]["name"] == "primary"
    assert list(docs["descriptor"][0]["data_keys"]) == data_keys
    assert [d["data_key"] for d in docs["stream_resource"]] == data_keys
    assert all(
        d["descriptor"] == docs["descriptor"][0]["uid"] for d in docs["stream_datum"]
    )
