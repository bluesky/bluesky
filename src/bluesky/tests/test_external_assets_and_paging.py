import re
from collections.abc import Iterator

import pytest
from event_model.documents import Datum
from event_model.documents.event import PartialEvent
from event_model.documents.event_descriptor import DataKey
from event_model.documents.event_page import PartialEventPage
from event_model.documents.resource import PartialResource
from event_model.documents.stream_datum import StreamDatum
from event_model.documents.stream_resource import StreamResource

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky.protocols import (
    Asset,
    Collectable,
    EventCollectable,
    EventPageCollectable,
    HasName,
    Readable,
    Reading,
    StreamAsset,
    WritesExternalAssets,
    WritesStreamAssets,
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


def collect_plan(*objs, pre_declare: bool, stream=False, stream_name=None):
    yield from bps.open_run()
    if pre_declare:
        yield from bps.declare_stream(*objs, name=stream_name, collect=True)
    yield from bps.collect(*objs, stream=stream, name=stream_name)
    yield from bps.close_run()


def collect_plan_no_stream_name_in_collect(*objs, pre_declare: bool, stream=False, stream_name=None):
    yield from bps.open_run()
    if pre_declare:
        yield from bps.declare_stream(*objs, name=stream_name, collect=True)
    yield from bps.collect(*objs, stream=stream)
    yield from bps.close_run()


class Named(HasName):
    name: str = ""
    parent = None

    def __init__(self, name: str) -> None:
        self.name = name
        self.counter = 0


def describe_datum(self: Named) -> dict[str, DataKey]:
    """Describe a single data key backed with Resource"""
    return {f"{self.name}-datum": DataKey(source="file", dtype="number", shape=[1000, 1500], external="OLD:")}


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


def read_datum(self: Named) -> dict[str, Reading]:
    """Read a single reference to a Datum"""
    return {f"{self.name}-datum": {"value": "RESOURCEUID/1", "timestamp": 456}}


class DatumReadable(Named, Readable, WritesExternalAssets):
    """A readable that produces a single Event backed by a Datum"""

    describe = describe_datum
    read = read_datum
    collect_asset_docs = collect_asset_docs_datum


def describe_stream_datum(self: Named) -> dict[str, DataKey]:
    """Describe 2 datasets which will be backed by StreamResources"""
    return {
        f"{self.name}-sd1": DataKey(source="file", dtype="number", shape=[1000, 1500], external="STREAM:"),
        f"{self.name}-sd2": DataKey(source="file", dtype="number", shape=[], external="STREAM:"),
    }


def get_index(self) -> int:
    """Report how many frames were written"""
    return 10


def collect_asset_docs_stream_datum(self: Named, index: int | None = None) -> Iterator[StreamAsset]:
    """Produce a StreamResource and StreamDatum for 2 data keys for 0:index"""
    index = index or 1
    for data_key in [f"{self.name}-sd1", f"{self.name}-sd2"]:
        uid = f"{data_key}-uid"
        if self.counter == 0:
            # Backward compatibility test, ignore typing errors
            stream_resource = StreamResource(  # type: ignore[typeddict-item]
                resource_kwargs={"dataset": f"/{data_key}/data"},
                data_key=data_key,
                root="/root",
                resource_path="/path.h5",
                spec="AD_HDF5_SWMR_STREAM",
                uid=uid,
            )
            yield "stream_resource", stream_resource

        stream_datum = StreamDatum(
            stream_resource=uid,
            descriptor="",
            uid=f"{uid}/{self.counter}",
            indices={"start": self.counter, "stop": self.counter + index},
            seq_nums={"start": 0, "stop": 0},
        )
        yield "stream_datum", stream_datum
    self.counter += index


def read_empty(self) -> dict[str, Reading]:
    """Produce an empty event"""
    return {}


def describe_pv(self: Named) -> dict[str, DataKey]:
    """Describe a single data_key backed by a PV value"""
    return {f"{self.name}-pv": DataKey(source="pv", dtype="number", shape=[])}


def read_pv(self: Named) -> dict[str, Reading]:
    """Read a single data_key from a PV"""
    return {f"{self.name}-pv": Reading(value=5.8, timestamp=123)}


class PvAndDatumReadable(Named, Readable, WritesExternalAssets):
    """An odd device that produces a single event with one datakey from a pv, and one backed by a Datum"""

    describe = merge_dicts(describe_pv, describe_datum)
    read = merge_dicts(read_pv, read_datum)
    collect_asset_docs = collect_asset_docs_datum


class PvAndStreamDatumReadable(Named, Readable, WritesStreamAssets):
    """An odd device that produces a single event with one datakey from a pv, and one backed by a StreamDatum"""

    describe = merge_dicts(describe_pv, describe_stream_datum)
    read = read_pv
    collect_asset_docs = collect_asset_docs_stream_datum
    get_index = get_index


def describe_collect_old_pv(self) -> dict[str, dict[str, DataKey]]:
    """Doubly nested old describe format with 2 pvs in each stream"""
    return {
        stream: {f"{stream}-{pv}": DataKey(source="pv", dtype="number", shape=[]) for pv in ["pv1", "pv2"]}
        for stream in ["stream1", "stream2"]
    }


def collect_old_pv(self) -> Iterator[PartialEvent]:
    """Produce 2 events in each of the 2 streams"""
    for i in range(2):
        for stream in ["stream1", "stream2"]:
            yield PartialEvent(
                data={f"{stream}-{pv}": i for pv in ["pv1", "pv2"]},
                timestamps={f"{stream}-{pv}": 100 + i for pv in ["pv1", "pv2"]},
                time=100 + i,
            )


class OldPvCollectable(Named, EventCollectable):
    """Produce events in 2 streams backed by PVs"""

    describe_collect = describe_collect_old_pv
    collect = collect_old_pv


def describe_collect_old_datum(self: Named):
    """Produces a single data key in a stream backed by a Datum"""
    return {
        "stream3": {
            f"{self.name}-datum": DataKey(source="file", dtype="number", shape=[1000, 1500], external="OLD:")
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
    return {stream: {"pv": DataKey(source="pv", dtype="number", shape=[])} for stream in ["stream1", "stream2"]}


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


def describe_collect_pv(self: Named) -> dict[str, DataKey]:
    """New style describe collect with 2 PVs"""
    return {f"{self.name}-{pv}": DataKey(source="pv", dtype="number", shape=[]) for pv in ["pv1", "pv2"]}


def collect_pv(self: Named) -> Iterator[PartialEvent]:
    """Collect for the new style describe collect of 2 data keys from PVs"""
    for i in range(2):
        yield PartialEvent(
            data={f"{self.name}-{pv}": i for pv in ["pv1", "pv2"]},
            timestamps={f"{self.name}-{pv}": 100 + i for pv in ["pv1", "pv2"]},
            time=100 + i,
        )


def collect_pages_pv(self: Named) -> Iterator[PartialEventPage]:
    """Same as collect_pv but in EventPage form"""
    yield PartialEventPage(
        data={f"{self.name}-{pv}": [0, 1] for pv in ["pv1", "pv2"]},
        timestamps={f"{self.name}-{pv}": [100, 101] for pv in ["pv1", "pv2"]},
        time=[100, 101],
    )


class PvCollectable(Named, EventCollectable):
    """Produces events with 2 data keys backed by PVs"""

    describe_collect = describe_collect_pv
    collect = collect_pv


class PvPageCollectable(Named, EventPageCollectable):
    """Produces event pages with 2 data keys backed by PVs"""

    describe_collect = describe_collect_pv
    collect_pages = collect_pages_pv


class StreamDatumReadableCollectable(Named, Readable, Collectable, WritesStreamAssets):
    """Produces no events, but only StreamResources for 2 data keys and can be read or collected"""

    describe = describe_stream_datum
    describe_collect = describe_stream_datum
    read = read_empty
    collect_asset_docs = collect_asset_docs_stream_datum
    get_index = get_index


def test_datum_readable_counts(RE):
    """Test that count-ing a datum-producing device results in expected documents."""
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
    """Test that count-ing a StreamDatum-producing device results in expected documents."""
    det = StreamDatumReadableCollectable(name="det")
    docs = DocHolder()
    RE(bp.count([det], 2), docs.append)
    docs.assert_emitted(start=1, descriptor=1, stream_resource=2, stream_datum=4, event=2, stop=1)
    assert docs["descriptor"][0]["name"] == "primary"
    assert list(docs["descriptor"][0]["data_keys"]) == ["det-sd1", "det-sd2"]
    assert all(sd["descriptor"] == docs["descriptor"][0]["uid"] for sd in docs["stream_datum"])
    assert all(e["data"] == {} for e in docs["event"])
    assert all(e["filled"] == {} for e in docs["event"])
    assert [sd["indices"] for sd in docs["stream_datum"]] == [
        {"start": 0, "stop": 1},
        {"start": 0, "stop": 1},
        {"start": 1, "stop": 2},
        {"start": 1, "stop": 2},
    ]


def test_combinations_counts(RE):
    """Test that mixing a StreamDatum- and Datum-producing device in one count works."""
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
    assert set(docs["descriptor"][0]["data_keys"]) == {
        "det1-pv",
        "det1-datum",
        "det2-pv",
        "det2-sd1",
        "det2-sd2",
    }
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
        match=re.escape(
            "Collect now emits EventPages (stream=False), so emitting Events (stream=True) is no longer supported"
        ),
    ):
        RE(collect_plan(OldPvCollectable("det"), pre_declare=False, stream=True))


def test_predeclare_requires_stream_name(RE):
    with pytest.raises(
        AssertionError,
        match=re.escape("A stream name that is not None is required for pre-declare"),
    ):
        RE(collect_plan(OldPvCollectable(name="det"), pre_declare=True))


def test_new_style_with_steam_name_requires_pre_declare(RE):
    with pytest.raises(
        AssertionError,
        match=re.escape("If a message stream name is provided declare stream needs to be called first."),
    ):
        RE(collect_plan(StreamDatumReadableCollectable(name="det"), pre_declare=False, stream_name="main"))


def test_new_style_with_no_stream_name_and_no_pre_declare_does_not_try_and_make_a_stream(RE):
    with pytest.raises(
        AssertionError,
        match=re.escape("Single nested data keys should be pre-declared"),
    ):
        RE(collect_plan(StreamDatumReadableCollectable(name="det"), pre_declare=False))


def test_same_key_in_multiple_streams_fails(RE):
    with pytest.raises(
        RuntimeError,
        match=re.escape("Can't use identical data keys in multiple streams"),
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
    docs.assert_emitted(start=1, descriptor=1, resource=1, datum=1, event_page=1, stop=1)
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
    docs.assert_emitted(start=1, descriptor=3, resource=1, datum=1, event_page=3, stop=1)
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


@pytest.mark.parametrize("cls", [PvCollectable, PvPageCollectable])
def test_pv_collectable(RE, cls):
    det = cls(name="det")
    docs = DocHolder()
    RE(collect_plan_no_stream_name_in_collect(det, pre_declare=True, stream_name="main"), docs.append)
    docs.assert_emitted(start=1, descriptor=1, event_page=1, stop=1)
    assert docs["descriptor"][0]["name"] == "main"
    assert list(docs["descriptor"][0]["data_keys"]) == ["det-pv1", "det-pv2"]
    assert docs["event_page"][0]["data"] == {
        "det-pv1": [0, 1],
        "det-pv2": [0, 1],
    }
    assert docs["event_page"][0]["timestamps"] == {
        "det-pv1": [100, 101],
        "det-pv2": [100, 101],
    }


def test_new_collect_needs_predeclare(RE):
    with pytest.raises(
        AssertionError,
        match="Single nested data keys should be pre-declared",
    ):
        RE(collect_plan(PvCollectable(name="det"), pre_declare=False))


def test_stream_datum_collectable(RE):
    det = StreamDatumReadableCollectable(name="det")
    docs = DocHolder()
    RE(collect_plan(det, pre_declare=True, stream_name="main"), docs.append)
    docs.assert_emitted(start=1, descriptor=1, stream_resource=2, stream_datum=2, stop=1)
    assert docs["descriptor"][0]["name"] == "main"
    assert list(docs["descriptor"][0]["data_keys"]) == ["det-sd1", "det-sd2"]
    assert docs["stream_resource"][0]["data_key"] == "det-sd1"
    assert docs["stream_resource"][1]["data_key"] == "det-sd2"
    assert docs["stream_datum"][0]["descriptor"] == docs["descriptor"][0]["uid"]
    assert docs["stream_datum"][1]["descriptor"] == docs["descriptor"][0]["uid"]


def test_stream_datum_collectable_no_stream_name(RE):
    det = StreamDatumReadableCollectable(name="det")
    docs = DocHolder()
    RE(collect_plan_no_stream_name_in_collect(det, pre_declare=True, stream_name="main"), docs.append)
    docs.assert_emitted(start=1, descriptor=1, stream_resource=2, stream_datum=2, stop=1)
    assert docs["descriptor"][0]["name"] == "main"
    assert list(docs["descriptor"][0]["data_keys"]) == ["det-sd1", "det-sd2"]
    assert docs["stream_resource"][0]["data_key"] == "det-sd1"
    assert docs["stream_resource"][1]["data_key"] == "det-sd2"
    assert docs["stream_datum"][0]["descriptor"] == docs["descriptor"][0]["uid"]
    assert docs["stream_datum"][1]["descriptor"] == docs["descriptor"][0]["uid"]


@pytest.mark.parametrize("cls1", [PvCollectable, PvPageCollectable])
@pytest.mark.parametrize("cls2", [PvCollectable, PvPageCollectable, StreamDatumReadableCollectable])
def test_many_collectables_fails(RE, cls1, cls2):
    """If there are multiple objects they must all WritesStreamAssests"""
    det1, det2 = cls1(name="det1"), cls2(name="det2")
    with pytest.raises(
        AssertionError,
        match=re.escape("does not implement all WritesStreamAssets methods"),
    ):
        RE(collect_plan(det1, det2, pre_declare=False))


def test_many_stream_datum_collectables(RE):
    """Test collecting from multiple StreamDatum-producing devices."""
    det1 = StreamDatumReadableCollectable(name="det1")
    det2 = StreamDatumReadableCollectable(name="det2")
    docs = DocHolder()
    RE(collect_plan(det1, det2, pre_declare=True, stream_name="main"), docs.append)
    docs.assert_emitted(start=1, descriptor=1, stream_resource=4, stream_datum=4, stop=1)
    data_keys = ["det1-sd1", "det1-sd2", "det2-sd1", "det2-sd2"]
    assert docs["descriptor"][0]["name"] == "main"
    assert set(docs["descriptor"][0]["data_keys"]) == set(data_keys)  # This only works in a set
    assert [d["data_key"] for d in docs["stream_resource"]] == data_keys
    assert all(d["descriptor"] == docs["descriptor"][0]["uid"] for d in docs["stream_datum"])


def tomo_plan(*objs):
    """An applied flyscanning example plan"""
    yield from bps.open_run()
    for name in ["flats", "darks", "projections"]:
        # projections is flyscan, others are step scan, so set collect accordingly
        yield from bps.declare_stream(*objs, name=name, collect=name == "projections")
    # take some flats, then darks
    yield from bps.collect(*objs, name="flats")
    yield from bps.collect(*objs, name="darks")
    # do the flyscan
    yield from bps.collect(*objs, name="projections")
    # take some flats at the end
    yield from bps.collect(*objs, name="flats")
    yield from bps.close_run()


def test_tomography_multi_stream_same_detectors(RE):
    """Test tomo_plan applied example"""
    det1 = StreamDatumReadableCollectable(name="det1")
    det2 = StreamDatumReadableCollectable(name="det2")
    docs = DocHolder()
    RE(tomo_plan(det1, det2), docs.append)
    # TODO: after https://github.com/bluesky/event-model/issues/296 this might be:
    #   stream_resource=12,  # one per stream per dataset per detector
    docs.assert_emitted(
        start=1,
        descriptor=3,  # one per stream
        stream_resource=4,  # one per dataset per detector
        stream_datum=16,  # 2 in flats, 1 in darks, one in projs, per dataset per detector
        stop=1,
    )
    data_keys = ["det1-sd1", "det1-sd2", "det2-sd1", "det2-sd2"]
    assert [d["name"] for d in docs["descriptor"]] == ["flats", "darks", "projections"]
    assert all(frozenset(d["data_keys"]) == frozenset(data_keys) for d in docs["descriptor"])
    assert [d["data_key"] for d in docs["stream_resource"]] == data_keys
    assert [d["descriptor"] for d in docs["stream_datum"]] == (
        [docs["descriptor"][0]["uid"]] * 4
        + [docs["descriptor"][1]["uid"]] * 4
        + [docs["descriptor"][2]["uid"]] * 4
        + [docs["descriptor"][0]["uid"]] * 4
    )


def change_conf_plan(*objs):
    """Re-emit a fresh EventDescriptor after the first Event."""
    yield from bps.open_run()
    for _ in range(2):
        yield from bps.declare_stream(*objs, name="main")
        yield from bps.collect(*objs)
    yield from bps.close_run()


def test_multiple_declare_in_same_stream(RE):
    """Test re-emitting an EventDescriptor while using StreamDatum-producing devices."""
    det1 = StreamDatumReadableCollectable(name="det1")
    det2 = StreamDatumReadableCollectable(name="det2")
    docs = DocHolder()
    RE(change_conf_plan(det1, det2), docs.append)
    docs.assert_emitted(start=1, descriptor=2, stream_resource=4, stream_datum=8, stop=1)
    data_keys = ["det1-sd1", "det1-sd2", "det2-sd1", "det2-sd2"]
    assert docs["descriptor"][0]["name"] == "main"
    assert frozenset(docs["descriptor"][0]["data_keys"]) == frozenset(data_keys)
    assert all(d["descriptor"] == docs["descriptor"][0]["uid"] for d in docs["stream_datum"][:4])
    assert docs["descriptor"][1]["name"] == "main"
    assert frozenset(docs["descriptor"][1]["data_keys"]) == frozenset(data_keys)
    assert all(d["descriptor"] == docs["descriptor"][1]["uid"] for d in docs["stream_datum"][4:])
