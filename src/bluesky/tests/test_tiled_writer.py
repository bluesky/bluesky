from typing import Dict, Iterator, Optional

import h5py
import numpy as np
import pytest
from event_model.documents.event_descriptor import DataKey
from event_model.documents.stream_datum import StreamDatum
from event_model.documents.stream_resource import StreamResource
from tiled.catalog import in_memory
from tiled.client import Context, from_context
from tiled.server.app import build_app

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky.callbacks.tiled_writer import TiledWriter
from bluesky.protocols import (
    Collectable,
    HasName,
    Readable,
    Reading,
    StreamAsset,
    WritesStreamAssets,
)


@pytest.fixture
def catalog(tmpdir):
    catalog = in_memory(writable_storage=tmpdir)
    yield catalog


@pytest.fixture
def app(catalog):
    app = build_app(catalog)
    yield app


@pytest.fixture
def context(app):
    with Context.from_app(app) as context:
        yield context


@pytest.fixture
def client(context):
    client = from_context(context)
    yield client


class Named(HasName):
    name: str = ""
    root: str = "/root"
    parent = None

    def __init__(self, name: str, root: str) -> None:
        self.name = name
        self.root = root
        self.counter = 0


def test_stream_datum_readable_counts(RE, client, tmpdir):
    tw = TiledWriter(client)
    det = StreamDatumReadableCollectable(name="det", root=str(tmpdir))
    RE(bp.count([det], 3), tw)
    arrs = client.values().last()["primary"]["external"].values()
    assert arrs[0].shape == (3,)
    assert arrs[1].shape == (3, 10, 15)
    assert arrs[0].read() is not None
    assert arrs[1][:] is not None


def test_stream_datum_collectable(RE, client, tmpdir):
    det = StreamDatumReadableCollectable(name="det", root=str(tmpdir))
    tw = TiledWriter(client)
    RE(collect_plan(det, pre_declare=False), tw)
    arrs = client.values().last()["primary"]["external"].values()
    assert arrs[0].read() is not None
    assert arrs[1][:] is not None


def collect_plan(*objs, pre_declare: bool, stream=True):
    yield from bps.open_run()
    if pre_declare:
        yield from bps.declare_stream(*objs, collect=True)
    yield from bps.collect(*objs, stream=stream, return_payload=False, name="primary")
    yield from bps.close_run()


def describe_stream_datum(self: Named) -> Dict[str, DataKey]:
    """Describe 2 datasets which will be backed by StreamResources"""
    return {
        f"{self.name}-sd1": DataKey(source="file", dtype="number", shape=[1], external="STREAM:"),
        f"{self.name}-sd2": DataKey(source="file", dtype="array", shape=[10, 15], external="STREAM:"),
    }


def get_index(self) -> int:
    """Report how many frames were written"""
    return 10


def collect_asset_docs_stream_datum(self: Named, index: Optional[int] = None) -> Iterator[StreamAsset]:
    """Produce a StreamResource and StreamDatum for 2 data keys for 0:index"""
    index = index or 1
    file_path = self.root + "/dataset.h5"
    for data_key in [f"{self.name}-sd1", f"{self.name}-sd2"]:
        uid = f"{data_key}-uid"
        data_desc = self.describe()[data_key]  # Descriptor dictionary for the current data key
        data_shape = tuple(data_desc["shape"])
        data_shape = data_shape if data_shape != (1,) else ()
        hdf5_path = f"/{data_key}/VALUE"
        if self.counter == 0:
            stream_resource = StreamResource(
                resource_kwargs={"path": hdf5_path, "chunked": False},
                data_key=data_key,
                root=self.root,
                resource_path="/dataset.h5",
                spec="ADHDF5_SWMR_STREAM",
                uid=uid,
            )
            # Initialize an empty HDF5 dataset (3D: var 1 dim, fixed 2 and 3 dims)
            with h5py.File(file_path, "a") as f:
                dset = f.require_dataset(
                    hdf5_path,
                    (0, *data_shape),
                    maxshape=(None, *data_shape),
                    dtype=np.dtype("double"),
                    chunks=(10, *data_shape),
                )

            yield "stream_resource", stream_resource

        indx_min, indx_max = self.counter, self.counter + index
        stream_datum = StreamDatum(
            stream_resource=uid,
            descriptor="",
            uid=f"{uid}/{self.counter}",
            indices={"start": indx_min, "stop": indx_max},
            seq_nums={"start": 0, "stop": 0},
        )

        # Write (append to) the hdf5 dataset
        with h5py.File(file_path, "a") as f:
            dset = f[hdf5_path]
            dset.resize([indx_max, *data_shape])
            dset[indx_min:indx_max, ...] = np.random.randn(indx_max - indx_min, *data_shape)

        yield "stream_datum", stream_datum
    self.counter += index


def read_empty(self) -> Dict[str, Reading]:
    """Produce an empty event"""
    return {}


class StreamDatumReadableCollectable(Named, Readable, Collectable, WritesStreamAssets):
    """Produces no events, but only StreamResources for 2 data keys and can be read or collected"""

    describe = describe_collect = describe_stream_datum
    read = read_empty
    collect_asset_docs = collect_asset_docs_stream_datum
    get_index = get_index
