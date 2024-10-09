from typing import Dict, Iterator, Optional, Tuple

import h5py
import numpy as np
import ophyd.sim
import pytest
import tifffile as tf
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
def catalog(tmp_path):
    catalog = in_memory(writable_storage=str(tmp_path))
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


class StreamDatumReadableCollectable(Named, Readable, Collectable, WritesStreamAssets):
    """Produces no events, but only StreamResources/StreamDatums and can be read or collected"""

    def _get_hdf5_stream(self, data_key: str, index: int) -> Tuple[StreamResource, StreamDatum]:
        file_path = self.root + "/dataset.h5"
        uid = f"{data_key}-uid"
        data_desc = self.describe()[data_key]  # Descriptor dictionary for the current data key
        data_shape = tuple(data_desc["shape"])
        data_shape = data_shape if data_shape != (1,) else ()
        hdf5_dataset = f"/{data_key}/VALUE"

        stream_resource = None
        if self.counter == 0:
            stream_resource = StreamResource(
                parameters={"dataset": hdf5_dataset, "chunk_size": False},
                data_key=data_key,
                root=self.root,
                resource_path="/dataset.h5",
                uri="file://localhost" + file_path,
                spec="AD_HDF5_SWMR_STREAM",
                mimetype="application/x-hdf5",
                uid=uid,
            )
            # Initialize an empty HDF5 dataset (3D: var 1 dim, fixed 2 and 3 dims)
            with h5py.File(file_path, "a") as f:
                dset = f.require_dataset(
                    hdf5_dataset,
                    (0, *data_shape),
                    maxshape=(None, *data_shape),
                    dtype=np.dtype("double"),
                    chunks=(100, *data_shape),
                )

        indx_min, indx_max = self.counter, self.counter + index
        stream_datum = StreamDatum(
            stream_resource=uid,
            descriptor="",
            uid=f"{uid}/{self.counter}",
            indices={"start": indx_min, "stop": indx_max},
            seq_nums={"start": 0, "stop": 0},  # seq_nums will be overwritten by RunBundler
        )

        # Write (append to) the hdf5 dataset
        with h5py.File(file_path, "a") as f:
            dset = f[hdf5_dataset]
            dset.resize([indx_max, *data_shape])
            dset[indx_min:indx_max, ...] = np.random.randn(indx_max - indx_min, *data_shape)

        return stream_resource, stream_datum

    def _get_tiff_stream(self, data_key: str, index: int) -> Tuple[StreamResource, StreamDatum]:
        file_path = self.root
        for data_key in [f"{self.name}-sd3"]:
            uid = f"{data_key}-uid"
            data_desc = self.describe()[data_key]  # Descriptor dictionary for the current data key
            data_shape = tuple(data_desc["shape"])
            stream_resource = None
            if self.counter == 0:
                stream_resource = StreamResource(
                    parameters={"chunk_size": 1, "template": "{:05d}.tif"},
                    data_key=data_key,
                    root=self.root,
                    uri="file://localhost" + self.root + "/",
                    spec="AD_TIFF",
                    mimetype="multipart/related;type=image/tiff",
                    uid=uid,
                )

            indx_min, indx_max = self.counter, self.counter + index
            stream_datum = StreamDatum(
                stream_resource=uid,
                descriptor="",
                uid=f"{uid}/{self.counter}",
                indices={"start": indx_min, "stop": indx_max},
                seq_nums={"start": 0, "stop": 0},  # seq_nums will be overwritten by RunBundler
            )

            # Write a tiff file
            data = np.random.randint(0, 255, data_shape, dtype="uint8")
            tf.imwrite(file_path + f"/{self.counter:05}.tif", data)

        return stream_resource, stream_datum

    def describe(self) -> Dict[str, DataKey]:
        """Describe datasets which will be backed by StreamResources"""
        return {
            f"{self.name}-sd1": DataKey(source="file", dtype="number", shape=[1], external="STREAM:"),
            f"{self.name}-sd2": DataKey(source="file", dtype="array", shape=[10, 15], external="STREAM:"),
            f"{self.name}-sd3": DataKey(
                source="file", dtype="array", dtype_numpy="uint8", shape=[5, 7, 4], external="STREAM:"
            ),
        }

    def describe_collect(self) -> Dict[str, DataKey]:
        return self.describe()

    def collect_asset_docs(self, index: Optional[int] = None) -> Iterator[StreamAsset]:
        """Produce a StreamResource and StreamDatum for all data keys for 0:index"""
        index = index or 1
        data_keys_methods = {
            f"{self.name}-sd1": self._get_hdf5_stream,
            f"{self.name}-sd2": self._get_hdf5_stream,
            f"{self.name}-sd3": self._get_tiff_stream,
        }

        for data_key, method in data_keys_methods.items():
            stream_resource, stream_datum = method(data_key, index)
            if stream_resource is not None:
                yield "stream_resource", stream_resource
            yield "stream_datum", stream_datum

        self.counter += index

    def get_index(self) -> int:
        """Report how many frames were written"""
        return self.counter

    def read(self) -> Dict[str, Reading]:
        """Produce an empty event"""
        return {}


class SynSignalWithRegistry(ophyd.sim.SynSignalWithRegistry):
    """A readable image detector that writes a sequence of files and generates relevant Bluesky documents.

    Subclassed from ophyd.sim to match the updated schema of Resource documents.
    """

    def __init__(self, *args, dtype_numpy="uint8", **kwargs):
        self.dtype_numpy = dtype_numpy
        super().__init__(*args, **kwargs)

    def stage(self):
        super().stage()
        parameters = {"chunk_size": 1, "template": "_{:d}." + self.save_ext}
        self._asset_docs_cache[-1][1]["resource_kwargs"].update(parameters)

    def describe(self):
        res = super().describe()
        for key in res:
            res[key]["external"] = "FILESTORE"
            res[key]["dtype_numpy"] = self.dtype_numpy
        return res


def test_stream_datum_readable_counts(RE, client, tmp_path):
    tw = TiledWriter(client)
    det = StreamDatumReadableCollectable(name="det", root=str(tmp_path))
    RE(bp.count([det], 3), tw)
    arrs = client.values().last()["primary"]["external"].values()
    assert arrs[0].shape == (3,)
    assert arrs[1].shape == (3, 10, 15)
    assert arrs[2].shape == (3, 5, 7, 4)
    assert arrs[0].read() is not None
    assert arrs[1].read() is not None
    assert arrs[2].read() is not None


def test_stream_datum_collectable(RE, client, tmp_path):
    det = StreamDatumReadableCollectable(name="det", root=str(tmp_path))
    tw = TiledWriter(client)
    RE(collect_plan(det, name="primary"), tw)
    arrs = client.values().last()["primary"]["external"].values()

    assert arrs[0].read() is not None
    assert arrs[1].read() is not None
    assert arrs[2].read() is not None


def test_handling_non_stream_resource(RE, client, tmp_path):
    det = SynSignalWithRegistry(
        func=lambda: np.random.randint(0, 255, (10, 15), dtype="uint8"),
        dtype_numpy="uint8",
        name="img",
        labels={"detectors"},
        save_func=tf.imwrite,
        save_path=str(tmp_path),
        save_spec="AD_TIFF",
        save_ext="tif",
    )
    tw = TiledWriter(client)
    RE(bp.count([det], 3), tw)
    extr = client.values().last()["primary"]["external"]["img"]
    intr = client.values().last()["primary"]["internal"]["events"]
    conf = client.values().last()["primary"]["config"]["img"]

    assert extr.shape == (3, 10, 15)
    assert extr.read() is not None
    assert set(intr.columns) == {"seq_num", "ts_img"}
    assert len(intr.read()) == 3
    assert (intr["seq_num"].read() == [1, 2, 3]).all()
    assert set(conf.columns) == {"descriptor_uid", "img", "ts_img"}
    assert len(conf.read()) == 1


def collect_plan(*objs, name="primary"):
    yield from bps.open_run()
    yield from bps.declare_stream(*objs, collect=True, name=name)
    yield from bps.collect(*objs, return_payload=False, name=name)
    yield from bps.close_run()
