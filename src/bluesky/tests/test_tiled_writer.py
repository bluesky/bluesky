import os
from collections.abc import Iterator
from typing import Optional, Union

import h5py
import numpy as np
import ophyd.sim
import pytest
import tifffile as tf
from event_model.documents.event_descriptor import DataKey
from event_model.documents.stream_datum import StreamDatum
from event_model.documents.stream_resource import StreamResource

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
    tiled_catalog = pytest.importorskip("tiled.catalog")
    yield tiled_catalog.in_memory(writable_storage=str(tmp_path))


@pytest.fixture
def app(catalog):
    tsa = pytest.importorskip("tiled.server.app")
    yield tsa.build_app(catalog)


@pytest.fixture
def context(app):
    tc = pytest.importorskip("tiled.client")
    with tc.Context.from_app(app) as context:
        yield context


@pytest.fixture
def client(context):
    tc = pytest.importorskip("tiled.client")
    yield tc.from_context(context)


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

    def _get_hdf5_stream(self, data_key: str, index: int) -> tuple[Optional[StreamResource], StreamDatum]:
        file_path = os.path.join(self.root, "dataset.h5")
        uid = f"{data_key}-uid"
        data_desc = self.describe()[data_key]  # Descriptor dictionary for the current data key
        data_shape = tuple(data_desc["shape"])
        hdf5_dataset = f"/{data_key}/VALUE"

        stream_resource = None
        if self.counter == 0:
            # Backward compatibility test, ignore typing errors
            stream_resource = StreamResource(  # type: ignore[typeddict-unknown-key]
                parameters={"dataset": hdf5_dataset, "chunk_shape": (100, *data_shape[1:])},
                data_key=data_key,
                root=self.root,
                resource_path="/dataset.h5",
                uri="file://localhost/" + file_path,
                spec="AD_HDF5_SWMR_STREAM",
                mimetype="application/x-hdf5",
                uid=uid,
            )
            # Initialize an empty HDF5 dataset (3D: var 1 dim, fixed 2 and 3 dims)
            with h5py.File(file_path, "a") as f:
                dset = f.require_dataset(
                    hdf5_dataset,
                    data_shape,
                    maxshape=(None, *data_shape[1:]),
                    dtype=np.dtype("float64"),
                    chunks=(100, *data_shape[1:]),
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
            dset.resize([indx_max * data_shape[0], *data_shape[1:]])
            dset[indx_min * data_shape[0] : indx_max * data_shape[0], ...] = np.random.randn(
                (indx_max - indx_min) * data_shape[0], *data_shape[1:]
            )

        return stream_resource, stream_datum

    def _get_tiff_stream(self, data_key: str, index: int) -> tuple[Optional[StreamResource], StreamDatum]:
        file_path = self.root
        for data_key in [f"{self.name}-sd3"]:
            uid = f"{data_key}-uid"
            data_desc = self.describe()[data_key]  # Descriptor dictionary for the current data key
            data_shape = tuple(data_desc["shape"])
            stream_resource = None
            if self.counter == 0:
                # Backward compatibility test, ignore typing errors
                stream_resource = StreamResource(  # type: ignore[typeddict-unknown-key]
                    parameters={"chunk_shape": (1, *data_shape), "template": "{:05d}.tif", "stackable": True},
                    data_key=data_key,
                    root=self.root,
                    uri="file://localhost/" + self.root + "/",
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
            tf.imwrite(os.path.join(file_path, f"{self.counter:05}.tif"), data)

        return stream_resource, stream_datum

    def describe(self) -> dict[str, DataKey]:
        """Describe datasets which will be backed by StreamResources"""
        return {
            # Numerical data with 1 number per event in hdf5 format
            f"{self.name}-sd1": DataKey(
                source="file",
                dtype="number",
                dtype_numpy=np.dtype("float64").str,
                shape=[
                    1,
                ],
                external="STREAM:",
            ),
            # 2-D data with 5 frames per event in hdf5 format
            f"{self.name}-sd2": DataKey(
                source="file",
                dtype="array",
                dtype_numpy=np.dtype("float64").str,
                shape=[5, 10, 15],
                external="STREAM:",
            ),
            # 3-D data with 10 frames per event in tiff format
            f"{self.name}-sd3": DataKey(
                source="file",
                dtype="array",
                dtype_numpy=np.dtype("uint8").str,
                shape=[10, 5, 7, 4],
                external="STREAM:",
            ),
        }

    def describe_collect(self) -> Union[dict[str, DataKey], dict[str, dict[str, DataKey]]]:
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

    def read(self) -> dict[str, Reading]:
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
        parameters = {"chunk_shape": (1,), "template": "_{:d}." + self.save_ext, "stackable": True}
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
    assert arrs[1].shape == (15, 10, 15)
    assert arrs[2].shape == (3, 10, 5, 7, 4)
    assert arrs[0].read() is not None
    assert arrs[1].read() is not None
    assert arrs[2].read() is not None


def test_stream_datum_readable_with_two_detectors(RE, client, tmp_path):
    det1 = StreamDatumReadableCollectable(name="det1", root=str(tmp_path))
    det2 = StreamDatumReadableCollectable(name="det2", root=str(tmp_path))
    tw = TiledWriter(client)
    RE(bp.count([det1, det2], 3), tw)
    arrs = client.values().last()["primary"]["external"].values()

    assert arrs[0].shape == (3,)
    assert arrs[1].shape == (15, 10, 15)
    assert arrs[2].shape == (3, 10, 5, 7, 4)
    assert arrs[3].shape == (3,)
    assert arrs[4].shape == (15, 10, 15)
    assert arrs[5].shape == (3, 10, 5, 7, 4)
    assert arrs[0].read() is not None
    assert arrs[1].read() is not None
    assert arrs[2].read() is not None
    assert arrs[3].read() is not None
    assert arrs[4].read() is not None
    assert arrs[5].read() is not None


def test_stream_datum_collectable(RE, client, tmp_path):
    det = StreamDatumReadableCollectable(name="det", root=str(tmp_path))
    tw = TiledWriter(client)
    RE(collect_plan(det, name="primary"), tw)
    arrs = client.values().last()["primary"]["external"].values()

    assert arrs[0].read() is not None
    assert arrs[1].read() is not None
    assert arrs[2].read() is not None


@pytest.mark.parametrize("frames_per_event", [1, 5, 10])
def test_handling_non_stream_resource(RE, client, tmp_path, frames_per_event):
    det = SynSignalWithRegistry(
        func=lambda: np.random.randint(0, 255, (frames_per_event, 10, 15), dtype="uint8"),
        dtype_numpy=np.dtype("uint8").str,
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
    # breakpoint()
    assert extr.shape == (3, frames_per_event, 10, 15)
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
