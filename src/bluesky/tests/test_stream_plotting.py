from typing import Dict, Iterator, Optional, Tuple

import h5py
import numpy as np
import tifffile as tf
from event_model.documents.event_descriptor import DataKey
from event_model.documents.stream_datum import StreamDatum
from event_model.documents.stream_resource import StreamResource

import bluesky.plan_stubs as bps
from bluesky.callbacks.core import CollectLiveStream
from bluesky.protocols import (
    Collectable,
    HasName,
    Readable,
    Reading,
    StreamAsset,
    WritesStreamAssets,
)


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
        hdf5_path = f"/{data_key}/VALUE"

        stream_resource = None
        if self.counter == 0:
            stream_resource = StreamResource(
                parameters={"path": hdf5_path, "chunk_size": False},
                data_key=data_key,
                root=self.root,
                resource_path="/dataset.h5",
                uri="file://localhost" + file_path,
                spec="ADHDF5_SWMR_STREAM",
                mimetype="application/x-hdf5",
                uid=uid,
            )
            # Initialize an empty HDF5 dataset (3D: var 1 dim, fixed 2 and 3 dims)
            with h5py.File(file_path, "a") as f:
                dset = f.require_dataset(
                    hdf5_path,
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
            dset = f[hdf5_path]
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
                    uri="file://localhost" + self.root,
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
                source="file", dtype="array", dtype_str="uint8", shape=[5, 7, 4], external="STREAM:"
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


def test_stream_datum_collectable(RE, tmp_path):
    det = StreamDatumReadableCollectable(name="det", root=str(tmp_path))
    # tw = TiledWriter(client)
    collector = CollectLiveStream()
    RE(collect_plan(det, name="primary"), collector)
    # arrs = client.values().last()["primary"]["external"].values()
    # assert arrs[0].read() is not None
    # assert arrs[1].read() is not None
    # assert arrs[2].read() is not None


def collect_plan(*objs, name="primary"):
    yield from bps.open_run()
    yield from bps.declare_stream(*objs, collect=True, name=name)
    yield from bps.collect(*objs, return_payload=False, name=name)
    yield from bps.close_run()
