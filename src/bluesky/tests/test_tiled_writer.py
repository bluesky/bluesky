import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import h5py
import jinja2
import numpy as np
import ophyd.sim
import pytest
import tifffile as tf
from event_model.documents.event_descriptor import DataKey
from event_model.documents.stream_datum import StreamDatum
from event_model.documents.stream_resource import StreamResource
from tiled.client import record_history

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

rng = np.random.default_rng(12345)


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    tiled_catalog = pytest.importorskip("tiled.catalog")
    tmp_path = tmp_path_factory.mktemp("tiled_catalog")
    yield tiled_catalog.in_memory(
        writable_storage={"filesystem": str(tmp_path), "sql": f"duckdb:///{tmp_path}/test.db"},
        readable_storage=[str(tmp_path.parent)],
    )


@pytest.fixture(scope="module")
def app(catalog):
    tsa = pytest.importorskip("tiled.server.app")
    yield tsa.build_app(catalog)


@pytest.fixture(scope="module")
def context(app):
    tc = pytest.importorskip("tiled.client")
    with tc.Context.from_app(app) as context:
        yield context


@pytest.fixture(scope="module")
def client(context):
    tc = pytest.importorskip("tiled.client")
    yield tc.from_context(context)


@pytest.fixture(scope="module")
def external_assets_folder(tmp_path_factory):
    """External data files used with the saved documents."""
    # Create a temporary directory
    temp_dir = tmp_path_factory.mktemp("example_files")

    # Create an external hdf5 file
    with h5py.File(temp_dir.joinpath("dataset.h5"), "w") as file:
        grp = file.create_group("entry").create_group("data")
        grp.create_dataset("data_1", data=rng.random(size=(3,), dtype="float64"))
        grp.create_dataset("data_2", data=rng.integers(-10, 10, size=(3, 13, 17)), dtype="<i8")

    # Create a second external hdf5 file to be declared in a different stream resource
    with h5py.File(temp_dir.joinpath("dataset_part2.h5"), "w") as file:
        grp = file.create_group("entry").create_group("data")
        grp.create_dataset("data_2", data=rng.integers(-10, 10, size=(5, 13, 17)), dtype="<i8")

    # Create a sequence of tiff files
    (temp_dir / "tiff_files").mkdir(parents=True, exist_ok=True)
    for i in range(3):
        data = rng.integers(0, 255, size=(1, 10, 15), dtype="uint8")
        tf.imwrite(temp_dir.joinpath("tiff_files", f"img_{i:05}.tif"), data)

    yield str(temp_dir.absolute()).replace("\\", "/")


def render_templated_documents(fname: str, data_dir: str):
    dirpath = str(Path(__file__).parent.joinpath("examples"))
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(dirpath), autoescape=False)
    template = env.get_template(fname)
    rendered = template.render(root_path=data_dir, uuid=str(uuid.uuid4())[:-12])

    yield from json.loads(rendered)


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

    def _get_hdf5_stream(self, data_key: str, index: int) -> tuple[StreamResource | None, StreamDatum]:
        file_path = os.path.join(self.root, "dataset.h5")
        uid = f"{data_key}-uid"
        data_desc = self.describe()[data_key]  # Descriptor dictionary for the current data key
        data_shape = cast(tuple[int, ...], tuple(data_desc["shape"]))
        hdf5_dataset = f"/{data_key}/VALUE"

        stream_resource = None
        if self.counter == 0:
            # Backward compatibility test, ignore typing errors
            stream_resource = StreamResource(  # type: ignore[typeddict-unknown-key]
                parameters={"dataset": hdf5_dataset, "chunk_shape": (100, *data_shape[1:]), "_validate": True},
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

    def _get_tiff_stream(self, data_key: str, index: int) -> tuple[StreamResource | None, StreamDatum]:
        file_path = self.root
        for data_key in [f"{self.name}-sd3"]:
            uid = f"{data_key}-uid"
            data_desc = self.describe()[data_key]  # Descriptor dictionary for the current data key
            data_shape = cast(tuple[int, ...], tuple(data_desc["shape"]))
            stream_resource = None
            if self.counter == 0:
                # Backward compatibility test, ignore typing errors
                stream_resource = StreamResource(  # type: ignore[typeddict-unknown-key]
                    parameters={
                        "chunk_shape": (1, *data_shape),
                        "template": "{:05d}.tif",
                        "join_method": "stack",
                        "_validate": True,
                    },
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

    def describe_collect(self) -> dict[str, DataKey] | dict[str, dict[str, DataKey]]:
        return self.describe()

    def collect_asset_docs(self, index: int | None = None) -> Iterator[StreamAsset]:
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
        parameters = {
            "chunk_shape": (1,),
            "template": "_{:d}." + self.save_ext,
            "join_method": "stack",
            "_validate": True,
        }
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
    stream = client.values().last()["primary"]
    keys = sorted(set(stream.base.keys()).difference({"internal"}))

    assert stream[keys[0]].shape == (3,)
    assert stream[keys[1]].shape == (15, 10, 15)
    assert stream[keys[2]].shape == (3, 10, 5, 7, 4)
    assert stream[keys[0]].read() is not None
    assert stream[keys[1]].read() is not None
    assert stream[keys[2]].read() is not None


def test_stream_datum_readable_with_two_detectors(RE, client, tmp_path):
    det1 = StreamDatumReadableCollectable(name="det1", root=str(tmp_path))
    det2 = StreamDatumReadableCollectable(name="det2", root=str(tmp_path))
    tw = TiledWriter(client)
    RE(bp.count([det1, det2], 3), tw)
    stream = client.values().last()["primary"]
    keys = sorted(set(stream.base.keys()).difference({"internal"}))

    assert stream[keys[0]].shape == (3,)
    assert stream[keys[1]].shape == (15, 10, 15)
    assert stream[keys[2]].shape == (3, 10, 5, 7, 4)
    assert stream[keys[3]].shape == (3,)
    assert stream[keys[4]].shape == (15, 10, 15)
    assert stream[keys[5]].shape == (3, 10, 5, 7, 4)
    assert stream[keys[0]].read() is not None
    assert stream[keys[1]].read() is not None
    assert stream[keys[2]].read() is not None
    assert stream[keys[3]].read() is not None
    assert stream[keys[4]].read() is not None
    assert stream[keys[5]].read() is not None


def test_stream_datum_collectable(RE, client, tmp_path):
    det = StreamDatumReadableCollectable(name="det", root=str(tmp_path))
    tw = TiledWriter(client)
    RE(collect_plan(det, name="primary"), tw)
    stream = client.values().last()["primary"]
    keys = sorted(set(stream.base.keys()).difference({"internal"}))

    assert stream[keys[0]].read() is not None
    assert stream[keys[1]].read() is not None
    assert stream[keys[2]].read() is not None


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
    extr = client.values().last()["primary"].base["img"]
    intr = client.values().last()["primary"].base["internal"]
    assert extr.shape == (3, frames_per_event, 10, 15)
    assert extr.read() is not None
    assert set(intr.columns) == {"seq_num", "time"}
    assert len(intr.read()) == 3
    assert (intr["seq_num"].read() == [1, 2, 3]).all()


def collect_plan(*objs, name="primary"):
    yield from bps.open_run()
    yield from bps.declare_stream(*objs, collect=True, name=name)
    yield from bps.collect(*objs, return_payload=False, name=name)
    yield from bps.close_run()


@pytest.mark.parametrize("fname", ["internal_events", "external_assets", "external_assets_legacy"])
@pytest.mark.parametrize("batch_size", [0, 1, 1000, None])
def test_with_correct_sample_runs(client, batch_size, external_assets_folder, fname):
    if batch_size is None:
        tw = TiledWriter(client)
    else:
        tw = TiledWriter(client, batch_size=batch_size)
    for item in render_templated_documents(fname + ".json", external_assets_folder):
        if item["name"] == "start":
            uid = item["doc"]["uid"]
        tw(**item)

    run = client[uid]

    for stream in run.values():
        assert stream.read() is not None


@pytest.mark.parametrize("error_type", ["shape", "chunks", "dtype"])
@pytest.mark.parametrize("validate", [True, False])
def test_validate_external_data(client, external_assets_folder, error_type, validate):
    tw = TiledWriter(client)

    documents = render_templated_documents("external_assets_single_key.json", external_assets_folder)
    for item in documents:
        name, doc = item["name"], item["doc"]
        if name == "start":
            uid = doc["uid"]

        # Modify the document to introduce an error
        if (error_type == "shape") and (name == "descriptor"):
            doc["data_keys"]["det-key2"]["shape"] = [1, 2, 3]  # should be [1, 13, 17]
        elif (error_type == "chunks") and name in {"resource", "stream_resource"}:
            doc["parameters"]["chunk_shape"] = [1, 2, 3]  # should be [100, 13, 17]
        elif (error_type == "dtype") and (name == "descriptor"):
            doc["data_keys"]["det-key2"]["dtype_numpy"] = np.dtype("int32").str  # should be "int64"

        # Add flag to trigger validation
        if name in {"resource", "stream_resource"} and validate:
            doc["parameters"]["_validate"] = True

        # Check that the warning is issued when data changes during the validation
        if name == "stop" and validate:
            with pytest.warns(UserWarning):
                tw(name, doc)
        else:
            tw(name, doc)

    # Try reading the imported data
    run = client[uid]
    if not validate and not error_type == "chunks":
        with pytest.raises(ValueError):
            assert run["primary"].read() is not None
    else:
        assert run["primary"].read() is not None
        assert run["primary"]["det-key2"].read().shape == (8, 13, 17)


@pytest.mark.parametrize("squeeze", [True, False])
def test_slice_and_squeeze(client, external_assets_folder, squeeze):
    tw = TiledWriter(client)

    documents = render_templated_documents("external_assets_single_key.json", external_assets_folder)
    for item in documents:
        name, doc = item["name"], item["doc"]
        if name == "start":
            uid = doc["uid"]

        # Modify the documents to add slice and squeeze parameters
        if name == "descriptor":
            doc["data_keys"]["det-key2"]["shape"] = [1, 17] if squeeze else [1, 5, 17]
        elif name in {"resource", "stream_resource"}:
            doc["parameters"]["slice"] = ":,5,:" if squeeze else ":,:5,:"
            doc["parameters"]["squeeze"] = squeeze
            doc["parameters"]["chunk_shape"] = [1]

        tw(name, doc)

    # Try reading the imported data
    assert client[uid]["primary"].read() is not None


def test_legacy_multiplier_parameter(client, external_assets_folder):
    tw = TiledWriter(client)

    documents = render_templated_documents("external_assets_single_key.json", external_assets_folder)
    for item in documents:
        name, doc = item["name"], item["doc"]
        if name == "start":
            uid = doc["uid"]

        # Modify the documents to add slice and squeeze parameters
        if name == "descriptor":
            doc["data_keys"]["det-key2"]["shape"] = [13, 17]
        elif name in {"resource", "stream_resource"}:
            doc["parameters"]["multiplier"] = 1

        tw(name, doc)

    # Try reading the imported data
    assert client[uid]["primary"].read() is not None


def test_streams_with_no_events(client, external_assets_folder):
    tw = TiledWriter(client)

    for item in render_templated_documents("external_assets_single_key.json", external_assets_folder):
        name, doc = item["name"], item["doc"]
        if name == "start":
            uid = doc["uid"]

        # Skip the resource and datum documents
        if name in {"resource", "stream_resource", "datum", "stream_datum", "event"}:
            continue

        tw(name, doc)

    # Try reading the data -- should return an empty dataset
    assert client[uid]["primary"].read() is not None
    assert client[uid]["primary"].read().data_vars == {}
    assert client[uid]["primary"].metadata is not None


@pytest.mark.parametrize("include_data_sources", [True, False])
@pytest.mark.parametrize("fname", ["internal_events", "external_assets", "external_assets_legacy"])
def test_zero_gets(client, external_assets_folder, fname, include_data_sources):
    client = client.new_variation(include_data_sources=include_data_sources)
    assert client._include_data_sources == include_data_sources
    tw = TiledWriter(client)
    assert bool(tw.client._include_data_sources)

    with record_history() as history:
        for item in render_templated_documents(fname + ".json", external_assets_folder):
            tw(**item)

    # Count the number of GET requests
    num_gets = sum(1 for req in history.requests if req.method == "GET")
    assert num_gets == 0


def test_bad_document_order(client, external_assets_folder):
    """Test that the TiledWriter can handle documents in a different order than expected

    Emit datum documents in the end, before the Stop document, but after corresponding Event documents.
    """
    tw = TiledWriter(client)

    document_cache = []
    for item in render_templated_documents("external_assets_legacy.json", external_assets_folder):
        name, doc = item["name"], item["doc"]
        if name == "start":
            uid = doc["uid"]

        if name == "datum":
            document_cache.append({"name": name, "doc": doc})
            continue

        if name == "stop":
            for cached_item in document_cache:
                tw(**cached_item)

        tw(**item)

    run = client[uid]

    for stream in run.values():
        assert stream.read() is not None
        assert "time" in stream.keys()
        assert "seq_num" in stream.keys()
        assert len(stream.keys()) > 2  # There's at least one data key in addition to time and seq_num


def test_json_backup(client, tmpdir, monkeypatch):
    def patched_event(name, doc):
        raise RuntimeError("This is a test error to check the backup functionality")

    monkeypatch.setattr("bluesky.callbacks.tiled_writer._RunWriter.event", patched_event)

    tw = TiledWriter(client, backup_directory=str(tmpdir))

    for item in render_templated_documents("internal_events.json", ""):
        name, doc = item["name"], item["doc"]
        if name == "start":
            uid = doc["uid"]
        print(name)

        tw(**item)

    run = client[uid]

    assert "primary" in run  # The Descriptor was processed and the primary stream was created
    assert run["primary"].read() is not None  # The stream can be read
    assert len(run["primary"].read()) == 0  # No events were processed due to the error
    assert "stop" in run.metadata  # The TiledWriter did not crash

    # Check that the backup file was created
    filepath = tmpdir / f"{uid[:8]}.jsonl"
    assert filepath.exists()
    with open(filepath) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 7
    assert lines[0]["name"] == "start"
    assert lines[1]["name"] == "descriptor"
    assert lines[2]["name"].startswith("event")
    assert lines[6]["name"] == "stop"
