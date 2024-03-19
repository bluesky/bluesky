import h5py
import pandas as pd
from event_model import DocumentRouter, RunRouter
from tiled.client import from_profile, from_uri
from tiled.structures.array import ArrayStructure, BuiltinDtype
from tiled.structures.core import Spec, StructureFamily
from tiled.structures.data_source import Asset, DataSource, Management

MIMETYPE_LOOKUP = {
    "hdf5": "application/x-hdf5",
    "ADHDF5_SWMR_STREAM": "application/x-hdf5",
    "AD_HDF5_SWMR_SLICE": "application/x-hdf5",
}


class TiledWriter:
    "Write metadata and data from Bluesky documents into Tiled."

    def __init__(self, client):
        self.client = client
        self._run_router = RunRouter([self._factory])

    def _factory(self, name, doc):
        return [_RunWriter(self.client)], []

    @classmethod
    def from_uri(cls, uri, **kwargs):
        client = from_uri(uri, **kwargs)
        return cls(client)

    @classmethod
    def from_profile(cls, profile, **kwargs):
        client = from_profile(profile, **kwargs)
        return cls(client)

    def __call__(self, name, doc):
        self._run_router(name, doc)


class _RunWriter(DocumentRouter):
    "Write the document from one Bluesky Run into Tiled."

    def __init__(self, client):
        self.client = client
        self.node = None
        self._descriptor_nodes = {}  # references to descriptor containers by uid's
        self._SR_nodes = {}
        self._SR_cache = {}

    def start(self, doc):
        self.node = self.client.create_container(
            key=doc["uid"],
            metadata={"start": doc},
            specs=[Spec("BlueskyRun", version="1.0")],
        )

    def stop(self, doc):
        metadata = dict(self.node.metadata) | {"stop": doc}
        self.node.update_metadata(metadata=metadata)

    def descriptor(self, doc):
        self._descriptor_nodes[doc["uid"]] = self.node.create_container(
            key=doc["name"], metadata=doc
        )

    def event(self, doc):
        parent_node = self._descriptor_nodes[doc["descriptor"]]
        parent_node.write_dataframe(
            pd.DataFrame({column: [value] for column, value in doc["data"].items()}),
            key="data",
        )
        parent_node.write_dataframe(
            pd.DataFrame(
                {column: [value] for column, value in doc["timestamps"].items()}
            ),
            key="timestamps",
        )

    def stream_resource(self, doc):
        # Only cache the StreamResource; add the node when at least one StreamDatum is added
        self._SR_cache[doc["uid"]] = doc

    def stream_datum(self, doc):
        descriptor_node = self._descriptor_nodes[doc["descriptor"]]

        num_rows = (
            doc["indices"]["stop"] - doc["indices"]["start"]
        )  # Number of rows added by new StreamDatum

        # Get the Stream Resource node if it already exists or register if from a cached SR document
        try:
            SR_node = self._SR_nodes[doc["stream_resource"]]

        except KeyError:
            # Register a new (empty) Stream Resource
            SR_doc = self._SR_cache.pop(doc["stream_resource"])

            # POST /api/v1/register/{path}
            file_path = (
                "/"
                + SR_doc["root"].strip("/")
                + "/"
                + SR_doc["resource_path"].strip("/")
            )
            data_path = SR_doc["resource_kwargs"]["path"].strip("/")
            data_uri = "file://localhost" + file_path
            assets = [
                Asset(data_uri=data_uri, is_directory=False, parameter="data_uri")
            ]
            data_key = SR_doc["data_key"]
            desc = dict(descriptor_node.metadata)["data_keys"][data_key]
            if desc["dtype"] == "array":
                data_shape = desc["shape"]
            elif desc["dtype"] == "number":
                data_shape = ()
            with h5py.File(file_path, "r") as f:
                data_type = BuiltinDtype.from_numpy_dtype(f[data_path].dtype)

            SR_node = descriptor_node.new(
                structure_family=StructureFamily.array,
                data_sources=[
                    DataSource(
                        assets=assets,
                        mimetype=MIMETYPE_LOOKUP[SR_doc["spec"]],
                        structure_family=StructureFamily.array,
                        structure=ArrayStructure(
                            data_type=data_type,
                            shape=[0, *data_shape],
                            chunks=[[0]] + [[d] for d in data_shape],
                        ),
                        parameters={"path": data_path.split("/")},
                        management=Management.external,
                    )
                ],
                metadata={},
                specs=[],
            )

            self._SR_nodes[SR_doc["uid"]] = SR_node

        # Append StreamDatum to an existing StreamResource (by overwriting it with changed shape)
        url = SR_node.uri.replace("/metadata/", "/data_source/")
        SR_node.refresh()
        ds_dict = SR_node.data_sources()[0]
        ds_dict["structure"]["shape"][0] += num_rows
        ds_dict["structure"]["chunks"][0] = [ds_dict["structure"]["shape"][0]]
        SR_node.context.http_client.put(
            url, json={"data_source": ds_dict}, params={"data_source": 1}
        )
