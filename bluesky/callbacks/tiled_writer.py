from collections.abc import Iterable

import numpy as np
import pandas as pd
from event_model import DocumentRouter, RunRouter
from tiled.client import from_profile, from_uri
from tiled.structures.array import ArrayStructure, BuiltinDtype
from tiled.structures.core import Spec, StructureFamily
from tiled.structures.data_source import Asset, DataSource, Management
from tiled.structures.table import TableStructure

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
        descriptor_node = self.node.create_container(key=doc["name"], metadata=doc)
        self._descriptor_nodes[doc["uid"]] = descriptor_node
        descriptor_node.create_container(key="external")
        descriptor_node.create_container(key="internal")

    def event(self, doc):
        descriptor_node = self._descriptor_nodes[doc["descriptor"]]
        parent_node = descriptor_node["internal"]
        df = pd.DataFrame(
            {c: [v] for c, v in doc["data"].items()}
            | {f"ts_{c}": [v] for c, v in doc["timestamps"].items()}
        )
        if "data" in parent_node.keys():
            parent_node["data"].append_partition(df, 0)
        else:
            parent_node.new(
                structure_family=StructureFamily.table,
                data_sources=[
                    DataSource(
                        structure_family=StructureFamily.table,
                        structure=TableStructure.from_pandas(df),
                        mimetype="text/csv",
                    ),  # or PARQUET_MIMETYPE
                ],
                key="data",
            )
            parent_node["data"].write_partition(df, 0)

    def stream_resource(self, doc):
        # Only cache the StreamResource; add the node when at least one StreamDatum is added
        self._SR_cache[doc["uid"]] = doc

    def stream_datum(self, doc):
        descriptor_node = self._descriptor_nodes[doc["descriptor"]]
        parent_node = descriptor_node["external"]

        num_new_rows = (
            doc["indices"]["stop"] - doc["indices"]["start"]
        )  # Number of rows added by new StreamDatum

        # Get the Stream Resource node if it already exists or register if from a cached SR document
        SR_uid = doc["stream_resource"]
        if SR_uid in self._SR_nodes.keys():
            SR_node = self._SR_nodes[SR_uid]
        elif SR_uid in self._SR_cache.keys():
            # Register a new (empty) Stream Resource
            SR_doc = self._SR_cache.pop(SR_uid)

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
            data_desc = dict(descriptor_node.metadata)["data_keys"][data_key]
            if data_desc["dtype"] == "array":
                data_shape = data_desc["shape"]
            elif data_desc["dtype"] == "number":
                data_shape = ()
            data_type = np.dtype(
                data_desc.get("dtype_str", "<f8")
            )  # Find machine dtype; assume '<f8' by default

            SR_node = parent_node.new(
                structure_family=StructureFamily.array,
                data_sources=[
                    DataSource(
                        assets=assets,
                        mimetype=MIMETYPE_LOOKUP[SR_doc["spec"]],
                        structure_family=StructureFamily.array,
                        structure=ArrayStructure(
                            data_type=BuiltinDtype.from_numpy_dtype(data_type),
                            shape=[0, *data_shape],
                            chunks=[[]] + [[d] for d in data_shape],
                        ),
                        parameters={"path": data_path.split("/")},
                        management=Management.external,
                    )
                ],
                metadata=SR_doc,
                specs=[],
            )
            self._SR_nodes[SR_uid] = SR_node
        else:
            raise RuntimeError(
                f"Stream Resource {SR_uid} is referenced before being declared."
            )

        # Append StreamDatum to an existing StreamResource (by overwriting it with changed shape)
        url = SR_node.uri.replace("/metadata/", "/data_source/")
        SR_node.refresh()
        ds_dict = SR_node.data_sources()[0]
        ds_dict["structure"]["shape"][0] += num_new_rows

        # Set up the chunk size based on the Stream Resource parameter `chunk_size`:
        #    None -- single chunk for all existing and new elements
        #    int -- fixed-sized chunks with at most `chunk_size` elements, last chunk can be smaller
        #    list[int] -- new elements are chunked according to the provided specs
        chunk_size = SR_node.metadata["resource_kwargs"].get("chunk_size", None)
        chunk_spec = ds_dict["structure"]["chunks"][0]
        if isinstance(chunk_size, Iterable):
            chunk_spec.extend(chunk_size)
        else:
            num_all_rows = ds_dict["structure"]["shape"][0]
            chunk_size = chunk_size or num_all_rows
            chunk_spec.clear()
            chunk_spec.extend([chunk_size] * int(num_all_rows / chunk_size))
            if num_all_rows % chunk_size:
                chunk_spec.append(num_all_rows % chunk_size)
        SR_node.context.http_client.put(
            url, json={"data_source": ds_dict}, params={"data_source": 1}
        )
