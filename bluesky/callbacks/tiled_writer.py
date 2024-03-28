from collections.abc import Iterable

import numpy as np
import pandas as pd
from event_model import DocumentRouter, RunRouter
from event_model.documents import StreamDatum, StreamResource
from pydantic.utils import deep_update
from tiled.client import from_profile, from_uri
from tiled.structures.array import ArrayStructure, BuiltinDtype
from tiled.structures.core import Spec, StructureFamily
from tiled.structures.data_source import Asset, DataSource, Management
from tiled.structures.table import TableStructure

MIMETYPE_LOOKUP = {
    "hdf5": "application/x-hdf5",
    "ADHDF5_SWMR_STREAM": "application/x-hdf5",
    "AD_HDF5_SWMR_SLICE": "application/x-hdf5",
    "AD_TIFF": "image/tiff",
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
        self.root_node = None
        self._descriptor_nodes = {}  # references to descriptor containers by uid's
        self._sr_nodes = {}
        self._sr_cache = {}

    def start(self, doc):
        self.root_node = self.client.create_container(
            key=doc["uid"],
            metadata={"start": doc},
            specs=[Spec("BlueskyRun", version="1.0")],
        )

    def stop(self, doc):
        doc = dict(doc)
        doc.pop("start", None)
        metadata = dict(self.root_node.metadata) | {"stop": doc}
        self.root_node.update_metadata(metadata=metadata)

    def descriptor(self, doc):
        desc_name = doc["name"]
        metadata = dict(doc)

        # Remove variable fields of the metadata and encapsulate them into sub-dictionaries
        uid = metadata.pop("uid")
        conf_dict = {uid: metadata.pop("configuration", {})}
        time_dict = {uid: metadata.pop("time")}
        var_fields = {"configuration": conf_dict, "time": time_dict}

        if desc_name not in self.root_node.keys():
            # Create a new descriptor node; write only the fixed part of the metadata
            desc_node = self.root_node.create_container(
                key=desc_name, metadata=metadata
            )
            desc_node.create_container(key="external")
            desc_node.create_container(key="internal")
        else:
            # Get existing descriptor node (with fixed and variable metadata saved before)
            desc_node = self.root_nodes[desc_name]

        # Update (add new values to) variable fields of the metadata
        metadata = deep_update(dict(desc_node.metadata), var_fields)
        desc_node.update_metadata(metadata)
        self._descriptor_nodes[uid] = desc_node

    def event(self, doc: dict):
        descriptor_node = self._descriptor_nodes[doc["descriptor"]]
        parent_node = descriptor_node["internal"]
        df = pd.DataFrame(
            {c: [v] for c, v in doc["data"].items()}
            | {f"ts_{c}": [v] for c, v in doc["timestamps"].items()}
        )
        if "events" in parent_node.keys():
            parent_node["events"].append_partition(df, 0)
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
                key="events",
            )
            parent_node["events"].write_partition(df, 0)

    def stream_resource(self, doc: StreamResource):
        # Only cache the StreamResource; add the node when at least one StreamDatum is added
        self._sr_cache[doc["uid"]] = doc

    def stream_datum(self, doc: StreamDatum):
        descriptor_node = self._descriptor_nodes[doc["descriptor"]]
        parent_node = descriptor_node["external"]

        num_new_rows = (
            doc["indices"]["stop"] - doc["indices"]["start"]
        )  # Number of rows added by new StreamDatum

        # Get the Stream Resource node if it already exists or register if from a cached SR document
        sr_uid = doc["stream_resource"]
        if sr_uid in self._sr_nodes.keys():
            sr_node = self._sr_nodes[sr_uid]
        elif sr_uid in self._sr_cache.keys():
            # Register a new (empty) Stream Resource
            sr_doc = self._sr_cache.pop(sr_uid)

            # POST /api/v1/register/{path}
            file_path = (
                sr_doc["root"].strip("/") + "/" + sr_doc["resource_path"].strip("/")
            )
            data_path = sr_doc["resource_kwargs"]["path"].strip("/")
            data_uri = "file://localhost/" + file_path
            data_key = sr_doc["data_key"]
            assets = [
                Asset(data_uri=data_uri, is_directory=False, parameter="data_uri")
            ]
            data_desc = dict(descriptor_node.metadata)["data_keys"][data_key]
            data_shape = tuple(data_desc["shape"])
            data_shape = data_shape if data_shape != (1,) else ()
            data_type = np.dtype(
                data_desc.get("dtype_str", "<f8")
            )  # Find machine dtype; assume '<f8' by default

            sr_node = parent_node.new(
                key=data_key,
                structure_family=StructureFamily.array,
                data_sources=[
                    DataSource(
                        assets=assets,
                        mimetype=MIMETYPE_LOOKUP[sr_doc["spec"]],
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
                metadata=sr_doc,
                specs=[],
            )
            self._sr_nodes[sr_uid] = sr_node
        else:
            raise RuntimeError(
                f"Stream Resource {sr_uid} is referenced before being declared."
            )

        # Append StreamDatum to an existing StreamResource (by overwriting it with changed shape)
        url = sr_node.uri.replace("/metadata/", "/data_source/")
        sr_node.refresh()
        ds_dict = sr_node.data_sources()[0]
        ds_dict["structure"]["shape"][0] += num_new_rows

        # Set up the chunk size based on the Stream Resource parameter `chunk_size`:
        #    None -- single chunk for all existing and new elements
        #    int -- fixed-sized chunks with at most `chunk_size` elements, last chunk can be smaller
        #    list[int] -- new elements are chunked according to the provided specs
        chunk_size = sr_node.metadata["resource_kwargs"].get("chunk_size", None)
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
        sr_node.context.http_client.put(
            url, json={"data_source": ds_dict}, params={"data_source": 1}
        )
