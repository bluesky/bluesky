import dataclasses
from abc import abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
from event_model import DocumentRouter, RunRouter
from event_model.documents import StreamDatum, StreamResource
from pydantic.utils import deep_update
from tiled.client import from_profile, from_uri
from tiled.client.utils import handle_error
from tiled.structures.array import ArrayStructure, BuiltinDtype
from tiled.structures.core import Spec, StructureFamily
from tiled.structures.data_source import Asset, DataSource, Management
from tiled.structures.table import TableStructure
from tiled.utils import safe_json_dump

MIMETYPE_LOOKUP = {
    "hdf5": "application/x-hdf5",
    "ADHDF5_SWMR_STREAM": "application/x-hdf5",
    "AD_HDF5_SWMR_SLICE": "application/x-hdf5",
    "AD_TIFF": "multipart/related;type=image/tiff",
}
DTYPE_LOOKUP = {"number": "<f8", "array": "<f8", "boolean": "bool", "string": "str", "integer": "int"}


# TODO (probably in a future PR?):
# Move these into external repo (probably area-detector-handlers)
# and use the existing handler discovery mechanism.

"""
A Handler consumes documents from Bluesky. It composes details (DataSource and its Assets) that will go into the
Tiled database. Each Handler (more specifically, a StreamHandler) is instantiated per a Stream Resource.

Tiled Adapters will later use this to read the data, with good random access and bulk access support.

We put this code into handlers so that additional, possibly very unusual, formats can be supported by users without
getting a PR merged into Bluesky or Tiled.

The STREAM_HANDLER_REGISTRY (see example below) and the Tiled catalog paramter adapters_by_mimetype can together be
used to support:
    - ingesting a new mimetype from Bluesky documents and generating DataSource and Asset with appropriate
      parameters (the handler's job);
    - Interpreting those DataSource and Asset parameters to do I/O (the adapter's job).
"""

# The below is confusing (to Dan).
# We have a use for StreamDatum(s) in, data out.
# No dependency on Tiled, its database, HTTP, etc.
# We just want fast incremental access to streaming data.

# In common:
# - knowledge of format details (meaning of parameters like swmr)
# - instance state to track things like num_total_rows.

# Different:
# - The "at rest" use case has a conceptual dependency on
#   DataSource Asset.
# - The streaming data use case has an import dependency o
#   whatever the I/O library is (e.g. h5py).


class Thing:
    def __init__(self, data_uri, swmr):  # TODO Add reshape, skips.
        ...

    def read_stream_datum(self, *stream_datum_docs):
        array = ...
        return array


class StreamHandlerBase:
    mimetype: str = ""
    _num_rows: int = 0
    _skips: List = dataclasses.field(default_factory=list)

    def __init__(self):
        pass

    @abstractmethod
    def consume_stream_datum(self, doc):
        """Process a new StreamDatum and update the internal data structure

        This will be called for every new StreamDatum received.
        Actions:
          - Consume new StreamDatum
          - Update shape and chunks
        """
        pass

    @abstractmethod
    def get_data_source(self):
        """Return a DataSource object reflecting the current state of the streamed dataset.

        The returned DataSource is conceptually similar (and can be an instance of) tiled.structures.DataSource. In
        general, it describes associated Assets (filepaths, mimetype) along with their internal data structure
        (array shape, chunks, additional parameters) and should contain all information necessary to read the file.
        """
        pass


class HDF5StreamHandler(StreamHandlerBase):
    mimetype = "application/x-hdf5"

    def __init__(
        self,
        data_uri,
        path,
        data_shape,
        data_type,
        chunk_size: Optional[int] = None,
        swmr: Optional[bool] = True,
        **kwargs,
    ):
        self.assets = [Asset(data_uri=data_uri, is_directory=False, parameter="data_uri")]
        self.path = path.strip("/").split("/")
        self.data_shape = data_shape  # Intrinsic data shape (single frame as written in HDF5 file)
        self.data_type = np.dtype(data_type)
        self.chunk_size = chunk_size
        self.adapter_parameters = {"path": self.path}
        self.swmr = swmr

    @property
    def shape(self):
        """Full shape including the 0-th dimension -- number of rows"""
        # TODO Add reshape, skips.
        return [self._num_rows, *self.data_shape]

    @property
    def chunks(self):
        """Chunking specification based on the Stream Resource parameter `chunk_size`:
        None or 0 -- single chunk for all existing and new elements
        int -- fixed-sized chunks with at most `chunk_size` elements, last chunk can be smaller
        """
        if not self.chunk_size:
            dim0_chunk = [self._num_rows]
        else:
            dim0_chunk = [self.chunk_size] * int(self._num_rows / self.chunk_size)
            if self._num_rows % self.chunk_size:
                dim0_chunk.append(self._num_rows % self.chunk_size)

        return [dim0_chunk] + [[d] for d in self.data_shape]

    def consume_stream_datum(self, doc):
        """Append StreamDatum to an existing Handler (StreamResource)"""
        # TODO Add reshape, skips.
        self._num_rows += doc["indices"]["stop"] - doc["indices"]["start"]  # Number of rows added by StreamDatum

    def get_data_source(self):
        """Return an instance of Tiled DataSource associated with the handler/StreamResource

        This will be called when we want to update Tiled. If the rate of StreamDatum consumed is high, we may
        not update Tiled _every_ time we consume a new StreamDatum, but do some batching of update. This is why
        this is its own method.
        """

        return DataSource(
            mimetype=self.mimetype,
            assets=self.assets,
            structure_family=StructureFamily.array,
            structure=ArrayStructure(
                data_type=BuiltinDtype.from_numpy_dtype(self.data_type),
                shape=self.shape,
                chunks=self.chunks,
            ),
            parameters=self.adapter_parameters,
            management=Management.external,
        )


class TIFFStreamHandler(StreamHandlerBase):
    mimetype = "multipart/related;type=image/tiff"

    def __init__(self, data_uri, data_shape, data_type, **kwargs):
        self.data_uri = data_uri
        self.assets = []
        self.data_shape = data_shape
        self.data_type = np.dtype(data_type)
        self.adapter_parameters = {"data_uris": []}

    @property
    def shape(self):
        """Shape including the 0-th dimension -- number of rows"""
        # TODO Add reshape, skips.
        return [self._num_rows, *self.data_shape]

    def consume_stream_datum(self, doc):
        indx = int(doc["uid"].split("/")[1])
        new_data_uri = self.data_uri.strip("/") + "/" + f"{indx:05d}.tif"
        new_asset = Asset(
            data_uri=new_data_uri,
            is_directory=False,
            parameter="data_uris",
            num=len(self.assets) + 1,
        )
        self.assets.append(new_asset)
        self.adapter_parameters["data_uris"].append(new_data_uri)
        self._num_rows += doc["indices"]["stop"] - doc["indices"]["start"]  # Number of rows added by StreamDatum

    def get_data_source(self):
        return DataSource(
            mimetype=self.mimetype,
            assets=self.assets,
            structure_family=StructureFamily.array,
            structure=ArrayStructure(
                data_type=BuiltinDtype.from_numpy_dtype(self.data_type),
                shape=self.shape,
                chunks=[[1] * self.shape[0]] + [[d] for d in self.data_shape],
            ),
            parameters=self.adapter_parameters,
            management=Management.external,
        )


STREAM_HANDLER_REGISTRY = {
    "application/x-hdf5": HDF5StreamHandler,
    "multipart/related;type=image/tiff": TIFFStreamHandler,
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
        self._desc_nodes = {}  # references to descriptor containers by their uid's
        self._sr_nodes = {}
        self._sr_cache = {}
        self._handlers = {}

    def start(self, doc):
        self.root_node = self.client.create_container(
            key=doc["uid"],
            metadata={"start": doc},
            specs=[Spec("BlueskyRun", version="1.0")],
        )

    def stop(self, doc):
        doc = dict(doc)
        doc.pop("start", None)
        metadata = {"stop": doc, **dict(self.root_node.metadata)}
        self.root_node.update_metadata(metadata=metadata)

    def descriptor(self, doc):
        desc_name = doc["name"]
        metadata = dict(doc)

        # Remove variable fields of the metadata and encapsulate them into sub-dictionaries with uids as the keys
        uid = metadata.pop("uid")
        conf_dict = {uid: metadata.pop("configuration", {})}
        time_dict = {uid: metadata.pop("time")}
        var_fields = {"configuration": conf_dict, "time": time_dict}

        if desc_name not in self.root_node.keys():
            # Create a new descriptor node; write only the fixed part of the metadata
            desc_node = self.root_node.create_container(key=desc_name, metadata=metadata)
            desc_node.create_container(key="external")
            desc_node.create_container(key="internal")
        else:
            # Get existing descriptor node (with fixed and variable metadata saved before)
            desc_node = self.root_nodes[desc_name]

        # Update (add new values to) variable fields of the metadata
        metadata = deep_update(dict(desc_node.metadata), var_fields)
        desc_node.update_metadata(metadata)
        self._desc_nodes[uid] = desc_node

    def event(self, doc: dict):
        descriptor_node = self._desc_nodes[doc["descriptor"]]
        parent_node = descriptor_node["internal"]
        df_dict = {c: [v] for c, v in doc["data"].items()}
        df_dict.update({f"ts_{c}": [v] for c, v in doc["timestamps"].items()})
        df = pd.DataFrame(df_dict)
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
                    ),
                ],
                key="events",
            )
            parent_node["events"].write_partition(df, 0)

    def stream_resource(self, doc: StreamResource):
        # Only _cache_ the StreamResource for now; add the node when at least one StreamDatum is added
        self._sr_cache[doc["uid"]] = doc

    def get_sr_node(self, sr_uid: str, desc_uid: str = None):
        """Get the Stream Resource node if it already exists or register if from a cached SR document"""

        if sr_uid in self._sr_nodes.keys():
            sr_node = self._sr_nodes[sr_uid]
            handler = self._handlers[sr_uid]

        elif sr_uid in self._sr_cache.keys():
            if not desc_uid:
                raise RuntimeError("Descriptor uid must be specified to initialise a Stream Resource node")

            sr_doc = self._sr_cache.pop(sr_uid)

            # POST /api/v1/register/{path}
            desc_node = self._desc_nodes[desc_uid]
            data_key = sr_doc["data_key"]

            # Find data shape and machine dtype; dtype_str takes precedence if specified
            data_desc = dict(desc_node.metadata)["data_keys"][data_key]
            data_shape = tuple(data_desc["shape"])
            data_shape = data_shape if data_shape != (1,) else ()
            data_type = data_desc["dtype"]
            data_type = DTYPE_LOOKUP[data_type] if data_type in DTYPE_LOOKUP.keys() else data_type
            data_type = np.dtype(data_desc.get("dtype_str", data_type))

            # Kept for back-compatibility with old StreamResource schema from event_model<1.20.0
            if ("mimetype" not in sr_doc.keys()) and ("spec" not in sr_doc.keys()):
                raise RuntimeError("StreamResource document is missing a mimetype or spec")
            else:
                sr_doc["mimetype"] = sr_doc.get("mimetype", MIMETYPE_LOOKUP[sr_doc["spec"]])
            if "parameters" not in sr_doc.keys():
                sr_doc["parameters"] = sr_doc.pop("resource_kwargs", {})
            if "uri" not in sr_doc.keys():
                file_path = sr_doc.pop("root").strip("/") + "/" + sr_doc.pop("resource_path").strip("/")
                sr_doc["uri"] = "file://localhost/" + file_path

            # Initialise a bluesky handler for the StreamResource
            handler_class = STREAM_HANDLER_REGISTRY[sr_doc["mimetype"]]
            handler = handler_class(
                sr_doc["uri"], data_shape=data_shape, data_type=data_type, **sr_doc["parameters"]
            )

            sr_node = desc_node["external"].new(
                key=data_key,
                structure_family=StructureFamily.array,
                data_sources=[handler.get_data_source()],
                metadata=sr_doc,
                specs=[],
            )

            self._handlers[sr_uid] = handler
            self._sr_nodes[sr_uid] = sr_node
        else:
            raise RuntimeError(f"Stream Resource {sr_uid} is referenced before being declared.")

        return sr_node, handler

    def stream_datum(self, doc: StreamDatum):
        # Get the Stream Resource node and handler
        sr_node, handler = self.get_sr_node(doc["stream_resource"], desc_uid=doc["descriptor"])
        handler.consume_stream_datum(doc)

        # Update StreamResource node in Tiled
        # TODO: Assigning data_source.id in the object and passing it in http params is superflous, but currently Tiled requires it.  # noqa
        sr_node.refresh()
        data_source = handler.get_data_source()
        data_source.id = sr_node.data_sources()[0]["id"]  # ID of the exisiting DataSource record
        endpoint = sr_node.uri.replace("/metadata/", "/data_source/", 1)
        handle_error(
            sr_node.context.http_client.put(
                endpoint,
                content=safe_json_dump({"data_source": data_source}),
                params={"data_source": data_source.id},
            )
        ).json()
