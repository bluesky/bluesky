from abc import abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from event_model import DocumentRouter, RunRouter
from event_model.documents import EventDescriptor, StreamDatum, StreamResource
from pydantic.utils import deep_update
from tiled.client import from_profile, from_uri
from tiled.client.base import BaseClient
from tiled.client.container import Container
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


# TODO: Move StreamHandler classes into external repo (probably area-detector-handlers) and use the existing
# handler discovery mechanism.
# GitHub Issue: https://github.com/bluesky/bluesky/issues/1740


class StreamHandlerBase:
    """
    A Handler consumes documents from Bluesky. It composes details (DataSource and its Assets) that will go into
    the Tiled database. Each Handler (more specifically, a StreamHandler) is instantiated per a Stream Resource.

    Tiled Adapters will later use this to read the data, with good random access and bulk access support.

    We put this code into handlers so that additional, possibly very unusual, formats can be supported by users
    without getting a PR merged into Bluesky or Tiled.

    The STREAM_HANDLER_REGISTRY (see example below) and the Tiled catalog paramter adapters_by_mimetype can be used
    together to support:
        - Ingesting a new mimetype from Bluesky documents and generating DataSource and Asset with appropriate
          parameters (the handler's job);
        - Interpreting those DataSource and Asset parameters to do I/O (the adapter's job).

    To implement new StreamHandlers for other mimetypes, subclass StreamHandlerBase, possibly expand the
    `consume_stream_datum` and `get_data_source` methods, and ensure that the returned the `adapter_parameters`
    property matches the expected adapter signature. Declare a specific mimetype or a set of acceptable mimetypes
    to allow valiadtion and automated discovery of the subclassed StreamHandler.
    """

    mimetype: Union[str, Set[str]] = ""

    def __init__(self, sres: StreamResource, desc: EventDescriptor):
        self._validate_mimetype(sres["mimetype"])

        self.data_key = sres["data_key"]
        self.uri = sres["uri"]
        self.assets: List[Asset] = []
        self._sres_parameters = sres["parameters"]

        # Find data shape and machine dtype; dtype_str takes precedence if specified
        data_desc = desc["data_keys"][self.data_key]
        self.datum_shape = tuple(data_desc["shape"])
        self.datum_shape = self.datum_shape if self.datum_shape != (1,) else ()
        self.dtype = data_desc["dtype"]
        self.dtype = DTYPE_LOOKUP[self.dtype] if self.dtype in DTYPE_LOOKUP.keys() else self.dtype
        self.dtype = np.dtype(data_desc.get("dtype_str", self.dtype))
        self.chunk_size = self._sres_parameters.get("chunk_size", None)

        self._num_rows: int = 0  # Number of rows in the Data Source (all rows, includung skips)
        self._has_skips: bool = False
        self._seqnums_to_indices_map: Dict[int, int] = {}

    def _validate_mimetype(self, mimetype):
        if isinstance(self.mimetype, str) and (mimetype == self.mimetype):
            return None
        elif isinstance(self.mimetype, set) and (mimetype in self.mimetype):
            self.mimetype = mimetype
            return None
        else:
            raise ValueError(f"A data source of {mimetype} type can not be handled by {self.__class__.__name__}.")

    @property
    def shape(self) -> Tuple[int]:
        """Native shape of the data stored in assets

        This includes the leading (0-th) dimension corresponding to the number of rows, including skipped rows, if
        any. The number of relevant usable data rows may be lower, which is determined by the `seq_nums` field of
        StreamDatum documents."""
        return self._num_rows, *self.datum_shape

    @property
    def chunks(self) -> Tuple[Tuple[int]]:
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

        return tuple(dim0_chunk), *[(d,) for d in self.datum_shape]

    @property
    def has_skips(self) -> bool:
        """Indicates whether any rows should be skipped when mapping their indices to frame numbers

        This flag is intended to provide a shortcut for more efficient data access when there are no skips, and the
        mapping between indices and seq_nums is straightforward. In other case, the _seqnums_to_indices_map needs
        to be taken into account.
        """
        return self._num_rows > len(self._seqnums_to_indices_map)

    @property
    @abstractmethod
    def adapter_parameters(self) -> Dict:
        """A dictionary of parameters passed to an Adapter

        These parameters are intended to provide any additional information required to read a data source of a
        specific mimetype, e.g. "path" or "data_uris".
        """
        pass

    def consume_stream_datum(self, doc: StreamDatum):
        """Process a new StreamDatum and update the internal data structure

        This will be called for every new StreamDatum received to account for the new added rows.
        This method _may_need_ to be subclassed and expanded depending on a specific mimetype.
        Actions:
          - Parse the fields in a new StreamDatum
          - Increment the number of rows (implemented by the Base class)
          - Keep track of the correspondence between indices and seq_nums (implemented by the Base class)
          - Update the list of assets, including their uris, if necessary
          - Update shape and chunks
        """
        self._num_rows += doc["indices"]["stop"] - doc["indices"]["start"]
        new_seqnums = range(doc["seq_nums"]["start"], doc["seq_nums"]["stop"])
        new_indices = range(doc["indices"]["start"], doc["indices"]["stop"])
        self._seqnums_to_indices_map.update(dict(zip(new_seqnums, new_indices)))

    def get_data_source(self) -> DataSource:
        """Return a DataSource object reflecting the current state of the streamed dataset.

        The returned DataSource is conceptually similar (and can be an instance of) tiled.structures.DataSource. In
        general, it describes associated Assets (filepaths, mimetype) along with their internal data structure
        (array shape, chunks, additional parameters) and should contain all information necessary to read the file.
        """
        return DataSource(
            mimetype=self.mimetype,
            assets=self.assets,
            structure_family=StructureFamily.array,
            structure=ArrayStructure(
                data_type=BuiltinDtype.from_numpy_dtype(self.dtype),
                shape=self.shape,
                chunks=self.chunks,
            ),
            parameters=self.adapter_parameters,
            management=Management.external,
        )


class HDF5StreamHandler(StreamHandlerBase):
    mimetype = "application/x-hdf5"

    def __init__(self, sres: StreamResource, desc: EventDescriptor):
        super().__init__(sres, desc)
        self.assets.append(Asset(data_uri=self.uri, is_directory=False, parameter="data_uri"))
        self.swmr = self._sres_parameters.get("swmr", True)

    @property
    def adapter_parameters(self) -> Dict:
        return {"path": self._sres_parameters["path"].strip("/").split("/")}


class TIFFStreamHandler(StreamHandlerBase):
    mimetype = "multipart/related;type=image/tiff"

    def __init__(self, sres: StreamResource, desc: EventDescriptor):
        super().__init__(sres, desc)
        self.chunk_size = 1
        self.data_uris: List[str] = []

    def consume_stream_datum(self, doc: StreamDatum):
        indx = int(doc["uid"].split("/")[1])
        new_datum_uri = self.uri.strip("/") + "/" + f"{indx:05d}.tif"
        new_asset = Asset(
            data_uri=new_datum_uri,
            is_directory=False,
            parameter="data_uris",
            num=len(self.assets) + 1,
        )
        self.assets.append(new_asset)
        self.data_uris.append(new_datum_uri)

        super().consume_stream_datum(doc)

    @property
    def adapter_parameters(self) -> Dict:
        return {"data_uris": self.data_uris}


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

    def __init__(self, client: BaseClient):
        self.client = client
        self.root_node = None
        self._desc_nodes: Dict[str, Container] = {}  # references to descriptor containers by their uid's
        self._sres_nodes: Dict[str, BaseClient] = {}
        self._sres_cache: Dict[str, StreamResource] = {}
        self._handlers: Dict[str, StreamHandlerBase] = {}

    def _ensure_sres_backcompat(self, sres: StreamResource) -> StreamResource:
        """Kept for back-compatibility with old StreamResource schema from event_model<1.20.0

        Will make changes to and return the _same_instance_ of StreamRsource dictionary.
        """

        if ("mimetype" not in sres.keys()) and ("spec" not in sres.keys()):
            raise RuntimeError("StreamResource document is missing a mimetype or spec")
        else:
            sres["mimetype"] = sres.get("mimetype", MIMETYPE_LOOKUP[sres["spec"]])
        if "parameters" not in sres.keys():
            sres["parameters"] = sres.pop("resource_kwargs", {})
        if "uri" not in sres.keys():
            file_path = sres.pop("root").strip("/") + "/" + sres.pop("resource_path").strip("/")
            sres["uri"] = "file://localhost/" + file_path

        return sres

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

    def descriptor(self, doc: EventDescriptor):
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

    def event(self, doc):
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
        """Process a StreamResource document

        Only _cache_ the StreamResource for now; add the node when at least one StreamDatum is added
        """

        self._sres_cache[doc["uid"]] = self._ensure_sres_backcompat(doc)

    def get_sres_node(self, sres_uid: str, desc_uid: Optional[str] = None) -> Tuple[BaseClient, StreamHandlerBase]:
        """Get Stream Resource node from Tiled, if it already exists, or register it from a cached SR document"""

        if sres_uid in self._sres_nodes.keys():
            sres_node = self._sres_nodes[sres_uid]
            handler = self._handlers[sres_uid]

        elif sres_uid in self._sres_cache.keys():
            if not desc_uid:
                raise RuntimeError("Descriptor uid must be specified to initialise a Stream Resource node")

            sres_doc = self._sres_cache.pop(sres_uid)
            desc_node = self._desc_nodes[desc_uid]

            # Initialise a bluesky handler for the StreamResource
            handler_class = STREAM_HANDLER_REGISTRY[sres_doc["mimetype"]]
            handler = handler_class(sres_doc, dict(desc_node.metadata))

            sres_node = desc_node["external"].new(
                key=handler.data_key,
                structure_family=StructureFamily.array,
                data_sources=[handler.get_data_source()],
                metadata=sres_doc,
                specs=[],
            )

            self._handlers[sres_uid] = handler
            self._sres_nodes[sres_uid] = sres_node
        else:
            raise RuntimeError(f"Stream Resource {sres_uid} is referenced before being declared.")

        return sres_node, handler

    def stream_datum(self, doc: StreamDatum):
        # Get the Stream Resource node and handler
        sres_node, handler = self.get_sres_node(doc["stream_resource"], desc_uid=doc["descriptor"])
        handler.consume_stream_datum(doc)

        # Update StreamResource node in Tiled
        # NOTE: Assigning data_source.id in the object and passing it in http params is superflous, but it is currently required by Tiled.  # noqa
        sres_node.refresh()
        data_source = handler.get_data_source()
        data_source.id = sres_node.data_sources()[0]["id"]  # ID of the exisiting DataSource record
        endpoint = sres_node.uri.replace("/metadata/", "/data_source/", 1)
        handle_error(
            sres_node.context.http_client.put(
                endpoint,
                content=safe_json_dump({"data_source": data_source}),
                params={"data_source": data_source.id},
            )
        ).json()
