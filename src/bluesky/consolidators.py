import collections
import dataclasses
import enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from event_model.documents import EventDescriptor, StreamDatum, StreamResource
from tiled.mimetypes import DEFAULT_ADAPTERS_BY_MIMETYPE
from tiled.structures.array import ArrayStructure, BuiltinDtype

DTYPE_LOOKUP = {"number": "<f8", "array": "<f8", "boolean": "bool", "string": "str", "integer": "int"}


# TODO: Move Consolidator classes into external repo (probably area-detector-handlers) and use the existing
# handler discovery mechanism.
# GitHub Issue: https://github.com/bluesky/bluesky/issues/1740


class StructureFamily(str, enum.Enum):
    array = "array"
    awkward = "awkward"
    container = "container"
    sparse = "sparse"
    table = "table"


class Management(str, enum.Enum):
    external = "external"
    immutable = "immutable"
    locked = "locked"
    writable = "writable"


@dataclasses.dataclass
class Asset:
    data_uri: str
    is_directory: bool
    parameter: Optional[str]
    num: Optional[int] = None
    id: Optional[int] = None


@dataclasses.dataclass
class DataSource:
    structure_family: StructureFamily
    structure: Any
    id: Optional[int] = None
    mimetype: Optional[str] = None
    parameters: dict = dataclasses.field(default_factory=dict)
    assets: List[Asset] = dataclasses.field(default_factory=list)
    management: Management = Management.writable


class ConsolidatorBase:
    """Consolidator of StremDatums

    A Consolidator consumes documents from RE; it is similar to usual Bluesky Handlers but is designed to work
    with streaming data (received via StreamResource and StreamDatum documents). It composes details (DataSource
    and its Assets) that will go into the Tiled database. Each Consolidator is instantiated per a Stream Resource.

    Tiled Adapters will later use this to read the data, with good random access and bulk access support.

    We put this code into consolidators so that additional, possibly very unusual, formats can be supported by
    users without getting a PR merged into Bluesky or Tiled.

    The CONSOLIDATOR_REGISTRY (see example below) and the Tiled catalog paramter adapters_by_mimetype can be used
    together to support:
        - Ingesting a new mimetype from Bluesky documents and generating DataSource and Asset with appropriate
          parameters (the consolidator's job);
        - Interpreting those DataSource and Asset parameters to do I/O (the adapter's job).

    To implement new Consolidators for other mimetypes, subclass ConsolidatorBase, possibly expand the
    `consume_stream_datum` and `get_data_source` methods, and ensure that the returned the `adapter_parameters`
    property matches the expected adapter signature. Declare a set of supported mimetypes to allow valiadtion and
    automated discovery of the subclassed Consolidator.
    """

    supported_mimetypes: Set[str] = set()

    def __init__(self, stream_resource: StreamResource, descriptor: EventDescriptor):
        self.mimetype = self.get_supported_mimetype(stream_resource)

        self.data_key = stream_resource["data_key"]
        self.uri = stream_resource["uri"]
        self.assets: List[Asset] = []
        self._sres_parameters = stream_resource["parameters"]

        # Find data shape and machine dtype; dtype_str takes precedence if specified
        data_desc = descriptor["data_keys"][self.data_key]
        self.datum_shape = tuple(data_desc["shape"])
        self.datum_shape = self.datum_shape if self.datum_shape != (1,) else ()
        self.dtype = data_desc["dtype"]
        self.dtype = DTYPE_LOOKUP[self.dtype] if self.dtype in DTYPE_LOOKUP.keys() else self.dtype
        self.dtype = np.dtype(data_desc.get("dtype_str", self.dtype))
        self.chunk_size = self._sres_parameters.get("chunk_size", None)

        self._num_rows: int = 0  # Number of rows in the Data Source (all rows, includung skips)
        self._has_skips: bool = False
        self._seqnums_to_indices_map: Dict[int, int] = {}

    @classmethod
    def get_supported_mimetype(cls, sres):
        if sres["mimetype"] not in cls.supported_mimetypes:
            raise ValueError(f"A data source of {sres['mimetype']} type can not be handled by {cls.__name__}.")
        return sres["mimetype"]

    @property
    def shape(self) -> Tuple[int]:
        """Native shape of the data stored in assets

        This includes the leading (0-th) dimension corresponding to the number of rows, including skipped rows, if
        any. The number of relevant usable data rows may be lower, which is determined by the `seq_nums` field of
        StreamDatum documents."""
        return self._num_rows, *self.datum_shape

    @property
    def chunks(self) -> Tuple[Tuple[int, ...], ...]:
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
    def adapter_parameters(self) -> Dict:
        """A dictionary of parameters passed to an Adapter

        These parameters are intended to provide any additional information required to read a data source of a
        specific mimetype, e.g. "path" the path into an HDF5 file or "template" the filename pattern of a TIFF
        sequence.
        """
        return {}

    def consume_stream_datum(self, doc: StreamDatum):
        """Process a new StreamDatum and update the internal data structure

        This will be called for every new StreamDatum received to account for the new added rows.
        This method _may need_ to be subclassed and expanded depending on a specific mimetype.
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

    def get_adapter(self, adapters_by_mimetype=None):
        """Return an Adapter suitable for reading the data

        Uses a dictionary mapping of a mimetype to a callable that returns an Adapter instance.
        This might be a class, classmethod constructor, factory function...
        it does not matter here; it is just a callable.
        """

        # User-provided adapters take precedence over defaults.
        all_adapters_by_mimetype = collections.ChainMap((adapters_by_mimetype or {}), DEFAULT_ADAPTERS_BY_MIMETYPE)
        adapter_factory = all_adapters_by_mimetype[self.mimetype]

        # Construct kwargs to pass to Adapter.
        parameters = collections.defaultdict(list)
        for asset in self.assets:
            if asset.parameter is None:
                # This asset is not directly opened by the Adapter. It is used indirectly, such as the case of HDF5
                # virtual dataset 'data' files are referenced from 'master' files.
                continue
            if asset.num is None:
                # This parameters takes the URI as a scalar value.
                parameters[asset.parameter] = asset.data_uri
            else:
                # This parameters takes a list of URIs.
                parameters[asset.parameter].append(asset.data_uri)

        parameters["structure"] = ArrayStructure(
            data_type=BuiltinDtype.from_numpy_dtype(self.dtype),
            shape=self.shape,
            chunks=self.chunks,
        )
        adapter_kwargs = dict(parameters)
        adapter_kwargs.update(self.adapter_parameters)

        return adapter_factory(**adapter_kwargs)


class HDF5Consolidator(ConsolidatorBase):
    supported_mimetypes = {"application/x-hdf5"}

    def __init__(self, stream_resource: StreamResource, descriptor: EventDescriptor):
        super().__init__(stream_resource, descriptor)
        self.assets.append(Asset(data_uri=self.uri, is_directory=False, parameter="data_uri"))
        self.swmr = self._sres_parameters.get("swmr", True)

    @property
    def adapter_parameters(self) -> Dict:
        return {"path": self._sres_parameters["path"].strip("/").split("/"), "swmr": self.swmr}


class TIFFConsolidator(ConsolidatorBase):
    supported_mimetypes = {"multipart/related;type=image/tiff"}

    def __init__(self, stream_resource: StreamResource, descriptor: EventDescriptor):
        super().__init__(stream_resource, descriptor)
        self.data_uris: List[str] = []

    def get_datum_uri(self, indx: int):
        """Return a full uri for a datum (an individual TIFF file) based on its index in the sequence.

        This relies on the `template` parameter passed in the StreamResource, which is a string either in the "new"
        Python formatting style that can be evaluated to a file name using the `.format(indx)` method given an
        integer index, e.g. "{:05}.tif".
        """
        return self.uri + self._sres_parameters["template"].format(indx)

    def consume_stream_datum(self, doc: StreamDatum):
        indx = int(doc["uid"].split("/")[1])
        new_datum_uri = self.get_datum_uri(indx)
        new_asset = Asset(
            data_uri=new_datum_uri,
            is_directory=False,
            parameter="data_uris",
            num=len(self.assets) + 1,
        )
        self.assets.append(new_asset)
        self.data_uris.append(new_datum_uri)

        super().consume_stream_datum(doc)


CONSOLIDATOR_REGISTRY = {
    "application/x-hdf5": HDF5Consolidator,
    "multipart/related;type=image/tiff": TIFFConsolidator,
}


def consolidator_factory(stream_resource_doc, descriptor_doc):
    consolidator_class = CONSOLIDATOR_REGISTRY[stream_resource_doc["mimetype"]]
    return consolidator_class(stream_resource_doc, descriptor_doc)
