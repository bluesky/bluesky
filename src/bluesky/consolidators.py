import collections
import dataclasses
import enum
import re
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
    """Consolidator of StreamDatums

    A Consolidator consumes documents from RE; it is similar to usual Bluesky Handlers but is designed to work
    with streaming data (received via StreamResource and StreamDatum documents). It composes details (DataSource
    and its Assets) that will go into the Tiled database. Each Consolidator is instantiated per a Stream Resource.

    Tiled Adapters will later use this to read the data, with good random access and bulk access support.

    We put this code into consolidators so that additional, possibly very unusual, formats can be supported by
    users without getting a PR merged into Bluesky or Tiled.

    The CONSOLIDATOR_REGISTRY (see example below) and the Tiled catalog parameter adapters_by_mimetype can be used
    together to support:
        - Ingesting a new mimetype from Bluesky documents and generating DataSource and Asset with appropriate
          parameters (the consolidator's job);
        - Interpreting those DataSource and Asset parameters to do I/O (the adapter's job).

    To implement new Consolidators for other mimetypes, subclass ConsolidatorBase, possibly expand the
    `consume_stream_datum` and `get_data_source` methods, and ensure that the returned `adapter_parameters`
    property matches the expected adapter signature. Declare a set of supported mimetypes to allow validation and
    automated discovery of the subclassed Consolidator.
    """

    supported_mimetypes: Set[str] = {"application/octet-stream"}

    def __init__(self, stream_resource: StreamResource, descriptor: EventDescriptor):
        self.mimetype = self.get_supported_mimetype(stream_resource)

        self.data_key = stream_resource["data_key"]
        self.uri = stream_resource["uri"]
        self.assets: List[Asset] = []
        self._sres_parameters = stream_resource["parameters"]

        # Find data shape and machine dtype; dtype_numpy, dtype_str take precedence if specified
        data_desc = descriptor["data_keys"][self.data_key]
        self.datum_shape = tuple(data_desc["shape"])
        self.datum_shape = self.datum_shape if self.datum_shape != (1,) else ()
        # Get data type. From highest precedent to lowest:
        # 1. Try 'dtype_numpy', optional in the document schema.
        # 2. Try 'dtype_str', an old convention predataing 'dtype_numpy', not in the schema.
        # 3. Get 'dtype', required by the schema, which is a fuzzy JSON spec like 'number'
        #    and make a best effort to convert it to a numpy spec like '<u8'.
        # 4. If unable to do any of the above, pass through whatever string is in 'dtype'.
        self.dtype = np.dtype(
            data_desc.get("dtype_numpy")  # standard location
            or data_desc.get(
                "dtype_str",  # legacy location
                # try to guess numpy dtype from JSON type
                DTYPE_LOOKUP.get(data_desc["dtype"], data_desc["dtype"]),
            )
        )
        self.chunk_shape = self._sres_parameters.get("chunk_shape", ())
        if 0 in self.chunk_shape:
            raise ValueError(f"Chunk size in all dimensions must be at least 1: chunk_shape={self.chunk_shape}.")

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
        """Explicit (dask-style) specification of chunk sizes

        The produced chunk specification is a tuple of tuples of int that specify the sizes of each chunk in each
        dimension; it is based on the StreamResource parameter `chunk_shape`.

        If `chunk_shape` is an empty tuple -- assume the dataset is stored as a single chunk for all existing and
        new elements. Usually, however, `chunk_shape` is a tuple of int, in which case, we assume fixed-sized
        chunks with at most `chunk_shape[0]` elements (i.e. `_num_rows`); last chunk can be smaller. If chunk_shape
        is a tuple with only one element -- assume it defines the chunk size along the leading (event) dimension.
        """

        def list_summands(A, b):
            # Generate a list with repeated b summing up to A; append the remainder if necessary
            return tuple([b] * (A // b) + ([A % b] if A % b > 0 else [])) or (0,)

        if len(self.chunk_shape) == 0:
            return (self._num_rows,), *[(d,) for d in self.datum_shape]

        elif len(self.chunk_shape) == 1:
            return list_summands(self._num_rows, self.chunk_shape[0]), *[(d,) for d in self.datum_shape]

        elif len(self.chunk_shape) == len(self.shape):
            return tuple([list_summands(ddim, cdim) for cdim, ddim in zip(self.chunk_shape, self.shape)])

        else:
            raise ValueError(
                f"The shape of chunks, {self.chunk_shape}, is not consistent with the shape of data, {self.shape}."
            )

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

        This property is to be subclassed as necessary.
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


class CSVConsolidator(ConsolidatorBase):
    supported_mimetypes: Set[str] = {"text/csv;header=absent"}

    def __init__(self, stream_resource: StreamResource, descriptor: EventDescriptor):
        super().__init__(stream_resource, descriptor)
        self.assets.append(Asset(data_uri=self.uri, is_directory=False, parameter="data_uris"))
        self.swmr = self._sres_parameters.get("swmr", True)

    @property
    def adapter_parameters(self) -> Dict:
        """Parameters to be passed to the HDF5 adapter, a dictionary with the keys:

        dataset: List[str] - a path to the dataset within the hdf5 file represented as list split at `/`
        swmr: bool -- True to enable the single writer / multiple readers regime
        """
        return {"dataset": self._sres_parameters["dataset"].strip("/").split("/"), "swmr": self.swmr}


class HDF5Consolidator(ConsolidatorBase):
    supported_mimetypes = {"application/x-hdf5"}

    def __init__(self, stream_resource: StreamResource, descriptor: EventDescriptor):
        super().__init__(stream_resource, descriptor)
        self.assets.append(Asset(data_uri=self.uri, is_directory=False, parameter="data_uri"))
        self.swmr = self._sres_parameters.get("swmr", True)

    @property
    def adapter_parameters(self) -> Dict:
        """Parameters to be passed to the HDF5 adapter, a dictionary with the keys:

        dataset: List[str] - a path to the dataset within the hdf5 file represented as list split at `/`
        swmr: bool -- True to enable the single writer / multiple readers regime
        """
        return {"dataset": self._sres_parameters["dataset"].strip("/").split("/"), "swmr": self.swmr}


class TIFFConsolidator(ConsolidatorBase):
    supported_mimetypes = {"multipart/related;type=image/tiff"}

    def __init__(self, stream_resource: StreamResource, descriptor: EventDescriptor):
        super().__init__(stream_resource, descriptor)
        self.data_uris: List[str] = []
        self.chunk_shape = self.chunk_shape or (1,)  # Assume one frame (chunk) per tiff file

        # Normalize filename template
        # self.template = self._sres_parameters["template"]
        def int_replacer(match):
            flags, width, precision, type_char = match.groups()

            # Handle the flags
            flag_str = ""
            if "-" in flags:
                flag_str += "<"  # Left-align
            elif "0" in flags:
                flag_str += "0"  # Zero padding
            if "+" in flags:
                flag_str += "+"  # Show positive sign
            elif " " in flags:
                flag_str += " "  # Space before positive numbers

            # Build width and precision if they exist
            width_str = width if width else ""
            precision_str = f".{precision}" if precision else ""

            # Handle cases like "%6.6d", which should be converted to "{:06d}"
            if precision and width:
                flag_str = "0"
                precision_str = ""
                width_str = str(max(precision, width))

            # Construct the new-style format specifier
            return f"{{:{flag_str}{width_str}{precision_str}{type_char}}}"

        self.template = (
            self._sres_parameters["template"]
            .replace("%s", "{:s}", 1)
            .replace("%s", "")
            .format(self._sres_parameters.get("filename", ""))
        )
        self.template = re.sub(r"%([-+#0 ]*)(\d+)?(?:\.(\d+))?([d])", int_replacer, self.template)

    def get_datum_uri(self, indx: int):
        """Return a full uri for a datum (an individual TIFF file) based on its index in the sequence.

        This relies on the `template` parameter passed in the StreamResource, which is a string in the "new"
        Python formatting style that can be evaluated to a file name using the `.format(indx)` method given an
        integer index, e.g. "{:05d}.tif".
        """

        return self.uri.rstrip("/") + "/" + self.template.format(indx)

    def consume_stream_datum(self, doc: StreamDatum):
        # Determine the indices in the names of tiff files from indices of frames and number of frames per file
        first_file_indx = int(doc["indices"]["start"] / self.chunk_shape[0])
        last_file_indx = int((doc["indices"]["stop"] - 1) / self.chunk_shape[0])
        for indx in range(first_file_indx, last_file_indx + 1):
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


CONSOLIDATOR_REGISTRY = collections.defaultdict(
    lambda: ConsolidatorBase,
    {
        "text/csv;header=absent": CSVConsolidator,
        "application/x-hdf5": HDF5Consolidator,
        "multipart/related;type=image/tiff": TIFFConsolidator,
    },
)


def consolidator_factory(stream_resource_doc, descriptor_doc):
    consolidator_class = CONSOLIDATOR_REGISTRY[stream_resource_doc["mimetype"]]
    return consolidator_class(stream_resource_doc, descriptor_doc)
