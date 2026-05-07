import collections
import dataclasses
import enum
import os
import re
import warnings
from typing import Any, Literal, cast

import numpy as np
from event_model.documents import EventDescriptor, StreamDatum, StreamResource
from tiled.mimetypes import DEFAULT_ADAPTERS_BY_MIMETYPE
from tiled.structures.array import ArrayStructure, BuiltinDtype, StructDtype

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
    parameter: str | None
    num: int | None = None
    id: int | None = None


@dataclasses.dataclass
class DataSource:
    structure_family: StructureFamily
    structure: Any
    id: int | None = None
    mimetype: str | None = None
    parameters: dict = dataclasses.field(default_factory=dict)
    assets: list[Asset] = dataclasses.field(default_factory=list)
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
    `consume_stream_datum` and `get_data_source` methods, and ensure that the keys of returned `adapter_parameters`
    dictionary matches the expected adapter signature. Declare a set of supported mimetypes to allow validation and
    automated discovery of the subclassed Consolidator.

    Attributes:
    -----------

    supported_mimetypes : set[str]
        a set of mimetypes that can be handled by a derived Consolidator class; raises ValueError if attempted to
        pass Resource documents related to unsupported mimetypes.
    join_method : Literal["stack", "concat"]
        a method to join the data; if "stack", the resulting consolidated dataset is produced by joining all datums
        along a new dimension added on the left, e.g. a stack of tiff images, otherwise -- datums will be appended
        to the end of the existing leftmost dimension, e.g. rows of a table (similarly to concatenation in numpy).
    join_chunks : bool
        if True, the chunking of the resulting dataset will be determined after consolidation, otherwise each part
        is considered to be chunked separately.
    """

    supported_mimetypes: set[str] = {"application/octet-stream"}
    join_method: Literal["stack", "concat"] = "concat"
    join_chunks: bool = True

    def __init__(self, stream_resource: StreamResource, descriptor: EventDescriptor):
        self.mimetype = self.get_supported_mimetype(stream_resource)

        self.data_key = stream_resource["data_key"]
        self.uri = stream_resource["uri"]
        self.assets: list[Asset] = []
        self._sres_parameters = stream_resource["parameters"]

        # Find datum shape and machine dtype
        data_desc = descriptor["data_keys"][self.data_key]
        if None in data_desc["shape"]:
            raise NotImplementedError(f"Consolidator for {self.mimetype} does not support variable-sized data")
        self.datum_shape: tuple[int, ...] = cast(tuple[int, ...], tuple(data_desc["shape"]))
        self.datum_shape = () if self.datum_shape == (1,) and self.join_method == "stack" else self.datum_shape

        # Check that the datum shape is consistent between the StreamResource and the Descriptor
        if multiplier := self._sres_parameters.get("multiplier"):
            self.datum_shape = self.datum_shape or (multiplier,)  # If datum_shape is not set
            if self.datum_shape[0] != multiplier:
                if self.datum_shape[0] == 1:
                    self.datum_shape = (multiplier,) + self.datum_shape[1:]
                else:
                    self.datum_shape = (multiplier,) + self.datum_shape
                    # TODO: Check consistency with chunk_shape

        # Determine the machine data type
        self.data_type: BuiltinDtype | StructDtype
        dtype_numpy = np.dtype(data_desc.get("dtype_numpy"))  # Falls back to np.dtype("float64") if not set
        if dtype_numpy.kind == "V":
            self.data_type = StructDtype.from_numpy_dtype(dtype_numpy)
        else:
            self.data_type = BuiltinDtype.from_numpy_dtype(dtype_numpy)

        # Set chunk (or partition) shape
        self.chunk_shape = self._sres_parameters.get("chunk_shape", ())
        if any(d <= 0 for d in self.chunk_shape):
            raise ValueError(f"Chunk size in all dimensions must be at least 1: chunk_shape={self.chunk_shape}.")

        # Possibly overwrite the join_method and join_chunks attributes
        self.join_method = self._sres_parameters.get("join_method", self.join_method)
        self.join_chunks = self._sres_parameters.get("join_chunks", self.join_chunks)

        self._num_rows: int = 0  # Number of rows in the Data Source (all rows, includung skips)
        self._seqnums_to_indices_map: dict[int, int] = {}

    @classmethod
    def get_supported_mimetype(cls, sres):
        if sres["mimetype"] not in cls.supported_mimetypes:
            raise ValueError(f"A data source of {sres['mimetype']} type can not be handled by {cls.__name__}.")
        return sres["mimetype"]

    @property
    def shape(self) -> tuple[int, ...]:
        """Native shape of the data stored in assets

        This includes the leading (0th) dimension corresponding to the number of rows (if the join_method is stack)
        including skipped rows, if any. The number of relevant usable data rows may be lower, which is determined
        by the `seq_nums` field of StreamDatum documents."""

        if (self.join_method == "concat") and len(self.datum_shape) > 0:
            return self._num_rows * self.datum_shape[0], *self.datum_shape[1:]

        return self._num_rows, *self.datum_shape

    @property
    def chunks(self) -> tuple[tuple[int, ...], ...]:
        """Explicit (dask-style) specification of chunk sizes

        The produced chunk specification is a tuple of tuples of int that specify the sizes of each chunk in each
        dimension; it is based on the StreamResource parameter `chunk_shape`.

        If `chunk_shape` is an empty tuple -- assume the dataset is stored as a single chunk for all existing and
        new elements. Usually, however, `chunk_shape` is a tuple of int, in which case, we assume fixed-sized
        chunks with at most `chunk_shape[0]` elements (i.e. `_num_rows`); last chunk can be smaller. If chunk_shape
        is a tuple with less than `self.shape` elements -- assume it defines the chunk sizes along the leading
        dimensions.

        If the joining method is "concat", and `join_chunks = False`, the chunking along the leftmost dimensions
        is assumed to be preserved in each appended data point, i.e. consecutive chunks do not join, e.g. for a 1d
        array with chunks (3,3,1), the resulting chunking after 3 repeats is (3,3,1,3,3,1,3,3,1).
        When `join_chunks = True` (default), the chunk size along the leftmost dimension is determined by the
        chunk_shape parameter; this is the case when `join_method == "stack"` well.
        Chunking along the trailing dimensions is always preserved as in the original (single) array.
        """

        def list_summands(A: int, b: int, repeat: int = 1) -> tuple[int, ...]:
            # Generate a list with repeated b summing up to A; append the remainder if necessary
            # e.g. list_summands(13, 3) = [3, 3, 3, 3, 1]
            # if `repeat = n`, n > 1, copy and repeat the entire result n times
            return tuple([b] * (A // b) + ([A % b] if A % b > 0 else [])) * repeat or (0,)

        # If chunk shape is less than or equal to the total shape dimensions, chunk each specified dimension
        # starting from the leading dimension
        if len(self.chunk_shape) <= len(self.shape):
            if (
                self.join_method == "stack"
                or (self.join_method == "concat" and self.join_chunks)
                or len(self.chunk_shape) == 0
            ):
                result = tuple(
                    list_summands(ddim, cdim)
                    for ddim, cdim in zip(self.shape[: len(self.chunk_shape)], self.chunk_shape)
                )
            else:
                result = (
                    list_summands(self.datum_shape[0], self.chunk_shape[0], repeat=self._num_rows),
                    *[
                        list_summands(ddim, cdim)
                        for ddim, cdim in zip(self.shape[1 : len(self.chunk_shape)], self.chunk_shape[1:])
                    ],
                )
            return result + tuple((d,) for d in self.shape[len(self.chunk_shape) :])

        # If chunk shape is longer than the total shape dimensions, raise an error
        else:
            raise ValueError(
                f"The shape of chunks, {self.chunk_shape}, should be less than or equal to the shape of data, "
                f"{self.shape}."
            )

    @property
    def has_skips(self) -> bool:
        """Indicates whether any rows should be skipped when mapping their indices to frame numbers

        This flag is intended to provide a shortcut for more efficient data access when there are no skips, and the
        mapping between indices and seq_nums is straightforward. In other case, the _seqnums_to_indices_map needs
        to be taken into account.
        """
        return self._num_rows > len(self._seqnums_to_indices_map)

    def adapter_parameters(self) -> dict:
        """A dictionary of parameters passed to an Adapter

        These parameters are intended to provide any additional information required to read a data source of a
        specific mimetype, e.g. "path" the path into an HDF5 file or "template" the filename pattern of a TIFF
        sequence.

        This method is to be subclassed as necessary.
        """
        return {}

    def structure(self) -> ArrayStructure:
        return ArrayStructure(
            data_type=self.data_type,
            shape=self.shape,
            chunks=self.chunks,
        )

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
            structure=self.structure(),
            parameters=self.adapter_parameters(),
            management=Management.external,
        )

    def get_adapter(self, adapters_by_mimetype=None):
        """Return an Adapter suitable for reading the data

        Uses a dictionary mapping of a mimetype to a callable that returns an Adapter instance from_catalog.
        """

        # User-provided adapters take precedence over defaults.
        all_adapters_by_mimetype = collections.ChainMap((adapters_by_mimetype or {}), DEFAULT_ADAPTERS_BY_MIMETYPE)
        adapter_class = all_adapters_by_mimetype[self.mimetype]

        # Mimic the necessary aspects of a tiled node with a namedtuple
        _Node = collections.namedtuple("Node", ["metadata_", "specs"])
        return adapter_class.from_catalog(self.get_data_source(), _Node({}, []), **self.adapter_parameters())

    def update_from_stream_resource(self, stream_resource: StreamResource):
        """Consume an additional related StreamResource document for the same data_key"""

        raise NotImplementedError("This method is not implemented in the base Consolidator class.")

    def validate(self, adapters_by_mimetype=None, fix_errors=False):
        """Validate the Consolidator's state against the expected structure"""

        # User-provided adapters take precedence over defaults.
        all_adapters_by_mimetype = collections.ChainMap((adapters_by_mimetype or {}), DEFAULT_ADAPTERS_BY_MIMETYPE)
        adapter_class = all_adapters_by_mimetype[self.mimetype]

        # Initialize adapter from uris and determine the structure
        uris = [asset.data_uri for asset in self.assets if asset.parameter == "data_uris"]
        structure = adapter_class.from_uris(*uris, **self.adapter_parameters()).structure()

        if self.shape != structure.shape:
            if not fix_errors:
                raise ValueError(f"Shape mismatch: {self.shape} != {structure.shape}")
            else:
                warnings.warn(f"Fixing shape mismatch: {self.shape} -> {structure.shape}", stacklevel=2)
                if self.join_method == "stack":
                    self._num_rows = structure.shape[0]
                    self.datum_shape = structure.shape[1:]
                elif self.join_method == "concat":
                    # Estimate the number of frames_per_event (multiplier)
                    multiplier = 1 if structure.shape[0] % structure.chunks[0][0] else structure.chunks[0][0]
                    self._num_rows = structure.shape[0] // multiplier
                    self.datum_shape = (multiplier,) + structure.shape[1:]

        if self.chunks != structure.chunks:
            if not fix_errors:
                raise ValueError(f"Chunk shape mismatch: {self.chunks} != {structure.chunks}")
            else:
                _chunk_shape = tuple(c[0] for c in structure.chunks)
                warnings.warn(f"Fixing chunk shape mismatch: {self.chunk_shape} -> {_chunk_shape}", stacklevel=2)
                self.chunk_shape = _chunk_shape

        if self.data_type != structure.data_type:
            if not fix_errors:
                raise ValueError(f"dtype mismatch: {self.data_type} != {structure.data_type}")
            else:
                warnings.warn(
                    f"Fixing dtype mismatch: {self.data_type.to_numpy_dtype()} -> {structure.data_type.to_numpy_dtype()}",  # noqa
                    stacklevel=2,
                )
                self.data_type = structure.data_type

        assert self.get_adapter() is not None, "Adapter can not not initialized"


class CSVConsolidator(ConsolidatorBase):
    supported_mimetypes: set[str] = {"text/csv;header=absent"}
    join_method: Literal["stack", "concat"] = "concat"
    join_chunks: bool = False

    def __init__(self, stream_resource: StreamResource, descriptor: EventDescriptor):
        super().__init__(stream_resource, descriptor)
        self.assets.append(Asset(data_uri=self.uri, is_directory=False, parameter="data_uris"))

    def adapter_parameters(self) -> dict:
        return {**self._sres_parameters}


class HDF5Consolidator(ConsolidatorBase):
    supported_mimetypes = {"application/x-hdf5"}

    def __init__(self, stream_resource: StreamResource, descriptor: EventDescriptor):
        super().__init__(stream_resource, descriptor)
        self.assets.append(Asset(data_uri=self.uri, is_directory=False, parameter="data_uris", num=0))
        self.swmr = self._sres_parameters.get("swmr", True)

    def adapter_parameters(self) -> dict:
        """Parameters to be passed to the HDF5 adapter, a dictionary with the keys:

        dataset: list[str] - a path to the dataset within the hdf5 file represented as list split at `/`
        swmr: bool -- True to enable the single writer / multiple readers regime
        """
        params = {"dataset": self._sres_parameters["dataset"], "swmr": self.swmr}
        if slice := self._sres_parameters.get("slice", False):
            params["slice"] = slice
        if squeeze := self._sres_parameters.get("squeeze", False):
            params["squeeze"] = squeeze

        return params

    def update_from_stream_resource(self, stream_resource: StreamResource):
        """Add an Asset for a new StreamResource document"""
        if stream_resource["parameters"]["dataset"] != self._sres_parameters["dataset"]:
            raise ValueError("All StreamResource documents must have the same dataset path.")
        if stream_resource["parameters"].get("chunk_shape", ()) != self._sres_parameters.get("chunk_shape", ()):
            raise ValueError("All StreamResource documents must have the same chunk shape.")

        asset = Asset(
            data_uri=stream_resource["uri"], is_directory=False, parameter="data_uris", num=len(self.assets)
        )
        self.assets.append(asset)


class MultipartRelatedConsolidator(ConsolidatorBase):
    def __init__(
        self, permitted_extensions: set[str], stream_resource: StreamResource, descriptor: EventDescriptor
    ):
        super().__init__(stream_resource, descriptor)
        self.permitted_extensions: set[str] = permitted_extensions
        self.data_uris: list[str] = []
        self.chunk_shape = self.chunk_shape or (1,)  # I.e. number of frames per file (tiff, jpeg, etc.)
        if self.join_method == "concat":
            assert self.datum_shape[0] % self.chunk_shape[0] == 0, (
                f"Number of frames per file ({self.chunk_shape[0]}) must divide the total number of frames per "
                f"datum ({self.datum_shape[0]}): variable-sized files are not allowed."
            )

        def int_replacer(match):
            """Normalize filename template

            Replace an integer format specifier with a new-style format specifier, i.e. convert the template string
            from "old" to "new" Python style, e.g. "%s%s_%06d.tif" to "filename_{:06d}.tif"

            """
            flags, width, precision, type_char = match.groups()

            # Handle the flags
            flag_str = ""
            if "-" in flags:
                flag_str = "<"  # Left-align
            if "+" in flags:
                flag_str += "+"  # Show positive sign
            elif " " in flags:
                flag_str += " "  # Space before positive numbers
            if "0" in flags:
                flag_str += "0"  # Zero padding

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
            .replace("{:s}", self._sres_parameters.get("filename", ""), 1)
        )
        self.template = re.sub(r"%([-+#0 ]*)(\d+)?(?:\.(\d+))?([d])", int_replacer, self.template)

    def get_datum_uri(self, indx: int):
        """Return a full uri for a datum (an individual image file) based on its index in the sequence.

        This relies on the `template` parameter passed in the StreamResource, which is a string in the "new"
        Python formatting style that can be evaluated to a file name using the `.format(indx)` method given an
        integer index, e.g. "{:05d}.ext".
        """
        assert os.path.splitext(self.template)[1] in self.permitted_extensions
        return self.uri + self.template.format(indx)

    def consume_stream_datum(self, doc: StreamDatum):
        """Determine the number and names of files from indices of datums and the number of files per datum.

        In the most general case, each file may be a multipage tiff or a stack of images (frames) and a single
        datum may be composed of multiple such files, leading to a total of self.datum_shape[0] frames.
        Since each file necessarily represents a single chunk (tiffs can not be sub-chunked), the number of
        frames per file is equal to the leftmost chunk_shape dimension, self.chunk_shape[0].
        The number of files produced per each datum is then the ratio of these two numbers.

        If `join_method == "stack"`, we assume that each datum becomes its own index in the new leftmost dimension
        of the resulting dataset, and hence corresponds to a single file.
        """

        files_per_datum = self.datum_shape[0] // self.chunk_shape[0] if self.join_method == "concat" else 1
        first_file_indx = doc["indices"]["start"] * files_per_datum
        last_file_indx = doc["indices"]["stop"] * files_per_datum
        for indx in range(first_file_indx, last_file_indx):
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


class TIFFConsolidator(MultipartRelatedConsolidator):
    supported_mimetypes = {"multipart/related;type=image/tiff"}

    def __init__(self, stream_resource: StreamResource, descriptor: EventDescriptor):
        super().__init__({".tif", ".tiff"}, stream_resource, descriptor)


class JPEGConsolidator(MultipartRelatedConsolidator):
    supported_mimetypes = {"multipart/related;type=image/jpeg"}

    def __init__(self, stream_resource: StreamResource, descriptor: EventDescriptor):
        super().__init__({".jpeg", ".jpg"}, stream_resource, descriptor)


class NPYConsolidator(MultipartRelatedConsolidator):
    supported_mimetypes = {"multipart/related;type=application/x-npy"}
    join_method: Literal["stack", "concat"] = "stack"

    # NOTE: NPYConsolidator is tailored for tests in databroker with ophyd.sim devices.
    # Use with caution in other settings!

    def __init__(self, stream_resource: StreamResource, descriptor: EventDescriptor):
        # Unlike other image sequence formats (e.g. TIFF) the filename
        # template is hard-coded in the NPY_SEQ handler. We inject it
        # here so that the rest of the processing can be handled
        # generically by ConsolidatorBase.
        stream_resource["parameters"]["template"] = "%s_%d.npy"
        data_key = stream_resource["data_key"]
        datum_shape = descriptor["data_keys"][data_key]["shape"]
        stream_resource["parameters"]["chunk_shape"] = (1, *datum_shape)
        super().__init__({".npy"}, stream_resource, descriptor)


CONSOLIDATOR_REGISTRY = collections.defaultdict(
    lambda: ConsolidatorBase,
    {
        "text/csv;header=absent": CSVConsolidator,
        "application/x-hdf5": HDF5Consolidator,
        "multipart/related;type=image/tiff": TIFFConsolidator,
        "multipart/related;type=image/jpeg": JPEGConsolidator,
        "multipart/related;type=application/x-npy": NPYConsolidator,
    },
)


def consolidator_factory(stream_resource_doc, descriptor_doc):
    consolidator_class = CONSOLIDATOR_REGISTRY[stream_resource_doc["mimetype"]]
    return consolidator_class(stream_resource_doc, descriptor_doc)
