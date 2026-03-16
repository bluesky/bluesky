import copy
import itertools
import logging
from collections import defaultdict, deque, namedtuple
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from warnings import warn

import pyarrow
from event_model import (
    DocumentNames,
    RunRouter,
    schema_validators,
    unpack_datum_page,
    unpack_event_page,
)
from event_model.documents import (
    Datum,
    DatumPage,
    DocumentType,
    Event,
    EventDescriptor,
    EventPage,
    Resource,
    RunStart,
    RunStop,
    StreamDatum,
    StreamResource,
)
from event_model.documents.event_descriptor import DataKey
from event_model.documents.stream_datum import StreamRange
from tiled.client import from_profile, from_uri
from tiled.client.base import BaseClient
from tiled.client.container import Container
from tiled.client.dataframe import DataFrameClient
from tiled.client.utils import handle_error
from tiled.structures.core import Spec
from tiled.utils import safe_json_dump

from ..consolidators import ConsolidatorBase, DataSource, StructureFamily, consolidator_factory
from ..run_engine import Dispatcher
from ..utils import truncate_json_overflow
from .core import CallbackBase
from .json_writer import JSONLinesWriter

# Aggregare the Event table rows and StreamDatums in batches before writing to Tiled
BATCH_SIZE = 10000

# Disallow using reserved words as data_keys identifiers
# Related: https://github.com/bluesky/event-model/pull/223
RESERVED_DATA_KEYS = ["time", "seq_num"]

# A lookup table for converting broad JSON types to numpy dtypes
JSON_TO_NUMPY_DTYPE = {"number": "<f8", "array": "<f8", "boolean": "|b1", "string": "<U0", "integer": "<i8"}

# A lookup table for converting Bluesky spec names to MIME types
MIMETYPE_LOOKUP = defaultdict(
    lambda: "application/octet-stream",
    {
        "hdf5": "application/x-hdf5",
        "AD_HDF5_SWMR_STREAM": "application/x-hdf5",
        "AD_HDF5_SWMR_SLICE": "application/x-hdf5",
        "PIL100k_HDF5": "application/x-hdf5",
        "XSP3": "application/x-hdf5",
        "XPS3": "application/x-hdf5",
        "XSP3_BULK": "application/x-hdf5",
        "XSP3_STEP": "application/x-hdf5",
        "AD_TIFF": "multipart/related;type=image/tiff",
        "AD_HDF5_GERM": "application/x-hdf5",
        "PIZZABOX_ENC_FILE_TXT_PD": "text/csv",
        "PANDA": "application/x-hdf5",
        "ROI_HDF5_FLY": "application/x-hdf5",
        "ROI_HDF51_FLY": "application/x-hdf5",
        "SIS_HDF51_FLY_STREAM_V1": "application/x-hdf5",
        "MERLIN_FLY_STREAM_V2": "application/x-hdf5",
        "MERLIN_HDF5_BULK": "application/x-hdf5",
        "TPX_HDF5": "application/x-hdf5",
        "EIGER2_STREAM": "application/x-hdf5",
        "NPY_SEQ": "multipart/related;type=application/x-npy",
        "ZEBRA_HDF51_FLY_STREAM_V1": "application/x-hdf5",
    },
)

logger = logging.getLogger(__name__)


def concatenate_stream_datums(*docs: StreamDatum):
    """Concatenate consecutive StreamDatum documents into a single StreamDatum document"""

    if len(docs) == 1:
        return docs[0]

    if len({doc["descriptor"] for doc in docs}) > 1:
        raise ValueError("All StreamDatum documents must reference the same descriptor.")
    if len({doc["stream_resource"] for doc in docs}) > 1:
        raise ValueError("All StreamDatum documents must reference the same stream_resource.")
    docs = tuple(sorted(docs, key=lambda doc: doc["indices"]["start"]))
    for d1, d2 in zip(docs[:-1], docs[1:]):  # TODO: use itertools.pairwise(docs) in python 3.10+
        if d1["indices"]["stop"] != d2["indices"]["start"]:
            raise ValueError("StreamDatum documents must be consecutive.")

    return StreamDatum(
        uid=docs[-1]["uid"],
        stream_resource=docs[-1]["stream_resource"],
        descriptor=docs[-1]["descriptor"],
        indices=StreamRange(start=docs[0]["indices"]["start"], stop=docs[-1]["indices"]["stop"]),
        seq_nums=StreamRange(start=docs[0]["seq_nums"]["start"], stop=docs[-1]["seq_nums"]["stop"]),
    )


# A named tuple to cache references to external data from Event documents.
ExternalEventDataReference = namedtuple(
    "ExternalEventDataReference",
    [
        "datum_id",  # The UID of the Datum document that references this external data
        "data_key",  # The data_key of the external data
        "desc_uid",  # The UID of the EventDescriptor document that this datum belongs to
        "seq_num",  # The sequence number of the Event document
    ],
)


class _ConditionalBackup:
    """Callback that tries to call the primary callback and, if it fails, flushes the buffer to backup callbacks.

    Once an error has been encountererd in the primary callback, all subsequent documents would be sent to the
    backup callbacks as well.

    This callback is intended to be used with a `RunRouter` and process documents from a single Bluesky run.
    """

    def __init__(self, primary_callback: Callable, backup_callbacks: list[Callable], maxlen: int = 1_000_000):
        self.primary_callback = primary_callback
        self.backup_callbacks = backup_callbacks
        self._buffer: deque[tuple[str, DocumentType]] = deque(maxlen=maxlen)
        self._push_to_backup = False

    def __call__(self, name: str, doc: DocumentType):
        self._buffer.append((name, doc))

        try:
            self.primary_callback(name, doc)
        except Exception as e:
            logger.warning(
                f"Primary callback {type(self.primary_callback).__name__} failed: {e}. "
                "Flushing buffer to backup callbacks.",
                stacklevel=2,
            )
            self._push_to_backup = True

        if self._push_to_backup:
            for name, doc in self._buffer:
                for bcb in self.backup_callbacks:
                    try:
                        bcb(name, doc)
                    except Exception as e:
                        logger.warning(
                            f"Backup callback {bcb.__class__.__name__} failed with error: {e}", stacklevel=2
                        )
            self._buffer.clear()


class RunNormalizer(CallbackBase):
    """Callback for updating Bluesky documents to their latest schema.

    This callback can be used to subscribe additional consumers that require the updated documents.
    Returns a shallow copy of the document to avoid modifying the original one.

    Parameters
    ----------
        patches : dict[str, Callable], optional
            A dictionary of patch functions to apply to the documents before modifying them.
            The keys are document names (e.g., "start", "stop", "descriptor", etc.), and the values
            are functions that take a document as input and return a modified document.
        spec_to_mimetype : dict[str, str], optional
            A dictionary mapping spec names to MIME types. This is used to convert `Resource` documents
            to the latest `StreamResource` schema.
            The supplied dictionary updates the default `MIMETYPE_LOOKUP` dictionary.
    """

    def __init__(
        self,
        patches: dict[str, Callable] | None = None,
        spec_to_mimetype: dict[str, str] | None = None,
    ):
        self._token_refs: dict[str, Callable] = {}
        self.dispatcher = Dispatcher()
        self.patches = patches or {}
        self.spec_to_mimetype = MIMETYPE_LOOKUP | (spec_to_mimetype or {})

        self._next_frame_index: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"carry": 0, "index": 0}
        )
        self._datum_cache: dict[str, Datum] = {}
        self._ext_ref_cache: list[ExternalEventDataReference] = []  # Cache for references to external Event data
        self._desc_name_by_uid: dict[str, str] = {}
        self._sres_cache: dict[str, StreamResource] = {}
        self._emitted: set[str] = set()  # UIDs of the StreamResource documents that have been emitted
        self._int_keys: set[str] = set()  # Names of internal data_keys
        self._ext_keys: set[str] = set()

    def _convert_resource_to_stream_resource(self, doc: Resource | StreamResource) -> StreamResource:
        """Make changes to and return a shallow copy of StreamRsource dictionary adhering to the new structure.

        Kept for back-compatibility with old StreamResource schema from event_model<1.20.0
        or Resource documents that are converted to StreamResources.
        """
        stream_resource_doc = cast(StreamResource, doc)

        if "mimetype" not in doc:
            # The document is a `Resource` or a < v1.20 `StreamResource`.
            # Both are converted to latest version `StreamResource`.
            for expected_key in ("spec", "root", "resource_path", "resource_kwargs"):
                if expected_key not in doc:
                    raise RuntimeError(
                        f"`Resource` or `StreamResource` legacy document is missing a '{expected_key}'"
                    )

            # Convert the Resource (or old StreamResource) document to a StreamResource document
            resource_dict = cast(dict, doc)
            stream_resource_doc["mimetype"] = self.spec_to_mimetype[resource_dict.pop("spec")]
            stream_resource_doc["parameters"] = resource_dict.pop("resource_kwargs", {})
            file_path = Path(resource_dict.pop("root").strip("/")).joinpath(
                resource_dict.pop("resource_path").strip("/")
            )
            stream_resource_doc["uri"] = "file://localhost/" + str(file_path).lstrip("/")

        # Ensure that the internal path within HDF5 files is referenced with "dataset" parameter
        if stream_resource_doc["mimetype"] == "application/x-hdf5":
            stream_resource_doc["parameters"]["dataset"] = stream_resource_doc["parameters"].pop(
                "path", stream_resource_doc["parameters"].pop("dataset", "")
            )

        # Ensure that only the necessary fields are present in the StreamResource document
        stream_resource_doc["data_key"] = stream_resource_doc.get("data_key", "")
        required_keys = {"data_key", "mimetype", "parameters", "uid", "uri"}
        for key in set(stream_resource_doc.keys()).difference(required_keys):
            stream_resource_doc.pop(key)  # type: ignore

        return stream_resource_doc

    def _convert_datum_to_stream_datum(
        self, datum_doc: Datum, data_key: str, desc_uid: str, seq_num: int
    ) -> tuple[StreamResource | None, StreamDatum]:
        """Convert the Datum document to the StreamDatum format

        This conversion requires (and is triggered when) the Event document is received. The function also returns
        a corresponding StreamResource document, if it hasn't been emitted yet.

        Parameters
        ----------
        datum_doc : Datum
            The Datum document to convert.
        data_key : str
            The data_key of the external data in the Event document; this parameter must be included in the new
            StreamResource document.
        desc_uid : str
            The UID of the EventDescriptor document that this datum belongs to.
        seq_num : int
            The sequence number of the Event document that this datum belongs to; 1-base index.

        Returns
        -------
        sres_doc : StreamResource, optional
            The corresponding StreamResource document, if it hasn't been emitted yet, otehrwise -- None.
        sdat_doc : StreamDatum
            The StreamDatum document corresponding to the Datum document.
        """

        # Some Datums contain datum_kwargs and the 'frame' field, which indicates the last index of the
        # frame. This should take precedence over the 'seq_num' field in the Event document. Keep the
        # last frame index in memory, since next Datums may refer to more than one frame (it is
        # assumed that Events always refer to a single frame).
        # There are cases when the frame_index is reset during the scan (e.g. if Datums for the same
        # data_key belong to different Resources), so the 'carry' field is used to keep track of the
        # previous frame index.
        datum_kwargs = datum_doc.get("datum_kwargs", {})
        frame = datum_kwargs.pop("frame", None)
        if frame is not None:
            desc_name = self._desc_name_by_uid[desc_uid]  # Name of the descriptor (stream)
            _next_index = self._next_frame_index[(desc_name, data_key)]
            index_start = sum(_next_index.values())
            _next_index["index"] = frame + 1
            index_stop = sum(_next_index.values())
            if index_stop < index_start:
                # The datum is likely referencing a next Resource, but the indexing must continue
                _next_index["carry"] = index_start
                index_stop = sum(_next_index.values())
        else:
            index_start, index_stop = seq_num - 1, seq_num
        indices = StreamRange(start=index_start, stop=index_stop)
        seq_nums = StreamRange(start=index_start + 1, stop=index_stop + 1)

        # produce the Resource document, if needed (add data_key to match the StreamResource schema)
        # Emit a copy of the StreamResource document with a new uid; this allows to account for cases
        # where one Resource is used by several data streams with different data_keys and datum_kwargs.
        sres_doc = None
        sres_uid = datum_doc["resource"]
        new_sres_uid = sres_uid + "-" + data_key
        if (sres_uid in self._sres_cache) and (new_sres_uid not in self._emitted):
            sres_doc = copy.deepcopy(self._sres_cache[sres_uid])
            sres_doc["data_key"] = data_key
            sres_doc["parameters"].update(datum_kwargs)
            sres_doc["uid"] = new_sres_uid

        # Produce the StreamDatum document
        sdat_doc = StreamDatum(
            uid=datum_doc["datum_id"],
            stream_resource=new_sres_uid,
            descriptor=desc_uid,
            indices=indices,
            seq_nums=seq_nums,
        )

        return sres_doc, sdat_doc

    def start(self, doc: RunStart):
        doc = copy.copy(doc)
        if patch := self.patches.get("start"):
            doc = patch(doc)
        self.emit(DocumentNames.start, doc)

    def stop(self, doc: RunStop):
        doc = copy.copy(doc)
        if patch := self.patches.get("stop"):
            doc = patch(doc)

        # If there are any cached references to external data, emit StreamResources and StreamDatums now
        for datum_id, data_key, desc_uid, seq_num in self._ext_ref_cache:
            if datum_doc := self._datum_cache.pop(datum_id, None):
                sres_doc, sdat_doc = self._convert_datum_to_stream_datum(datum_doc, data_key, desc_uid, seq_num)
                if (sres_doc is not None) and (sres_doc["uid"] not in self._emitted):
                    self.emit(DocumentNames.stream_resource, sres_doc)
                    self._emitted.add(sres_doc["uid"])
                self.emit(DocumentNames.stream_datum, sdat_doc)
            else:
                raise RuntimeError(
                    f"Cannot emit StreamDatum for {data_key} because the corresponding Datum document is missing."
                )

        self.emit(DocumentNames.stop, doc)

    def descriptor(self, doc: EventDescriptor):
        doc = copy.deepcopy(doc)
        if patch := self.patches.get("descriptor"):
            doc = patch(doc)

        # Rename data_keys that use reserved words, "time" and "seq_num"
        for name in RESERVED_DATA_KEYS:
            if name in doc["data_keys"].keys():
                if f"_{name}" in doc["data_keys"].keys():
                    raise ValueError(f"Cannot rename {name} to _{name} because it already exists")
                doc["data_keys"][f"_{name}"] = doc["data_keys"].pop(name)
                for obj_data_keys_list in doc["object_keys"].values():
                    if name in obj_data_keys_list:
                        obj_data_keys_list.remove(name)
                        obj_data_keys_list.append(f"_{name}")

        # Rename some fields (in-place) to match the current schema for the descriptor
        # Loop over all dictionaries that specify data_keys (both event data_keys or configuration data_keys)
        conf_data_keys = (obj["data_keys"].values() for obj in doc["configuration"].values())
        for data_keys_spec in itertools.chain(doc["data_keys"].values(), *conf_data_keys):
            # Determine numpy data type. From highest precedent to lowest:
            # 1. Try 'dtype_descr', optional, if present -- this is a structural dtype
            # 2. Try 'dtype_numpy', optional in the document schema.
            # 3. Try 'dtype_str', an old convention predataing 'dtype_numpy', not in the schema.
            # 4. Get 'dtype', required by the schema, which is a fuzzy JSON spec like 'number'
            #    and make a best effort to convert it to a numpy spec like '<u8'.
            # 5. If unable to do any of the above, do not set 'dtype_numpy' at all.
            dtype_descr = data_keys_spec.pop("dtype_descr", [])  # type: ignore
            dtype_str = data_keys_spec.pop("dtype_str", None)  # type: ignore
            if dtype_numpy := (
                list(map(list, dtype_descr))
                or data_keys_spec.get("dtype_numpy", dtype_str)
                or JSON_TO_NUMPY_DTYPE.get(data_keys_spec["dtype"])
            ):
                data_keys_spec["dtype_numpy"] = dtype_numpy

        # Ensure that all event data_keys have object_name assigned (for consistency)
        for obj_name, data_keys_list in doc["object_keys"].items():
            for key in data_keys_list:
                doc["data_keys"][key]["object_name"] = obj_name

        # Keep names of external and internal data_keys
        data_keys = doc.get("data_keys", {})
        self._int_keys.update({k for k, v in data_keys.items() if "external" not in v.keys()})
        self._ext_keys.update({k for k, v in data_keys.items() if "external" in v.keys()})
        for key in self._ext_keys:
            if key in data_keys:
                data_keys[key]["external"] = data_keys[key].pop("external", "")  # Make sure the value is not None

        # Keep a reference to the descriptor name (stream) by its uid
        self._desc_name_by_uid[doc["uid"]] = doc["name"]

        # Emit the updated descriptor document
        self.emit(DocumentNames.descriptor, doc)

    def event(self, doc: Event):
        doc = copy.deepcopy(doc)
        if patch := self.patches.get("event"):
            doc = patch(doc)

        # Part 0. ----- Preprocessing -----
        # Rename data_keys that use reserved words, "time" and "seq_num"
        for name in RESERVED_DATA_KEYS:
            if name in doc["data"].keys():
                doc["data"][f"_{name}"] = doc["data"].pop(name)
                doc["timestamps"][f"_{name}"] = doc["timestamps"].pop(name)
            if name in doc["filled"].keys():
                doc["filled"][f"_{name}"] = doc["filled"].pop(name)

        # Part 1. ----- Internal Data -----
        # Emit a new Event with _internal_ data: select only keys without 'external' flag or those that are filled
        filled = doc.pop("filled", {})
        event_keys = [k for k in self._int_keys if filled.get(k, True)] + [
            k for k in self._ext_keys if filled.get(k, False)
        ]
        event_doc = copy.copy(doc)  # Keep another copy with the external data_keys intact
        event_doc["data"] = {k: v for k, v in doc["data"].items() if k in event_keys}
        event_doc["timestamps"] = {k: v for k, v in doc["timestamps"].items() if k in event_keys}
        self.emit(DocumentNames.event, event_doc)

        # Part 2. ----- External Data -----
        # Process _external_ data: Loop over all referenced Datums and all external data keys that are not filled
        for data_key, datum_id in doc["data"].items():
            if data_key not in set(self._ext_keys).difference(event_keys):
                continue  # Skip internal data_keys
            if datum_doc := self._datum_cache.pop(datum_id, None):
                sres_doc, sdat_doc = self._convert_datum_to_stream_datum(
                    datum_doc, data_key, desc_uid=doc["descriptor"], seq_num=doc["seq_num"]
                )
                if (sres_doc is not None) and (sres_doc["uid"] not in self._emitted):
                    self.emit(DocumentNames.stream_resource, sres_doc)
                    self._emitted.add(sres_doc["uid"])  # Mark the StreamResource as emitted
                self.emit(DocumentNames.stream_datum, sdat_doc)
            else:
                # This Event references a Datum that has not been received yet; cache and process it later
                missing = ExternalEventDataReference(datum_id, data_key, doc["descriptor"], doc["seq_num"])
                self._ext_ref_cache.append(missing)

    def resource(self, doc: Resource):
        doc = copy.copy(doc)
        if patch := self.patches.get("resource"):
            doc = patch(doc)

        # Convert the Resource document to StreamResource format
        self._sres_cache[doc["uid"]] = self._convert_resource_to_stream_resource(doc)

    def stream_resource(self, doc: StreamResource):
        doc = copy.copy(doc)
        if patch := self.patches.get("stream_resource"):
            doc = patch(doc)

        # Convert the StreamResource document to the latest schema
        doc = self._convert_resource_to_stream_resource(doc)
        self.emit(DocumentNames.stream_resource, doc)

    def stream_datum(self, doc: StreamDatum):
        doc = copy.copy(doc)
        if patch := self.patches.get("stream_datum"):
            doc = patch(doc)
        self.emit(DocumentNames.stream_datum, doc)

    def datum(self, doc: Datum):
        doc = copy.copy(doc)
        if patch := self.patches.get("datum"):
            doc = patch(doc)

        self._datum_cache[doc["datum_id"]] = doc

    def datum_page(self, doc: DatumPage):
        for _doc in unpack_datum_page(doc):
            self.datum(_doc)

    def event_page(self, doc: EventPage):
        for _doc in unpack_event_page(doc):
            self.event(_doc)

    def emit(self, name, doc):
        """Check the document schema and send to the dispatcher"""
        schema_validators[name].validate(doc)
        self.dispatcher.process(name, doc)

    def subscribe(self, func, name="all"):
        """Convenience function for dispatcher subscription"""
        token = self.dispatcher.subscribe(func, name)
        self._token_refs[token] = func
        return token

    def unsubscribe(self, token):
        """Convenience function for dispatcher un-subscription"""
        self._token_refs.pop(token, None)
        self.dispatcher.unsubscribe(token)


class _RunWriter(CallbackBase):
    """Write documents from a single Bluesky Run into Tiled.

    This callback is intended to be used with a `RunRouter` and process documents from a single Bluesky run.
    It creates a new Tiled Container for the run and writes the internal data provided in Event documents
    as well as registers external files provided in StreamResource and StreamDatum documents.

    The callback is intended to be used with the most recent version of EventModel; to support legacy
    schemas, use the `RunNormalizer` callback first to update the documents before writing them to Tiled.

    Parameters
    ----------
        client : BaseClient
            The Tiled client to use for writing the data.
    """

    def __init__(self, client: BaseClient, batch_size: int = BATCH_SIZE):
        self.client = client
        self.root_node: None | Container = None
        self._desc_nodes: dict[str, Container] = {}  # references to the descriptor nodes by their uid's and names
        self._sres_nodes: dict[str, BaseClient] = {}
        self._internal_tables: dict[str, DataFrameClient] = {}  # references to the internal tables by desc_names
        self._stream_resource_cache: dict[str, StreamResource] = {}
        self._consolidators: dict[str, ConsolidatorBase] = {}
        self._internal_data_cache: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._external_data_cache: dict[str, StreamDatum] = {}  # sres_uid : (concatenated) StreamDatum
        self._batch_size = batch_size
        self.data_keys: dict[str, DataKey] = {}
        self.access_tags = None

    def _write_internal_data(self, data_cache: list[dict[str, Any]], desc_node: Container):
        """Write the internal data table to Tiled and clear the cache."""

        desc_name = desc_node.item["id"]  # Name of the descriptor (stream)
        table = pyarrow.Table.from_pylist(data_cache)

        if not (df_client := self._internal_tables.get(desc_name)):
            # Create a new "internal" data node and write the initial piece of data
            metadata = {k: v for k, v in self.data_keys.items() if k in table.column_names}
            metadata = truncate_json_overflow(metadata)
            # Replace any nulls in the schema with string type
            schema = copy.copy(table.schema)
            for i, field in enumerate(table.schema):
                if pyarrow.types.is_null(field.type):
                    schema = schema.set(i, field.with_type(pyarrow.string()))
                elif pyarrow.types.is_list(field.type) and pyarrow.types.is_null(field.type.value_type):
                    schema = schema.set(i, field.with_type(pyarrow.list_(pyarrow.string())))
            # Initialize the table and keep a reference to the client
            df_client = desc_node.create_appendable_table(
                schema=schema, key="internal", metadata=metadata, access_tags=self.access_tags
            )
            self._internal_tables[desc_name] = df_client

        df_client.append_partition(table, 0)

    def _write_external_data(self, doc: StreamDatum):
        """Register the external data provided in StreamDatum in Tiled"""

        sres_uid, desc_uid = doc["stream_resource"], doc["descriptor"]
        sres_node, consolidator = self.get_sres_node(sres_uid, desc_uid)
        consolidator.consume_stream_datum(doc)
        self._update_data_source_for_node(sres_node, consolidator.get_data_source())

    def _update_data_source_for_node(self, node: BaseClient, data_source: DataSource):
        """Update StreamResource node in Tiled"""
        data_source.id = node.data_sources()[0].id  # ID of the existing DataSource record
        handle_error(
            node.context.http_client.put(
                node.uri.replace("/metadata/", "/data_source/", 1),
                content=safe_json_dump({"data_source": data_source}),
            )
        ).json()

    def start(self, doc: RunStart):
        doc = copy.copy(doc)
        self.access_tags = doc.pop("tiled_access_tags", None)  # type: ignore
        self.root_node = self.client.create_container(
            key=doc["uid"],
            metadata={"start": truncate_json_overflow(dict(doc))},
            specs=[Spec("BlueskyRun", version="3.0")],
            access_tags=self.access_tags,
        )

    def stop(self, doc: RunStop):
        if self.root_node is None:
            raise RuntimeError("RunWriter is not properly initialized: no Start document has been recorded.")

        # Write the cached internal data
        for desc_name, data_cache in self._internal_data_cache.items():
            if data_cache:
                self._write_internal_data(data_cache, desc_node=self._desc_nodes[desc_name])
                data_cache.clear()

        # Write the cached StreamDatums data
        for stream_datum_doc in self._external_data_cache.values():
            self._write_external_data(stream_datum_doc)

        # Validate structure for some StreamResource nodes
        for sres_uid, sres_node in self._sres_nodes.items():
            consolidator = self._consolidators[sres_uid]
            if consolidator._sres_parameters.get("_validate", False):
                try:
                    consolidator.validate(fix_errors=True)
                except Exception as e:
                    msg = f"{type(e).__name__}: " + str(e).replace("\n", " ").replace("\r", "").strip()
                    warn(f"Validation of StreamResource {sres_uid} failed with error: {msg}", stacklevel=2)
                self._update_data_source_for_node(sres_node, consolidator.get_data_source())

        # Write the stop document to the metadata
        self.root_node.update_metadata(metadata={"stop": doc, **dict(self.root_node.metadata)}, drop_revision=True)

    def descriptor(self, doc: EventDescriptor):
        if self.root_node is None:
            raise RuntimeError("RunWriter is not properly initialized: no Start document has been recorded.")

        desc_name = doc["name"]  # Name of the descriptor/stream
        self.data_keys.update(doc.get("data_keys", {}))

        # Create a new Container with "composite" spec for the stream if it does not exist
        # Since the data_keys are guaranteed to be unique, we don't need to perform client-side validation of the
        # "composite" spec constraints and can use the `.base` (Container) client directly.
        if desc_name not in self._desc_nodes.keys():
            metadata = {k: v for k, v in doc.items() if k not in {"name", "object_keys", "run_start"}}
            desc_node = self.root_node.create_container(
                key=desc_name,
                metadata=truncate_json_overflow(metadata),
                specs=[Spec("BlueskyEventStream", version="3.0"), Spec("composite")],
                access_tags=self.access_tags,
            ).base
        else:
            # Rare Case: This new descriptor likely updates stream configs mid-experiment
            # We assume tha the full descriptor has been already received, so we don't need to store everything
            # but only the uid, timestamp, and also data and timestamps in configuration (without conf specs).
            desc_node = self._desc_nodes[desc_name]
            updates = desc_node.metadata.get("_config_updates", []) + [{"uid": doc["uid"], "time": doc["time"]}]
            if conf_meta := doc.get("configuration"):
                updates[-1].update({"configuration": conf_meta})
            # Update the metadata with the new configuration
            metadata = {"_config_updates": truncate_json_overflow(updates)}
            desc_node.update_metadata(metadata=metadata, drop_revision=True)

        self._desc_nodes[doc["uid"]] = self._desc_nodes[desc_name] = desc_node  # Keep a reference to the node

    def event(self, doc: Event):
        desc_uid = doc["descriptor"]
        desc_name = self._desc_nodes[desc_uid].item["id"]  # Name of the descriptor (stream)

        # Do not write the data immediately; collect it in a cache and write in bulk later
        data_cache = self._internal_data_cache[desc_name]
        row = {"seq_num": doc["seq_num"], "time": doc["time"], **doc["data"]}
        row.update({f"ts_{k}": v for k, v in doc["timestamps"].items()})
        data_cache.append(row)

        if len(data_cache) >= self._batch_size:
            self._write_internal_data(data_cache, desc_node=self._desc_nodes[desc_uid])
            data_cache.clear()

    def event_page(self, doc: EventPage):
        for _doc in unpack_event_page(doc):
            self.event(_doc)

    def stream_resource(self, doc: StreamResource):
        self._stream_resource_cache[doc["uid"]] = doc

    def get_sres_node(self, sres_uid: str, desc_uid: str | None = None) -> tuple[BaseClient, ConsolidatorBase]:
        """Get the Tiled node and the associate Consolidator corresponding to the data_key in StreamResource

        If the node does not exist, register it from a cached StreamResource document. Keep a reference to the
        node and the corresponding Consolidator object. If the node already exists, return the existing one.

        The nodes and consolidators are referenced by both:
        - sres_uid: the uid of the StreamResource document
        - desc_name + data_key: the name of the descriptor (stream) and the data_key
        """

        if sres_uid in self._sres_nodes.keys():
            sres_node = self._sres_nodes[sres_uid]
            consolidator = self._consolidators[sres_uid]

        elif sres_uid in self._stream_resource_cache.keys():
            if not desc_uid:
                raise RuntimeError("Descriptor uid must be specified to initialise a Stream Resource node")

            sres_doc = self._stream_resource_cache[sres_uid]
            desc_node = self._desc_nodes[desc_uid]
            full_data_key = f"{desc_node.item['id']}_{sres_doc['data_key']}"  # desc_name + data_key

            # Check if there already exists a Node and a Consolidator for this data_key
            # i.e. this is an additional StreamResource, whose data should be concatenated with the existing one
            if full_data_key in self._sres_nodes.keys():
                sres_node = self._sres_nodes[full_data_key]
                consolidator = self._consolidators[full_data_key]
                consolidator.update_from_stream_resource(sres_doc)
            else:
                consolidator = consolidator_factory(sres_doc, desc_node.metadata)
                sres_node = desc_node.new(
                    key=consolidator.data_key,
                    structure_family=StructureFamily.array,
                    data_sources=[consolidator.get_data_source()],
                    metadata={},
                    specs=[],
                    access_tags=self.access_tags,
                )

            self._consolidators[sres_uid] = self._consolidators[full_data_key] = consolidator
            self._sres_nodes[sres_uid] = self._sres_nodes[full_data_key] = sres_node
        else:
            raise RuntimeError(f"Stream Resource {sres_uid} is referenced before being received.")

        return sres_node, consolidator

    def stream_datum(self, doc: StreamDatum):
        if self._batch_size <= 1:
            # If batch size is 1, write the StreamDatum immediately
            self._write_external_data(doc)
            return

        # Try to concatenate and cache the StreamDatum document to process it later
        sres_uid = doc["stream_resource"]
        if cached_stream_datum_doc := self._external_data_cache.pop(sres_uid, None):
            try:
                _doc = concatenate_stream_datums(cached_stream_datum_doc, doc)
                if _doc["indices"]["stop"] - _doc["indices"]["start"] >= self._batch_size:
                    self._write_external_data(_doc)
                else:
                    self._external_data_cache[sres_uid] = _doc
            except ValueError:
                # If concatenation fails, write the cached document and then the new one immediately
                self._write_external_data(cached_stream_datum_doc)
                self._write_external_data(doc)
        else:
            self._external_data_cache[sres_uid] = doc


class TiledWriter:
    """Callback for write metadata and data from Bluesky documents into Tiled.

    This callback relies on the `RunRouter` to route documents from one or more runs into
    independent instances of the `_RunWriter` callback. The `RunRouter` is responsible for
    creating a new instance of the `_RunWriter` for each run.

    Parameters
    ----------
        client : `tiled.client.BaseClient`
            The Tiled client to use for writing data. This client must be initialized with
            the appropriate credentials and connection parameters to access the Tiled server.
        normalizer : Optional[CallbackBase]
            A callback for normalizing Bluesky documents to the latest schema. If not provided,
            the default `RunNormalizer` will be used. The supplied normalizer should accept
            `patches` and `spec_to_mimetype` (or `**kwargs`) for initialization.
            To disable normalization and pass the incoming document directly to _RunWriter,
            set this parameter to `None`.
        patches : Optional[dict[str, Callable]]
            A dictionary of patch functions to apply to specific document types before normalizing
            and writing them. The keys should be the document names (e.g., "start", "stop",
            "descriptor", etc.), and the values should be functions that take a document and return
            a modified document of the same type.
            This argument is ignored if `normalizer` is set to `None`.
        spec_to_mimetype : Optional[dict[str, str]]
            A dictionary mapping spec names to MIME types. This is used to convert `Resource` documents
            to the latest `StreamResource` schema. If not provided, the default mapping will be used.
            This argument is ignored if `normalizer` is set to `None`.
        backup_directory : Optional[str]
            If specified, this directory will be used to back up runs that fail to be written
            to Tiled. All documents for the entire Bluesky Run will be written in JSONLines format,
            allowing for recovery in case of errors during the writing process.
        batch_size : int
            The number of Events or StreamDatums collect before writing them to Tiled.
            This is useful for reducing the number of write operations and improving performance when
            writing large amounts of data (e.g. database migration). For streaming applications,
            it is recommended to set this parameter to <= 1, so that each Event or StreamDatum is written
            to Tiled immediately after they are received.
    """

    def __init__(
        self,
        client: BaseClient,
        *,
        normalizer: type[CallbackBase] | None = RunNormalizer,
        patches: dict[str, Callable] | None = None,
        spec_to_mimetype: dict[str, str] | None = None,
        backup_directory: str | None = None,
        batch_size: int = BATCH_SIZE,
    ):
        self.client = client.include_data_sources()
        self.patches = patches or {}
        self.spec_to_mimetype = spec_to_mimetype or {}
        self.backup_directory = backup_directory
        self._normalizer = normalizer
        self._run_router = RunRouter([self._factory])
        self._batch_size = batch_size

    def _factory(self, name, doc):
        """Factory method to create a callback for writing a single run into Tiled."""
        cb = run_writer = _RunWriter(self.client, batch_size=self._batch_size)

        if self._normalizer:
            # If normalize is True, create a RunNormalizer callback to update documents to the latest schema
            cb = self._normalizer(patches=self.patches, spec_to_mimetype=self.spec_to_mimetype)
            cb.subscribe(run_writer)

        if self.backup_directory:
            # If backup_directory is specified, create a conditional backup callback writing documents to JSONLines
            cb = _ConditionalBackup(cb, [JSONLinesWriter(self.backup_directory)])

        return [cb], []

    @classmethod
    def from_uri(
        cls,
        uri,
        *,
        normalizer: type[CallbackBase] | None = RunNormalizer,
        patches: dict[str, Callable] | None = None,
        spec_to_mimetype: dict[str, str] | None = None,
        backup_directory: str | None = None,
        batch_size: int = BATCH_SIZE,
        **kwargs,
    ):
        client = from_uri(uri, **kwargs)
        return cls(
            client,
            normalizer=normalizer,
            patches=patches,
            spec_to_mimetype=spec_to_mimetype,
            backup_directory=backup_directory,
            batch_size=batch_size,
        )

    @classmethod
    def from_profile(
        cls,
        profile,
        *,
        normalizer: type[CallbackBase] | None = RunNormalizer,
        patches: dict[str, Callable] | None = None,
        spec_to_mimetype: dict[str, str] | None = None,
        backup_directory: str | None = None,
        batch_size: int = BATCH_SIZE,
        **kwargs,
    ):
        client = from_profile(profile, **kwargs)
        return cls(
            client,
            normalizer=normalizer,
            patches=patches,
            spec_to_mimetype=spec_to_mimetype,
            backup_directory=backup_directory,
            batch_size=batch_size,
        )

    def __call__(self, name, doc):
        self._run_router(name, doc)
