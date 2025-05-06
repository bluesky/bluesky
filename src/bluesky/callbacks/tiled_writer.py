import copy
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union, cast
from warnings import warn

import pyarrow
from event_model import RunRouter, unpack_datum_page, unpack_event_page
from event_model.documents import (
    Datum,
    DatumPage,
    Event,
    EventDescriptor,
    EventPage,
    Resource,
    RunStart,
    RunStop,
    StreamDatum,
    StreamResource,
)
from event_model.documents.stream_datum import StreamRange
from tiled.client import from_profile, from_uri
from tiled.client.base import BaseClient
from tiled.client.composite import Composite
from tiled.client.container import Container
from tiled.client.dataframe import DataFrameClient
from tiled.client.utils import handle_error
from tiled.structures.core import Spec
from tiled.utils import safe_json_dump

from ..consolidators import ConsolidatorBase, DataSource, StructureFamily, consolidator_factory
from ..utils import truncate_json_overflow
from .core import MIMETYPE_LOOKUP, CallbackBase

TABLE_UPDATE_BATCH_SIZE = 10000


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


class _RunWriter(CallbackBase):
    """Write documents from one Bluesky Run into Tiled.

    Datum, Resource, and StreamResource documents are cached until Event or StreamDatum documents are received,
    after which corresponding nodes are created in Tiled.
    """

    def __init__(self, client: BaseClient):
        self.client = client.include_data_sources()
        self.root_node: Union[None, Container] = None
        self._desc_nodes: dict[str, Composite] = {}  # references to the descriptor nodes by their uid's and names
        self._sres_nodes: dict[str, BaseClient] = {}
        self._internal_tables: dict[str, DataFrameClient] = {}  # references to the internal tables by desc_names
        self._datum_cache: dict[str, Datum] = {}
        self._stream_resource_cache: dict[str, StreamResource] = {}
        self._consolidators: dict[str, ConsolidatorBase] = {}
        self._internal_data_cache: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._external_data_cache: dict[str, StreamDatum] = {}
        self._next_frame_index: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"carry": 0, "index": 0}
        )
        self.data_keys_int: dict[str, dict[str, Any]] = {}
        self.data_keys_ext: dict[str, dict[str, Any]] = {}

    def _convert_resource_to_stream_resource(self, doc: Union[Resource, StreamResource]) -> StreamResource:
        """Make changes to and return a shallow copy of StreamRsource dictionary adhering to the new structure.

        Kept for back-compatibility with old StreamResource schema from event_model<1.20.0
        or Resource documents that are converted to StreamResources.
        """
        doc = copy.copy(doc)
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
            stream_resource_doc["mimetype"] = MIMETYPE_LOOKUP[resource_dict.pop("spec")]
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

        return stream_resource_doc

    def _write_internal_data(self, data_cache: list[dict[str, Any]], desc_node: Composite):
        """Write the internal data table to Tiled and clear the cache."""

        desc_name = desc_node.item["id"]  # Name of the descriptor (stream)
        table = pyarrow.Table.from_pylist(data_cache)

        if not (df_client := self._internal_tables.get(desc_name)):
            # Create a new "internal" data node and write the initial piece of data
            metadata = {
                k: v for k, v in (self.data_keys_ext | self.data_keys_int).items() if k in table.column_names
            }
            # Replace any nulls in the schema with string type
            schema = copy.copy(table.schema)
            for i, field in enumerate(table.schema):
                if pyarrow.types.is_null(field.type):
                    schema = schema.set(i, field.with_type(pyarrow.string()))
                elif pyarrow.types.is_list(field.type) and pyarrow.types.is_null(field.type.value_type):
                    schema = schema.set(i, field.with_type(pyarrow.list_(pyarrow.string())))
            # Initialize the table and keep a reference to the client
            df_client = desc_node.create_appendable_table(schema=schema, key="internal", metadata=metadata)
            self._internal_tables[desc_name] = df_client

        df_client.append_partition(table, 0)

    def start(self, doc: RunStart):
        self.root_node = self.client.create_container(
            key=doc["uid"],
            metadata={"start": truncate_json_overflow(dict(doc))},
            specs=[Spec("BlueskyRun", version="3.0")],
        )
        self._streams_node = self.root_node.create_container(key="streams")

    def stop(self, doc: RunStop):
        if self.root_node is None:
            raise RuntimeError("RunWriter is not properly initialized: no Start document has been recorded.")

        # Write the cached internal data
        for desc_name, data_cache in self._internal_data_cache.items():
            if data_cache:
                self._write_internal_data(data_cache, desc_node=self._desc_nodes[desc_name])
                data_cache.clear()

        # Write the cached external data
        for stream_datum_doc in self._external_data_cache.values():
            self.stream_datum(stream_datum_doc)

        # Validate structure for some StreamResource nodes
        for sres_uid, sres_node in self._sres_nodes.items():
            consolidator = self._consolidators[sres_uid]
            if consolidator._sres_parameters.get("_validate", False):
                try:
                    consolidator.validate(fix_errors=True)
                except Exception as e:
                    warn(f"Validation of StreamResource {sres_uid} failed with error: {e}", stacklevel=2)
                self._update_data_source_for_node(sres_node, consolidator.get_data_source())

        # Write the stop document to the metadata
        self.root_node.update_metadata(metadata={"stop": doc, **dict(self.root_node.metadata)})

    def descriptor(self, doc: EventDescriptor):
        if self.root_node is None:
            raise RuntimeError("RunWriter is not properly initialized: no Start document has been recorded.")

        # Rename some fields to match the current schema (in-place)
        # Loop over all dictionaries that specify data_keys (both event data_keys or configuration data_keys)
        conf_data_keys = (obj["data_keys"].values() for obj in doc["configuration"].values())
        for data_keys_spec in itertools.chain(doc["data_keys"].values(), *conf_data_keys):
            if dtype_str := data_keys_spec.pop("dtype_str", None):
                data_keys_spec["dtype_numpy"] = data_keys_spec.get("dtype_numpy") or dtype_str
        # Ensure that all event data_keys have object_name assigned
        for obj_name, data_keys_list in doc["object_keys"].items():
            for key in data_keys_list:
                doc["data_keys"][key]["object_name"] = obj_name

        # Keep specifications for external and internal data_keys for faster access
        data_keys = doc.get("data_keys", {})
        self.data_keys_int.update({k: v for k, v in data_keys.items() if "external" not in v.keys()})
        self.data_keys_ext.update({k: v for k, v in data_keys.items() if "external" in v.keys()})
        for item in self.data_keys_ext.values():
            item["external"] = item.pop("external", "")  # Make sure the value set to external key is not None

        # Create a new Composite node for the stream if it does not exist
        desc_name = doc["name"]  # Name of the descriptor/stream
        if desc_name not in self._desc_nodes.keys():
            metadata = {k: v for k, v in doc.items() if k not in {"name", "object_keys", "run_start"}}
            desc_node = self._streams_node.create_composite(
                key=desc_name,
                metadata=truncate_json_overflow(metadata),
                specs=[Spec("BlueskyEventStream", version="3.0")],
            )
        else:
            # Rare Case: This new descriptor likely updates stream configs mid-experiment
            # We assume tha the full descriptor has been already received, so we don't need to store everything
            # but only the uid, timestamp, and also data and timestamps in configuration (without conf specs).
            desc_node = self._desc_nodes[desc_name]
            updates = desc_node.metadata.get("_config_updates", []) + [{"uid": doc["uid"], "time": doc["time"]}]
            if conf_meta := doc.get("configuration"):
                updates[-1].update({"configuration": conf_meta})
            # Update the metadata with the new configuration
            desc_node.update_metadata(metadata={"_config_updates": truncate_json_overflow(updates)})

        self._desc_nodes[doc["uid"]] = self._desc_nodes[desc_name] = desc_node  # Keep a reference to the node

    def event(self, doc: Event):
        desc_uid = doc["descriptor"]
        desc_name = self._desc_nodes[desc_uid].item["id"]  # Name of the descriptor (stream)

        # Process _internal_ data -- those keys without 'external' flag or those that have been filled
        data_cache = self._internal_data_cache[desc_name]
        data_keys_spec = {k: v for k, v in self.data_keys_int.items() if doc["filled"].get(k, True)}
        data_keys_spec.update({k: v for k, v in self.data_keys_ext.items() if doc["filled"].get(k, False)})
        row = {"seq_num": doc["seq_num"], "time": doc["time"]}
        row.update({k: v for k, v in doc["data"].items() if k in data_keys_spec.keys()})
        row.update({f"ts_{k}": v for k, v in doc["timestamps"].items() if k in data_keys_spec.keys()})
        data_cache.append(row)

        # Do not write the data immediately; collect it in a cache and write in bulk later
        if len(data_cache) >= TABLE_UPDATE_BATCH_SIZE:
            self._write_internal_data(data_cache, desc_node=self._desc_nodes[desc_uid])
            data_cache.clear()

        # Process _external_ data: Loop over all referenced Datums
        for data_key in self.data_keys_ext.keys():
            if doc["filled"].get(data_key, False):
                continue

            if datum_id := doc["data"].get(data_key):
                if datum_id in self._datum_cache.keys():
                    # Convert the Datum document to the StreamDatum format
                    datum_doc = self._datum_cache.pop(datum_id)
                    uid = datum_doc["datum_id"]
                    sres_uid = datum_doc["resource"]

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
                        _next_index = self._next_frame_index[(desc_name, data_key)]
                        index_start = sum(_next_index.values())
                        _next_index["index"] = frame + 1
                        index_stop = sum(_next_index.values())
                        if index_stop < index_start:
                            # The datum is likely referencing a next Resource, but the indexing must continue
                            _next_index["carry"] = index_start
                            index_stop = sum(_next_index.values())
                    else:
                        index_start, index_stop = doc["seq_num"] - 1, doc["seq_num"]
                    indices = StreamRange(start=index_start, stop=index_stop)
                    seq_nums = StreamRange(start=index_start + 1, stop=index_stop + 1)

                    # Update the Resource document (add data_key to match the StreamResource schema)
                    # Save a copy of the StreamResource document; this allows to account for cases where one
                    # Resource is used by several data streams with different data_keys and datum_kwargs.
                    sres_uid_key = sres_uid + "-" + data_key
                    if (
                        sres_uid in self._stream_resource_cache.keys()
                        and sres_uid_key not in self._stream_resource_cache.keys()
                    ):
                        sres_doc = copy.deepcopy(self._stream_resource_cache[sres_uid])
                        sres_doc["data_key"] = data_key
                        sres_doc["parameters"].update(datum_kwargs)
                        self._stream_resource_cache[sres_uid_key] = sres_doc

                    # Produce the StreamDatum document
                    stream_datum_doc = StreamDatum(
                        uid=uid,
                        stream_resource=sres_uid_key,
                        descriptor=desc_uid,
                        indices=indices,
                        seq_nums=seq_nums,
                    )

                    # Try to concatenate and cache the StreamDatum document to process it later
                    if cached_stream_datum_doc := self._external_data_cache.pop(data_key, None):
                        try:
                            _doc = concatenate_stream_datums(cached_stream_datum_doc, stream_datum_doc)
                            if _doc["indices"]["stop"] - _doc["indices"]["start"] > TABLE_UPDATE_BATCH_SIZE:
                                # Write the (large) concatenated StreamDatum document immediately
                                self.stream_datum(_doc)
                            else:
                                # Keep it in cache for further concatenation
                                self._external_data_cache[data_key] = _doc
                        except ValueError:
                            # If concatenation fails, write the cached document and the new one separately
                            self.stream_datum(cached_stream_datum_doc)
                            self.stream_datum(stream_datum_doc)
                    else:
                        self._external_data_cache[data_key] = stream_datum_doc
                else:
                    raise RuntimeError(f"Datum {datum_id} is referenced before being declared.")

    def event_page(self, doc: EventPage):
        for _doc in unpack_event_page(doc):
            self.event(_doc)

    def datum(self, doc: Datum):
        self._datum_cache[doc["datum_id"]] = copy.copy(doc)

    def datum_page(self, doc: DatumPage):
        for _doc in unpack_datum_page(doc):
            self.datum(_doc)

    def resource(self, doc: Resource):
        self._stream_resource_cache[doc["uid"]] = self._convert_resource_to_stream_resource(doc)

    def stream_resource(self, doc: StreamResource):
        # Backwards compatibility: old StreamResource schema is converted to the new one (event-model<1.20.0)
        self._stream_resource_cache[doc["uid"]] = self._convert_resource_to_stream_resource(doc)

    def get_sres_node(self, sres_uid: str, desc_uid: Optional[str] = None) -> tuple[BaseClient, ConsolidatorBase]:
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
                )

            self._consolidators[sres_uid] = self._consolidators[full_data_key] = consolidator
            self._sres_nodes[sres_uid] = self._sres_nodes[full_data_key] = sres_node
        else:
            raise RuntimeError(f"Stream Resource {sres_uid} is referenced before being received.")

        return sres_node, consolidator

    def _update_data_source_for_node(self, node: BaseClient, data_source: DataSource):
        """Update StreamResource node in Tiled"""
        data_source.id = node.data_sources()[0].id  # ID of the existing DataSource record
        handle_error(
            node.context.http_client.put(
                node.uri.replace("/metadata/", "/data_source/", 1),
                content=safe_json_dump({"data_source": data_source}),
            )
        ).json()

    def stream_datum(self, doc: StreamDatum):
        # Get the Stream Resource node and the associtaed Consolidator
        sres_node, consolidator = self.get_sres_node(doc["stream_resource"], desc_uid=doc["descriptor"])
        consolidator.consume_stream_datum(doc)
        self._update_data_source_for_node(sres_node, consolidator.get_data_source())
