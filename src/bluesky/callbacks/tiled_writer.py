import copy
import itertools
from collections import defaultdict
from datetime import datetime
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
from tiled.client.container import Container
from tiled.client.utils import handle_error
from tiled.structures.core import Spec
from tiled.utils import safe_json_dump

from ..consolidators import ConsolidatorBase, DataSource, StructureFamily, consolidator_factory
from .core import MIMETYPE_LOOKUP, CallbackBase

BATCH_SIZE = 100


def build_summary(start_doc, stop_doc, stream_names):
    summary = {
        "uid": start_doc["uid"],
        "scan_id": start_doc.get("scan_id"),
        "timestamp": start_doc["time"],
        "datetime": datetime.fromtimestamp(start_doc["time"]).isoformat(),
        "plan_name": start_doc.get("plan_name"),
        "stream_names": stream_names,
    }
    if stop_doc is None:
        summary["duration"] = None
    else:
        summary["duration"] = stop_doc["time"] - start_doc["time"]
    return summary


def concatenate_stream_datums(*docs: StreamDatum):
    """Concatenate consecutive StreamDatum documents into a single StreamDatum document"""

    if len(docs) == 1:
        return docs[0]

    if len({doc["descriptor"] for doc in docs}) > 1:
        raise ValueError("All StreamDatum documents must reference the same descriptor.")
    if len({doc["stream_resource"] for doc in docs}) > 1:
        raise ValueError("All StreamDatum documents must reference the same stream_resource.")
    docs = sorted(docs, key=lambda doc: doc["indices"]["start"])
    for d1, d2 in itertools.pairwise(docs):
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
        self.client = client
        self.root_node: Union[None, Container] = None
        self._desc_nodes: dict[str, Container] = {}  # references to descriptor containers by their uid's
        self._sres_nodes: dict[str, BaseClient] = {}
        self._datum_cache: dict[str, Datum] = {}
        self._stream_resource_cache: dict[str, StreamResource] = {}
        self._handlers: dict[str, ConsolidatorBase] = {}
        self._internal_data_cache: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._external_data_cache: dict[str, StreamDatum] = {}
        self._node_exists: dict[str, bool] = defaultdict(lambda: False)  # Keep track of existing nodes
        self._next_frame_index: dict[tuple[str, str], int] = defaultdict(lambda: {"carry": 0, "index": 0})
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
            file_path = resource_dict.pop("root").strip("/") + "/" + resource_dict.pop("resource_path").strip("/")
            stream_resource_doc["uri"] = "file://localhost/" + file_path

        # Ensure that the internal path within HDF5 files is referenced with "dataset" parameter
        if stream_resource_doc["mimetype"] == "application/x-hdf5":
            stream_resource_doc["parameters"]["dataset"] = stream_resource_doc["parameters"].pop(
                "path", stream_resource_doc["parameters"].pop("dataset", "")
            )

        return stream_resource_doc

    def start(self, doc: RunStart):
        self.root_node = self.client.create_container(
            key=doc["uid"],
            metadata={"start": dict(doc)},
            specs=[Spec("BlueskyRun", version="1.0")],
        )

        # Create the backbone structure for the BlueskyRun container
        self.root_node.create_container(key="config")
        self.root_node.create_container(key="streams")
        self.root_node.create_container(key="views")

    def stop(self, doc: RunStop):
        if self.root_node is None:
            raise RuntimeError("RunWriter is not properly initialized: no Start document has been recorded.")

        # Write the cached internal data
        for desc_name, data_cache in self._internal_data_cache.items():
            if data_cache:
                table = pyarrow.Table.from_pylist(data_cache)
                self.root_node[f"streams/{desc_name}/internal"].append_partition(table, 0)
                data_cache.clear()

        # Write the cached external data
        for stream_datum_doc in self._external_data_cache.values():
            self.stream_datum(stream_datum_doc)

        # Validate structure for some StreamResource nodes
        for sres_uid, sres_node in self._sres_nodes.items():
            handler = self._handlers[sres_uid]
            if handler._sres_parameters.get("_validate", False):
                handler.validate(fix_errors=True)
                self._update_data_source_for_node(sres_node, handler.get_data_source())

        # Update the summary metadata with the stop document
        stream_names = list(self.root_node.keys())
        summary = build_summary(self.root_node.metadata["start"], dict(doc), stream_names)
        metadata = {"stop": doc, "summary": summary, **dict(self.root_node.metadata)}
        self.root_node.update_metadata(metadata=metadata)

    def descriptor(self, doc: EventDescriptor):
        if self.root_node is None:
            raise RuntimeError("RunWriter is not roperly initialized: no Start document has been recorded.")

        desc_name = doc["name"]  # Name of the descriptor/stream

        # Create a new stream node (container) for the descriptor if it does not exist
        data_keys = doc.get("data_keys", {})
        if desc_name not in self.root_node["streams"].keys():
            # NOTE: Maybe don't store data_keys in metadata?
            metadata = data_keys | {"uid": doc["uid"], "time": doc["time"]}
            desc_node = self.root_node["streams"].create_container(key=desc_name, metadata=metadata)
        else:
            desc_node = self.root_node["streams"][desc_name]
        self._desc_nodes[doc["uid"]] = desc_node  # Keep a reference to the descriptor node by its uid

        # Keep specifications for external and internal data_keys for faster access
        self.data_keys_int.update({k: v for k, v in data_keys.items() if "external" not in v.keys()})
        self.data_keys_ext.update({k: v for k, v in data_keys.items() if "external" in v.keys()})

        # Write the configuration data
        ### Option 1. Values and TS in awkward arrays, data_keys specs in metadata
        # if conf_dict := doc.get("configuration", None):
        #     # Put configuration data_keys specs into the metadata
        #     conf_data_keys = {}
        #     for obj_name, obj_dict in conf_dict.items():
        #         if _conf_data_keys := obj_dict.pop("data_keys", None): # Remove data_keys from the obj dictionary
        #             for item in _conf_data_keys.values():
        #                 item.update({"object_name": obj_name})   # Make consistent with usual data_keys
        #             conf_data_keys[obj_name] = _conf_data_keys
        #     conf_meta = {"data_keys": conf_data_keys} if conf_data_keys else {}
        #     conf_meta.update({"uid": doc["uid"], "time": doc["time"]})
        #     conf_list = [conf_dict]

        ### Option 2. Everything in an array of "records" (dicts)
        conf_list = []
        for obj_name, obj_dict in doc.get("configuration", {}).items():
            for key, val in obj_dict.get("data_keys", {}).items():
                val.update({"object_name": obj_name, "data_key": key})
                val["value"] = obj_dict.get("data", {}).get(key, None)
                val["timestamp"] = obj_dict.get("timestamps", {}).get(key, None)
                conf_list.append(val)  # awkward does not like None

        ### Option 2b. Add the usual data_keys specs as well
        for obj_name, data_keys_list in doc.get("object_keys", {}).items():
            for key in data_keys_list:
                val = doc.get("data_keys", {})[key]
                val.update({"data_key": key, "object_name": obj_name})
                conf_list.append(val)

        if conf_list:
            # Define the key (name) and metadata for the configuration node
            conf_meta = {"uid": doc["uid"], "time": doc["time"]}
            conf_key = desc_name
            if conf_key in self.root_node["config"].keys():
                conf_key = f"{desc_name}-{doc['uid'][:8]}"
                warn(
                    f"Configuration node for the '{desc_name}' stream already exists."
                    f"The updated configuration will be stored under a new key, '{conf_key}'.",
                    stacklevel=2,
                )
            self.root_node["config"].write_awkward(conf_list, key=conf_key, metadata=conf_meta)

    def event(self, doc: Event):
        desc_uid = doc["descriptor"]
        parent_node = self._desc_nodes[desc_uid]
        desc_name = parent_node.item["id"]  # Name of the descriptor/stream

        # Process _internal_ data -- those keys without 'external' flag or those that have been filled
        data_cache = self._internal_data_cache[desc_name]
        data_keys_spec = {k: v for k, v in self.data_keys_int.items() if doc["filled"].get(k, True)}
        data_keys_spec.update({k: v for k, v in self.data_keys_ext.items() if doc["filled"].get(k, False)})
        row = {"seq_num": doc["seq_num"]}
        row.update({k: v for k, v in doc["data"].items() if k in data_keys_spec.keys()})
        row.update({f"ts_{k}": int(v) for k, v in doc["timestamps"].items()})  # Keep all timestamps
        data_cache.append(row)

        if self._node_exists[f"{desc_name}/internal"]:
            # Do not write the data immediately; collect it in a cache and write in bulk
            if len(data_cache) >= BATCH_SIZE:
                table = pyarrow.Table.from_pylist(data_cache)
                parent_node["internal"].append_partition(table, 0)
                data_cache.clear()
        else:
            # Create a new "internal" data node and write the initial piece of data
            table = pyarrow.Table.from_pylist(data_cache)
            parent_node.create_appendable_table(schema=table.schema, key="internal", metadata=data_keys_spec)
            parent_node["internal"].append_partition(table, 0)
            data_cache.clear()
            # Mark the node as existing to avoid making API calls for each subsequent inserts
            self._node_exists[f"{desc_name}/internal"] = True

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

                    # Produce the StreamDatum document and process it as usual
                    stream_datum_doc = StreamDatum(
                        uid=uid,
                        stream_resource=sres_uid_key,
                        descriptor=desc_uid,
                        indices=indices,
                        seq_nums=seq_nums,
                    )

                    # Try to concatenate and cache the StreamDatum document to process later
                    if cached_stream_datum_doc := self._external_data_cache.pop(data_key, None):
                        try:
                            _doc = concatenate_stream_datums(cached_stream_datum_doc, stream_datum_doc)
                            if _doc["indices"]["stop"] - _doc["indices"]["start"] > BATCH_SIZE:
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
        """Get Stream Resource node from Tiled, if it already exists, or register it from a cached SR document"""

        if sres_uid in self._sres_nodes.keys():
            sres_node = self._sres_nodes[sres_uid]
            handler = self._handlers[sres_uid]

        elif sres_uid in self._stream_resource_cache.keys():
            if not desc_uid:
                raise RuntimeError("Descriptor uid must be specified to initialise a Stream Resource node")

            sres_doc = self._stream_resource_cache.get(sres_uid)
            desc_node = self._desc_nodes[desc_uid]

            # Check if there already exists a Node and a Handler for this data_key
            if sres_doc["data_key"] in desc_node.keys():
                sres_node = desc_node[sres_doc["data_key"]]
                # Find the id of the original cached StreamResource node in the tree
                for id, node in self._sres_nodes.items():
                    if node.uri == sres_node.uri:
                        sres_uid_old = id
                        sres_node = node  # Keep the reference to the same node
                        break
                handler = self._handlers[sres_uid_old]
                handler.consume_stream_resource(sres_doc)
            else:
                # Initialise a bluesky handler (consolidator) for the StreamResource
                handler = consolidator_factory(sres_doc, {"data_keys": desc_node.metadata})
                sres_node = desc_node.new(
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

    def _update_data_source_for_node(self, node: BaseClient, data_source: DataSource):
        """Update StreamResource node in Tiled"""
        # NOTE: Assigning data_source.id in the object and passing it in http
        # params is superfluous, but it is currently required by Tiled.
        node.refresh()
        data_source.id = node.data_sources()[0].id  # ID of the existing DataSource record
        endpoint = node.uri.replace("/metadata/", "/data_source/", 1)
        handle_error(
            node.context.http_client.put(
                endpoint,
                content=safe_json_dump({"data_source": data_source}),
                params={"data_source": data_source.id},
            )
        ).json()

    def stream_datum(self, doc: StreamDatum):
        # Get the Stream Resource node and the associtaed handler (consolidator)
        sres_node, handler = self.get_sres_node(doc["stream_resource"], desc_uid=doc["descriptor"])
        handler.consume_stream_datum(doc)
        self._update_data_source_for_node(sres_node, handler.get_data_source())
