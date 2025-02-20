import copy
import itertools
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional, Union, cast

import pandas as pd
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
from pydantic.v1.utils import deep_update
from tiled.client import from_profile, from_uri
from tiled.client.base import BaseClient
from tiled.client.container import Container
from tiled.client.utils import handle_error
from tiled.structures.core import Spec
from tiled.structures.table import TableStructure
from tiled.utils import safe_json_dump

from ..consolidators import ConsolidatorBase, DataSource, StructureFamily, consolidator_factory
from .core import MIMETYPE_LOOKUP, CallbackBase

BULK_SIZE = 100


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
        for desc_uid, data_cache in self._internal_data_cache.items():
            if data_cache:
                df = pd.DataFrame(data_cache, index=range(len(data_cache)), columns=data_cache[0].keys())
                self._desc_nodes[desc_uid]["internal"].append_partition(df, 0)
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
        # Copy the document items, excluding the variable fields
        metadata = {k: v for k, v in doc.items() if k not in ("uid", "time", "configuration")}

        # Encapsulate variable fields into sub-dictionaries with uids as the keys
        uid = doc["uid"]
        conf_dict = {uid: doc.get("configuration", {})}
        time_dict = {uid: doc["time"]}
        var_fields = {"configuration": conf_dict, "time": time_dict}

        if desc_name not in self.root_node["streams"].keys():
            # Create a new descriptor node; write only the fixed part of the metadata
            desc_node = self.root_node["streams"].create_container(key=desc_name, metadata=metadata)
        else:
            # Get existing descriptor node (with fixed and variable metadata saved before)
            desc_node = self.root_node["streams"][desc_name]

        # Update (add new values to) variable fields of the metadata
        metadata = deep_update(dict(desc_node.metadata), var_fields)
        desc_node.update_metadata(metadata)

        # Keep specifications for external and internal data_keys for faster access
        if not isinstance(metadata["data_keys"], dict):
            raise RuntimeError("Expected data_keys to be a dictionary")
        self.data_keys_int.update({k: v for k, v in metadata["data_keys"].items() if "external" not in v.keys()})
        self.data_keys_ext.update({k: v for k, v in metadata["data_keys"].items() if "external" in v.keys()})

        # Write the configuration data: loop over all detectors
        if conf_dict[uid]:
            if desc_name not in self.root_node["config"].keys():
                conf_node = self.root_node["config"].create_container(key=desc_name)
            else:
                conf_node = self.root_node["config"][desc_name]
        for det_name, det_dict in doc.get("configuration", {}).items():
            df_dict = {"descriptor_uid": uid}
            df_dict.update(det_dict.get("data", {}))
            df_dict.update({f"ts_{c}": v for c, v in det_dict.get("timestamps", {}).items()})
            conf_node.write_awkward([df_dict], key=det_name)

            # TODO: clean-up AwkwardClient: better __repr__, .read(), etc.

            # df = pd.DataFrame([df_dict], index=[0], columns=df_dict.keys())
            # if det_name in conf_node.keys():
            #     conf_node[det_name].append_partition(df, 0)
            # else:
            #     conf_node.new(
            #         structure_family=StructureFamily.table,
            #         data_sources=[
            #             DataSource(
            #                 structure_family=StructureFamily.table,
            #                 structure=TableStructure.from_pandas(df),
            #                 mimetype="text/csv",
            #             ),
            #         ],
            #         key=det_name,
            #         metadata=det_dict["data_keys"],
            #     )
            #     conf_node[det_name].write_partition(df, 0)

        self._desc_nodes[uid] = desc_node

    def event(self, doc: Event):
        parent_node = self._desc_nodes[doc["descriptor"]]
        desc_name = parent_node.item["id"]  # Name of the descriptor/stream

        # Process _internal_ data -- those keys without 'external' flag or those that have been filled
        data_keys_spec = {k: v for k, v in self.data_keys_int.items() if doc["filled"].get(k, True)}
        data_keys_spec.update({k: v for k, v in self.data_keys_ext.items() if doc["filled"].get(k, False)})
        df_dict = {"seq_num": doc["seq_num"]}
        df_dict.update({k: v for k, v in doc["data"].items() if k in data_keys_spec.keys()})
        df_dict.update({f"ts_{k}": v for k, v in doc["timestamps"].items()})  # Keep all timestamps
        if self._node_exists[f"{desc_name}/internal"]:
            # Do not add the data immediately; collect it in a cache and write in bulk
            data_cache = self._internal_data_cache[doc["descriptor"]]
            data_cache.append(df_dict)
            if len(data_cache) >= BULK_SIZE:
                df = pd.DataFrame(data_cache, index=range(len(data_cache)), columns=df_dict.keys())
                parent_node["internal"].append_partition(df, 0)
                data_cache.clear()
        else:
            # Create a new internal data node and write the initial piece of data
            df = pd.DataFrame([df_dict], index=[0], columns=df_dict.keys())
            parent_node.new(
                structure_family=StructureFamily.table,
                data_sources=[
                    DataSource(
                        structure_family=StructureFamily.table,
                        structure=TableStructure.from_pandas(df),
                        mimetype="text/csv",
                    ),
                ],
                key="internal",
                metadata=data_keys_spec,
            )
            parent_node["internal"].write_partition(df, 0)
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
                    desc_uid = doc["descriptor"]  # From Event document

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
                            if _doc["indices"]["stop"] - _doc["indices"]["start"] > BULK_SIZE:
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
                handler = consolidator_factory(sres_doc, dict(desc_node.metadata))
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
