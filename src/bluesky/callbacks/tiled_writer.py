import copy
from typing import Any, Optional, Union, cast

import pandas as pd
from event_model import RunRouter
from event_model.documents import (
    Datum,
    Event,
    EventDescriptor,
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
        self._consolidators: dict[str, ConsolidatorBase] = {}
        self.data_keys_int: dict[str, dict[str, Any]] = {}
        self.data_keys_ext: dict[str, dict[str, Any]] = {}

    def _convert_resource_to_stream_resource(self, doc: Union[Resource, StreamResource]) -> StreamResource:
        """Kept for back-compatibility with old StreamResource schema from event_model<1.20.0
        or Resource documents that are converted to StreamResources.

        Will make changes to and return a shallow copy of StreamRsource dictionary adhering to the new structure.
        """
        doc = copy.copy(doc)
        stream_resource_doc = cast(StreamResource, doc)
        # If the document already adheres to StreamResource schema, return it
        if "mimetype" in doc:
            return stream_resource_doc

        # The document is a `Resource` or a < v1.20 `StreamResource`.
        # Both are converted to latest version `StreamResource`.
        for expected_key in ("spec", "root", "resource_path", "resource_kwargs"):
            if expected_key not in doc:
                raise RuntimeError(f"`Resource` or `StreamResource` legacy document is missing a '{expected_key}'")

        # Convert the Resource (or old StreamResource) document to a StreamResource document
        resource_dict = cast(dict, doc)
        stream_resource_doc["mimetype"] = MIMETYPE_LOOKUP[resource_dict.pop("spec")]
        stream_resource_doc["parameters"] = resource_dict.pop("resource_kwargs", {})
        file_path = resource_dict.pop("root").strip("/") + "/" + resource_dict.pop("resource_path").strip("/")
        stream_resource_doc["uri"] = "file://localhost/" + file_path

        return stream_resource_doc

    def start(self, doc: RunStart):
        self.root_node = self.client.create_container(
            key=doc["uid"],
            metadata={"start": doc},
            specs=[Spec("BlueskyRun", version="1.0")],
        )

    def stop(self, doc: RunStop):
        if self.root_node is None:
            raise RuntimeError("RunWriter is properly initialized: no Start document has been recorded.")

        metadata = {"stop": doc, **dict(self.root_node.metadata)}
        self.root_node.update_metadata(metadata=metadata)

    def descriptor(self, doc: EventDescriptor):
        if self.root_node is None:
            raise RuntimeError("RunWriter is properly initialized: no Start document has been recorded.")

        desc_name = doc["name"]
        # Copy the document items, excluding the variable fields
        metadata = {k: v for k, v in doc.items() if k not in ("uid", "time", "configuration")}

        # Encapsulate variable fields into sub-dictionaries with uids as the keys
        uid = doc["uid"]
        conf_dict = {uid: doc.get("configuration", {})}
        time_dict = {uid: doc["time"]}
        var_fields = {"configuration": conf_dict, "time": time_dict}

        if desc_name not in self.root_node.keys():
            # Create a new descriptor node; write only the fixed part of the metadata
            desc_node = self.root_node.create_container(key=desc_name, metadata=metadata)
            desc_node.create_container(key="external")
            desc_node.create_container(key="internal")
            desc_node.create_container(key="config")
        else:
            # Get existing descriptor node (with fixed and variable metadata saved before)
            desc_node = self.root_node[desc_name]

        # Update (add new values to) variable fields of the metadata
        metadata = deep_update(dict(desc_node.metadata), var_fields)
        desc_node.update_metadata(metadata)

        # Keep specifications for external and internal data_keys for faster access
        if not isinstance(metadata["data_keys"], dict):
            raise RuntimeError("Expected data_keys to be a dictionary")
        self.data_keys_int.update({k: v for k, v in metadata["data_keys"].items() if "external" not in v.keys()})
        self.data_keys_ext.update({k: v for k, v in metadata["data_keys"].items() if "external" in v.keys()})

        # Write the configuration data: loop over all detectors
        conf_node = desc_node["config"]
        for det_name, det_dict in doc.get("configuration", {}).items():
            df_dict = {"descriptor_uid": uid}
            df_dict.update(det_dict.get("data", {}))
            df_dict.update({f"ts_{c}": v for c, v in det_dict.get("timestamps", {}).items()})
            df = pd.DataFrame(df_dict, index=[0], columns=df_dict.keys())
            if det_name in conf_node.keys():
                conf_node[det_name].append_partition(df, 0)
            else:
                conf_node.new(
                    structure_family=StructureFamily.table,
                    data_sources=[
                        DataSource(
                            structure_family=StructureFamily.table,
                            structure=TableStructure.from_pandas(df),
                            mimetype="text/csv",
                        ),
                    ],
                    key=det_name,
                    metadata=det_dict["data_keys"],
                )
                conf_node[det_name].write_partition(df, 0)

        self._desc_nodes[uid] = desc_node

    def event(self, doc: Event):
        desc_node = self._desc_nodes[doc["descriptor"]]

        # Process _internal_ data -- those keys without 'external' flag or those that have been filled
        data_keys_spec = {k: v for k, v in self.data_keys_int.items() if doc["filled"].get(k, True)}
        data_keys_spec.update({k: v for k, v in self.data_keys_ext.items() if doc["filled"].get(k, False)})
        parent_node = desc_node["internal"]
        df_dict = {"seq_num": doc["seq_num"]}
        df_dict.update({k: v for k, v in doc["data"].items() if k in data_keys_spec.keys()})
        df_dict.update({f"ts_{k}": v for k, v in doc["timestamps"].items()})  # Keep all timestamps
        df = pd.DataFrame(df_dict, index=[0], columns=df_dict.keys())
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
                metadata=data_keys_spec,
            )
            parent_node["events"].write_partition(df, 0)

        # Process _external_ data: Loop over all referenced Datums
        for data_key in self.data_keys_ext.keys():
            if doc["filled"].get(data_key, False):
                continue

            if datum_id := doc["data"].get(data_key):
                if datum_id in self._datum_cache.keys():
                    # Convert the Datum document to the StreamDatum format
                    datum_doc = self._datum_cache.pop(datum_id)
                    uid = datum_doc["datum_id"]
                    stream_resource = datum_doc["resource"]
                    descriptor = doc["descriptor"]  # From Event document
                    indices = StreamRange(start=doc["seq_num"] - 1, stop=doc["seq_num"])
                    seq_nums = StreamRange(start=doc["seq_num"], stop=doc["seq_num"] + 1)
                    stream_datum_doc = StreamDatum(
                        uid=uid,
                        stream_resource=stream_resource,
                        descriptor=descriptor,
                        indices=indices,
                        seq_nums=seq_nums,
                    )
                    # Update the Resource document (add data_key as in StreamResource)
                    if stream_datum_doc["stream_resource"] in self._stream_resource_cache.keys():
                        self._stream_resource_cache[stream_datum_doc["stream_resource"]]["data_key"] = data_key

                    self.stream_datum(stream_datum_doc)
                else:
                    raise RuntimeError(f"Datum {datum_id} is referenced before being declared.")

    def datum(self, doc: Datum):
        self._datum_cache[doc["datum_id"]] = copy.copy(doc)

    def resource(self, doc: Resource):
        self._stream_resource_cache[doc["uid"]] = self._convert_resource_to_stream_resource(doc)

    def stream_resource(self, doc: StreamResource):
        # Backwards compatibility: old StreamResource schema is converted to the new one (event-model<1.20.0)
        self._stream_resource_cache[doc["uid"]] = self._convert_resource_to_stream_resource(doc)

    def get_sres_node(self, sres_uid: str, desc_uid: Optional[str] = None) -> tuple[BaseClient, ConsolidatorBase]:
        """Get Stream Resource node from Tiled, if it already exists, or register it from a cached SR document"""

        if sres_uid in self._sres_nodes.keys():
            sres_node = self._sres_nodes[sres_uid]
            consolidator = self._consolidators[sres_uid]

        elif sres_uid in self._stream_resource_cache.keys():
            if not desc_uid:
                raise RuntimeError("Descriptor uid must be specified to initialise a Stream Resource node")

            sres_doc = self._stream_resource_cache.pop(sres_uid)
            desc_node = self._desc_nodes[desc_uid]

            # Initialise a bluesky consolidator for the StreamResource
            consolidator = consolidator_factory(sres_doc, dict(desc_node.metadata))

            sres_node = desc_node["external"].new(
                key=consolidator.data_key,
                structure_family=StructureFamily.array,
                data_sources=[consolidator.get_data_source()],
                metadata={},
                specs=[],
            )

            self._consolidators[sres_uid] = consolidator
            self._sres_nodes[sres_uid] = sres_node
        else:
            raise RuntimeError(f"Stream Resource {sres_uid} is referenced before being declared.")

        return sres_node, consolidator

    def stream_datum(self, doc: StreamDatum):
        # Get the Stream Resource node and the associtaed consolidator
        sres_node, consolidator = self.get_sres_node(doc["stream_resource"], desc_uid=doc["descriptor"])
        consolidator.consume_stream_datum(doc)

        # Update StreamResource node in Tiled
        # NOTE: Assigning data_source.id in the object and passing it in http
        # params is superfluous, but it is currently required by Tiled.
        sres_node.refresh()
        data_source = consolidator.get_data_source()
        data_source.id = sres_node.data_sources()[0].id  # ID of the existing DataSource record
        endpoint = sres_node.uri.replace("/metadata/", "/data_source/", 1)
        handle_error(
            sres_node.context.http_client.put(
                endpoint,
                content=safe_json_dump({"data_source": data_source}),
                params={"data_source": data_source.id},
            )
        ).json()
