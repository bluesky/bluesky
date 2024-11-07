import copy
from datetime import datetime
from typing import Any, Optional, Union

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
        self._docs_cache: dict[str, Union[Datum, Resource, StreamResource]] = {}
        self._handlers: dict[str, ConsolidatorBase] = {}
        self.data_keys_int: dict[str, dict[str, Any]] = {}
        self.data_keys_ext: dict[str, dict[str, Any]] = {}

    def _ensure_resource_backcompat(self, doc: Union[Resource, StreamResource]) -> Union[Resource, StreamResource]:
        """Kept for back-compatibility with old StreamResource schema from event_model<1.20.0

        Will make changes to and return a shallow copy of StreamRsource dictionary adhering to the new structure.
        """

        doc = copy.copy(doc)
        if ("mimetype" not in doc.keys()) and ("spec" not in doc.keys()):
            raise RuntimeError("StreamResource document is missing a mimetype or spec")
        else:
            doc["mimetype"] = doc.get("mimetype") or MIMETYPE_LOOKUP[doc.get("spec")]
        if "parameters" not in doc.keys():
            doc["parameters"] = doc.pop("resource_kwargs", {})
        if "uri" not in doc.keys():
            file_path = doc.pop("root").strip("/") + "/" + doc.pop("resource_path").strip("/")
            doc["uri"] = "file://localhost/" + file_path

        return doc

    def start(self, doc: RunStart):
        self.root_node = self.client.create_container(
            key=doc["uid"],
            metadata={"start": dict(doc)},
            specs=[Spec("BlueskyRun", version="1.0")],
        )

    def stop(self, doc: RunStop):
        if self.root_node is None:
            raise RuntimeError("RunWriter is properly initialized: no Start document has been recorded.")

        stream_names = list(self.root_node.keys())
        summary = build_summary(self.root_node.metadata["start"], dict(doc), stream_names)
        metadata = {"stop": dict(doc), "summary": summary, **dict(self.root_node.metadata)}
        self.root_node.update_metadata(metadata=metadata)

    def descriptor(self, doc: EventDescriptor):
        if self.root_node is None:
            raise RuntimeError("RunWriter is properly initialized: no Start document has been recorded.")

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
        else:
            # Get existing descriptor node (with fixed and variable metadata saved before)
            desc_node = self.root_node[desc_name]

        # Update (add new values to) variable fields of the metadata
        metadata = deep_update(dict(desc_node.metadata), var_fields)
        desc_node.update_metadata(metadata)

        # Keep specifications for external and internal data_keys for faster access
        self.data_keys_int.update({k: v for k, v in metadata["data_keys"].items() if "external" not in v.keys()})
        self.data_keys_ext.update({k: v for k, v in metadata["data_keys"].items() if "external" in v.keys()})

        # Write the configuration data: loop over all detectors
        if conf_dict[uid]:
            if "config" not in desc_node.keys():
                conf_node = desc_node.create_container(key="config")
            else:
                conf_node = desc_node["config"]
        for det_name, det_dict in conf_dict[uid].items():
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
        parent_node = self._desc_nodes[doc["descriptor"]]

        # Process _internal_ data -- those keys without 'external' flag or those that have been filled
        data_keys_spec = {k: v for k, v in self.data_keys_int.items() if doc["filled"].get(k, True)}
        data_keys_spec.update({k: v for k, v in self.data_keys_ext.items() if doc["filled"].get(k, False)})
        df_dict = {"seq_num": doc["seq_num"]}
        df_dict.update({k: v for k, v in doc["data"].items() if k in data_keys_spec.keys()})
        df_dict.update({f"ts_{k}": v for k, v in doc["timestamps"].items()})  # Keep all timestamps
        df = pd.DataFrame(df_dict, index=[0], columns=df_dict.keys())
        if "internal" in parent_node.keys():
            parent_node["internal"].append_partition(df, 0)
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
                key="internal",
                metadata=data_keys_spec,
            )
            parent_node["internal"].write_partition(df, 0)

        # Process _external_ data: Loop over all referenced Datums
        for data_key in self.data_keys_ext.keys():
            if doc["filled"].get(data_key, False):
                continue

            if datum_id := doc["data"].get(data_key):
                if datum_id in self._docs_cache.keys():
                    # Convert the Datum document to the StreamDatum format
                    datum_doc = self._docs_cache.pop(datum_id)
                    datum_doc["uid"] = datum_doc.pop("datum_id")
                    datum_doc["stream_resource"] = datum_doc.pop("resource")
                    datum_doc["descriptor"] = doc["descriptor"]  # From Event document
                    datum_doc["indices"] = {"start": doc["seq_num"] - 1, "stop": doc["seq_num"]}
                    datum_doc["seq_nums"] = {"start": doc["seq_num"], "stop": doc["seq_num"] + 1}

                    # Update the Resource document (add data_key as in StreamResource)
                    if datum_doc["stream_resource"] in self._docs_cache.keys():
                        self._docs_cache[datum_doc["stream_resource"]]["data_key"] = data_key

                    self.stream_datum(datum_doc)
                else:
                    raise RuntimeError(f"Datum {datum_id} is referenced before being declared.")

    def event_page(self, doc: EventPage):
        for _doc in unpack_event_page(doc):
            self.event(_doc)

    def datum(self, doc: Datum):
        self._docs_cache[doc["datum_id"]] = copy.copy(doc)

    def datum_page(self, doc: DatumPage):
        for _doc in unpack_datum_page(doc):
            self.datum(_doc)

    def resource(self, doc: Resource):
        self._docs_cache[doc["uid"]] = self._ensure_resource_backcompat(doc)

    def stream_resource(self, doc: StreamResource):
        self._docs_cache[doc["uid"]] = self._ensure_resource_backcompat(doc)

    def get_sres_node(self, sres_uid: str, desc_uid: Optional[str] = None) -> tuple[BaseClient, ConsolidatorBase]:
        """Get Stream Resource node from Tiled, if it already exists, or register it from a cached SR document"""

        if sres_uid in self._sres_nodes.keys():
            sres_node = self._sres_nodes[sres_uid]
            handler = self._handlers[sres_uid]

        elif sres_uid in self._docs_cache.keys():
            if not desc_uid:
                raise RuntimeError("Descriptor uid must be specified to initialise a Stream Resource node")

            sres_doc = self._docs_cache.pop(sres_uid)
            desc_node = self._desc_nodes[desc_uid]

            # Initialise a bluesky handler (consolidator) for the StreamResource
            handler = consolidator_factory(sres_doc, dict(desc_node.metadata))

            if "external" not in desc_node.keys():
                desc_node.create_container(key="external")
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
        # Get the Stream Resource node and the associtaed handler (consolidator)
        sres_node, handler = self.get_sres_node(doc["stream_resource"], desc_uid=doc["descriptor"])
        handler.consume_stream_datum(doc)

        # Update StreamResource node in Tiled
        # NOTE: Assigning data_source.id in the object and passing it in http
        # params is superfluous, but it is currently required by Tiled.
        sres_node.refresh()
        data_source = handler.get_data_source()
        data_source.id = sres_node.data_sources()[0].id  # ID of the existing DataSource record
        endpoint = sres_node.uri.replace("/metadata/", "/data_source/", 1)
        handle_error(
            sres_node.context.http_client.put(
                endpoint,
                content=safe_json_dump({"data_source": data_source}),
                params={"data_source": data_source.id},
            )
        ).json()
