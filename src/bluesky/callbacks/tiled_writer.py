import copy
from typing import Dict, Optional, Tuple, Union

import pandas as pd
from event_model import RunRouter
from event_model.documents import EventDescriptor, StreamDatum, StreamResource
from pydantic.utils import deep_update
from tiled.client import from_profile, from_uri
from tiled.client.base import BaseClient
from tiled.client.container import Container
from tiled.client.utils import handle_error
from tiled.structures.core import Spec
from tiled.structures.table import TableStructure
from tiled.utils import safe_json_dump

from ..consolidators import CONSOLIDATOR_REGISTRY, ConsolidatorBase, DataSource, StructureFamily
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
    "Write the document from one Bluesky Run into Tiled."

    def __init__(self, client: BaseClient):
        self.client = client
        self.root_node: Union[None, Container] = None
        self._desc_nodes: Dict[str, Container] = {}  # references to descriptor containers by their uid's
        self._sres_nodes: Dict[str, BaseClient] = {}
        self._sres_cache: Dict[str, StreamResource] = {}
        self._handlers: Dict[str, ConsolidatorBase] = {}

    def _ensure_sres_backcompat(self, sres: StreamResource) -> StreamResource:
        """Kept for back-compatibility with old StreamResource schema from event_model<1.20.0

        Will make changes to and return a shallow copy of StreamRsource dictionary adhering to the new structure.
        """

        sres = copy.copy(sres)
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
        if self.root_node is None:
            raise RuntimeError("RunWriter is properly initialized: no Start document has been recorded.")

        doc = dict(doc)
        metadata = {"stop": doc, **dict(self.root_node.metadata)}
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
            desc_node.create_container(key="external")
            desc_node.create_container(key="internal")
        else:
            # Get existing descriptor node (with fixed and variable metadata saved before)
            desc_node = self.root_node[desc_name]

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

    def get_sres_node(self, sres_uid: str, desc_uid: Optional[str] = None) -> Tuple[BaseClient, ConsolidatorBase]:
        """Get Stream Resource node from Tiled, if it already exists, or register it from a cached SR document"""

        if sres_uid in self._sres_nodes.keys():
            sres_node = self._sres_nodes[sres_uid]
            handler = self._handlers[sres_uid]

        elif sres_uid in self._sres_cache.keys():
            if not desc_uid:
                raise RuntimeError("Descriptor uid must be specified to initialise a Stream Resource node")

            sres_doc = self._sres_cache.pop(sres_uid)
            desc_node = self._desc_nodes[desc_uid]

            # Initialise a bluesky handler (consolidator) for the StreamResource
            handler_class = CONSOLIDATOR_REGISTRY[sres_doc["mimetype"]]
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
        # Get the Stream Resource node and the associtaed handler (consolidator)
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
