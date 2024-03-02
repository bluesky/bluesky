import pandas as pd
from event_model import DocumentRouter, RunRouter
from tiled.client import from_profile, from_uri
from tiled.structures.core import Spec


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


class _RunWriter(DocumentRouter):
    "Write the document from one Bluesky Run into Tiled."

    def __init__(self, client):
        self.client = client
        self.node = None
        self.descriptors = {}

    def start(self, doc):
        self.node = self.client.create_container(
            key=doc["uid"],
            metadata={"start": doc},
            specs=[Spec("BlueskyRun", version="1.0")],
        )

    def stop(self, doc):
        metadata = dict(self.node.metadata)
        metadata["stop"] = doc
        self.node.update_metadata(metadata=metadata)

    def descriptor(self, doc):
        self.descriptors[doc["uid"]] = self.node.create_container(
            key=doc["name"], metadata={"descriptors": doc}
        )

    def event(self, doc):
        parent_node = self.descriptors[doc["descriptor"]]
        parent_node.write_dataframe(
            pd.DataFrame({column: [value] for column, value in doc["data"].items()}),
            key="data",
        )
        parent_node.write_dataframe(
            pd.DataFrame(
                {column: [value] for column, value in doc["timestamps"].items()}
            ),
            key="timestamps",
        )
