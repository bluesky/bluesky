import numpy as np
import pandas as pd
from event_model import DocumentRouter, RunRouter
from tiled.client import from_profile, from_uri
from tiled.structures.array import ArrayStructure, BuiltinDtype
from tiled.structures.core import Spec, StructureFamily
from tiled.structures.data_source import Asset, DataSource, Management

MIMETYPE_LOOKUP = {"hdf5": "application/x-hdf5"}


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
        self._descriptor_nodes = {}  # references to descriptor containers by uid's
        self._SR_nodes = {}
        self._SR_cache = {}

    def start(self, doc):
        self.node = self.client.create_container(
            key=doc["uid"],
            metadata={"start": doc},
            specs=[Spec("BlueskyRun", version="1.0")],
        )

    def stop(self, doc):
        metadata = dict(self.node.metadata) | {"stop": doc}
        self.node.update_metadata(metadata=metadata)

    def descriptor(self, doc):
        self._descriptor_nodes[doc["uid"]] = self.node.create_container(
            key=doc["name"], metadata=doc
        )

    def event(self, doc):
        parent_node = self._descriptor_nodes[doc["descriptor"]]
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

    def stream_resource(self, doc):
        # Cache the StreamResource
        self._SR_cache[doc["uid"]] = doc

    def stream_datum(self, doc):
        descriptor_node = self._descriptor_nodes[doc["descriptor"]]
        arr_shape = dict(descriptor_node.metadata)["data_keys"]["image"]["shape"]
        num_rows = (
            doc["indices"]["stop"] - doc["indices"]["start"]
        )  # Number of rows added by new StreamDatum

        # Get the Stream Resource node if it already exists or register if from a cached SR document
        print(f"Processing Stream Datum \n {doc}")

        try:
            SR_node = self._SR_nodes[doc["stream_resource"]]

        except KeyError:
            # Register a new (empty) Stream Resource
            SR_doc = self._SR_cache.pop(doc["stream_resource"])

            # POST /api/v1/register/{path}
            assets = [
                Asset(
                    data_uri="file://localhost"
                    + SR_doc["root"]
                    + SR_doc["resource_path"],
                    is_directory=False,
                    parameter="data_uri",
                )
            ]

            SR_node = descriptor_node.new(
                structure_family=StructureFamily.array,
                data_sources=[
                    DataSource(
                        assets=assets,
                        mimetype=MIMETYPE_LOOKUP[SR_doc["spec"]],
                        structure_family=StructureFamily.array,
                        structure=ArrayStructure(
                            data_type=BuiltinDtype.from_numpy_dtype(np.dtype("double")),
                            shape=[0] + arr_shape,
                            chunks=[[0]] + [[d] for d in arr_shape],
                        ),
                        # structure=ArrayStructure.from_array(np.ones(arr_shape)),
                        parameters={"path": ["test"]},
                        management=Management.external,
                    )
                ],
                metadata={},
                specs=[],
            )

            self._SR_nodes[SR_doc["uid"]] = SR_node

        # Append StreamDatum to an existing StereamResource (by overwriting it with changed shape)
        url = SR_node.uri.replace("/metadata/", "/data_source/")
        SR_node.refresh()
        ds_dict = SR_node.data_sources()[0]
        # SR_node.include_data_sources() ?
        ds_dict["structure"]["shape"][0] += num_rows
        ds_dict["structure"]["chunks"][0] = [min(ds_dict["structure"]["shape"][0], 100)]
        SR_node.context.http_client.put(
            url, json={"data_source": ds_dict}, params={"data_source": 1}
        )

        # breakpoint()
