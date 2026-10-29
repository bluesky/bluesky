**********************
Integration with Tiled
**********************

`Tiled <https://blueskyproject.io/tiled/>`_ is a data management system that allows for the storage and retrieval of structured data. In the context of Bluesky, it provides a way to store the data and metadata for runs in a structured format that can be easily accessed and queried.


Representation of Bluesky Runs in Tiled
=======================================

The `TiledWriter` callback is designed specifically for converting Bluesky run documents into a format suitable for storage in a Tiled database.

It implicitly distinguishes between "internal" and "external" data. The internal data are associated with the `Event` documents generated during a run; typically this represents scalar measurements from sensors, motor positions, etc, which is stored in a form of a table with columns corresponding to different data keys and each row representing a measurement at a single timestamp.

On the other hand, the external data are written by detectors directly on disk and usually take the form of images or multidimensional arrays. The references to the external files are provided in `StreamResource` (`Resource` in legacy implementations) documents, which register the corresponding array-like `DataSources` in Tiled. `StreamDatum` (or `Datum`) documents are processed via the mechanism of `Consolidators` and determine the correspondence between the indexing within these external arrays and the physically-meaningful sequence of timestamps.

The time dimension (that is, the sequence of measurements) is usually shared between internal and external data. Tiled handles this by writing all data from the same Bluesky stream into a container with a dedicated `"composite"` spec, which tells the Tiled client how the data are aligned. Each stream node's metadata includes the specifications for the related data keys as well as the configuration parameters provided in the `EventDescriptor` document.

Finally, nodes for multiple streams are grouped together and placed into a container for the entire run; its metadata contains the `Start` and `Stop` documents. The Run container created by TiledWriter is designated with the `BlueskyRun` version `3.0` spec to enable its back-compatibility with legacy code via bluesky-tiled-plugins.

An example of the Tiled catalog structure for a Bluesky run might look like this:

.. code-block:: text

    BlueskyRun <Container ("BlueskyRun_v3")>
    │
    ├─ baseline <Container ("composite")>
    │       ├─ internal <Table>   -- written by Tiled
    │       ├─ image_1 <Array>    -- external data from files
    │       │      ...
    │       └─ image_n <Array>
    ├─ primary <Container ("composite")>
    │       ├─ internal <Table>   -- written by Tiled
    │       ├─ image_1 <Array>    -- external data from files
    │       │      ...
    │       └─ image_n <Array>
    └─ third_stream <Container ("composite")>


.. note::

    To be able to use TiledWriter, the Tiled server must be configured with an SQL catalog and an SQL-backed storage database for tabular data.


Callback Architecture
=====================

Structurally, TiledWriter consists of two main parts: `RunNormalizer` and `_RunWriter`.

The former is responsible for converting legacy document schemas to their latest version; this ensures that existing Bluesky code that relies on older versions of the Bluesky Event Model can still function correctly with TiledWriter. For example, while TiledWriter natively works with the modern `StreamResource` and `StreamDatum` documents commonly used in asynchronous plans, the `Resource` and `Datum` documents are automatically converted to their modern counterparts prior to being written to the Tiled catalog. The schema normalization is mostly done by renaming and restructuring certain document fields, but `RunNormalizer` also allows the user to invoke use-case-specific patches for each type of documents and achieve high flexibility.

The simplified flowchart of the `RunNormalizer` logic is shown below. It illustrates how the input documents (top) are processed and emitted as output documents (bottom) after specific transformations or caching operations.

.. mermaid::

    flowchart TD
        %% Input documents
        subgraph Input [ ]
            style Input fill:#ffffff,stroke-width:0
            StartIn["Start"]
            DescriptorIn["Descriptor"]
            ResourceIn["Resource"]
            DatumIn["Datum"]
            EventIn["Event"]
            StopIn["Stop"]
        end

        %% Emitted documents
        subgraph Output [ ]
            style Output fill:#ffffff,stroke-width:0
            StartOut["Start"]
            DescriptorOut["Descriptor"]
            EventOut["Event"]
            StreamResourceOut["StreamResource"]
            StreamDatumOut["StreamDatum"]
            StopOut["Stop"]
        end

        %% Processing steps
        StartIn --> P1["start():<br/>patch → emit"]
        P1 --> StartOut

        DescriptorIn --> P2["descriptor():<br/>patch → rename fields →<br/>track internal/external keys → emit"]
        P2 --> DescriptorOut

        ResourceIn --> P3["resource():<br/>patch → convert to StreamResource → cache"]
        P3 --> SResCache[(SRes Cache)]

        DatumIn --> P4["datum():<br/>patch → cache"]
        P4 --> DatumCache[(Datum Cache)]

        EventIn --> P5["event():<br/>patch → split internal/external keys → emit"]
        P5 -->|internal data| EventOut
        P5 -->|external data| P6["convert_datum_to_stream_datum()<br/>move datum_kwargs to parameters on SRes"]
        P6 --> StreamDatumOut
        P6 --> |only before first SDatum| StreamResourceOut

        StopIn --> P7["stop():<br/>patch → flush cached StreamDatum"]
        P7 --> StopOut
        P7 --> StreamDatumOut
        P7 --> |if not emitted<br/>already| StreamResourceOut

        %% Extra connections
        SResCache --> P6
        DatumCache --> P6

        %% Styling
        classDef doc fill:#e0f7fa,stroke:#00796b,stroke-width:1px;
        classDef emit fill:#f1f8e9,stroke:#33691e,stroke-width:1px;
        classDef proc fill:#fff3e0,stroke:#e65100,stroke-width:1px;

        class StartIn,DescriptorIn,ResourceIn,DatumIn,EventIn,StopIn doc;
        class StartOut,DescriptorOut,EventOut,StreamResourceOut,StreamDatumOut,StopOut emit;
        class P1,P2,P3,P4,P5,P6,P7 proc;


The second component, `_RunWriter`, is the callback that directly communicates with the Tiled server. It uses the `RunRouter` to manage the routing of documents from multiple runs into separate instances of the internal `_RunWriter` callback, ensuring that each Bluesky run is handled separately.

Furthermore, TiledWriter implements a backup mechanism that allows to save the documents to a local file system in case the Tiled server is not available or any other error occurs during the writing process. This ensures that no data is lost and can be retried later.


Usage
========

A minimal simulated example of using TiledWriter in a Bluesky plan is shown below:

.. code-block:: python

    from bluesky import RunEngine
    import bluesky.plans as bp
    from bluesky.callbacks.tiled_writer import TiledWriter
    from tiled.server import SimpleTiledServer
    from tiled.client import from_uri
    from ophyd.sim import det
    from ophyd.sim import hw

    # Initialize the Tiled server and client
    save_path = "/path/to/save/detector_data"
    tiled_server = SimpleTiledServer(readable_storage=[save_path])
    tiled_client = from_uri(tiled_server.uri)

    # Initialize the RunEngine and subscribe TiledWriter
    RE = RunEngine()
    tw = TiledWriter(tiled_client)
    RE.subscribe(tw)

    # Run an experiment collecting internal data
    uid, = RE(bp.count([det], 3))
    data = tiled_client[uid]['primary/det'].read()

    # Run an experiment collecting external data
    uid, = RE(bp.count([hw(save_path=save_path).img], 2))
    data = tiled_client[uid]['primary/img'].read()
