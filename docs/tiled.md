# Tiled Integration

[Tiled][] can be used to store and retrieve data produces by Bluesky. It aims
to reduce the "background noise" of data management and let users keep their
attention on the science.

## Start a Basic Tiled Server

In this example, we will use embedded databases and simple security
designed for a single user. (Read on for production-scale databases
and multi-user configuration.)

Install tiled, with the dependencies needed to run a server.

```sh
pip install "tiled[server]"
```

We will start Tiled with:
- a "catalog" database of metadata and pointers (filepaths, etc.) to large data
- an [OLAP][]

```sh
tiled serve catalog --init catalog.db -w duckdb://./storage.db
```

This will print a URL with a random secret key, which will be use below. (For
testing purposes, it can be convenient to set key manually to something easy to
type like `--api-key secret`.)

If you are going to connect to this from other hosts, set `--host 0.0.0.0`. By
default, for security reasons, the server only accepts connections from
`localhost`.

## Write From Bluesky to Tiled

Install tiled, with the dependencies needed to run a client. (This can be
in the same Python environment as the server, or a different one.)

```sh
pip install "tiled[client]"
```

It is also recommended (but not strictly necessary) to install a small
libraries with conveniences that provides a nicer user experience for
Tiled with Bluesky:

```sh
pip install bluesky-tiled-plugins
```

Set the environment variable ``TILED_API_KEY`` to match the api_key that was
set above when the server started.

```sh
export TILED_API_KEY=...
```

```python
from bluesky import RunEngine
from bluesky.callbacks.tiled_writer import TiledWriter
from tiled.client import from_uri

# Connect to Tiled.
client = from_uri("http://localhost:8000")  # will automatically use TILED_API_KEY

RE = RunEngine()
tw = TiledWriter(client, batch_size=1)
RE.subscribe(tw)
```

When the TiledWriter receives Bluesky documents from the RunEngine, it makes
API calls to Tiled. These include:

- uploading metadata from Run Start, Event Descriptor, and Run Stop documents
- uploading tabular data from the Event documents
- registering externally-written array (e.g. image) data references in
  Resource and Datum or StreamResource and StreamDatum documents
 
## Access the Data

```python
r = client.values().last()
r["streams"]
r["streams/primary"].read()
```

## Backward-compatibility with "Databroker"

**The [Databroker][] project is superseded by Tiled. We recommend that new users
ignore it; there is no need for them to install it.**

For backward-compatibility with existing user code, Databroker will continue to
be maintained, but it has been reimplemented as a compatibility layer, a
wrapper around Tiled.

```sh
pip install --pre "databroker[client]"
```

```python
from databroker import Broker
from tiled.client import from_uri

# Connect to Tiled.
client = from_uri("http://localhost:8000")  # will automatically use TILED_API_KEY

# Construct the backward-compatibility layer.
db = Broker(client)
```

[OLAP]: https://en.wikipedia.org/wiki/Online_analytical_processing
[Tiled]: https://blueskyproject.io/tiled
