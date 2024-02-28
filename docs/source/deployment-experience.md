# Deployment Experience

Several facilities deploy Bluesky across many instruments (up to 25 or so).
In this document we summarize lessons learned from that experience. At the
moment, only NSLS-II (Bluesky's facility of origin) is represented, but we
hope to add comments from other facilities in the future.

## NSLS-II

### Python environment management

We deploy conda environments, verioned by operating cycle, in a central (NFS)
software mount. The instrument staff and users have only read-only access to
these environments. The environments contains the union of all requirements
across all instruments: they include core scientific Python libraries (e.g.
scipy), Bluesky libraries, and other more instrument-specific libraries, both
internally and externally developed.

The environments read-only in order to avoid accidentally breaking changes.
To support short-term patching and minor variations betweeen instruments,
we add an instrument-specific directory to the `PYTHONPATH` at each
instrument respectively, granting instrument staff read/write permission
on that directory. This enables them to install extra packages and upgrade
or downgrade existing ones as needed, without the risk of unintended
consequences for other instruments.

The conda environments are build via a
[CI service](https://github.com/nsls2-conda-envs),
hosted on [Zenodo](https://zenodo.org/record/6555136), and
deployed via conda-pack. Using conda-pack speeds up installation and
improves reliability.

We recommend:

* Create centrally-managed read only conda enviornments.
* Use `PYTHONPATH` to "overlay" custom additions or version changes.

### Database Management

NSLS-II and some other facilities we know of currently use a separate
single-node MongoDB instance per instrument, but _we do not recommend this_. We
are moving toward a central MongoDB cluster shared by all instrumenents. This
reduces maintainence burden (e.g. less work to update MongoDB version) and
improves reliability (avoid single-node deployments which are single points
of failure).

Each instrument will have its own MongoDB database on the cluster.

We recommend:

* Use a central cluster, shared by all instruments.
* Use authentication between the MongoDB client and server.

### Message Bus

We use a message to "publish" Bluesky documents from the RunEngine so
that they are available for consumers in other processes or on
other machines on the network.

To start, we made good use of 0MQ as a simple message bus. However, we
are moving to use Kafka, a much more fully-featured and robust solution.
The Bluesky document publishers and consumers and fairly agnostic to
the transport technology used, so switching is easy.

Finally, we current subscribe the database-insertion code (`db.insert`)
_directly_ to the RunEngine. We have begun to move that step downstream
of a message bus, i.e. switching from:

```
RunEngine --> MongoDB
```

to

```
RunEngine --> Kafka --> MongoDB
```

We find that this improves performance because Kafka can accept data buffer it
faster than MongoDB can insert it. We have yet to commit to this architecture
at scale.

We recommend:
* Start with 0MQ to play because it is easy to set up.
* Move to Kafka early, once you are relying on these systems.

### Managing facility-specific code

Where possible, we encourage facilities to push embellishments to Bluesky
libraires so the benefits and maintenace can be shared. But each facility also
accumulates utilies, bluesky plans, ophyd Devices, and other code that is
specific to them. Some of this work may eventually be pushed upstream but is
deployed locally at first to meet local deadlines. Other work may not be
useful at other facilties and therefore not appropriate to contribute upstream.

At NSLS-II, we have a catch-all library [nslsii](https://github.com/NSLS-II/nslsii).
APS and LCLS have similar packages.

We recommend:
* Make a facility-specific package.
* Make it public.

### Managing instrument configuration

Each instrument needs Python code to:

* Define instrument-specific ophyd Device classes
* Instantiate ophyd Devices
* Define instrument-specific plans
* Subscribe instrument-specific convenience
* Customize the user experience

Setting up an environment for collection is too complex for a declarative configuration
file (like a YAML file) and requires executing Python code.

The initial target of Bluesky development was a command-line tool. We chose the
IPython application, we used the "profiles" feature of IPython to run
instrument-specific code and populate the user namespace at IPython start-up
with variables representing ophyd Device, bluesky plans, and so on.
We currently have at least one GitHub repository per instrument. These are public
under separate GitHub organization. The repositories are all named `profile_collection`.

Over time, these large IPython profiles have become unwieldy and difficult to
maintain. We recommend moving at least class and plan definitions to a normal
Python package that you can import from. This makes the code easier to
organize, maintain, reuse, and test.

### Deployment tools

For deploying services (MongoDB, Kafka, Tiled) and configuration, we use
Ansible and have embraced "configuration as code" as much as we can.

We recommend:
* Automate all deployment.
* Reduce the variation between instruments as much as is technically feasible.
