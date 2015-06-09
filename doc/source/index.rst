Bluesky Data Collection Interface
=================================

Bluesky is Python package for interactive data collection. There are three
components:

* *Messages,* simple, single-step instructions,
* a *Run Engine,* which processes the messages and coordinates collection, 
* and *Documents*, Python dictionaries containing data and metadata, organized
  in a
  `specified but flexible <http://nsls-ii.github.io/arch/metadatastore-format.html>`__
  way.


Key Features
------------
* Running can be **paused, aborted, and resumed** with user-defined
  checkpoints.
* In addition to the built-in scans, you can specify **custom scans and 
  "macros"** in a simple, procedural way using basic Python sytnax (e.g.,
  for-loops) and a small set of commands. There are many documented examples.
* You can register **custom commands** -- say, to control your robot -- and
  immediately integrate them with existing ones.
* You can write your own high-level **"motors" that control many PVs**. Examples
  include coordinated motion and controllers like XPD's "gas switcher" that don't
  really map onto a continuous single-axis motor.
* While IPython can be used to set up a convenient interactive environment,
  **IPython is not required**. Data collection can be run from plain Python
  scripts. This will provide easy integration with beamline-specific control
  GUIs.
* Optional **automatic data export** at the end of each run. This paves the way
  for prompt HPC.
* Highly customizable **live plotting** is possible. Working examples are
  included.
* **Adaptive scans** can sample fast-changing regions more. There are
  ready-to-use functions and examples for writing your own.
* Automated **tests** ensure stability of the project going forward.
* With optional **data validation** at the end of a run, you can be sure that
  the data will be available from Data Broker.
* This documentation!

Relationship to EPICS, pyepics, ophyd
-------------------------------------

Bluesky is a high-level interface: it communicates with hardware through
Python objects that are expected to have methods like ``read`` and ``set``.
These objects might be implemented with ophyd, pyepics, cothread, or any other
Python package. Bluesky does not import or require on any of them.

Relationship to DataBroker, metadatastore, filestore
----------------------------------------------------

Bluesky includes a module called ``standard_config``. At NSLS-II, it will used
every beamline, but it is technically optional.
Among other things, it configures metadatastore to listen for and store data
collected and emitted by bluesky. (It is configured in such a way that, if data
is for any reason not saved, the run will fail and the user will be immediately
notified.) 
Results will be available from the DataBroker as they are collected.

Because integration with metadatastore is not strictly assumed, other
storage or processing pipelines can be integrated in the future without making
changes to bluesky.

Integration with filestore is handled by individual detector interfaces, such as
``AreaDetector``.

Contents
--------

.. toctree::
   :maxdepth: 1

   workflow
   scans
   callbacks
   state-machine
   custom-scans
   msg
   object_api
   nsls2_checklist
   legacy_api
