Bluesky Data Collection Framework
=================================

Bluesky is a library for experiment control and collection of scientific data
and metadata. It emphasizes the following virtues:

* **Live, Streaming Data:** Available for inline visualization and processing.
* **Rich Metadata:** Captured and organized to facilitate reproducibility and
  searchability.
* **Experiment Generality:** Seamlessly reuse a procedure on completely
  different hardware.
* **Interruption Recovery:** Experiments are "rewindable," recovering cleanly
  from interruptions.
* **Automated Suspend/Resume:** Experiments can be run unattended,
  automatically suspending and resuming if needed.
* **Pluggable I/O:** Export data (live) into any desired format or database.
* **Customizability:** Integrate custom experimental procedures and commands,
  and get the I/O and interruption features for free.
* **Integration with Scientific Python:** Interface naturally with numpy and
  Python scientific stack.

How to Use This Documentation
-----------------------------

Start with the :doc:`tutorial`. It's a good place to start for everyone, and it
gives a good overview of the project in a narrative style. Read as far as you
need to solve your problem, and come back again if your needs change. Each
section of the tutorial adds a piece of complexity in exchange for deeper
customization.

The remaining sections document bluesky's behavior in a less narrative style,
providing clear API documentation intermixed with some examples and explanation
of design and intent.

Index
-----

.. toctree::
   :caption: User Documentation
   :maxdepth: 1

   tutorial
   plans
   documents
   metadata
   callbacks
   state-machine
   simulation
   progress-bar
   event_descriptors
   async
   debugging
   run_engine_api
   magics
   from-pyepics-to-bluesky
   comparison-with-spec
   appendix

.. toctree::
   :caption: Developer Documentation
   :maxdepth: 1

   hardware
   msg
   run_engine
   api_changes

.. toctree::
   :hidden:
   :caption: Data Collection

   bluesky <https://nsls-ii.github.io/bluesky>
   ophyd <https://nsls-ii.github.io/ophyd>

.. toctree::
   :hidden:
   :caption: Data Access and Management

   databroker <https://nsls-ii.github.io/databroker>
   amostra <https://nsls-ii.github.io/amostra>
   datamuxer <https://nsls-ii.github.io/datamuxer>
   suitcase <https://nsls-ii.github.io/suitcase>

.. toctree::
   :hidden:
   :caption: GitHub Links

   NSLS-II Repositories <https://github.com/NSLS-II/>
   Bug Reports <https://github.com/NSLS-II/Bug-Reports/issues>

