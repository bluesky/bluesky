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

Bluesky interacts with hardware through Python objects that are expected to
have a specified interface. This interface is implemented for "simulated"
motors and detectors included in the ``bluesky.examples`` module, which we use
here in documented examples and tests.

To control actual hardware, an additional package is required. The `ophyd
<https://nsls-ii.github.io/ophyd>`_ package implements the bluesky interface
for controlling motors, detectors, etc. via
`EPICS <http://www.aps.anl.gov/epics/>`_. Other control systems could be
integrated with bluesky in the future by presenting this same interface.

Index
-----

.. toctree::
   :caption: User Documentation
   :maxdepth: 1

   plans_intro
   documents
   metadata
   plans
   callbacks
   state-machine
   event_descriptors
   async
   debugging
   cookbook/index

.. toctree::
   :caption: Developer Documentation
   :maxdepth: 1

   hardware
   msg
   run_engine
   api_changes
