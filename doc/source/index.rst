Bluesky Data Collection Interface
=================================

Bluesky is Python package for interactive data collection. There are three
components:

* *Messages,* simple, single-step instructions,
* a *Run Engine,* which processes the messages and coordinates collection, 
* and *Documents*, Python dictionaries containing data and metadata, organized
  in a
  `specified but flexible <http://nsls-ii.github.io/architecture-overview.html>`__
  way.

Basic Operation
---------------

.. code-block:: python

    from bluesky import RunEngine, Msg
    RE = RunEngine()
    plan = [Msg('set', motor), Msg('read', detector), ...]
    RE(plan, f)  # where f is any function that does something with the data

Key Features
------------

Supervised Execution of an Experimental Plan
++++++++++++++++++++++++++++++++++++++++++++

* The logic of an experiment is given an **iterable expression** (e.g., a
  list). The RunEngine manages formatting the data, handling interruptions,
  and other details general to most experiments.
* Running can be cleanly **paused and later aborted or resumed** at
  user-defined checkpoints.
* Runs can automatically supend and resume in response to external conditions.
* In addition to the built-in plans, you can specify **custom plans** in a
  simple, procedural way using basic Python sytnax (e.g., for-loops) and a
  small set of commands. There are many documented examples.
* You can register **custom commands**---say, to control your robot---and
  immediately integrate them with existing ones.

Powerful Features with Working Examples
+++++++++++++++++++++++++++++++++++++++

* Highly customizable **live plotting** and other real-time processing
  pipelines are possible. Useful working examples are included.
* **Adaptive plans** can sample fast-changing regions more. There are
  ready-to-use functions and examples for writing your own.
* Optional **automatic data export** at the end of each run.

Robustness
++++++++++

* Every Document undergoes **validation** to ensure immediate feedback in
  the event of trouble. Additional, customized data validation is possible.
* Strong code coverage by automated **tests** ensures stability.

.. toctree::
   :maxdepth: 1
   :hidden:

   simple_api
   plans 
   callbacks
   state-machine
   metadata 
   custom-plans
