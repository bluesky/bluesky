Bluesky Data Collection Framework
=================================

Bluesky is light-weight Python package for interactive data collection. There
are three components:

* A *Plan* --- an experimental procedures expressed as a iterable of
  simple, granular instructions dubbed *Messages*
* a *Run Engine,* which processes the messages and coordinates collection, 
* and *Documents*, Python dictionaries containing data and metadata, organized
  in a
  `specified but flexible <http://nsls-ii.github.io/architecture-overview.html>`__
  way.

Basic Operation
---------------

1. Make a "RunEngine", a kind of interpreter for plans. (It is initialized with
   a dictionary of metadata --- here empty.)

.. code-block:: python

    from bluesky import RunEngine
    RE = RunEngine({})

2. Define a plan from scratch (as we do here) or use one of the numerous built-in
plans.

.. code-block:: python

    from bluesky.plans import open_run, trigger_and_read, close_run
    from bluesky.examples import det  # simulated hardware

    def plan()
        yield from open_run()
        yield from trigger_and_read([det])
        yield from close_run()

3. Execute the plan, directing the generated "documents" to the print function
or any other function that does something with the data (like save it).

.. code-block:: python

    RE(plan(), print)

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

   plans 
   callbacks
   state-machine
   metadata 
   simple_api
   api_changes
