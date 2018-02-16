.. currentmodule:: bluesky.plans

Documents
=========

A primary design goal of bluesky is to enable better research by recording
rich metadata alongside measured data for use in later analysis. Documents are
how we do this.

A *document* is our term for a Python dictionary with a schema --- that is,
organized in a
`formally specified <https://github.com/NSLS-II/event-model>`_ way --- created
by the RunEngine during plan execution.  All of the metadata and data generated
by executing the plan is organized into documents.

A :doc:`later section <callbacks>` describes how outside functions can
"subscribe" to a stream of these documents, visualizing, processing, or saving
them. This section provides an outline of documents themselves, aiming to give
a sense of the structure and familiarity with useful components.

.. _run_overview:

Overview of a "Run"
-------------------

Each document belongs to a *run* --- loosely speaking, a dataset. Executing any
of the :ref:`built-in pre-assembled plans <preassembled_plans>`, like
:func:`scan` and :func:`count`, creates one run.

.. note::

    Fundamentally, the scope of a run is intentionally vague and flexible. One
    plan might generate many runs or one long run. It just depends on how you
    want to organize your data, both at collection time and analysis time.

    The tutorial's :ref:`tutorial_capture_data` section explores this.

The documents in each run are:

- A **Run Start document**, containg all of the metadata known at the start of
  the run. Highlights:

    - time --- the start time
    - plan_name --- e.g., ``'scan'`` or ``'count'``
    - uid --- unique ID that identifies this run
    - scan_id --- human-friendly integer scan ID (not necessarily unique)
    - any other :doc:`metadata captured at execution time <metadata>` from the
      plan or the user

- **Event documents**, containing the actual measurements. These are your data.

    - time --- a timestamp for this group of readings
    - seq_num --- sequence number, counting up from 1
    - data --- a dictionary of readings like
      ``{'temperature': 5.0, 'position': 3.0}``
    - timestamps --- a dictionary of individual timestamps for each reading,
      from the hardware

- **Event Descriptor documents** provide a schema for the data in the Event
  documents. They list all of the keys in the Event's data and give useful
  information about them, such as units and precision. They also contain
  information about the configuration of the hardware.

- A **Run Stop document**, containing metadata known only at the end of the
  run. Highlights:

    - time --- the time when the run was completed
    - exit_status --- "success", "abort", or "fail"

Every document has a ``time`` (its creation time) and a separate ``uid`` to
identify it. The Event documents also have a ``descriptor`` field linking them
to the Event Descriptor with their metadata. And the Event Descriptor and
Run Stop documents have a ``run_start`` field linking them to their Run
Start. Thus, all the documents in a run are linked back to the Run Start.

Documents in Detail
-------------------

Run Start
+++++++++

Again, a 'start' document marks the beginning of the run. It comprises
everything we know before we start taking data, including all metadata provided
by the user and the plan. (More on this in the :doc:`next section <metadata>`.)

All built-in plans provide some useful metadata like the names of the
detector(s) and motor(s) used. (User-definied plans may also do this; see
:ref:`this section <tutorial_plan_metadata>` of the tutorial.)

The command:

.. code-block:: python

    from bluesky.plans import scan
    from ophyd.sim import det, motor  # simulated detector, motor

    # Scan 'motor' from -3 to 3 in 10 steps, taking readings from 'det'.
    RE(scan([det], motor, -3, 3, 16), purpose='calibration',
       sample='kryptonite')

generates a 'start' document like this:

.. code-block:: python

    # 'start' document
    {'purpose': 'calibration',
     'sample': 'kryptonite',
     'detectors': ['det'],
     'motors': ['motor'],
     'plan_name': 'scan',
     'plan_type': 'generator',
     'plan_args': {'detectors': '[det]',
                   'motor': 'Mover(...)',
                   'num': '16',
                   'start': '-3',
                   'stop': '3'},
     'scan_id': 282,
     'time': 1442521005.6099606,
     'uid': '<randomly-generated unique ID>',
    }

.. note::

    Time is given in UNIX time (seconds since 1970). Software for looking at
    the data would, of course, translate that into a more human-readable form.

Event
+++++

An 'event' records one or more measurements with an associated time.

.. code-block:: python

    # 'event' document
    {'data':
        {'temperature': 5.0,
          'x_setpoint': 3.0,
          'x_readback': 3.05},
     'timestamps':
        {'temperature': 1442521007.9258342,
         'x_setpoint': 1442521007.5029348,
         'x_readback': 1442521007.5029348},
     'time': 1442521007.3438923,
     'seq_num': 1
     'uid': '<randomly-generated unique ID>',
     'descriptor': '<reference to a descriptor document>'}

From a data analysis perspective, these readings were simultaneous, but in
actuality the occurred at separate times.  The separate times of the individual
readings are not thrown away (they are recorded in 'timestamps') but the
overall event 'time' is often more useful.

Run Stop
++++++++

A 'stop' document marks the end of the run. It contains metadata that is not
known until the run completes.

The most commonly useful fields here are 'time' and 'exit_status'.

.. code-block:: python

    # 'stop' document
    {'exit_status': 'success',  # or 'fail' or 'abort'
     'reason': '',  # The RunEngine can provide reason for failure here.
     'time': 1442521012.1021606,
     'uid': '<randomly-generated unique ID>',
     'start': '<reference to the start document>',
     'num_events': {'primary': 16}
    }

Event Descriptor
++++++++++++++++

As stated above, a 'descriptor' document provides a schema for the data in the
Event documents. It provides useful information about each key in the data and
about the configuration of the hardware. The layout of a descriptor is detailed
and takes some time to cover, so we defer it to a
:doc:`later section <event_descriptors>`.
