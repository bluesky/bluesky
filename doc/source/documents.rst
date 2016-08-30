.. currentmodule:: bluesky.plans

Documents
=========

The RunEngine coordinates I/O while executing a plan. It provides a live stream
of data and metadata *documents* to outside functions that can visualize,
process, or store them. A :doc:`later section <callbacks>` describes how to
subscribe functions to this live stream. This section provides an outline of
documents themselves, aiming to give a sense of the structure and familiarity
with useful components.

When a plan instructs the RunEngine to read a detector, the RunEngine records
that reading in a Python dictionary that relates the measurement to associated
metadata. We dub this dictionary, which is organized in a
`formally specified <https://github.com/NSLS-II/event-model>`_ way, a
*document*.

Overview of a "Run"
-------------------

Each document belongs to a *run* --- loosely speaking, a dataset. Executing any
of the :ref:`built-in pre-assembled plans <preassembled_plans>`, like
:func:`scan` and :func:`count`, creates one run.

.. note::

    Fundamentally, the scope of a run is intentionally vague and flexible. One
    plan might generate many runs or one long run. It just depends on how you
    want to organize your data, both at collection time and analysis time.

    The section :ref:`reimplementing_count` explores this.

The documents in each run are:

- A Run Start document, containg all of the metadata known at the start.
  Highlights:

    - time --- the start time
    - plan_name --- e.g., ``'scan'`` or ``'count'``
    - uid --- randomly-generated ID that uniquely identifies this run
    - scan_id --- human-friendly integer scan ID (not necessarily unique)
    - any other :doc:`metadata captured execution time <metadata>` from the
      plan or the user

- Event documents, containing the actual measurements. Highlights:

    - time --- a timestamp for this group of readings
    - data --- a dictionary of readings like
      ``{'temperature': 5.0, 'position': 3.0}``
    - timestamps --- a dictionary of individual timestamps for each reading,
      from the hardware

- Event Descriptor documents, with metadata about the measurements in the
  events (units, precision, etc.) and about the configuration of the hardware
  that generated them.

- A Run Stop document, containing metadata known only at the end. Highlights:

    - time --- the time when the run was completed
    - exit_status --- "success", "abort", or "fail"

Every document has a ``time`` (its creation time) and a separate ``uid`` to
idenify it. The Event documents also have a ``descriptor`` field referring back
to the Event Descriptor with their metadata. And the Event Descriptor and Run
Stop documents have a ``run_start`` field referring back to their Run Start.
Thus, all the doucments in a run are linked back to the Run Start.

Documents in Detail
-------------------

Event
+++++

An 'event' is a data point with an associated time. One event might contain
more than one reading, but it is presumed that these readings took place at
roughly the same time --- that, for the purposes of later analysis, the
measurements in an can usually be treated as synchronous. For example, they
could be presented together as one row in a table.

.. code-block:: python

    # 'event' document
    {'data': {'temperature': 5.0, 'position': 3.0},
     'timestamps': {'temperature': 1442521007.9258342, 'position': 1442521007.5029348}
     'time': 1442521007.3438923,
     'uid': '<randomly-generated unique ID>', 
     'descriptor': '<reference to a descriptor>'}

The separate times of the individual readings are not thrown away (they are in
'timestamps') but the overall event 'time' is more often used.

.. note::

    Time is given in UNIX time (seconds since 1970). Tools for looking at the
    data would, of course, translate that into a more human-readable form.

Run Start
+++++++++

A 'start' document marks the beginning of the run. It comprises everything we
know before we start taking data, including all metadata provided by the user
and the plan.

All the built-in plans provide some automatic metadata like the names of the
detector(s) and motor(s) used, which can be very useful in searching for data.

The RunEngine guarantees that an entry for ``'plan_name'`` (and
somewhat-less-useful ``plan_type``) will be present, as well as ``'time'``,
``'uid'``, and ``'scan_id'``.

The command:

.. code-block:: python

    from bluesky.plans import scan
    from bluesky.examples import det, motor  # simulated detector, motor

    RE(scan([det], motor, -3, 3, 16), purpose='calibration',
       sample='kryptonite')

generates a 'start' document like this:

.. code-block:: python

    # 'start' document
    {'purpose': 'calibration',
     'sample': 'kryptonite',
     'detectors': ['det'],
     'motors': ['motor1'],
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

Run Stop
++++++++

A 'stop' document marks the end of the run. It contains metadata that is not
known until the run completes.

.. code-block:: python

    # 'stop' document
    {'exit_status': 'success',  # or 'fail' or 'abort'
     'reason': '',  # can describe reason for failure
     'time': 1442521012.1021606,
     'uid': '<randomly-generated unique ID>',
     'start': '<reference to the start document>'
    }

Event Descriptor
++++++++++++++++

TO DO
