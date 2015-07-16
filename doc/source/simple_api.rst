Simple Scan Interface
=====================

.. ipython:: python

    from bluesky.examples import det1, det2, det3, det
    %run -i ../bluesky/standard_config.py

.. ipython:: python

    DETS = [det]

The simple scan interface provides a condensed syntax to execute common tasks.

Some of the names and signatures of the functions in this module closely match
some core "macros" in SPEC, control software used in X-ray diffraction
experiments. Other functions introduce new functionality.

The simple interface is not necessarily recommended for developing new scans.
It is backed by a more explicit :doc:`scans` suited for development work.

Specify Detectors
-----------------

.. note::

    If you are using a IPython profile, a list of detectors might be
    automatically specified at startup. In that case, you may not need to do
    anything unless you need to inspect or customize that list.

The global variable ``DETS`` is a list of a detector objects. It controls
which detectors are triggered and read by all the simple scans.

.. ipython:: python

    DETS = [det1, det2]

Like any Python list, you can append and remove elements.

.. ipython:: python

    DETS.append(det3)
    DETS.remove(det1)
    DETS

There are other global variables particular to certain kinds of scan.
They are addressed below.

Count
-----

A ``ct`` ("count") scan reads all the detectors in the global list ``DETS`` for 
a given acquisition time. If no time is specified, 1 second is the default.

.. ipython:: python

    ct(time=1)

An Aside: Record Metadata with the Scan
---------------------------------------

Basic metadata, such as the time, is automatically recorded. To include custom
metadata with the scan, add keyword arguments like so::

    ct(color='red')

These can be strings, numbers, or even Python dictionaries. This is a useful
way to capture detailed information in a structured way.::

    ct(attempt=3, sample={name: 'A', 'width': 5, 'height': 10})

By breaking out the sample information into separate fields, it becomes
possible to search on individual attributes and use them in later analysis.
This is much more useful than, say, ``'sampleA_width5_height10'``. In fact,
to encourage good practices, we do not allow the ``sample`` field to be a
string or a number: it *must* be a Python dictionary.

Motor Scans
-----------

Like ``ct``, the motor scans read from all the detectors in the global list
``DETS``.

Absolute Scans
^^^^^^^^^^^^^^

An ``ascan`` ("absolute scan") scans one motor in equal-sized steps.

.. code-block:: python

    ascan(motor, start, finish, intervals, time)

Note that ``intervals`` counts the number of *steps* which is one less
than the number of *data points*. This follows the convention in SPEC.
Outside of the simple API, we revert to the Python convention of counting
data points, not steps.

An ``a2scan`` scans two motors together along different trajectories,
again in equal-sized steps. (We think of this as the "inner product" of two
trajectories.)

.. code-block:: python

    a2scan(motor1, start1, finish1, motor2, start2, finish2, intervals, time)

.. code-block:: python

    a3scan(motor1, start1, finish1, motor2, start2, finish2, motor3, 
           start3, finish3, intervals, time)

We provide ``a2scan`` and ``a3scan`` for convenience, but in fact both of them
support any number of motors. This is valid:::

    a2scan(motor1, start1, finish1, motor2, start2, finish2, motor3, start3,
           finish3, motor4, start4, finish4, intervals, time)

Delta Scans
^^^^^^^^^^^

A ``dscan`` ("delta scan") scans one motor in equal-size steps, specified
relative to the motor's current position.::

    dscan(motor, start, finish, intervals, time)

``lup`` is an alias for ``dscan``. And as with ``ascan`` above, there is a
``d2scan`` and a ``d3scan``, each of which accept an unlimited number of
motors.

Mesh Scan
^^^^^^^^^

A ``mesh`` scan scans any number of motors in a mesh. (We think of this as the
"other product" of the trajectories.)::

    mesh(motor1, start1, finish1, intervals1, motor2, start2, finish2,
         intervals2, time)

As with ``a2scan`` and ``a3scan``, ``mesh`` accepts any number of motors.
Notice that the number of intervals is specified sepraately for each motor.

Scans Tied to Particular Motors / Controllers
---------------------------------------------

Theta Two Theta
^^^^^^^^^^^^^^^

This scan requires the global variables ``TH_MOTOR`` ("theta motor") and
``TTH_MOTOR`` ("two theta motor").

A ``th2th`` ("theta two theta") scans steps the two theta motor through a
given range while stepping the theta motor through half that range.::

    th2th(start, finish, intervals, time)

Temperature Scans
^^^^^^^^^^^^^^^^^

Temperature scans require the global variable ``TEMP_CONTROLLER``.

A ``tscan`` steps the temperature controller through equally-spaced temperature
set points. An optional ``sleep`` argument specifies a thermalization time. As
in SPEC, it is zero by default.::

    tscan(start, finish, intervals, time, sleep=0)

There is also ``dtscan``, a relative temperature scan.

Tweak
-----

Tweak is an interactive scan that reads a field from one detector, displays
the result, and prompts the user to specify where to step the motor next.
It requires the global variable ``MASTER_DET`` (which detectors to use,
such as ``sclr``) and ``MASTER_DET_FIELD`` (the name of the field in that
detector to read out, such as ``'sclr_chan4'``). Note that the former is a
readable object and the latter is a string of text.::

    tw(motor, step)
