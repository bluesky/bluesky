(Optional) SPEC-like Interface
==============================

Conceptual Differences Between Bluesky and SPEC
-----------------------------------------------

Some scientists are familiar with `SPEC <http://www.certif.com/spec.html>`_,
a domain-specific language for hardware control. It is possible to imitate the
SPEC workflow on top of bluesky. Of course, we still adhere to the Python
syntax so that we can employ the full power of the general-purpose Python
language.

It is useful to understand a key conceptual difference between bluesky and
SPEC. SPEC treats the *specification* of an experiment ("move a motor from 1
to 5 in 5 strides") and its *execution* in one step. For example, in SPEC,
typing

.. code-block:: bash

    ascan th, 1, 5, 5

both specifies and executes the scan. Fundamentally, bluesky separates these
steps: first we generate a "plan" (a set of granular instructions)

.. code-block:: python

    plan = AbsScanPlan(detectors, th, 1, 5, 4)

and then we pass the plan to a RunEninge for execution.

.. code-block:: python

    RE(plan)

To imitate the SPEC workflow, these two steps are lumped together.

.. code-block:: python

    ascan(th, 1, 5, 5)  # this is bluesky's imitation of SPEC

For simple tasks, the condensed syntax is clearly convenient; for others,
maintaining that logical separation can be empowering. (See the section on
:doc:`plans` for why.)

Addiontally, SPEC maintains a global list of detectors that apply to all scans.
Bluesky specifies detectors on a per-plan basis: for example, a "count" might
involve different detectors than an "ascan". But, to imitate SPEC, bluesky
provides a global "stash" of settings, ``gs``, including a customizable list of
"standard" detectors, as illustrated below.

.. ipython:: python
   :suppress:

   from bluesky.examples import det1, det2, det3, det
   from bluesky.standard_config import gs
   from bluesky.standard_config import *  # the simple scan instances
   from bluesky.tests.utils import setup_test_run_engine
   gs.DETS = [det]
   gs.RE = setup_test_run_engine()


Specify Detectors
-----------------

.. note::

    If you are using a IPython profile, a list of detectors might be
    automatically specified at startup. In that case, you may not need to do
    anything unless you need to inspect or customize that list.

The setting ``gs.DETS`` is a list of a detector objects. It controls
which detectors are triggered and read by all the simple scans.
(Incidentally, ``gs`` stands for "global state" or "global settings." Why
can't it just be plain ``DETS``? Global variables are best avoided in Python,
and the ``gs.`` part provides useful input validation.)

.. ipython:: python

    gs.DETS = [det1, det2]

Like any Python list, you can append and remove elements.

.. ipython:: python

    gs.DETS.append(det3)
    gs.DETS.remove(det1)
    gs.DETS

There are other settings which control the output of the scans --
``gs.TABLE_COLS`` and ``gs.PLOT_Y``  for example. Explore the contents of
``gs`` by typing ``gs.<TAB>``.

Count
-----

A ``ct`` ("count") scan reads all the detectors in the list ``DETS`` for 
a given acquisition time. If no time is specified, 1 second is the default.

.. code-block:: python

    ct(time=1)

Motor Scans
-----------

Like ``ct``, the motor scans read from all the detectors in the list
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
support any number of motors. This is valid:

.. code-block:: python

    a2scan(motor1, start1, finish1, motor2, start2, finish2, motor3, start3,
           finish3, motor4, start4, finish4, intervals, time)

Delta Scans
^^^^^^^^^^^

A ``dscan`` ("delta scan") scans one motor in equal-size steps, specified
relative to the motor's current position.

.. code-block:: python

    dscan(motor, start, finish, intervals, time)

``lup`` is an alias for ``dscan``. And as with ``ascan`` above, there is a
``d2scan`` and a ``d3scan``, each of which accept an unlimited number of
motors.

Mesh Scan
^^^^^^^^^

A ``mesh`` scan scans any number of motors in a mesh. (We think of this as the
"other product" of the trajectories.)

.. code-block:: python

    mesh(motor1, start1, finish1, intervals1, motor2, start2, finish2,
         intervals2, time)

As with ``a2scan`` and ``a3scan``, ``mesh`` accepts any number of motors.
Notice that the number of intervals is specified sepraately for each motor.

Scans Tied to Particular Motors / Controllers
---------------------------------------------

Theta Two Theta
^^^^^^^^^^^^^^^

This scan requires the settings ``gs.TH_MOTOR`` ("theta motor") and
``gs.TTH_MOTOR`` ("two theta motor").

A ``th2th`` ("theta two theta") scans steps the two theta motor through a
given range while stepping the theta motor through half that range.

.. code-block:: python

    th2th(start, finish, intervals, time)

Temperature Scans
^^^^^^^^^^^^^^^^^

Temperature scans require the setting ``gs.TEMP_CONTROLLER``.

A ``tscan`` steps the temperature controller through equally-spaced temperature
set points. An optional ``sleep`` argument specifies a thermalization time. As
in SPEC, it is zero by default.

.. code-block:: python

    tscan(start, finish, intervals, time, sleep=0)

There is also ``dtscan``, a relative temperature scan.

Tweak
-----

Tweak is an interactive scan that reads a field from one detector, displays
the result, and prompts the user to specify where to step the motor next.
It requires the setting ``gs.MASTER_DET`` (which detector to use,
such as ``sclr``) and ``MASTER_DET_FIELD`` (the name of the field in that
detector to read out, such as ``'sclr_chan4'``). Note that the former is a
readable object and the latter is a string of text.

.. code-block:: python

    tw(motor, step)
