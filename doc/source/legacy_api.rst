.. currentmodule:: bluesky.standard_config

Legacy API
**********

.. warning::

   The commands covered below are supported for backward-compatibility with
   the ophyd Run Engine. While they will ensure that old code can be run, they
   are not compatible with the new features offered by bluesky.

Basic Built-in Scans
--------------------

Overview
========

At startup, ophyd creates Python objects representing different kinds of
scan: ``ct``, ``ascan``, ``dscan``, and others. Various scan settings
can be inspected and configured using their attribures, such as::

    ct.default_detectors

To run a scan, call it like a function.::

    ct()

Count
=====

A *count* triggers all detectors once.

Specify Detectors
^^^^^^^^^^^^^^^^^
To specify which detectors to scan, set the ``default_detectors`` attribute of
a Python object called ``ct``.::

    ct.default_detectors = [my_detector1, my_detector2]

Note that ``my_detector1`` and ``my_detector2`` are not in quotes. They are
ophyd objects already defined by the configuration profile at startup.

Run the Scan
^^^^^^^^^^^^

To run the scan, call ``ct`` like a function.::

    ct()

The data is automatically stored and also printed to the screen.

An Aside: Record Metadata with the Scan
=======================================

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

Absolute Scan
=============

An *absolute scan* or "A-scan" varies something in even-sized steps and
triggers detectors after each step.

Set the detectors the same as with ``ct()`` above.

``ascan`` requires some arguments. The first is an ophyd object representing
a "positioner" -- a motor, a temperature controller, or any hardware that can be written to. The others are a start position, a stop position, and the
number of points to sample.

For example, if I have a positioner called ``motor``, I can take measurements
at 0, 1, 2, and 3.::

    ascan(motor1, 0, 3, 4)

Alternatively, you may specify the step size instead. The number of points will
be inferred.::

    ascan(motor1, 0, 3, step=1)

As illustrated with ``ct`` in the previous section, ``ascan`` accepts
custom keyword arguments, passed with the scan as metadata.::

    ascan(motor1, 0, 3, 4, label='B', mood='skeptical')

Delta Scan
==========

``dscan`` matches the syntax of ``ascan``, but ``start`` and ``stop`` are
interpreted as relative to the current position. Recall that one can check
the current positions with the command ``wh_pos()``.

.. currentmodule:: ophyd.cli_api

Commands
--------

.. autosummary::
   :toctree: generated/

   wh_pos
   set_pos
   mov
   movr
   set_lm
   log_pos
   log_pos_diff
