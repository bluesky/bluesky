.. currentmodule:: bluesky.scans

Object-Oriented Scan Interface
==============================

In the object-oriented interface, the scan object is not bundled with a copy
of the RunEngine. That is, the *instructions* and the *execution* are treated
separately.

Each of the scans in the :doc:`simple_api` is backed by a Scan object. Once an
instance of a Scan is created, it can be reused without respecifying its
parameters.

Usage Example
-------------

.. ipython:: python
    :suppress:

    from bluesky.tests.utils import setup_test_run_engine
    RE = setup_test_run_engine()
    from bluesky.examples import motor, det1, det2, det3

Make an instance of a scan, passing in parameters.

.. ipython:: python

    from bluesky.scans import AbsScan
    my_scan = AbsScan([det1, det2], motor, 1, 5, 10)

Running the scan is a separate step. The same scan can be run multiple times
without respecifying its parameters.

    RE(my_scan)
    RE(my_scan)

.. ipython:: python

Any of the scan'ss parameters can be updated individually.

.. ipython:: python

    my_scan.num = 4  # change number of data points from 10 to 4
    RE(my_scan)
    my_scan.detectors.append(det3)  # add another detector
    RE(my_scan)

The ``set`` method is a convenient way to update multiple parameters at once.

.. ipython:: python

    my_scan.set(start=20, stop=25)

Count
-----

.. autofunction:: Count

Absolute Scans
--------------

.. autofunction:: AbsScan
.. autofunction:: LogAbsScan
.. autofunction:: AbsListScan
.. autofunction:: InnerProductAbsScan
.. autofunction:: OuterProductAbsScan

Relative (Delta) Scans
----------------------

.. autofunction:: DeltaScan
.. autofunction:: LogDeltaScan
.. autofunction:: DeltaListScan
.. autofunction:: InnerProductDeltaScan
.. autofunction:: OuterProductDeltaScan

.. _builtin-adaptive-scans:

Adative Scans
-------------

.. autofunction:: AdaptiveAbsScan
.. autofunction:: AdaptiveDeltaScan

Interactive Scans
-----------------

.. autofunction:: Tweak
