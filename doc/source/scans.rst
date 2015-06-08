Basic Scans
===========

.. currentmodule:: bluesky.scans

An Example Workflow
-------------------

.. ipython:: python
   :suppress:

   from bluesky.examples import det
   point_det = det
   temp1 = det
   temp2 = det
   from bluesky import RunEngine
   RE = RunEngine()
   RE.verbose = False
   RE.memory['owner'] = 'demo'
   RE.memory['beamline_id'] = 'demo'
   from bluesky.scans import Count

.. note::

   This example assumes that you are running with a standard configuration.
   If not, simply run ``from bluesky.standard_config import *``.

1. Decide on the detectors of interest. Define a list for convenience.

.. ipython:: python

   d = [point_det, temp1, temp2]

2. Create a new scan.

.. ipython:: python

   c = Count(d)

``c`` encapsulates scan instructions and the detector list.

3. Run the scan.

.. ipython:: python

   RE(c)

4. Rerun, adjusting parameters as desired.

.. ipython:: python

   c.detectors = [temp1, temp2]
   RE(c)

If a scan is commonly used, it can be defined in an IPython profile run at
startup.

Count
-----

.. autofunction:: Count

Absolute Scans
--------------

.. autofunction:: LinAscan
.. autofunction:: LogAscan
.. autofunction:: Ascan

Relative (Delta) Scans
----------------------

.. autofunction:: LinDscan
.. autofunction:: LogDscan
.. autofunction:: Dscan

Adative Scans
-------------

.. autofunction:: AdaptiveAscan
.. autofunction:: AdaptiveDscan
