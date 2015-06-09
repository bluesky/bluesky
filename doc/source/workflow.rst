
An Example Workflow
===================

.. ipython:: python
   :okwarning:
   :suppress:

   from bluesky.examples import det1, det2, det
   from bluesky.callbacks import live_table
   from bluesky import RunEngine
   RE = RunEngine()
   RE.verbose = False
   RE.memory['owner'] = 'Jane the Scientist'
   RE.memory['beamline_id'] = 'demo'
   from bluesky.scans import Count

   def print_metadata(start):
       for field, value in sorted(start.items()):
           if field not in ['time', 'uid']:
               print('{0}: {1}'.format(field, value))

.. note::

   This example assumes that you are running with a standard configuration.
   If not, simply run ``from bluesky.standard_config import *``.

Defining and Running a Scan
---------------------------

Decide on the detectors of interest and instantiate a scan.

.. ipython:: python

   dets = [det1, det2]  # favorite detectors
   c = Count(dets)

Now ``c`` encapsulates scan instructions and the detector list. Run the scan.

.. ipython:: python

   RE(c)

To see a live-updating table during collection, we'll add a
:doc:`subscription <callbacks>`.

.. ipython:: python

    RE.subscribe('all', live_table(dets))

Run the scan again.

.. ipython:: python

   RE(c)

Make a second instance ``Count``, configured to take multiple measurements.

.. ipython:: python

   more_c = Count(dets, 5)

Make some changes to the parameters---say, take 2 measurements intead of 5---
and run again.

.. ipython:: python

    more_c.num = 2
    RE(more_c)

Handling Metadata
-----------------

.. ipython:: python

    RE.subscribe('start', print_metadata)
    RE(c)

Metadata can be specified like so. It will be stored with the data.

.. ipython:: python

    RE.memory['project'] = 'my xray project'
    RE.memory['sample'] = {'color': 'red', 'dimensions': [10, 20, 5]}
    RE.memory['my_custom_field'] = 'zebra'
    RE(c)

.. note::

    Structured data, such as

    .. code-block:: python

        {'color': 'red', 'dimensions': [10, 20, 5]}

    is much better than a long string like

    .. code-block:: python

        'red_10_20_5'

    because it is searchable and self-describing. To encourage good practices,
    the RunEngine inists that 'sample' be a dictionary. Any other fields
    you invent can be anything you want.

Additional metadata can be specified when the scan is run.

.. ipython:: python

    RE(c, experimenter='Emily', mood='excited')

Metadata is automatically reused between runs unless overridden.

.. ipython:: python

    RE(c)
    RE(c, sample={'color': 'blue', 'dimensions': [3, 1, 4]})

To review the metadata before running ascan, check ``RE.memory``, which
behaves like a Python dictionary.

.. ipython:: python

    RE.memory['sample']

To start fresh:

.. ipython:: python

    RE.memory.clear()

Some fields and required, and the RunEngine will raise an error if they are
not set.

.. ipython:: python
    :okexcept:

    RE(c)
    # Filling in required metadata...
    RE.memory['owner'] = 'John'
    RE.memory['beamline_id'] = 'demo'
    RE(c)
