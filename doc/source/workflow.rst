An Example Workflow
===================

.. ipython:: python
   :okwarning:
   :suppress:

   from bluesky.examples import det1, det2, det
   from bluesky.callbacks import LiveTable, print_metadata
   from bluesky import RunEngine
   RE = RunEngine()
   RE.verbose = False
   RE.memory['owner'] = 'Jane'
   RE.memory['group'] = 'Grant No. 12345'
   RE.memory['config'] = {'detector_model': 'XYZ', 'pxiel_size': 10}
   RE.memory['beamline_id'] = 'demo'
   from bluesky.scans import Count

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

It worked, but we could use a little more feedback.

To see a live-updating table during collection, we'll add a
:doc:`subscription <callbacks>`. (Don't worry, you don't have to type this
every time. You can make it happen automatically at startup.)

.. ipython:: python

    token = RE.subscribe('all', LiveTable(['det1', 'det2']))

Run the scan again.

.. ipython:: python

   RE(c)

Make a second instance ``Count``, configured to take multiple measurements.

.. ipython:: python

   more_c = Count(dets, 5)
   RE(more_c)

Make some changes to the parameters---say, take 2 measurements intead of 5---
and run again.

.. ipython:: python

    more_c.num = 2
    RE(more_c)

.. ipython:: python
    :suppress:

    RE.unsubscribe(token)
    RE.subscribe('start', print_metadata)

Handling Metadata
-----------------

Usage
+++++

Metadata can be specified like so. It will be stored with the data.

.. ipython:: python

    RE.memory['project'] = 'my xray project'
    RE.memory['sample'] = {'color': 'red', 'dimensions': [10, 20, 5]}
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

Persistence
+++++++++++

The following fields are automatically reused between runs unless overridden.

* sample
* project
* owner
* group
* beamline_id
* config (which should rarely change; see below)
* scan_id (which is automatically incremented)

Custom fields, like 'experimenter' and 'mood' in the example above, are not
reused by default, as we can see below.

.. ipython:: python

    RE(c)
    RE(c, sample={'color': 'blue', 'dimensions': [3, 1, 4]})

To add a custom field to the list of peristent fields, use
``RE.persistent_fields.append('experimenter')``. Use
``RE.persistent_fields.remove('experimenter')`` to stop persisting it.
Fields that are required by our Document specification---owner, group,
beamline_id, and config---cannot be removed. (More on these below.)

To review the metadata before running ascan, check ``RE.memory``, which
behaves like a Python dictionary.

.. ipython:: python

    RE.memory['sample']

To start fresh:

.. ipython:: python

    RE.memory.clear()

Required Fields
+++++++++++++++

Some fields and required by our Document specification, and the RunEngine will
raise a ``KeyError`` if they are not set. These fields are:

* owner
* group
* beamline_id (e.g., 'csx')
* config, a dictionary describing the hardware, calibration, dead pixels on
  detectors, etc.

``standard_config.py`` fills some of these in automatically (e.g., 'owner'
defaults to the username of the UNIX user currently logged in).
