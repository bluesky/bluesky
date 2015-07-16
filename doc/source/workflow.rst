Experiment Metadata
===================

Usage Example
-------------

Metadata can be specified like so. It will be stored with the data.

.. ipython:: python

    RE.md['project'] = 'my xray project'
    RE.md['sample'] = {'color': 'red', 'dimensions': [10, 20, 5]}
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

Persistence Between Scans
-------------------------

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

To review the metadata before running ascan, check ``RE.md``, which
behaves like a Python dictionary.

.. ipython:: python

    RE.md['sample']

To start fresh:

.. ipython:: python

    RE.md.clear()

Required Fields
---------------

Some fields and required by our Document specification, and the RunEngine will
raise a ``KeyError`` if they are not set. These fields are:

* owner
* group
* beamline_id (e.g., 'csx')
* config, a dictionary describing the hardware, calibration, dead pixels on
  detectors, etc.

``standard_config.py`` fills some of these in automatically (e.g., 'owner'
defaults to the username of the UNIX user currently logged in).
