Progress Bar
************

Bluesky provides a progress bar add-on. For example, two motors moving
simulateously make a display like this:

.. code-block:: none

    mtr1  9%|███▊                                       | 0.09/1.0 [00:00<00:01,  1.21s/deg]
    mtr2100%|████████████████████████████████████████████| 1.0/1.0 [00:01<00:00,  1.12s/deg]

This display includes:

* the name of the device (motor, temperature controller, etc.)
* the distance (or degrees, etc.) traveled so far
* the total distance to be covered
* the time elapsed
* the estimated time remaining
* the rate (determined empirically)

The progress bar relies on the device to report its progress. If a device does
not provide comprehensive information, a simpler progress bar will be shown,
listing the names of devices being waited on and reporting which have
completed.

.. code-block:: none

    mtr1 [No progress bar available.]
    mtr2 [Complete.]

Any time the RunEngine waits on hardware the progress bar is notified. This
includes, for example, waiting for a motor to move or waiting for a detector to
trigger. (In bluesky jargon, the progress bar is notified any time the
RunEngine processes a 'wait' command).

The progress bar is not set up by default. It must be attached to a RunEngine.
This need only be done once (say, in a startup file).

.. code-block:: python

    from bluesky.utils import ProgressBarManager
    
    RE.waiting_hook = ProgressBarManager()

Some motions are very quick and not worth displaying a progress bar for. By
default, a progress bar is only drawn after 0.2 seconds. If an action completes
before then, the progress bar is never shown. To choose a shorter or longer
delay---say 5 seconds---use the parameter ``ProgressBarManager(delay_draw=5)``.

For more technical detail about communication between the device, the
RunEngine, and the ProgressBarManager, read about the ``watch`` method in the
:ref:`status_obj_api` and ``waiting_hook`` in the :doc:`run_engine_api`.

The implementation of the progress bar itself makes use of
`tqdm <https://github.com/tqdm/tqdm/>`_, a lovely Python package for making
a progress bar out of any iterable.
