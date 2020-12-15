
Utility classes and functions
=============================

.. automodule:: bluesky.utils


Msg
---
.. autosummary::
   :nosignatures:
   :toctree: generated

   Msg


Persistent metadata
-------------------

To maintain a peristent set of meta-data between Python sessions
we include a dictionary duck-type based on `zict.Func`.

.. autosummary::
   :nosignatures:
   :toctree: generated

   PersistentDict
   PersistentDict.directory



Internal exceptions
-------------------

We define a number of `Exception` sub-classes for internal signaling.

.. autosummary::
   :nosignatures:
   :toctree: generated

   RunEngineControlException
   RequestAbort
   RequestStop
   RunEngineInterrupted
   NoReplayAllowed
   IllegalMessageSequence
   FailedPause
   FailedStatus
   InvalidCommand
   PlanHalt
   RampFail


Progress bars
-------------

These are used by the RunEngine to display progress bars and
are the clients of the :obj:`~ophyd.status.MoveStatus.watch` API



.. autosummary::
   :nosignatures:
   :toctree: generated

   ProgressBar
   ProgressBar.update
   ProgressBar.draw
   ProgressBar.clear

   ProgressBarManager


During tasks
------------

These objects encapsulate what the  RunEngine should do on its thread while
waiting for the plan to complete in the background thread

.. autosummary::
   :nosignatures:
   :toctree: generated

   DuringTask
   DuringTask.block

   DefaultDuringTask
