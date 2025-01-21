
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

To save and restore metadata between Python sessions, we provide two support
classes. Choose from these alternatives:

======================================  ======================================
class                                   storage model
======================================  ======================================
:class:`~bluesky.utils.PersistentDict`  Directory of files backed by ``zict``.  
:class:`~bluesky.utils.StoredDict`      Single YAML file.
======================================  ======================================

.. autosummary::
   :nosignatures:
   :toctree: generated

   PersistentDict
   PersistentDict.directory
   StoredDict


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

   ProgressBar.update
   ProgressBar.draw
   ProgressBar.clear



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
