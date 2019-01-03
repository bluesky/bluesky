Debugging and Logging
=====================

Bluesky uses Python logging framework. See below for a list of the names of the
loggers it publishes to.

Useful Snippets
---------------

If the RunEngine is "hanging," running slowly, or repeatedly encountering an
error, it is useful to know exactly where in the plan the problem is occurring.
To follow the RunEngine's progress through the plan, crank up the verbosity of
the logging.

.. code-block:: python

   RE.log.setLevel('DEBUG')

To direct the output to a file instead of to the screen:

.. code-block:: python

   from bluesky import set_handler
   set_handler(file='debugging_bluesky.txt')

Logger Names
------------

Here is the complete list of loggers used by bluesky.

* ``'bluesky'`` --- the logger to which all bluesky log messages propagate
* ``'bluesky.RE'`` --- Messages from a RunEngine. INFO-level notes state
  changes. DEBUG-level notes when each message from a plan is about to be
  processed, when a document has been emitted to subscribed callbacks, and when
  a status object has completed.
* ``'bluesky.RE.<id>'`` --- Messages from a specific RunEngine instance,
  disambiguating the (rare) case where there are multiple RunEngine instances
  in the same process. This is the logger that the accessor ``RE.log`` refers
  to.
* ``'bluesky.RE.<id>.msg'`` --- DEBUG-level notes when each message from a plan
  is about to be processed.

Logging Handlers
----------------

By default, bluesky prints log messages to the standard out by adding a
:class:`logging.StreamHandler` to the ``'bluesky'`` logger at import time. You
can, of course, configure the handlers manually in the standard fashion
supported by Python. But a convenience function :func:`bluesky.set_handler`,
makes it easy to address common cases.

See the Examples section below.

.. autofunction:: bluesky.set_handler
