*********************
Debugging and Logging
*********************

.. versionchanged:: 1.6.0

   Bluesky's use of Python's logging framework has been completely reworked to
   follow Python's documented best practices for libraries.

Bluesky uses Python's logging framework, which enables sophisticated log
management. For common simple cases, including viewing logs in the terminal or
writing them to a file, the next section illustrates streamlined,
copy/paste-able examples. Users who are familiar with that framework or who
need to route logs to multiple destinations may wish to skip ahead to
:ref:`logger_api`.

Useful Snippets
===============

Log warnings
------------

This is the recommended standard setup.

.. code-block:: python

   from bluesky import config_bluesky_logging
   config_bluesky_logging()

It will display ``'bluesky'`` log records of ``WARNING`` level or higher in the
terminal (standard out) with a format tailored to bluesky.

Maximum verbosity
-----------------

If the RunEngine is "hanging," running slowly, or repeatedly encountering an
error, it is useful to know exactly where in the plan the problem is occurring.
To follow the RunEngine's progress through the plan, crank up the verbosity of
the logging.

This will display each message from the plan just before the RunEngine
processes it, giving a clear indication of when plan execution is stuck.

.. code-block:: python

   from bluesky import config_bluesky_logging
   config_bluesky_logging(level='DEBUG')

Log to a file
-------------

This will direct all log messages to a file instead of the terminal (standard
out).

.. code-block:: python

    from bluesky import config_bluesky_logging
    config_bluesky_logging(file='/tmp/bluesky.log', level='DEBUG')

.. important::

   We strongly recommend setting levels on *handlers* not on *loggers*.
   In previous versions of bluesky, we recommended adjusting the level on the
   *logger*, as in ``RE.log.setLevel('DEBUG')``. We now recommended
   that you *avoid* setting levels on loggers because it would affect all
   handlers downstream, potentially inhibiting some other part of the program
   from collecting the records it wants to collect.

.. _logger_api:

Bluesky's Logging-Related API
=============================

Logger Names
------------

Here are the primary loggers used by bluesky.

* ``'bluesky'`` --- the logger to which all bluesky log records propagate
* ``'bluesky.emit_document'`` --- A log record is emitted whenever a Document
  is emitted. The log record does not contain the full content of the
  Document.
* ``'bluesky.RE'`` --- Records from a RunEngine. INFO-level notes state
  changes. DEBUG-level notes when each message from a plan is about to be
  processed and when a status object has completed.
* ``'bluesky.RE.msg`` --- A log record is emitted when each
  :class:`~bluesky.utils.Msg` is about to be processed.
* ``'bluesky.RE.state`` --- A log record is emitted when the RunEngine's state
  changes.

There are also some module-level loggers for specific features.

Formatter
---------

.. autoclass:: bluesky.log.LogFormatter

Global Handler
---------------

Following Python's recommendation, bluesky does not install any handlers at
import time, but it provides a function to set up a basic useful configuration
in one line, similar to Python's :py:func:`logging.basicConfig` but with some
additional options---and scoped to the ``'bluesky'`` logger with bluesky's
:class:`bluesky.log.LogFormatter`. It streamlines common use cases without
interfering with more sophisticated use cases.

We recommend that facilities using bluesky leave this function for users and
configure any standardized, facility-managed logging handlers separately, as
described in the next section.

.. autofunction:: bluesky.log.config_bluesky_logging
.. autofunction:: bluesky.log.get_handler

Advanced Example
================

The flow of log event information in loggers and handlers is illustrated in the
following diagram:

.. image:: https://docs.python.org/3/_images/logging_flow.png

For further reference, see the Python 3 logging howto:
https://docs.python.org/3/howto/logging.html#logging-flow

As an illustrative example, we will set up two handlers using the Python
logging framework directly, ignoring bluesky's convenience function.

Suppose we set up a handler aimed at a file:

.. code-block:: python

    import logging
    file_handler = logging.FileHandler('bluesky.log')

And another aimed at `Logstash <https://www.elastic.co/products/logstash>`_:

.. code-block:: python

    import logstash  # requires python-logstash package
    logstash_handler = logstash.TCPLogstashHandler(<host>, <port>, version=1)

We can attach the handlers to the bluesky logger, to which all log records
created by bluesky propagate:

.. code-block:: python

    logger = logging.getLogger('bluesky')
    logger.addHandler(logstash_handler)
    logger.addHandler(file_filter)

We can set the verbosity of each handler. Suppose want maximum verbosity in the
file but only medium verbosity in logstash.

.. code-block:: python

    logstash_handler.setLevel('INFO')
    file_handler.setLevel('DEBUG')

Finally, ensure that "effective level" of ``logger`` is at least as verbose as
the most verbose handler---in this case, ``'DEBUG'``. By default, at import,
its level is not set

.. ipython:: python
   :verbatim:

    logging.getLevelName(logger.level)
    'NOTSET'

and so it inherits the level of Python's default
"handler of last resort," :py:obj:`logging.lastResort`, which is ``'WARNING'``.

.. ipython:: python
   :verbatim:

    logging.getLevelName(logger.getEffectiveLevel())
    'WARNING'

In this case we should set it to ``'DEBUG'``, to match the most verbose level
of the handler we have added.

.. code-block:: python

   logger.setLevel('DEBUG')

This makes DEBUG-level records *available* to all handlers. Our logstash
handler, set to ``'INFO'``, will filter out DEBUG-level records.

To globally disable the generation of any log records at or below a certain
verbosity, which may be helpful for optimizing performance, Python provides
:py:func:`logging.disable`.
