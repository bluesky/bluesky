Debugging and Logging
=====================

Message Hook
------------

If the RunEngine is "hanging," running slowly, or repeatedly encountering an
error, it is useful to know exactly where in the plan the problem is occurring.
To follow the RunEngine's progress through the plan, use the message hook:

.. code-block:: python

    RE.msg_hook = print

Before each message ``msg`` in the plan is processed by the RunEngine,
``print(msg)`` will be called. The result is *extremely* verbose and may even
significantly slow down plan execution, so it should only be used for
debugging.

Using ``print`` is easiest, but in generalfunction can be used. For example, to
write to a file:

.. code-block:: python

    def append_to_file(msg):
        with open('myfile.log', 'a') as f:
            f.write(str(msg))

    RE.msg_hook = append_to_file

To restore default behavior, set the hook back to ``None``:

.. code-block:: python

    RE.msg_hook = None


Logging
-------

The RunEngine integrates with Python's built-in logging framework. It provides
a convenient attribute for configuring logging quickly.

.. code-block:: python

    # standard Python logging setup
    import logging
    logging.basicConfig()
    
    RE.log.disabled = False

With this configuration, executing a plan prints log messages to the screen.

The logger issues INFO-level messages whenever the the RunEngine changes state
(idle -> running, running -> paused, etc.) and DEBUG-level messages whenever a
new Document is created and emitted to the subscriptions. Demo:

.. code-block:: python

    In [3]: RE(count([det]))
    Out[3]: ['f015945c-7e9f-4d2c-9a83-d1db1b31fb43']

    In [4]: RE.verbose = True

    In [5]: RE(count([det]))
    INFO:bluesky.run_engine_id4371931376:Change state on
    <bluesky.run_engine.RunEngine object at 0x1049660f0> from 'idle' ->
    'running'
    DEBUG:bluesky.run_engine_id4371931376:Starting new with uid
    '4f3e173f-3383-49a4-94bc-cac571144c4d'
    DEBUG:bluesky.run_engine_id4371931376:Emitted RunStart
    (uid='4f3e173f-3383-49a4-94bc-cac571144c4d')
    DEBUG:bluesky.run_engine_id4371931376:The object reader: det reports
    trigger is done with status True.
    DEBUG:bluesky.run_engine_id4371931376:Emitted Event Descriptor with name
    'primary' containing data keys dict_keys(['det'])
    (uid='400f6d4a-db5d-454b-9b1c-5e759eb8511b')
    DEBUG:bluesky.run_engine_id4371931376:Emitted Event with data keys
    dict_keys(['det']) (uid='dd2dedba-4ac7-4761-a94a-2e65c1579aa8')
    DEBUG:bluesky.run_engine_id4371931376:Stopping run
    '4f3e173f-3383-49a4-94bc-cac571144c4d'
    DEBUG:bluesky.run_engine_id4371931376:Emitted RunStop
    (uid='654f4bfb-043f-4b81-9a8f-371ce276caf3')
    INFO:bluesky.run_engine_id4371931376:Change state on
    <bluesky.run_engine.RunEngine object at 0x1049660f0> from 'running' ->
    'idle'
    Out[5]: ['4f3e173f-3383-49a4-94bc-cac571144c4d']

The log messages include the Python id of the RunEngine instance (``id(RE)``)
in case logs from multiple instances end up in the same file.

The ``RE.log`` attribute is a standard Python logger object. For example, to
change the log level to skip DEBUG-level messages:

.. code-block:: python

    RE.log.setLevel(logging.INFO)

.. info::

    For back-compatibility with old versions of bluesky, there is also an
    ``RE.verbose`` attribute. ``RE.verbose`` is a synonym for
    ``not RE.log.disabled``.

Debugging Callbacks
-------------------

    See :ref:`debugging_callbacks`.
