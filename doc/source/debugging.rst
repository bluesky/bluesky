Debugging and Logging
=====================

Message Hook
------------

If the RunEngine is "hanging," running slowly, or repeatedly encountering an
error, it is useful to know exactly where in the plan the problem is occurring.
To follow the RunEngine's progress through the plan, use the message hook. You
can set it to any callable, including the ``print`` function:

.. code-block:: python

    RE.msg_hook = print

or a small utility that formats a more readable message and includes a
timestamp:

.. code-block:: python

    from bluesky.utils import ts_msg_hook

    RE.msg_hook = ts_msg_hook

Now the RunEngine will call ``ts_msg_hook(msg)`` before processing each ``msg``
in the plan. Execute the plan that is causing the issue and watch the output.
Suppose we are running a ``count`` that freezes up while triggering a detector.
That would look like this:

.. ipython::
    :verbatim:

    In [8]: RE(count([det]))
    11:39:06.712560 stage             -> det             args: (), kwargs: {}
    11:39:06.715281 open_run          -> None            args: (), kwargs:
    {'detectors': ['det'], 'num_points': 1, 'num_intervals': 0, 'plan_args':
    {'detectors': ["SynGauss(name='det', value=1.0, timestamp=1516379938.010617)"],
    'num': 1}, 'plan_name': 'count', 'hints': {'dimensions': [(('time',),
    'primary')]}}
    11:39:06.717813 checkpoint        -> None            args: (), kwargs: {}
    11:39:06.718076 trigger           -> det             args: (), kwargs:
    {'group': 'trigger-0487d9'}
    11:39:06.718659 wait              -> None            args: (), kwargs:
    {'group': 'trigger-0487d9'}

The last message is is a 'wait' message, and it waiting for the 'trigger' just
above it to complete. If the plan freezes at this point, a likely culprit is
the triggering mechanism of the detector. In this example, we can see the
detector in question is a simulated detector, ``ophyd.sim.SynGauss``.
From here we would investigate whether the hardware was behaving properly and
whether its triggering behavior was programmed properly on the Python side.

When finished troubleshooting, set ``RE.msg_hook = None``.

We used ``print`` for simple debugging. Any user-defined function that accepts
a ``bluesky.Msg`` namedtuple as its argument could also be used. For example,
to write to a file:

.. code-block:: python

    def append_to_file(msg):
        with open('myfile.log', 'a') as f:
            f.write(str(msg))

    RE.msg_hook = append_to_file

State Hook
----------

The RunEngine can be in one of three states:

* 'idle' (ready to accept a new plan)
* 'running' (running the event loop and processing a plan)
* 'paused' (not running the event loop, but holding onto state in preparation
  for possibly resuming)

The state is exposed through the RunEngine's ``state`` attribute. To monitor
changes in state, use the ``state_hook`` attribute. Like ``msg_hook`` above, it
can be set to ``None`` (default) or a function. In this case, the function
should accept two arguments: the new state and the previous state.

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

The logger issues INFO-level messages whenever the RunEngine changes state
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

.. note::

    For back-compatibility with old versions of bluesky, there is also an
    ``RE.verbose`` attribute. ``RE.verbose`` is a synonym for
    ``not RE.log.disabled``.
