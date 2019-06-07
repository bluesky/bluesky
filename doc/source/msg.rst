.. _msg:

Message Protocol
================

*Note: This is a technical document not optimized for user readability.*

Overview
--------

A *plan* is a sequence of atomic operations describing a data acquisition
procedure. Each operation is represented by a ``bluesky.Msg`` ("message")
object. A plan may be implemented as a simple list of messages:

.. code-block:: python

    from bluesky import Msg

    # (Behold, the most boring data acquisition ever conducted!)
    plan = [Msg('open_run'), Msg('close_run')]

or as a generator the yields messages one at time:

.. code-block:: python

    def plan():
        yield Msg('open_run')
        yield Msg('close_run')

The above examples are equivalent. For more sophisticated uses, the second one
is more powerful, as it can incorporate loops, conditionals, adaptive logic ---
generally any Python code.

But, crucially, the plan code itself must not communicate with hardware.
(You should never put ``epics.caput(...)`` in a plan!) Rather, each operation
is represented by a ``Msg`` object that *describes* what should be done. This
makes it safe to introspect the plan for error-checking, simulation, and
visualization purposes --- without touching real hardware. For example, we
could print each message in the plan like so:

.. code-block:: python

    plan = [Msg('open_run'), Msg('close_run')]

    # a very, very simple 'plan simulator'
    for msg in plan:
        print(msg)

A ``Msg`` has four members, accessible as attributes:

- command
- obj
- args
- kwargs

where ``command`` must be one of a controlled list of commands, ``obj`` is the
object (i.e. Device) to apply the command to, if applicable, and ``args`` and
``kwargs`` are arguments to the command.

To execute the plan, the :doc:`RunEngine <run_engine>` consumes it, one message at a time.

.. code-block:: python

    def very_simple_run_engine(plan):
        for msg in plan:
            # Process the msg.

The ``RunEngine`` has a registry which is used to dispatch the ``Msg`` objects
based on the value of the ``Msg.command``. For example, if the RunEngine
receives the message ``Msg('set', motor, 5)``, the RunEngine will:

1. Identify that the command for this message is ``'set'``.
2. Look up ``'set'`` in its command registry and find that it is mapped to
   ``RunEngine._set``.
3. Pass ``Msg('set', motor, 5)`` to its ``_set`` method.
4. Inside ``_set``, call ``motor.set(5)``. (This is where the actual
   communication with hardware occurs.)
5. Update some internal caches that will be useful later. For example, it will
   keep track of that fact that ``motor`` may be in motion so that it can stop
   it safely if an error occurs. This illustrates another important reason that
   plans must always yield messages to interact with hardware and absolutely
   never communicate with hardware directly. Calling ``epics.caput`` inside a
   plan prevents the RunEngine from knowing about it and thus circumvents
   its facilities for putting devices in a safe state in the event of an
   unexpected exit or error.

A standard set of commands are registered by default.  By convention, a ``Msg``
with the command ``'name'`` is mapped to a coroutine method on the RunEngine
named ``_name``, as in ``'set'`` -> ``RunEngine._set`` in the example above.
Users can register their own coroutines to add custom commands, though this is
very rarely necessary.

Some commands do not involve communication with hardware. For example,
``Msg('sleep', None, 5)`` causes the RunEngine to sleep for 5 seconds. ``None``
is a placeholder for the "object" (Device) which is not applicable for a
``'sleep'`` command. Just as plans should never communicate with hardware
directly, they should also never employ long blocking calls like
``time.sleep()``. Instead, the ``'sleep'`` command, mapped to
``RunEngine._sleep``, integrates with the RunEngine's event loop to sleep in a
non-blocking way that allows for the RunEngine to stay responsive in the
meantime --- watching for user interruptions and possibility collecting data
asynchronously in the background.

Other commands are used to control metadata and I/O. For example,
``Msg('open_run')`` and ``Msg('close_run')`` delineate the scope of one run.
Any keyword arguments passed to the ``'open_run'`` message are interpreted as
metadata, encoded into the RunStart document.

The following is a comprehensive overview of the built-in commands.

.. _commands:

Commands
--------

.. warning::

    This section of the documentation is incomplete.

These are the 'built in' commands, some of which are deeply tied to the
state of the `RunEngine` instance.

create
++++++

This command tells the run engine that it should start to collect the results
of ``read`` to create an event.  If this is called twice without a ``save`` or
``drop`` between them it is an exception (as you can not have more than one
open event going at a time).

This relies very heavily on the internal state of the run engine and should not
be overridden by the user.

This call returns `None` back to the co-routine.

This ignores all parts of the `Msg` except the command.

save
++++

This is the pair to ``create`` which bundles and causes ``Event`` documents to
be emitted.  This must be called after a ``create`` or a the scan will die and
raise `IllegalMessageSequence`.

This relies very heavily on the internal state of the run engine and should not
be messed with.

This call returns `None` back to the co-routine.

This ignores all parts of the `Msg` except the command.

read
++++

This causes `read` to be called on the ``obj`` in the message ::

  msg.obj.read(*msg.args, **msg.kwargs)

Anything that is read between a ``create`` and ``save`` will be bundled into
a single event.

This relies very heavily on the internal state of the run engine and should not
be messed with.

Returns the dictionary returned by `read` to the co-routine.

The ``args`` and ``kwargs`` parts of the message are passed to the `read`
method.


null
++++

This is a null message and is ignored by the run engine.  This exists to make
the algebra work.

Returns `None` to the co-routine.

Ignores all values in the `Msg` except the command.

set
+++

Tells a ``Mover`` object to move.  Currently this mimics the epics-like logic
of immediate motion.

stage and unstage
+++++++++++++++++
Instruct the RunEngine to stage/unstage the object. This calls
``obj.stage()``/``obj.unstage``.

Expected message objects are::

    Msg('stage', object)
    Msg('unstage', object)

which results in these calls::

    staged_devices = object.stage()
    unstaged_devices = object.unstage()

where ``staged_devices``/``unstaged_devices`` are a list of the
``ophyd.Device`` (s) that were (un)staged, not status objects.

One may wonder why the return is a list of Devices as opposed to Status
objects, such as in ``set`` and similar ``Msg`` s.
This was debated for awhile. Operations performed during staging are supposed
to involve twiddling configuration, and should happen fast. Staging should not
involve lengthy set calls.

Why a list of the objects staged? Staging a Device causes that Device's
component Devices (if any) to also be staged. All of these children are added
to a list, along with [self], and returned by Device.stage(), so that the plan
can keep track of what has been staged, like so::

    devices_staged = yield Msg('stage', device)

Why would the plan want to know that? It needs to avoid accidentally trying to
stage something twice, such as a staging a parent and then trying to also stage
its child. It's important to avoid that because staging something redundantly
raises an error.


trigger
+++++++

This will call the ``obj.trigger`` method and cache the returned status object
and caches the returned status object.


sleep
+++++

Sleep the event loop.

wait
++++

Block progress until every object that was triggered or set the keyword
argument `group=<GROUP>` is done.

Expected message object is:

Msg('wait', group=<GROUP>)

where ``<GROUP>`` is any hashable key.

wait_for
++++++++
Instruct the ``RunEngine`` to wait for this ``asyncio.Future`` object to be
done. This allows for external arbitrary control of the ``RunEngine``.
Ex ::

    from asyncio.futures import Future
    future = Future()
    future.done() # will give false
    RE(Msg('wait_for', [lambda : future ,]))
    # this sets the future to done
    future.set_result(3)
    future.done() # will give True


input
+++++
Process an input. Allows for user input during a run.

Examples::

    Msg('input', None)
    Msg('input', None, prompt='>')  # customize prompt


checkpoint
++++++++++

Instruct the RunEngine to create a checkpoint so that we can rewind to this
point if necessary.

clear_checkpoint
++++++++++++++++
Clear a set checkpoint.

rewindable
++++++++++

pause
+++++

Request the run engine to pause

Expected message object is::

    Msg('pause', defer=False, name=None, callback=None)


kickoff
+++++++

Start a flyscan object.

collect
+++++++

Collect data cached by a flyer and emit descriptor and event documents.
This calls the ``obj.collect()`` method.

complete
++++++++

Tell a flyer, 'stop collecting, whenever you are ready'.

This calls the method ``obj.complete()`` of the given object. The flyer returns
a status object. Some flyers respond to this command by stopping collection and
returning a finished status object immediately. Other flyers finish their given
course and finish whenever they finish, irrespective of when this command is
issued.


configure
+++++++++

Configure an object.

Expected message object is::

    Msg('configure', object, *args, **kwargs)

which results in this call::

    object.configure(*args, **kwargs)


subscribe
+++++++++
Add a subscription after the run has started.

This, like subscriptions passed to __call__, will be removed at the
end by the RunEngine.

Expected message object is:

    Msg('subscribe', None, callback_function, document_name)

where `document_name` is one of:

    {'start', 'descriptor', 'event', 'stop', 'all'}

and `callback_function` is expected to have a signature of:

    ``f(name, document)``

    where name is one of the ``document_name`` options and ``document``
    is one of the document dictionaries in the event model.

See the docstring of bluesky.run_engine.Dispatcher.subscribe() for more
information.

unsubscribe
+++++++++++

Remove a subscription during a call -- useful for a multi-run call
where subscriptions are wanted for some runs but not others.

Expected message object is::

    Msg('unsubscribe', None, TOKEN)
    Msg('unsubscribe', token=TOKEN)

where ``TOKEN`` is the return value from ``RunEngine._subscribe()``

open_run
++++++++
Instruct the RunEngine to start a new "run"

Expected message object is::

    Msg('open_run', None, **kwargs)

where ``**kwargs`` are any additional metadata that should go into the RunStart
document

close_run
+++++++++

Instruct the RunEngine to write the RunStop document

Expected message object is::

    Msg('close_run', None, exit_status=None, reason=None)

if *exit_stats* and *reason* are not provided, use the values
stashed on the RE.


drop
++++

Drop a bundle of readings without emitting a completed Event document.

This is a command that abandons previous ``create`` and ``read`` commands
without emitting an event. This can be used to drop known bad events
(e.g. no beam) and keep the event document stream clean. It is safe to start
another ``create``, ``read``, ``save`` sequence after a ``drop``.

This must be called after a ``create`` or a the scan will die and raise
`IllegalMessageSequence`.

This call returns `None` back to the co-routine.

This ignores all parts of the `Msg` except the command.


monitor
+++++++
Monitor a signal. Emit event documents asynchronously.

A descriptor document is emitted immediately. Then, a closure is
defined that emits Event documents associated with that descriptor
from a separate thread. This process is not related to the main
bundling process (create/read/save).

Expected message object is::

    Msg('monitor', obj, **kwargs)
    Msg('monitor', obj, name='event-stream-name', **kwargs)

where kwargs are passed through to ``obj.subscribe()``


unmonitor
+++++++++

Stop monitoring; i.e., remove the callback emitting event documents.

Expected message object is::

    Msg('unmonitor', obj)


stop
++++

Stop a device.

Expected message object is::

    Msg('stop', obj)

This amounts to calling ``obj.stop()``.


Registering Custom Commands
---------------------------

The RunEngine can be taught any new commands. They can be registered using the
following methods.

.. automethod:: bluesky.run_engine.RunEngine.register_command
    :noindex:

.. automethod:: bluesky.run_engine.RunEngine.unregister_command
    :noindex:

.. autoattribute:: bluesky.run_engine.RunEngine.commands
    :noindex:

.. automethod:: bluesky.run_engine.RunEngine.print_command_registry
    :noindex:
