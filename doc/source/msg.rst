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

The execute the plan, the RunEngine consumes it, one message at a time.

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
   it safely if an error occurs.

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
Any keyword arguments to passed the ``'open_run'`` message are interpreted as
metadata, encoded into the RunStart document.

The following is a comprehensive overview of the built-in commands.

.. _commands:

Commands
--------

.. warning::

    This section of the documentation is incomplete.

These are the 'built in' commands, some of which are deeply tied to the
state of the `RunEnigne` instance.

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
of immediate motion

trigger
+++++++

sleep
+++++

wait
++++

checkpoint
++++++++++

pause
+++++

collect
+++++++

kickoff
+++++++

drop
++++

This is a command that abandons previous ``create`` and ``read`` commands
without emitting an event. This can be used to drop known bad events
(e.g. no beam) and keep the event document stream clean. It is safe to start
another ``create``, ``read``, ``save`` sequence after a ``drop``.

This must be called after a ``create`` or a the scan will die and raise
`IllegalMessageSequence`.

This call returns `None` back to the co-routine.

This ignores all parts of the `Msg` except the command.

Registering Custom Commands
---------------------------

The RunEngine can be taught any new commands. They can be registered using the
following methods.

.. automethod:: bluesky.run_engine.RunEngine.register_command
    :noindex:

.. automethod:: bluesky.run_engine.RunEngine.unregister_command
    :noindex:
