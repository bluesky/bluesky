.. _msg:

Message Protocol
================

*Note: This is a technical document not optimized for user readability.*

Overview
--------

The `RunEngine` oversees all commands of a run, intelligently rewinding and
re-running commands during a run in order to produce a clean result. How does
it work? On the basic level, it works by storing a list of atomistic operations
to be run, and running them in sequence. Each of these atomistic operations are
represented by a `bluesky.Msg` object. Thus, every plan designed for use by the
`RunEngine` is a list of `Msg` objects. To fully capture the provenance of a
command, four items are needed:

- command
- obj
- args
- kwargs

where ``command`` must be one of a controlled list of commands, ``obj`` is the
object to apply the command to and ``args`` and ``kwargs`` are arguments to the
command. For convenience, the `bluesky.Msg` object subclasses the `namedtuple`
class so that these members may be accessed as attributes (ie.  ``msg.command``
where ``msg`` is an instance of `bluesky.Msg`). Any ``args`` or ``kwargs`` not
consumed by the run engine are passed through to the calls on the objects.

The `RunEngine` has a registry which is used to dispatch the `Msg` objects
based on the value of the `Msg.cmd`. By default a basic set of commands are
registered, but users can register their own functions to add custom commands.
As of the time of writing of this code, the convention currently used is that
for any message string ``'name'`` is a reference to the `RunEngine` command
``RE._name(obj, *args, **kwargs)`` where ``RE`` is an instance of
``RunEngine``.

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
