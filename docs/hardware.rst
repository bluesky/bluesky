How Bluesky Interfaces with Hardware
====================================

.. _hardware_interface:

Overview
--------

Bluesky interacts with hardware through a high-level abstraction, leaving the
low-level details of communication as a separate concern. In bluesky's view,
*all* devices are in a sense "detectors," in that they can be read. A subset
of these devices are "positioners" that can also be set (i.e., written to or
moved).

In short, each device is represented by a Python object that has attributes and
methods with certain established names. We have taken pains to make this
interface as slim as possible, while still being general enough to address
every kind of hardware we have encountered.

.. seealso::
    :ref:`hardware_interface_packages`

Specification
-------------

.. _status_obj_api:

Status object
+++++++++++++

The interface of a "status" object, which the ``RunEngine`` uses to
asynchronously monitor the compeletion of having triggered or set a device.

.. autoclass:: bluesky.protocols.Status
   :members:
   :undoc-members:

If ``success`` is ``False`` when the Status is marked done, this is taken
to mean, "We have given up." For example, "The motor is stuck and will
never get where it is going." A ``FailedStatus`` exception will be raised
inside the RunEngine.

Additionally, ``Status`` objects may (optionally) add a watch function that
conforms to the following definition

   .. method:: watch(func)

        Subscribe to notifications about progress. Useful for progress bars.

        **Parameters**

        func : callable
            Expected to accept the keyword arguments:

                * ``name``
                * ``current``
                * ``initial``
                * ``target``
                * ``unit``
                * ``precision``
                * ``fraction``
                * ``time_elapsed``
                * ``time_remaining``

            Any given call to ``func`` may only include a subset of these
            parameters, depending on what the status object knows about its own
            progress.

            The ``fraction`` argument accepts a single float representing fraction
            remaining.
            A fraction of zero indicates completion.
            A fraction of one indicates progress has not started.


Named Device
++++++++++++

Some of the interfaces below require a ``name`` attribute, they implement this
interface:

.. autoclass:: bluesky.protocols.HasName
    :members:
    :show-inheritance:

Some of also require a ``parent`` attribute, they implement this interface:

.. autoclass:: bluesky.protocols.HasParent
    :members:
    :show-inheritance:


Readable Device
+++++++++++++++

To produce data in a step scan, a device must be Readable:

.. autoclass:: bluesky.protocols.Readable
    :members:
    :show-inheritance:

A dict of stream name to Descriptors is returned from :meth:`describe`, where a
`Descriptor` is a dictionary with the following keys:

.. autoclass:: bluesky.protocols.DataKey
    :members:

A dict of stream name to Reading is returned from :meth:`read`, where a
`Reading` is a dictionary with the following keys:

.. autoclass:: bluesky.protocols.Reading
    :members:

The following keys can optionally be present in a `Reading`:

.. autoclass:: bluesky.protocols.ReadingOptional
    :members:

If the device has configuration that only needs to be read once at the start of
scan, the following interface can be implemented:

.. autoclass:: bluesky.protocols.Configurable
    :members:
    :show-inheritance:

If a device needs to do something before it can be read, the following interface
can be implemented:

.. autoclass:: bluesky.protocols.Triggerable
    :members:
    :show-inheritance:


External Asset Writing Interface
++++++++++++++++++++++++++++++++

Devices that write their data in external files, rather than returning directly
from ``read()`` should implement the following interface:

.. autoclass:: bluesky.protocols.WritesExternalAssets
    :members:
    :show-inheritance:

The yielded values are a tuple of the document type and the document as a dictionary.

A Resource will be yielded to show that data will be written to an external resource
like a file on disk:

.. autoclass:: bluesky.protocols.PartialResource
    :members:

While a Datum will be yielded to specify a single frame of data in a Resource:

.. autoclass:: bluesky.protocols.Datum
    :members:

.. seealso:: https://blueskyproject.io/event-model/external.html


Movable (or "Settable") Device
++++++++++++++++++++++++++++++

The interface of a movable device extends the interface of a readable device
with the following additional methods and attributes.

.. autoclass:: bluesky.protocols.Movable
    :members:
    :show-inheritance:

    .. attribute:: position

        A optional heuristic that describes the current position of a device as
        a single scalar, as opposed to the potentially multi-valued description
        provided by ``read()``.

        .. note::

            The position attribute has been deprecated in favour of the
            Locatable protocol below

Certain plans like :func:`~bluesky.plan_stubs.mvr` would like to know where a
Device was last requested to move to, and other plans like
:func:`~bluesky.plan_stubs.rd` would like to know where a Device is currently
located. Devices may implement ``locate()`` to provide this information.

.. autoclass:: bluesky.protocols.Locatable
    :members:
    :show-inheritance:

``Location`` objects are dictionaries with the following entries:

.. autoclass:: bluesky.protocols.Location
    :members:


"Flyer" Interface
+++++++++++++++++

*For context on what we mean by "flyer", refer to the section on* :doc:`async`.

The interface of a "flyable" device is separate from the interface of a readable
or settable device, though there is some overlap.

.. autoclass:: bluesky.protocols.Flyable
    :members:
    :show-inheritance:

The yielded values from ``collect()`` are partial Event dictionaries:

.. autoclass:: bluesky.protocols.PartialEvent
    :members:

If any of the data keys are in external assets rather than including the data,
a ``filled`` key should be present:


Flyable devices can also implement :class:`Configurable` if they have
configuration that only needs to be read once at the start of scan


Optional Interfaces
-------------------

These are additional interfaces for providing *optional* behavior to ``Readable``, ``Movable``,
and ``Flyable`` devices.

The methods described here are either hooks for various plans/RunEngine messages which are
ignored if not present or required by only a subset of RunEngine messages.
In the latter case, the RunEngine may error if it tries to use a device which does not define
the required method.

.. autoclass:: bluesky.protocols.Stageable
    :members:
    :show-inheritance:

.. autoclass:: bluesky.protocols.Subscribable
    :members:
    :show-inheritance:

.. autoclass:: bluesky.protocols.Pausable
    :members:
    :show-inheritance:

.. autoclass:: bluesky.protocols.Stoppable
    :members:
    :show-inheritance:

.. autoclass:: bluesky.protocols.Checkable
    :members:
    :show-inheritance:

.. autoclass:: bluesky.protocols.HasHints
    :members:
    :show-inheritance:

.. autoclass:: bluesky.protocols.Hints
    :members:

.. autoclass:: bluesky.protocols.Preparable
   :members:
   :show-inheritance:

Checking if an object supports an interface
-------------------------------------------

You can check at runtime if an object supports an interface with ``isinstance``:

.. code-block:: python

    from bluesky.protocols import Readable

    assert isinstance(obj, Readable)
    obj.read()

This will check that the correct methods exist on the object, and are callable,
but will not check any types.

There is also a helper function for this:

.. autofunction:: bluesky.protocols.check_supports
