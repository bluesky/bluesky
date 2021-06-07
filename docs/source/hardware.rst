How Bluesky Interfaces with Hardware
====================================

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

Readable Device
+++++++++++++++

The interface of a readable device:

.. autoclass:: bluesky.protocols.Readable
    :members:
    :undoc-members:

Movable (or "Settable")  Device
+++++++++++++++++++++++++++++++

The interface of a movable device extends the interface of a readable device
with the following additional methods and attributes.


.. autoclass:: bluesky.protocols.Movable
    :members:
    :show-inheritance:

    .. attribute:: position

        A heuristic that describes the current position of a device as a
        single scalar, as opposed to the potentially multi-valued description
        provided by ``read()``.

        Optional: bluesky itself does not use the position attribute, but other
        parts of the ecosystem might.
        Developers are encouraged to implement this attribute where possible.


"Flyer" Interface
+++++++++++++++++

*For context on what we mean by "flyer", refer to the section on :doc:`async`.*

The interface of a "flyable" device is separate from the interface of a readable
or settable device, though there is some overlap.


.. autoclass:: bluesky.protocols.Flyable
    :members:
    :undoc-members:


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

.. autoclass:: bluesky.protocols.Subscribable
    :members:

.. autoclass:: bluesky.protocols.Pausable
    :members:

.. autoclass:: bluesky.protocols.Stoppable
    :members:

.. autoclass:: bluesky.protocols.Checkable
    :members:

.. autoclass:: bluesky.protocols.Hinted
    :members:


Implementations
---------------

Real Hardware
+++++++++++++

The `ophyd
<https://nsls-ii.github.io/ophyd>`_ package implements this interface for
a wide variety of hardware, communicating using
`EPICS <http://www.aps.anl.gov/epics/>`_ via the Python bindings
`pyepics <http://cars9.uchicago.edu/software/python/pyepics3/>`_.Other control
systems (Tango, LabView, etc.) could be integrated with bluesky in the future
by implementing this same interface.

Simulated Hardware
++++++++++++++++++

A toy "test" implementation the interface is included in the
:mod:`ophyd.sim` module. These implementations act as simulated hardware,
and we use them extensively in examples, demos, and the test suite. They can
also be useful for exercising analysis workflows before running a real
experiment. API documentation is below.
