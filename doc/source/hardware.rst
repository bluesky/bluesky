How Bluesky Interface to Hardware
=================================

Bluesky interacts with hardware through a high-level abstraction, leaving the
low-level details of communication as a separate concern. In bluesky's view,
*all* devices are in a sense "detectors," in that they can be read. A subset
of these devices are "positioners" that can also be set (i.e., written to or
moved).

In short, each device is represented by a Python object that has attributes and
methods with certain established names. We have taken pains to make this
interface as slim as possible, while still being general enough to address
every kind of hardware we have encountered.

A working reference implementation of a ``Reader`` (e.g., a detector) and a
``Mover`` (e.g., a motor or temperature controller) are in the
``bluesky.examples`` module. These implementations act as "simulated hardware,"
and we use them extensively in examples, demos, and the test suite.

The `ophyd
<https://nsls-ii.github.io/ophyd>`_ package implements this interface for
*real* hardware, communicating via `EPICS <http://www.aps.anl.gov/epics/>`_.
Other control systems (Tango, LabView, etc.) could be integrated with bluesky
in the future by implementing this same interface.
