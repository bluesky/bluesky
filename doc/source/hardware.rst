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

.. class:: Status:

    .. attribute:: done

        boolean

    .. attribute:: success

        boolean

    If ``success`` is ``False`` when the Status is marked done, this is taken
    to mean, "We have given up." For example, "The motor is stuck and will
    never get where it is going." A ``FailedStatus`` exception will be raised
    inside the RunEngine.

    .. attribute:: finished_cb

        a callback function that ``Status`` will call when it is marked done.

    It may be that ``Status`` is done before a function has been attached to
    ``finished_cb``. In that case, the function should be called as soon as it
    is attached.

    .. method:: watch(func)

        Subscribe to notifications about progress. Useful for progress bars.

        Parameters
        ----------
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

.. class:: ReadableDevice

    .. attribute:: name

        a human-readable string identifying the device

    .. attribute:: parent

        ``None``, or a reference to a parent device

        See the ``stage`` method below for the operational signifance of
        ``parent``.

    .. method:: read()

        Return an OrderedDict mapping field name(s) to values and timestamps.
        The field names must be strings. The values can be any JSON-encodable
        type or a numpy array, which the RunEngine will convert to (nested)
        lsits. The timestamps should be UNIX time (seconds since 1970).

        Example return value:

        .. code-block:: python

            OrderedDict(('channel1',
                         {'value': 5, 'timestamp': 1472493713.271991}),
                         ('channel2',
                         {'value': 16, 'timestamp': 1472493713.539238}))


    .. method:: describe()

        Return an OrderedDict with exactly the same keys as the ``read``
        method, here mapped to metadata about each field.

        Example return value:

        .. code-block:: python

            OrderedDict(('channel1',
                         {'source': 'XF23-ID:SOME_PV_NAME',
                          'dtype': 'number',
                          'shape': []}),
                        ('channel2',
                         {'source': 'XF23-ID:SOME_PV_NAME',
                          'dtype': 'number',
                          'shape': []}))

        We refer to each entry as a "data key." These fields are required:

        * source (a descriptive string --- e.g., an EPICS Process Variable)
        * dtype: one of the JSON data types: {'number', 'string', 'array'}
        * shape: ``None`` or a list of dimensions --- e.g., ``[5, 5]`` for a
          5x5 array

        Optional additional fields (precision, units, etc.) are allowed.
        The optional field ``external`` should be used to provide information
        about references to externally-stored data, such as large image arrays.

    .. method:: trigger()

        Return a ``Status`` that is marked done when the device is done
        triggering.

        If the device does not need to be triggered, simply return a ``Status``
        that is marked done immediately.

    .. method:: read_configuration()

        Same API as ``read`` but for slow-changing fields related to
        configuration (e.g., exposure time). These will typically be read only
        once per run.

        Of course, for simple cases, you can effectively omit this complexity
        by returning an empty dictionary.

    .. method:: describe_configuration()

        Same API as ``describe``, but corresponding to the keys in
        ``read_configuration``.

    .. attribute:: hints

        A dictionary of suggestions for best-effort visualization and
        processing. This does not affect what data is read or saved; it is only
        a suggestion to enable automated tools to provide helpful information
        with minimal guidance from the user. See :ref:`hints`.

    .. method:: configure(*args, **kwargs)

        This can change the device's configuration in an arbitrary way. When
        the RunEngine calls this method, it also emits a fresh Event Descriptor
        because it assumes that the configuration in the previous Event
        Descriptor might no longer be valid.

        Returns a tuple of the *old* result of ``read_configuration()`` and the
        *new* result of ``read_configuration()``.

    *This concludes the required API. The following are optional.*

    .. method:: stage()

        An optional hook for "setting up" the device for acquisition.

        It should return a list of devices including itself and any other
        devices that are staged as a result of staging this one.
        (The ``parent`` attribute expresses this relationship: a device should
        be staged/unstaged whenever its parent is staged/unstaged.)

    .. method:: unstage()

        A hook for "cleaning up" the device after acquisition.

        It should return a list of devices including itself and any other
        devices that are unstaged as a result of unstaging this one.

    .. method:: subscribe(function)

        Optional, needed only if the device will be :doc:`monitored <async>`.

        When the device has a new value ready, it should call ``function``
        asynchronously in a separate thread.

    .. method:: clear_sub(function)

        Remove a subscription. (See ``subscribe``, above.)

    .. method:: pause()

        An optional hook to do some device-specific work when the RunEngine
        pauses.

    .. method:: resume()

        An optional hook to do some device-specific work when the RunEngine
        resumes after a pause.


Settable (Movable) Device
+++++++++++++++++++++++++

The interface of a settable device extends the interface of a readable device
with the following additional methods and attributes:

.. class:: SettableDevice:

    .. method:: stop()

        Safely stop a device that may or may not be in motion.

    .. method:: set(*args, **kwargs)

        Return a ``Status`` that is marked done when the device is done
        moving.

    .. attribute:: position

        a heuristic that describes the current position of a device as a
        single scalar, as opposed to the potentially multi-valued description
        provided by ``read()``

"Flyer" Interface
+++++++++++++++++

*For context on what we mean by "flyer", refer to the section on :doc:`async`.*

The interace of a "flyable" device is separate from the interface of a readable
or settable device, though there is some overlap.

.. class:: FlyableDevice:

    .. method:: kickoff()

       Begin acculumating data. Return a ``Status`` and mark it done when
       acqusition has begun.

    .. method:: complete()

       Return a ``Status`` and mark it done when acquisition has completed.

    .. method:: collect()

        Yield dictionaries that are partial Event documents. They should
        contain the keys 'time', 'data', and 'timestamps'. A 'uid' is added by
        the RunEngine.

    .. method:: describe_collect()

        This is like ``describe()`` on readable devices, but with an extra
        layer of nesting. Since a flyer can potentially return more than one
        event stream, this is a dict of stream names (strings) mapped to a
        ``describe()``-type output for each.

    *The remaining methods and attributes match ReadableDevice.*

    .. method:: configure(*args, **kwargs)

        same as for a readable device

    .. method:: read_configuration()

        same as for a readable device

    .. method:: describe_configuration()

        same as for a readable device

    .. attribute:: name

        same as for a readable device

    .. attribute:: parent

        same as for a readable device

    .. method:: stage()

        optional, same as for a readable device

    .. method:: unstage()

        optional, same as for a readable device

    .. method:: pause()

        optional, same as for a readable device

    .. method:: resume()

        optional, same as for a readable device

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
