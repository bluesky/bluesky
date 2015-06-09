Expected API of Hardware Objects
================================

The `BlueSky` run engine code is completely independent of the details
of how the interaction with the hardware works, but the objects passed
to it must have a minimum api.  There are three rough categories of
objects, ``Reader`` objects which can be asked for a set of values,
``Mover`` objects which are ``Reader`` objects which can be instructed
to step-wise change their state and ``Flyer`` objects which can be
instructed to start a fly-scan.


Reader API
----------

The required functions and attributes are


.. py:class:: Reader

   .. py:method:: describe()

      Describe the data that will be returned by
      :py:meth:`Reader.read`.  The schema must match the ``DataKey``
      schema from metadatastore.

   .. py:method:: read(*args, **kwargs):

      Read the data from the object.  The ``*args`` and ``**kwargs``
      can be passed in to the runengine to control any special
      details specific to this object.

      The return value must be a `dict` and must conform to the schema
      defined by the :py:meth:`Reader.describe`

   .. py:method:: trigger(*args, **kwargs)

      Trigger the reader to begin taking data.  So that the event loop
      can run this should be non-blocking.

      This is not required to return anything.

   .. py:attribute:: ready

      If the :py:class:`Reader` is ready to be read.  This is used by
      the ``Wait`` messages.  This must be `True` only when the
      :py:class:`Reader` is fully done and `False` otherwise.  Beware
      if not setting this `False` soon enough and allowing loops to
      over-run data collection.


The objects can have any other methods or attributes required for their operation
or easy of use.  These objects do not need to sub-class any particular base class,
only have these methods.

These objects can be used with ``Read``, ``Trigger``, ``Wait``, and
``Describe`` `Msg`.


Mover API
---------

The required functions and attributes are


.. py:class:: Mover

   .. py:method:: describe()

      Describe the data that will be returned by
      :py:meth:`Mover.read`.  The schema must match the ``DataKey``
      schema from metadatastore.

   .. py:method:: read(*args, **kwargs):

      Read the data from the object.  The ``*args`` and ``**kwargs``
      can be passed in to the runengine to control any special
      details.

      The return value must be a `dict` and must conform to the schema
      defined by the :py:meth:`Mover.describe`

   .. py:method:: set(*args, **kwargs)

      Set the 'position' of the `Mover`.  The exact meaning of the ``*args``
      and ``**kwargs`` is left to the

   .. py:attribute:: ready

      If the :py:class:`Mover` is ready to be read.  This is used by
      the ``Wait`` messages.  This must be `True` only when the
      :py:class:`Mover` is fully done and `False` otherwise.  Beware
      if not setting this `False` soon enough and allowing loops to
      over-run motion.


The objects can have any other methods or attributes required for their operation
or easy of use.  These objects do not need to sub-class any particular base class,
only have these methods.

These objects can be used with ``Read``, ``Set``, ``Wait``, and ``Describe`` `Msg`.


Flyer API
---------

This is the first pass at implementing a generic interface to fly
scans.  It only supports the work flow of :

1. start the scan
2. check if it is done
3. when done collect all of the data

In the future this might be extended to allow for partial collection
of data and a way to stop or pause a running fly scan.

The required functions and attributes are


.. py:class:: Flyer

   .. py:method:: describe()

      Describe the data that will be returned by
      :py:meth:`Flyer.read`.  The schema must match the ``DataKey``
      schema from metadatastore.

   .. py:attribute:: ready

      If the :py:class:`Flyer` is ready to be collected.  This is used by
      the ``Wait`` messages.  This must be `True` only when the
      :py:class:`Mover` is fully done and `False` otherwise.  Beware
      of not setting this `False` soon enough and allowing loops to
      over-run motion.


The objects can have any other methods or attributes required for their operation
or ease of use.  These objects do not need to sub-class any particular base class,
only have these methods.

These objects can be used with ``Kickoff``, ``Collect``, and ``Wait`` `Msg`.
