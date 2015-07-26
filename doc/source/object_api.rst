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

      This is used implicitly by the default event bundling code in
      the run engine.

   .. py:method:: read(*args, **kwargs):

      Read the data from the object.  The ``*args`` and ``**kwargs``
      can be passed in to the runengine to control any special
      details specific to this object.

      The return value must be a `dict` and must conform to the schema
      defined by the :py:meth:`Reader.describe`

      This is used explicitly via the ``read`` message.

   .. py:method:: trigger(*args, **kwargs)

      Trigger the reader to begin taking data.  So that the event loop
      can run this should be non-blocking.

      This must return a an object with a ``callback`` attribute.  The
      callback must be called when the Reader is ready and must be
      called on assignment if the Reader was ready before the callback
      value was assigned.  This is used to implement block groups.

      This is used explicitly via the ``trigger`` message.

   .. py:method:: configure(*args, **kwargs)

      Setup detectors which have complicated
      state or may need change the settings of a detector from a 'live'
      mode to a 'collection' mode.

      If this method is defined, `deconfigure` must also be defined.

      There is no required return for this function.

      This method is explicitly called in the ``configure`` message.

    .. py:method:: deconfigure(*args, **kwargs)

       Remove reader from 'collection' state and return to 'live'
       state.

       There is no required return.

       This method is called explicitly in the ``deconfigure`` message.

       This method is called implicitly on exiting the run engine
       if the ``configure`` message has been used and ``deconfigure``
       message has not.



The objects can have any other methods or attributes required for
their operation or easy of use.  These objects do not need to
sub-class any particular base class, only have these methods.

These objects can be used with ``Read``, ``Trigger``, ``Wait``, and
``Describe``,  ``configure``, and ``deconfigure``  `Msg` objects.


Mover API
---------

The required functions and attributes are


.. py:class:: Mover

   .. py:method:: describe()

      Describe the data that will be returned by
      :py:meth:`Mover.read`.  The schema must match the ``DataKey``
      schema from metadatastore.

      This is used implicitly by the default event bundling code in
      the run engine.

   .. py:method:: read(*args, **kwargs):

      Read the data from the object.  The ``*args`` and ``**kwargs``
      can be passed in to the runengine to control any special
      details specific to this object.

      The return value must be a `dict` and must conform to the schema
      defined by the :py:meth:`Mover.describe`

      This is used explicitly via the ``read`` message.

   .. py:method:: trigger(*args, **kwargs)

      Trigger the reader to begin taking data.  So that the event loop
      can run this should be non-blocking.

      This must return a an object with a ``callback`` attribute.  The
      callback must be called when the Mover is ready and must be
      called on assignment if the Mover was ready before the callback
      value was assigned.  This is used to implement block groups.

      This is used explicitly via the ``trigger`` message.

      This is a trigger on reading, not starting the motion.

   .. py:method:: trigger(*args, **kwargs)

      Trigger the reader to begin taking data.  So that the event loop
      can run this should be non-blocking.

      This must return an object with a ``callback`` attribute.  The
      callback must be called when the Mover is ready and must be
      called on assignment if the Mover was ready before the callback
      value was assigned.  This is used to implement block groups.

      This is used explicitly via the ``trigger`` message.

   .. py:method:: set(*args, **kwargs)

      Set the target position and start moving.

      This must return an object with a ``callback`` attribute.  The
      callback must be called when the Mover is at the target position
      and must be called on assignment if the Mover was ready before
      the callback value was assigned.  This is used to implement
      block groups.

      This is used explicitly via the ``set`` message.

      This method maybe split into two steps (set target, start motion)
      in the future.


The objects can have any other methods or attributes required for
their operation or easy of use.  These objects do not need to
sub-class any particular base class, only have these methods.

These objects can be used with ``Read``, ``trigger``, ``Set``,
``Wait``, and ``Describe`` `Msg`.


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
      :py:meth:`Flyer.collect`.  The schema must match the
      ``DataKey`` schema from metadatastore.

      This is used implicitly by the default event bundling code in
      the run engine.

   .. py:method:: collect()

      Collects many events from the `Flyer`.  This function must return an
      iterable that will yield event `dict` that  conform to the schema
      defined by the :py:meth:`Flyer.describe`.

      This is an analogue of `Reader.read`

      This is explictily used by the ``Collect`` message

   .. py:method:: kickoff()


      Start the flyscan.  So that the event loop can run this should
      be non-blocking.

      This must return an object with a ``callback`` attribute.  The
      callback must be called when the flyscan is ready to be collected or
      called on assignment if the flyscan was ready before the callback
      value was assigned.  This is used to implement block groups.

      This is used explicitly via the ``Kickoff`` message.


The objects can have any other methods or attributes required for their operation
or ease of use.  These objects do not need to sub-class any particular base class,
only have these methods.

These objects can be used with ``Kickoff``, ``Collect``, and ``Wait`` `Msg`.

Eventually this API will be modified to enable incremental collection
of events.
