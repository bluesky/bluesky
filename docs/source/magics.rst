*******************************
IPython 'Magics' [Experimental]
*******************************

.. warning::

    This section covers an experimental feature of bluesky. It may be altered
    or removed in the future.

What Are 'Magics'?
------------------

IPython is an interactive Python interpreter designed for and by scientists. It includes a feature called "magics" --- convenience commands that aren't part of the Python language itself. For example, ``%history`` is a magic:

.. ipython:: python

    a = 1
    b = 2
    %history

The IPython documentation documents the
`complete list of built-in magics <https://ipython.readthedocs.io/en/stable/interactive/magics.html>`_
and, further,
`how to define custom magics <https://ipython.readthedocs.io/en/stable/config/custommagics.html>`_.

Bluesky Magics
--------------

Bundled with bluesky are some IPython magics. They are intended for maintenance
tasks or casual sanity checks.  **Intentionally, none of the magics save data;
for that you should use the RunEngine and plans.**

To use the magics, first register them with IPython:

.. ipython:: python

    from bluesky.magics import BlueskyMagics
    get_ipython().register_magics(BlueskyMagics)

For this example we'll use some simulated hardware.

.. ipython:: python

    from ophyd.sim import motor1, motor2

Moving a Motor
~~~~~~~~~~~~~~

Suppose you want to move a motor interactively. You can use the ``%mov`` magic:

.. ipython:: python

    %mov motor1 42

Where ``motor1`` refers to the actual ophyd object itself.
This is equivanent to:

.. code-block:: python

    from bluesky.plan_stubs import mv

    RE(mv(motor1, 42))

but less to type. There is also a ``%movr`` magic for "relative move". They can
move multiple devices in parallel like so:

.. ipython:: python

    %mov motor1 -3 motor2 3


Note: Magics has changed from version v1.3.0 onwards. The previous method will
be described in the next section.

Taking a reading using ``%ct`` (Post v1.3.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before we may make use of the power of magics for counting, we must "label"
this hardware. To add a label, we must give hardware a ``labels={'mylabel'}``
keyword argument. For example, here we initialize five simulated signals: two
motors, a shutter motor, an area detector and a point detector:

.. ipython:: python

    import numpy as np
    from ophyd.sim import SynAxis, SynSignal
    motor1 = SynAxis(name='motor1', labels={'motors', 'scan_motors'})
    motor2 = SynAxis(name='motor2', labels={'motors', 'scan_motors'})
    shutter_motor = SynAxis(name='shutter_motor', labels={'motors', 'shutter_motors'})
    # create a fake area detector that returns a 2x2 array
    area_detector = SynSignal(func=lambda: np.random.random((2, 2)),
                              name='adet1', labels={'detectors', 'area_detectors'})
    point_detector = SynSignal(func=lambda: np.random.random((1,)),
                               name='pointdet1', labels={'detectors', 'point_detectors'})

Now we have detectors and motors, with proper labels.

Now suppose you want to take a quick reading of some devices and print the
results to the screen without saving them or doing any fancy processing. Use
the ``%ct`` magic:

.. ipython:: python

    %ct area_detectors

Where the names after count are a list of whitespace separated labels. In this
case, only ``area_detector`` will be counted.

Running ``%ct`` without arguments looks for the ``detectors`` label by default:

.. ipython:: python

    %ct

In this case, we count both on the area detector and the point detector.


Aside on the automagic feature in IPython
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If IPython’s ‘automagic’ feature is enabled, IPython will even let you drop the
``%`` as long as the meaning is unambiguous:

.. ipython:: python

    ct
    ct = 3  # Now ct is a variable so automagic will not work...
    ct
    # ... but the magic still works.
    %ct

For what it’s worth, we recommend disabling 'automagic'. The ``%`` is useful
for flagging what follows as magical, non-Python code.

Listing available motors using ``%wa`` (Post v1.3.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Finally, the ``%wa`` magic displays the a list of labeled devices.

.. ipython:: python

    %wa scan_motors

will display all motors used for a scan.
If blank, will print all labeled devices.

.. ipython:: python

    %wa

Note: It is possible to give a device more than one label. Thus it is possible
to have the same device in more than one list when calling ``%wa``. It is up to
the user to decide whether they want overlapping labels or not.

    
Comparison with SPEC
~~~~~~~~~~~~~~~~~~~~

The names of these magics, and the order of the parameters they take, are meant
to feel familiar to users of :doc:`SPEC <comparison-with-spec>`.

Again, they must be registered with IPython before they can be used:

.. code-block:: python

    from bluesky.magics import BlueskyMagics
    get_ipython().register_magics(BlueskyMagics)


Taking a reading using ``%ct`` (Pre v1.3.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Previously, you could set a default list of detectors and them use ``%ct``
without any parameters. This behaviour is deprecated. Do not use this:

.. ipython:: python
    :okwarning:

    BlueskyMagics.detectors = [area_detector, point_detector]
    %ct

This is no longer supported.

Listing available motors using ``%wa`` (Pre v1.3.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Previously, it was possible to supply a list of motors. This feature is also
deprecated. Do not use this:

.. ipython:: python
    :okwarning:

    BlueskyMagics.positioners = [motor1, motor2]
    %wa

======================================================================= ==============================
Magic                                                                   Plan Invoked
======================================================================= ==============================
``%mov``                                                                :func:`~bluesky.plan_stubs.mv`
``%movr``                                                               :func:`~bluesky.plan_stubs.mvr`
``%ct``                                                                 :func:`~bluesky.plans.count`
``%wa``                                                                 ("where all") Survey positioners*
======================================================================= ==============================
