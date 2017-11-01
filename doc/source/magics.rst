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

    from ophyd.sim import det1, det2, motor1, motor2

Suppose you want to move a motor interactively. You can use the ``%mov`` magic:

.. ipython:: python

    %mov motor1 42

This is equivanent to:

.. code-block:: python

    from bluesky.plan_stubs import mv

    RE(mv(motor1, 42))

but less to type. There is also a ``%movr`` magic for "relative move". They can
move multiple devices in parallel like so:

.. ipython:: python

    %mov motor1 -3 motor2 3

Now suppose you want to take a quick reading of some devices and print the
results to the screen without saving them or doing any fancy processing. Use
the ``%ct`` magic:

.. ipython:: python

    %ct [det1, det2]

or, equivalently,

.. ipython:: python

    dets = [det1, det2]
    %ct dets

You can set a default list of detectors and them use ``%ct`` without any
parameters:

.. ipython:: python

    BlueskyMagics.detectors = [det1, det2]
    %ct

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

Finally, the ``%wa`` magic displays the current positions of movable
devices. Like ``%ct``, it accepts a list of devices

.. ipython:: python

    %wa [motor1, motor2]
    
or, if blank, falls back on a defaults list configured like so:

.. ipython:: python

    BlueskyMagics.positioners = [motor1, motor2]
    %wa

The names of these magics, and the order of the parameters they take, are meant
to feel familiar to users of :doc:`SPEC <comparison-with-spec>`.

Again, they must be registered with IPython before they can be used:

.. code-block:: python

    from bluesky.magics import BlueskyMagics
    get_ipython().register_magics(BlueskyMagics)

======================================================================= ==============================
Magic                                                                   Plan Invoked
======================================================================= ==============================
``%mov``                                                                :func:`~bluesky.plan_stubs.mv`
``%movr``                                                               :func:`~bluesky.plan_stubs.mvr`
``%ct``                                                                 :func:`~bluesky.plans.count`
``%wa``                                                                 ("where all") Survey positioners*
======================================================================= ==============================
