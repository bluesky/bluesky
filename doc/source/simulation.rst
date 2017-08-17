.. currentmodule:: bluesky.simulators

Simulation and Error Checking
=============================

Bluesky provides three different approaches for simulating a plan without
actually executing it:

1. Introspect a plan by passing it to a "simulator" instead of a RunEngine.
2. Execute a plan with the real RunEngine, but use simulated hardware objects.
3. Redefine the RunEngine commands to change their meanings.

Approaches (1) and (2) are the most straightforward and most common.

Introspection
-------------

Recall that plans yield messages that *describe* what should be done; they
do not communicate with hardware directly. Therefore it's easy to use (or
write) a simple function that iterates through the plan and summarizes or
analyzes its actions.

Summarize
^^^^^^^^^

The simulator :func:`summarize_plan` print a summary of what a plan would do if
executed by the RunEngine.

.. ipython:: python

    from bluesky.simulators import summarize_plan
    from bluesky.examples import det, motor
    from bluesky.plans import scan
    summarize_plan(scan([det], motor, 1, 3 ,3))

To see the unabridged contents of a plan, simply use the builtin Python
function :func:`list`. Note that it is not possible to summarize plans that
have adaptive logic because their contents are determined dynamically during
plan executation.

.. ipython:: python

    list(scan([det], motor, 1, 3 ,3))

Check Limits
^^^^^^^^^^^^

.. ipython:: python
    :suppress:

    motor.limits = (-1000, 1000)

Suppose that this motor is configured with limits on its range of motion at +/-
1000. The :func:`check_limits` simulator can verify whether or not a plan will
violate these limits, saving you from discovering this part way through a long
experiment.

.. ipython:: python
    :okexcept:

    from bluesky.simulators import check_limits

    check_limits(scan([det], motor, 1, 3 ,3))  # no problem here
    check_limits(scan([det], motor, 1, -3000, 3000))  # should raise an error

Simulated Hardware
------------------

.. warning::

    In a future release of bluesky, we plan to base the simulated hardware code
    more closely on the code that deals with real hardware in
    `ophyd <https://nsls-ii.github.io/ophyd>`_, and we may break
    backward-compatibility in doing so. In the event that objects from ophyd
    and these simulated hardware objects behave differently, ophyd is correct.

It's easy to mock up an object that simulates a detector or a motor. The
classes :class:`bluesky.examples.Reader` and :class:`bluesky.examples.Mover`
provide scaffolds for quickly building objects that behave like a detector or
a "settable" device such as a motor or a temperature controller.

A simluated detector needs a dictionary mapping field names to functions that
return each field's current value. In this simple example, the detector has one
field named 'intensity', whose value is always 1.

.. code-block:: python

    {'intensity': lambda: 1}

The detector also needs a name. We'll use 'd'.

.. code-block:: python

    d = Reader('d', {'intensity': lambda: 1})

Sometimes the field name matches the overall name, so we have fewer names to
remember.

.. code-block:: python

    d = Reader('d', {'d': lambda: 1})

Other times it's useful to disambiguate. Suppose a detector gives us two
readings:

.. code-block:: python

    d = Reader('d', {'d_right': lambda: 1, 'd_left': lambda: -1})

This example adds some Gaussian noise:

.. code-block:: python

    import numpy as np
    d = Reader('d', {'d': lambda: 1 + np.random.randn()})

Each time this simulated detector is triggered, the function is called and a
new random value is generated. This value is returned when the detector is
read.

A 'settable' device is defined similarly but with an important difference.
Above, the functions took the form ``lambda: ...`` with no arguments. Here,`
the functions take one or more arguments, like so:

.. code-block:: python

    {'x': lambda x: x}

When the device is 'set' to, say, 5, the function ``lambda x: x`` will be
called with the argument ``5``. The result (``5``, of course) will be cached
and then returned whenever the device is read.

Finally, we need to initialize the device to some value. (Suppose it is read
before it is set!) We provide a second dictionary with the same keys as the
first, mapped to the function argument(s) for initialization. Altogether:

.. code-block:: python

    m = Mover('m', {'x': lambda x: x}, {'x': 0})

For a less trivial example, here's a device that always goes +0.1 compared to
where it is told to go. Also note that its initial position will be 0.1, not 0.

.. code-block:: python

    m = Mover('m', {'x': lambda x: x + 0.1}, {'x': 0})

And here's a version of that device that reports both its setpoint and its
actual ("readback") position.

.. code-block:: python

    m = Mover('m', {'x_readback', lambda x: x + 0.1,
                    'x_setpoint': lambda x: x},
                   {'x_setpoint': 0, 'x_readback': 0})

It is easy to define complex, multiaxis devices.

.. code-block:: python

    m = Mover('m', {'x': lambda x: x, 'y': lambda y: y, 'z': lambda z: z},
              {'x': -1, 'y': 0, 'z': 100})

It's also easy to write a simulated detector whose value depends on the
position of a simulated motor.

.. code-block:: python

    # The 'intensity' of d is a linear function of the 'x' position of m.
    m = Mover('m', {'x': lambda x: x}, {'x': 0})
    d = Reader('d', {'intensity': lambda: 2 * m.read()['x']})
    
Customizing RunEngine Methods
-----------------------------

The RunEngine allows you to customize the meaning of commands (like 'set' and
'read'). One could use this feature to create a dummy RunEngine that, instead
of actually reading and writing to hardware, merely reports what it *would*
have done.

.. automethod:: bluesky.run_engine.RunEngine.register_command
.. automethod:: bluesky.run_engine.RunEngine.unregister_command
