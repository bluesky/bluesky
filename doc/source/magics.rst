*******************************
Writing Custom IPython 'Magics'
*******************************

This section is not about bluesky itself; it highlights a feature of
IPython.

Bluesky is designed to be usable in an interactive session and also as a
library for building higher-level tools, such as a Graphical User Interface.
Presently, there is no officially-supported GUI for bluesky. (We may provide
tools for building bluesky GUIs in the future.) There is, however, a way to
build a terse command-line interface on top of bluesky using a feature of
IPython.

IPython is an interactive Python interpreter designed for and by scientists. It
includes a feature called "magics" --- convenience commands that aren't part of
the Python language itself. For example, ``%history`` is a magic:

.. ipython:: python

    a = 1
    b = 2
    %history

The IPython documentation documents the
`complete list of built-in magics <https://ipython.readthedocs.io/en/stable/interactive/magics.html>`_
and, further,
`how to define custom magics <https://ipython.readthedocs.io/en/stable/config/custommagics.html>`_.

What follows are some examples for defining IPython magics for bluesky and some
suggestions on using them effectively. We emphasize that these examples are not
part of the official bluesky interface; they merely illustrate a reasonable
pattern for user customization.

Suppose you have imported the some plans, defined a RunEngine ``RE``, and
created a list of detectors ``d``.

.. code-block:: python

    from bluesky import RunEngine
    from bluesky.examples import det1, det2, motor  # simulated hardware
    from bluesky.plans import count, mv

    RE = RunEngine({})
    d = [det1, det2]

.. ipython:: python
    :suppress:

    from bluesky import RunEngine
    RE = RunEngine({})
    from bluesky.examples import det1, det2, motor  # simulated hardware
    d = [det1, det2]
    from bluesky.plans import count, mv


And suppose that, in typical interactive use, you often take a reading from
these detectors, move a motor, and repeat.

.. ipython:: python

    RE(count(d))
    RE(mv(motor, 3))

We can define IPython magics for these commands (see script below) to create
shortcuts, such as:

.. ipython:: python
    :suppress:

    import ast
    from IPython.core.magic import register_line_magic
    import bluesky.plans as bp
    def ct(line):
        global d
        global RE
        print('---> RE(count(d))')
        return RE(bp.count(d))

    register_line_magic(ct)
    def mov(line):
        global RE
        motor_varname, pos = line.split()
        motor = globals()[motor_varname]
        pos = ast.literal_eval(pos)
        print('---> RE(mv({motor}, {pos}))'.format(motor=motor_varname, pos=pos))
        return RE(bp.mv(motor, pos))

    register_line_magic(mov)
    del ct, mov

.. ipython:: python

    %ct

.. ipython:: python

    %mov motor 3

IPython's 'automagic' will even let you drop the ``%`` as long as the meaning
is unambiguous:

.. ipython:: python

    ct
    ct = 3  # Now ct is a variable so automagic will not work...
    ct
    # ... but the magic still works.
    %ct

It's still possible to capture the output of execution in a variable:

.. ipython:: python

    uids = %ct

.. ipython:: python

    uids

But it's not possible to access the underlying plan with introspection tools:

.. code-block:: python

    print_summary(count(d))  # This works
    print_summary(%ct)  # This does not!

Magics invoking the bluesky RunEngine do not combine well and should not be
used as building blocks. They should only be run one at a time. **Do not put
them into loops or scripts like this.**

.. ipython:: python

    # DANGER: This can go badly if it is interrupted with Ctrl+C!!!
    for i in range(3):
        %ct

Instead, :ref:`compose plans properly <composing_plans>`, writing
user-defined plans like:

.. ipython:: python

    def multi_count(N):
        for i in range(N):
            yield from count(d)

and executing them

.. ipython:: python
    
    RE(multi_count(3))

Then, if you wish, define a new magic for invoking your custom plan. Or skip
it and just use the plan directly, as above. The shortcuts are best for quick,
simple operations with few parameters and no need for simluation.

Wrting custom plans retains correct interruption behavior and retains your
ability to simulate the plans for error-checking, time estimation,
pre-visualization, etc.  Resist the temptation to invent a private macro
language out of magics. You'll find that there are unexpected corner-cases
everywhere, and that inventing a language is hard! Stick to Python for writing
any program logic, and use magics as one-off shortcuts.

The ``%ct`` and ``%mov`` magics were defined by execting this script:

.. code-block:: python

    # magics.py

    # This file must be run in the global namespace, not imported as a module.
    # This can be done using an IPython profile startup directory or with
    # `%run -i magics.py`.

    import ast
    from IPython.core.magic import register_line_magic
    from bluesky.plans import count, mv


    @register_line_magic
    def ct(line):
        # %ct --> RE(count(d))
        # Expect this magic to be defined and executed in a namespace with a
        # RunEngine instance named RE and a list of detectors named d.
        global d
        global RE
        print('---> RE(count(d))')
        return RE(count(d))


    @register_line_magic
    def mov(line):
        # %mov theta 3 --> RE(mv(theta, 3))
        # Expect this magic to be defined and executed in a namespace with a
        # RunEngine instance named RE.
        # Avoid clobbering the name '%mv', the built-in magic for moving files.
        global RE
        motor_varname, pos = line.split()
        motor = globals()[motor_varname]
        pos = ast.literal_eval(pos)
        print('---> RE(mv({motor}, {pos}))'.format(motor=motor_varname, pos=pos))
        return RE(mv(motor, pos))

    # In an interactive session, we need to delete these to avoid
    # name conflicts for automagic to work.
    del ct, mov
