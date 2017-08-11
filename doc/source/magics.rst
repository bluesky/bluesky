*******************************
IPython 'Magics' [Experimental]
*******************************

.. warning::

    This section covers an experimental feature of bluesky. It may be altered
    or removed in the future.

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

Suppose you have imported the some plans, defined a RunEngine ``RE``, and
created a list of detectors ``dets``, like so:

.. code-block:: python

    from bluesky import RunEngine
    from bluesky.examples import det1, det2, motor  # simulated hardware
    from bluesky.plans import count, mv

    RE = RunEngine({})
    dets = [det1, det2]

.. ipython:: python
    :suppress:

    from bluesky import RunEngine
    RE = RunEngine({})
    from bluesky.examples import det1, det2, motor  # simulated hardware
    dets = [det1, det2]
    from bluesky.plans import count, mv

And suppose that, in typical interactive use, you often take a reading from
these detectors, move a motor, and repeat.

.. ipython:: python

    RE(count(dets))
    RE(mv(motor, 3))

Bundled with bluesky are some IPython magics for executing these commands more
succinctly. To use them, register them with IPython:

.. ipython:: python

    from bluesky.magics import SPECMagics
    get_ipython().register_magics(SPECMagics)

The magics expect to find an instance of the RunEngine named ``RE`` and (when
applicable) a list of detectors named ``dets`` pre-defined by the user in the
global namespace. If those expectations are satisfied, ``RE(count(dets))`` can
be handily invoked like this:

.. ipython:: python

    %ct

And ``RE(mv(motor, 3))`` can be run like this:

.. ipython:: python

    %mov motor 3

If IPython’s ‘automagic’ feature is enabled, IPython will even let you drop the
% as long as the meaning is unambiguous:

.. ipython:: python

    ct
    ct = 3  # Now ct is a variable so automagic will not work...
    ct
    # ... but the magic still works.
    %ct

For what it’s worth, we recommend disabling 'automagic'. The ``%`` is useful
for flagging what follows as magical, non-Python code.

Using the magics, it’s still possible to capture the output of execution in a
variable:

.. ipython:: python

    uids = %ct

.. ipython:: python

    uids

But it's not possible to access the underlying plan with introspection tools:

.. code-block:: python

    summarize_plan(count(dets))  # This works
    sumamrize_plan(%ct)  # This does not!

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
            yield from count(dets)

and executing them

.. ipython:: python
    
    RE(multi_count(3))

The shortcuts are best for quick, simple operations with few parameters and no
need for simluation.

Resist the temptation to invent a private macro language out of magics. You'll
find that there are unexpected corner-cases everywhere, and that inventing a
language is hard! Stick to Python for writing any program logic, and use magics
as one-off shortcuts.

Built-in Magics
---------------

The names of these magics, and the order of the parameters they take, are meant
to feel familiar to users of :doc:`SPEC <comparison-with-spec>`. They encompass
only a subset of the plans available in bluesky.

These magics expect to find an instance of the RunEngine named ``RE`` and (when 
applicable) a list of detectors named ``dets`` pre-defined by the user in the
global namespace.

Again, they must be registered with IPython before they can be used:

.. code-block:: python

    from bluesky.magics import SPECMagics
    get_ipython().register_magics(SPECMagics)

.. currentmodule:: bluesky.plans

======================================================================= ==============================
Magic                                                                   Plan Invoked
======================================================================= ==============================
``%mov``                                                                :func:`mv`
``%movr``                                                               :func:`mvr`
``%ct``                                                                 :func:`count`
``%ascan motor start stop intervals``                                   :func:`scan`
``%dscan motor start stop intervals``                                   :func:`relative_scan`
``%a2scan motor1 start1 stop1 motor2 start2 stop2 intervals``           :func:`inner_product_scan`
``%d2scan motor1 start1 stop1 motor2 start2 stop2 intervals``           :func:`relative_inner_product_scan`
``%mesh motor1 start1 stop1 intervals1 motor2 start2 stop2 intervals2`` :func:`outer_product_scan`
``%wa``                                                                 ("where all") Survey positioners*
======================================================================= ==============================

\*The magic ``%wa`` differs from the others; it does not execute a plan. It
also requires some special configuration: it relies on a list of "positioners"
at ``SPECMagics.positioners`` that must be configured in advance. For example:

.. code-block:: python

    MY_POSITIONERS = [eta,
                      delta,
                      gamma,
                      temperature]

    SPECMagics.positioners.extend(MY_POSITIONERS)
