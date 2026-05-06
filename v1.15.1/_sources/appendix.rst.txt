Appendix
========

This section covers Python language features that may be new to some readers.
They are used by bluesky but not *unique* Wherever possible, we to bluesky.

.. _yield_from_primer:

A Primer on ``yield`` and ``yield from``
----------------------------------------

This is a very brief primer on the Python syntax ``yield`` and ``yield from``,
a feature of the core language that we will use extensively.

A Python *function* returns once:

.. ipython:: python

    def f():
        return 1

    f()

A Python *generator* is like a function with multiple exit points. Calling a
generator produces an *iterator* that yields one value at a time. After
each ``yield`` statement, its execution is suspended.

.. ipython:: python

    def f():
        yield 1
        yield 2

We can exhaust the generator (i.e., get all its values) by calling ``list()``.

.. ipython:: python

    list(f())

We can get one value at a time by calling ``next()``

.. ipython:: python

    it = f()
    next(it)
    next(it)

or by looping through the values.

.. ipython:: python

    for val in f():
        print(val)

To examine what is happening when, we can add prints.

.. ipython:: python

    def verbose_f():
        print("before 1")
        yield 1
        print("before 2")
        yield 2

.. ipython:: python

    it = verbose_f()
    next(it)
    next(it)

Notice that execution is suspended after the first yield statement. The
second ``print`` is not run until we resume execution by requesting a second
value. This is a useful feature of generators: they can express "lazy"
execution.

Generators can delegate to other generators using ``yield from``. This is
syntax we commonly use to combine plans.

.. ipython:: python

    def double_f():
        yield from f()
        yield from f()

The above is equivalent to:

.. ipython:: python

    def double_f():
        for val in f():
            yield val
        for val in f():
            yield val

The ``yield from`` syntax is just more succinct.

.. ipython:: python

    list(double_f())
