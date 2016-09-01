.. currentmodule:: bluesky.plans

Asynchronous Acquisition
========================

This section encompasses "fly scans," "monitoring," and in general handling
data acquisition that is occurring at different rates.

.. note::

    If you are here because you just want to "move two motors at once" or
    something in that category, you're in luck: you don't need anything as
    complex as what we present in this section. Read about multidimensional
    plans in the section on :doc:`plans`.

In short, "flying" is for acquisition at high rates and "monitoring" is for
acquisition an irregular or slow rate.

**Flying** means: "Let the hardware take control, cache data externally, and
then transfer all the data to the RunEngine at the end." This is essential when
the data acquisition rates are faster than the RunEngine or Python can go.

.. note::

    As a point of reference, the RunEngine processes message at a rate of
    about 35k/s (not including any time added by whatever the message *does*).


    .. code-block:: python

        In [3]: %timeit RE(Msg('null') for j in range(1000))
        10 loops, best of 3: 26.8 ms per loop

**Monitoring** a means acquiring readings whenever a new reading is available,
at a device's natural update rate. For example, we might monitor background
condition (e.g., beam current) on the side while executing the primary logic of
a plan. The documents are generated in real time --- not all at the end, like
flying --- so if the update rate is too high, monitoring can slow down the
execution of the plan.

Flying
------

In bluesky's view, there are three steps to "flying" a device during a scan.

1. **Kickoff**: Begin accumulating data. A 'kickoff' command completes once
   acquisition has successfully started.
2. **Complete**: This step tells the device, "I am ready whenever you are
   ready." If the device is just collecting until it is told to stop, it will
   report that is it ready immediately. If the device is executing some
   predetermined trajectory, it will finish before reporting ready.
3. **Collect**: Finally, the data accumulated by the device is transferred to
   the RunEngine and processed like any other data.

To "fly" one or more "flyable" devices during a plan, bluesky provides a
`preprocessor <preprocessors>`. It is available as a wrapper,
:func:`fly_during_wrapper`

.. code-block:: python

    from bluesky.examples import det, flyer1, flyer2  # simulated hardware
    from bluesky.plans import count, fly_during_wrapper

    RE(fly_during_wrapper(count([det], num=5), [flyer1, flyer2]))

and as a decorator, :func:`fly_during_wrapper`.

.. code-block:: python

    from bluesky.examples import det, flyer1, flyer2  # simulated hardware
    from bluesky.plans import count, fly_during_decorator

    # Define a new plan for future use.
    fly_and_count = fly_during_decorator([flyer1, flyer2])(count)

    RE(fly_and_count([det]))

Alternatively, if you are using the `SPEC-like Plan API <spec_api>`, simply add
flyers to the global state:

.. code-block:: python

    from bluesky.global_state import gs
    from bluesky.examples import flyer1, flyer2

    gs.FLYERS = [flyer1, flyer2]

They will be included with all plans until removed.

Monitoring
----------

To monitor some device during a plan, bluesky provides a
`preprocessor <preprcoessors>`. It
is available as a wrapper, :func:`monitor_during_wrapper`

.. code-block:: python

    from bluesky.examples import det, det1
    from bluesky.plans import count, monitor_during_wrapper

    # Record any updates from det1 while 'counting' det 5 times.
    RE(monitor_during_wrapper(count([det], num=5), [det1]))

and as a decorator, :func:`fly_during_wrapper`.

.. code-block:: python

    from bluesky.examples import det, det1
    from bluesky.plans import count, monitor_during_decorator

    # Define a new plan for future use.
    monitor_and_count = monitor_during_decorator([det1])(count)

    RE(monitor_and_count([det]))

Alternatively, if you are using the `SPEC-like Plan API <spec_api>`, simply add
devices to be monitored to the global state:

.. code-block:: python

    from bluesky.global_state import gs
    from bluesky.examples import det1

    gs.MONITORS = [det1]

They will be included with all plans until removed.
