Live Feedback and Processing
============================

.. ipython:: python
   :suppress:
   :okwarning:

   from bluesky.examples import det1, det2, det3, det, motor
   from bluesky.callbacks import *
   from bluesky import RunEngine, Msg
   RE = RunEngine()
   RE.verbose = False
   RE.md['owner'] = 'Jane'
   RE.md['group'] = 'Grant No. 12345'
   RE.md['beamline_id'] = 'demo'
   from bluesky.plans import Count

Demo: live-updating table
-------------------------

We begin with the simplest useful example of live feedback, a table.

.. ipython:: python
    :suppress:

    from bluesky.plans import AbsListScanPlan, AbsScanPlan
    from bluesky.callbacks import LiveTable
    dets = [det1, det2, det3]
    table = LiveTable(dets)
    RE.subscribe('all', table)  # Subscribe table to all future runs.
    my_plan = AbsListScanPlan(dets, motor, [1,2,4,8])

.. ipython:: python

    RE(my_plan)

.. ipython:: python
    :suppress:

    RE.unsubscribe(0)

Overview of Callbacks
---------------------

As the RunEngine processes instructions, it creates *Documents,* plain Python
dictionaries organized in a `specified but flexible
<http://nsls-ii.github.io/architecture-overview.html>`__ way. These Documents
contain the data and metadata generated during the plan's execution. Each time
a new Document is created, the RunEngine passes it to a list of functions.
These functions can do anyting: store the data to disk, transfer the data to a
cluster, update a plot, print a message, etc. The functions are called
"callbacks."

We subscribe callbacks to the live stream of Documents. You can think of a
subscription as a self-addressed stamped envelope. They tell the RunEngine,
"When you create a Document, send it to this callback for processing."

Ways to Invoke Callbacks
------------------------

Subscribe on a per-run basis
++++++++++++++++++++++++++++

To set up callbacks for a single use on a particular run, pass a second
argument to the call to the RunEngine.

.. ipython:: python

    dets = [det1, det2, det3]
    RE(my_plan, LiveTable(dets))

To use multiple callbacks, simply pass a list of them.

.. ipython:: python

    dets = [det1, det2, det3]
    RE(my_plan, [LiveTable(dets), LivePlot(det1)])

.. note::

    **Advanced:** Note that this more explicit syntax is equivalent.

    .. ipython:: python

        dets = [det1, det2, det3]
        RE(my_plan, subs={'all': [LiveTable(dets), LivePlot(det1)]})

    The allowed keys are 'all', 'start', 'stop', 'descriptor', and 'event',
    corresponding to the names of the Documents.

Subscribe each time a certain scan is used
++++++++++++++++++++++++++++++++++++++++++

Often, the same subscriptions are useful each time a certain kind of scan is
run. To associate particular callbacks with a given scan, give the scan
a ``subs`` attribute.

This simplest way is to simply "monkey-patch" the scan instance like so:

.. ipython:: python

    my_plan.subs = LiveTable(dets)

As above, this can one callback, a list of callbacks, or a dictionary. They
will be used automatically each time the scan is run.

.. ipython:: python

    RE(my_plan)

More complex subscriptions can configured using a property, which can inspect
the internal state --- for example, using the range of motor positions to
set plot limits in advance.

.. ipython:: python

    class PlottingAbsScan(AbsScanPlan):
        @property
        def subs(self):
            lp = LivePlot(self.detectors[0], self.motor,
                          xlim=(self.start, self.stop))
            return lp

Advanced: Subscribe for every scan
++++++++++++++++++++++++++++++++++

The RunEngine itself can store a collection of subscriptions to be applied to
every single scan it executes.

Usually, if a subscription is useful for every single run, it should be added
to a IPython configuration file and subscribed automatically at startup. This
involves a more explicit API that exposes some of the details of how
subscriptions work.

The method ``RE.subscribe`` passes through to this method:

.. automethod:: bluesky.run_engine.Dispatcher.subscribe


*Lossless subscriptions* are also applied to every scan. See below for more on
this topic.

Running Callbacks on Old Data
+++++++++++++++++++++++++++++

.. warning::

    This subsection documents a feature that has not been released yet.

If the data is being saved to metadatastore (as it is if you use the standard
configuration) then you can feed data from the Data Broker in the callbacks.

.. ipython:: python
    :verbatim:

    In [1]: from dataportal import DataBroker, stream
    
    In [2]: stream(header, LiveTable(cols))

Built-in Callbacks
------------------

LiveTable
+++++++++

As each data point is collected (i.e., as each Event Document is generated) a
row is added to the table. For nonscalar detectors, such as area detectors,
the sum is shown. (See LiveImage, below, to view the images themselves.)

The only crucial parameter is the first one, which specifies which fields to
include in the table. These can include specific fields (e.g., ``sclr_chan4``)
or readable objects (e.g., ``sclr``). The other parameters adjust the display
format.

.. autoclass:: bluesky.callbacks.LiveTable

LivePlot
++++++++

Plot scalars.

.. note::

    In order to keep up with the scan, subscriptions skip over some Documents
    when they fall behind. Be aware that plots may not show all points. (Don't
    worry: *all* the data is still being saved.)

.. autoclass:: bluesky.callbacks.LivePlot

LiveImage
+++++++++

.. note::

    In order to keep up with the scan, subscriptions skip over some Documents
    when they fall behind. Be aware that plots may not show all points. (Don't
    worry: *all* the data is still being saved.)

.. autoclass:: bluesky.broker_callbacks.LiveImage

Post-scan Data Export
+++++++++++++++++++++

.. warning::

    This isn't tested or documented yet, but it's possible.

Post-scan Data Validation
+++++++++++++++++++++++++

.. warning::

    This isn't tested or documented yet, but it's possible.

Writing Custom Callbacks
------------------------

Any function that accepts a Python dictionary as its argument can be used as
a callback. Refer to simple examples above to get started.

Two Simple Custom Callbacks
+++++++++++++++++++++++++++

These simple examples illustrate the concept and the usage.

First, we define a function that takes two arguments

#. the name of the Document type ('start', 'stop', 'event', or 'descriptor')
#. the Document itself, a dictionary

This is the *callback*.

.. ipython:: python

    def print_data(name, doc):
        print("Measured: %s" % doc['data'])

Then, we tell the RunEngine to call this function on each Event Document.
We are setting up a *subscription*.

.. ipython:: python

    s = Count([det])
    RE(s, {'event': print_data})

Each time the RunEngine generates a new Event Doucment (i.e., data point)
``print_data`` is called.

There are five kinds of subscriptions matching the four kinds of Documents plus
an 'all' subscription that receives all Documents.

* 'start'
* 'descriptor'
* 'event'
* 'stop'
* 'all'

We can use the 'stop' subscription to trigger automatic end-of-run activities.
For example:

.. ipython:: python

    def celebrate(name, doc):
        # Do nothing with the input; just use it as a signal that run is over.
        print("The run is finished!")

Let's use both ``print_data`` and ``celebrate`` at once.

.. ipython:: python

    RE(s, {'event': print_data, 'stop': celebrate})

Using multiple document types
+++++++++++++++++++++++++++++

Some tasks use only one Document type, but we often need to use more than one.
For example, LiveTable uses 'start' kick off the creation of a fresh table,
it uses 'event' to see the data, and it uses 'stop' to draw the bottom border.

A convenient pattern for this kind of subscription is a class with a method
for each Document type.

.. ipython:: python

    from bluesky.callbacks import CallbackBase
    class MyCallback(CallbackBase):
        def start(self, doc):
            print("I got a new 'start' Document")
            # Do something
        def descriptor(self, doc):
            print("I got a new 'descriptor' Document")
            # Do something
        def event(self, doc):
            print("I got a new 'event' Document")
            # Do something
        def stop(self, doc):
            print("I got a new 'stop' Document")
            # Do something

The base class, ``CallbackBase``, takes care of dispatching each Document to
the corresponding method. If your application does not need all four, you may
simple omit methods that aren't required.

Advanced: Subscribing During a Scan
-----------------------------------

.. warning::

    This section requires some familiarity with Messages, covered in
    :doc:`custom-plans`. If you haven't at least skimmed that section of the
    documents, head over to that page and then revisit this.

Subscriptions can added and removing during the course of a scan.

.. ipython:: python

    def count_with_table(detectors):
        table = LiveTable(detectors)
        yield Msg('subscribe', None, 'start', table)
        yield Msg('subscribe', None, 'descriptor', table)
        yield Msg('subscribe', None, 'event', table)
        yield Msg('subscribe', None, 'stop', table)
        yield from Count(detectors)
    RE(count_with_table(dets))

Critical Lossless Subscriptions
-------------------------------

Because subscriptions are processed during a scan, it's possible that they
can slow down data collection. We mitigate this by making the subscriptions
*lossy*. That is, some Documents will be skipped if the subscription
functions take too long and fall behind. For the purposes of real-time
feedback, this is usually acceptable. For other purposes, like saving data to
metadatastore, it is not.

Critical subscriptions are subscriptions that are executed on every Document
no matter how long they take to run, potentially slowing down data collection
but guaranteeing that all tasks are completed but the scan proceeds.

For example, in the standard configuration, metadatastore insertion functions
are registered as critical subscriptions.

If your subscription requires the complete, lossless stream of Documents
and you are will to accept the possibility of slowing down data
collection while that stream in processed, you can register your own critical
subscriptions. Use ``RE._subscribe_lossless(name, func)`` where ``name``
if one of ``'start'``, ``'descriptor'``, ``'event'``, ``'stop'``, and ``func``
is a callable that accepts a Python dictionary as its argument. Note that
there is no ``'all'`` callback implemented for critical subscriptions.
