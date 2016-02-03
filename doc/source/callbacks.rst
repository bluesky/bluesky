Live Feedback and Processing
****************************

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
   from bluesky.plans import Count, AbsScanPlan
   plan = AbsScanPlan([det, det1, det2, det3], motor, 1, 4, 4)

Overview of Callbacks
---------------------

As the RunEngine processes instructions, it creates *Documents,* plain Python
dictionaries organized in a `specified but flexible
<http://nsls-ii.github.io/architecture-overview.html>`__ way. These Documents
contain the data and metadata generated during the plan's execution. Each time
a new Document is created, the RunEngine passes it to a list of functions.
These functions can do anyting: store the data to disk, print a line of text
to the scren, add a point to a plot, or even transfer the data to a cluster
for immediate processing. These functions are called "callbacks."

We subscribe callbacks to the live stream of Documents. You can think of a
callback as a self-addressed stamped envelope. It tells the RunEngine,
"When you create a Document, send it to this function for processing."

In order to keep up with the scan and avoiding slowing down data collection,
most subscriptions skip some Documents when they fall behind. A table might
skip a row, a plot might skip a point. But *critical* subscriptions -- like
saving the data -- are run in a lossless mode guananteed to process all
the Docuemnts.

Simplest Example
----------------

This example passes every Document to the ``print`` function, printing
each Document as it is generated during data collection.

.. code-block:: python

    RE(plan, print)

We will not show the lenthy output of this command here; the documents are not
so nice to read in their raw form. See ``LiveTable`` below for a more refined
implementation of this basic example.

Ways to Invoke Callbacks
------------------------

Subscribe on a per-run basis
++++++++++++++++++++++++++++

As in the simple example above, pass a second argument to the RunEngine.

.. ipython:: python

    dets = [det1, det2, det3]
    RE(plan, LiveTable(dets))

``LiveTable`` takes a list of objects or names to tell it which data columns to
show. It prints the lines one at a time, as data collection proceeds.

To use multiple callbacks, you may pass a list of them.

.. ipython:: python

    RE(plan, [LiveTable(dets), LivePlot(det1)])

Use this more verbose form to filter the Documents by type, feeding only
certain document types to certain callbacks.

.. code-block:: python

    # Give all documents to LiveTable and LivePlot.
    # Send only 'start' Documents to the print function.
    RE(plan, {'all': [LiveTable(dets), LivePlot(det1)], 'start': print})

The allowed keys are 'all', 'start', 'stop', 'descriptor', and 'event',
corresponding to the names of the Documents.

Subscribe each time a certain plan is used
++++++++++++++++++++++++++++++++++++++++++

Often, the same subscriptions are useful each time a certain kind of plan is
run. To associate particular callbacks with a given plan, give the plan 
a ``subs`` attribute. All the built-in plans already have a ``subs``
attribute, primed with a dictionary of empty lists.

.. ipython:: python

    plan.subs

Append functions to these lists to route Documents to them every time the plan
is executed.

.. ipython:: python

    plan.subs['all'].append(LiveTable(dets))

Now our ``plan`` will invoke ``LiveTable`` every time.

.. ipython:: python

    RE(plan)

Now suppose we change the detectors used by the plan.

.. ipython:: python

    plan.detectors.remove(det3)
    plan.detectors

The ``LiveTable`` callback is now out of date; it still includes
``[det1, det2, det3]``. How can we make this more convenient?

To customize the callback based on the content of the plan, use a subscription
factory: a function that takes in a plan and returns a callback function.

.. code-block:: python

    def make_table_with_detectors(plan):
        dets = plan.detectors
        return LiveTable(dets)

    plan.sub_factories['all'].append(make_table_with_detectors)

When the plan is executed, it passes *itself* as an argument to its own
``sub_factories``, producing customized callbacks. In this examples, a new
``LiveTable`` is made on the fly. Each time the plan is executed, new
callbacks are made via factory functions like this one.

A plan can have both normal subscriptions in ``plan.subs`` and subscription
factories in ``plan.sub_factories``. All will be used.

Subscribe for every run
+++++++++++++++++++++++

The RunEngine itself can store a collection of subscriptions to be applied to
every single scan it executes.

Usually, if a subscription is useful for every single run, it should be added
to a IPython configuration file and subscribed automatically at startup.

The method ``RE.subscribe`` passes through to this method:

.. automethod:: bluesky.run_engine.Dispatcher.subscribe

.. automethod:: bluesky.run_engine.Dispatcher.unsubscribe


Running Callbacks on Saved Data
+++++++++++++++++++++++++++++++

Callbacks are designed to work live, but they also work retroactively on
completed runs with data that has been saved to disk.

.. warning::

    This subsection documents a feature that has not been released yet.

If the data is accessible from the Data Broker (as it is if you use the standard
configuration) then you can feed data from the Data Broker in the callbacks.

.. code-block:: python

    from dataportal import DataBroker, stream
    stream(header, callback_func) 

Live Table
----------

As each data point is collected (i.e., as each Event Document is generated) a
row is added to the table. For nonscalar detectors, such as area detectors,
the sum is shown. (See LiveImage, below, to view the images themselves.)

The only crucial parameter is the first one, which specifies which fields to
include in the table. These can include specific fields (e.g., the string
``'sclr_chan4'``) or readable objects (e.g., the object ``sclr``).

Numerous other parameters allow you to customize the display style.

.. autoclass:: bluesky.callbacks.LiveTable

Live Plot for Scalar Data
-------------------------

Plot scalars.

.. autoclass:: bluesky.callbacks.LivePlot

Live Image Plot
---------------

.. autoclass:: bluesky.broker_callbacks.LiveImage

Live Raster Plot (Heat Map)
---------------------------

.. autoclass:: bluesky.callbacks.LiveRaster

Automated Data Export
---------------------

Exporting Image Data as TIFF Files
++++++++++++++++++++++++++++++++++

First, compose a filename template. This is a simple working example.

.. code-block:: python

    template = "output_dir/{start.scan_id}_{event.seq_num}.tiff"

The template can include metadata or event data from the scan.

.. code-block:: python

    template = ("output_dir/{start.scan_id}_{start.sample_name}_"
                "{event.data.temperature}_{event.seq_num}.tiff")

It can be handy to use the metadata to sort the images into directories.

.. code-block:: python

    template = "{start.user}/{start.scan_id}/{event.seq_num}.tiff"

If each image data point is actually a stack of 2D image planes, the template
must also include ``{i}``, which will count through the iamge planes in the
stack.

(Most metadata comes from the "start" document, hence ``start.scan_id`` above.
See
`here <https://nsls-ii.github.io/architecture-overview.html>`_ for a more
comprehensive explanation of what is in the different documents.)

Next, create an exporter.

.. code-block:: python

    from bluesky.broker_callbacks import LiveTiffExporter

    exporter = LiveTiffExporter('image', template)

Finally, to export all the images from a run when it finishes running, wrap the
exporter in ``post_run`` and subscribe.

.. code-block:: python

    from bluesky.broker_callbacks import post_run

    RE.subscribe('all', post_run(exporter))

It also possible to write TIFFs live, hence the name ``LiveTiffExporter``, but
there is an important disadvantage to this: in order to ensure that every image
is saved, a lossless subscription must be used. And, as a consequence, the
progress of the experiment may be intermittently slowed while data is written
to disk. In some circumstances, this affect on the timing of the experiment may
not be acceptable.

.. code-block:: python

    RE.subscribe_lossless('all', exporter)

There are more configuration options avaiable, as given in detail below.

.. autoclass:: bluesky.broker_callbacks.LiveTiffExporter

Export All Data and Metadata in an HDF5 File
++++++++++++++++++++++++++++++++++++++++++++

A Stop Document is emitted at the end of every run. Subscribe to it, using it
as a cue to load the dataset via the DataBroker and export an HDF5 file
using `suitcase <https://nsls-ii.github.io/suitcase>`_.


Working example:

.. code-block:: python

    from databroker import DataBroker as db
    import suitcase

    def suitcase_as_callback(name, doc):
        if name != 'stop':
            return
        run_start_uid = doc['run_start']
        header = db[run_start_uid]
        filename = '{}.h5'.format(run_start_uid)
        suitcase.export(header, filename)

    RE.subscribe('stop', suitcase_as_callback)

Verify Data Has Been Saved
--------------------------

The following verifies that all Documents and external files from a run have
been saved to disk and are accessible from the DataBroker.  It prints a message
indicating success or failure.

Note: If the data collection machine is not able to access the machine where
some external data is being saved, it will indicate failure. This can be a
false alarm.

.. code-block:: python

    from bluesky.broker_callbacks import post_run, verify_files_saved

    RE.subscribe('all', post_run(verify_files_saved))

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

.. code-block:: python

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

Lossless Subscriptions for Critical Functions
---------------------------------------------

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
subscriptions.

.. automethod:: bluesky.run_engine.RunEngine.subscribe_lossless

.. automethod:: bluesky.run_engine.RunEngine.unsubscribe_lossless
