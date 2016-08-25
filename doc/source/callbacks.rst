Live Visualization and Export
*****************************

.. ipython:: python
   :suppress:
   :okwarning:

   from bluesky import RunEngine
   RE = RunEngine({})

.. _callbacks:

Overview of Callbacks
---------------------

As the RunEngine executes a plan, it organizes metadata and data into
*Documents,* Python dictionaries organized in a `specified but flexible
<http://nsls-ii.github.io/architecture-overview.html>`__ way. 
Each time a new Document is created, the RunEngine passes it to a list of
functions. These functions can do anyting: store the data to disk, print a
line of text to the scren, add a point to a plot, or even transfer the data to
a cluster for immediate processing. These functions are called "callbacks."

We "subscribe" callbacks to the live stream of Documents coming from the
RunEngine. You can think of a callback as a self-addressed stamped envelope: it
tells the RunEngine, "When you create a Document, send it to this function for
processing."

Simplest Working Example
------------------------

This example passes every Document to the ``print`` function, printing
each Document as it is generated during data collection.

.. code-block:: python

    from bluesky.plans import count
    from bluesky.examples import det

    RE(count([det]), print)

The ``print`` function is a blunt instrument; it dumps too much information to
the screen.  See ``LiveTable`` below for a more refined option.

Ways to Invoke Callbacks
------------------------

Interactively
+++++++++++++

As in the simple example above, pass a second argument to the RunEngine.
For some callback function ``cb``, the usage is:

.. code-block:: python

    RE(plan(), cb))

A working example:

.. code-block:: python

    from bluesky.examples import det, motor
    from bluesky.plans import scan
    from bluesky.callbacks import LiveTable
    dets = [det]
    RE(scan(dets, motor, 1, 5, 5), LiveTable(dets))

A *list* of callbacks --- ``[cb1, cb2]`` --- is also accepted; see
:ref:`filtering`, below, for addtional options.

Persistently
++++++++++++

The RunEngine keeps a list of callbacks to apply to *every* plan it executes.
For example, the callback that saves the data to a database is typically
invoked this way. For some callback function ``cb``, the usage is:

.. code-block:: python

    RE.subscribe('all', cb)

This step is usually performed in a startup file (i.e., IPython profile).

The method ``RunEngine.subscribe`` is an alias for this method:

.. automethod:: bluesky.run_engine.Dispatcher.subscribe

The method ``RunEngine.unsubscribe`` is an alias for this method:

.. automethod:: bluesky.run_engine.Dispatcher.unsubscribe

.. _subs_decorator:

Through a plan
++++++++++++++

Use the ``subs_decorator`` :ref:`plan preprocessor <preprocessors>` to attach
callbacks to a plan so that they are subscribed every time it is run.

In this example, we define a new plan, ``plan2``, that adds some callback
``cb`` to some existing plan, ``plan1``.

.. code-block:: python

    from bluesky.plans import subs_decorator

    @subs_decorator(cb)
    def plan2():
        yield from plan1()

or, equivalently,

.. code-block:: python

    plan2 = subs_decorator(cb)(plan1)

For example, to define a variant of ``scan`` that includes a table by default:

.. code-block:: python

    from bluesky.plans import scan, subs_decorator

    def my_scan(detectors, motor, start, stop, num, *, per_step=None, md=None):
        "This plan takes the same arguments as `scan`."

        table = LiveTable([motor] + list(detectors))

        @subs_decorator(table)
        def inner():
            yield from scan(detectors, motor, start, stop, num,
                            per_step=per_step, md=md)

        yield from inner()

Text and Visualization
----------------------

LiveTable
+++++++++

As each data point is collected (i.e., as each Event Document is generated) a
row is added to the table. Demo:

.. ipython:: python

    from bluesky.plans import scan
    from bluesky.examples import motor, det
    from bluesky.callbacks import LiveTable

    RE(scan([det], motor, 1, 5, 5), LiveTable([motor, det]))

.. autoclass:: bluesky.callbacks.LiveTable

.. _liveplot:

LivePlot (for scalar data)
++++++++++++++++++++++++++

Plot scalars.

.. autoclass:: bluesky.callbacks.LivePlot

Live Image
++++++++++

.. autoclass:: bluesky.callbacks.broker.LiveImage

.. _liveraster:

LiveRaster (Heat Map)
+++++++++++++++++++++

.. autoclass:: bluesky.callbacks.LiveRaster

LiveMesh (Heat Map)
+++++++++++++++++++

.. autoclass:: bluesky.callbacks.LiveMesh

PeakStats 
++++++++++

TO DO

Export
------

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

    from bluesky.callbacks.broker import LiveTiffExporter

    exporter = LiveTiffExporter('image', template)

Finally, to export all the images from a run when it finishes running, wrap the
exporter in ``post_run`` and subscribe.

.. code-block:: python

    from bluesky.callbacks.broker import post_run

    RE.subscribe('all', post_run(exporter))

It also possible to write TIFFs live, hence the name ``LiveTiffExporter``, but
there is an important disadvantage to doing this subscription in the same
process: progress of the experiment may be intermittently slowed while data is
written to disk. In some circumstances, this affect on the timing of the
experiment may not be acceptable.

.. code-block:: python

    RE.subscribe('all', exporter)

There are more configuration options avaiable, as given in detail below. It is
recommended to use these expensive callbacks in a separate process.

.. autoclass:: bluesky.callbacks.broker.LiveTiffExporter

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

    from bluesky.callbacks.broker import post_run, verify_files_saved

    RE.subscribe('all', post_run(verify_files_saved))

Debugging Callbacks
-------------------

If a callback raises an exception, the RunEngine catches that exception and
merely prints a warning. This behavior is intended to prevent some problem in
the callbacks from aborting the plan. Often, users to prefer to let data
collection complete and "pick up the pieces" later.

But this is not always desirable, especially when trying to debug problems with
callbacks. To stop the RunEngine from catching exceptions from the callbacks,
set

.. code-block:: python

    RE.ignore_callback_exceptions = False

.. _filtering:

Filtering by Document Type
--------------------------

There are four "subscriptions" that a callback to receive documents from:

* 'start'
* 'stop'
* 'event'
* 'descriptor'

Additionally, there is an 'all' subscription.

The command:

.. code-block:: python

    RE(plan(), cb)

is a shorthand that is normalized to ``{'all': [cb]}``. To receive only certain
documents, specify the document routing explicitly. Examples:

.. code-block:: python

    RE(plan(), {'start': [cb]}
    RE(plan(), {'all': [cb1, cb2], 'start': [cb3]})

The ``subs_decorator``, presented above, accepts the same variety of inputs.

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

    from bluesky.examples import det
    from bluesky.plans import count

    RE(count([det]), {'event': print_data})

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

.. code-block:: python

    def celebrate(name, doc):
        # Do nothing with the input; just use it as a signal that run is over.
        print("The run is finished!")

Let's use both ``print_data`` and ``celebrate`` at once.

.. code-block:: python

    RE(plan(), {'event': print_data, 'stop': celebrate})

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

Subscriptions in Separate Processes or Host
-------------------------------------------

Because subscriptions are processed during a scan, it's possible that they can
slow down data collection. We mitigate this by making the subscriptions run in
a separate process.

In the main process, where the RunEngine is executing the plan, a ``Publisher``
is created. It subscribes to the RunEngine. It serializes the documents it
receives and it sends them over a socket to a 0MQ "forwarder device," which
rebroadcasts the documents to any number of other processes or machines on the
network.

These other processes or machines set up a ``RemoteDispatcher`` which connects
to the "forwarder device," receives the documents, and then runs callbacks just
as they would be run if they were in the local ``RunEngine`` process.

Multiple Publishers (each with its own RunEngine) can send documents to the
same forwarder device. RemoteDispatchers can filter the document stream based
on host, process ID, and/or ``id(RunEngine)``.

Minimal Example
+++++++++++++++

Look for a forwarder device configuration file at
``/etc/zmq_forwarder_device.yml`` or
``~/.config/zmq_forwarder_device/connection.yml``. If there isn't one, create
one:

.. code-block:: yaml

    #~/.config/zmq_forwarder_device.yml
    {'frontend_port': 5577
    'backend_port': 5578
    'host': 'localhost'}  # optional

In production (e.g., at NSLS-II beamlines) the forwarder device should be
running in the background as a service. Here is how to start one just for play:

.. code-block:: bash

    # uses config in /etc/zmq_forwarder_device.yml
    #  or ~/.config/zmq_forwarder_device/connection.yml
    $ python bluesky/examples/forwarder_device.py

Start a callback that will receive documents from the forwarder and, in this
simple example, just print them.

.. code-block:: python

    from bluesky.callbacks.zmqsub import RemoteDispatcher
    d = RemoteDispatcher('localhost', 5578)
    d.subscribe('all', print)
    d.start()  # runs event loop forever

On the machine/process where you want to actually collect data,
hook up a subscription to publish documents to the forwarder. Finally,
generate some documents with a simple plan.

.. code-block:: python

    # Assume you have already create a RunEngine, RE.

    from bluesky.callbacks.zmqpub import Publisher
    Publisher(RE, 'localhost', 5577)
    RE([Msg('open_run'), Msg('close_run')])

As a result, the callback prints:

.. code-block:: python

    start
    stop

The connection between the publisher and the subscriber is lossless. (Messages
are cached on the publisher side if the subscriber is slow.)

Example: Plotting in separate process
+++++++++++++++++++++++++++++++++++++

As in the minimal example above, start a forwarder device. Then:

On the plotting machine:

.. code-block:: python

    import matplotlib
    matplotlib.use('Qt4Agg')

    import matplotlib.pyplot as plt
    plt.ion()

    from bluesky.utils import install_qt_kicker
    from bluesky.callbacks import LivePlot
    from bluesky.callbacks.zmqsub import RemoteDispatcher

    d = RemoteDispatcher('localhost', 5578)
    install_qt_kicker(d.loop)
    d.subscribe('all', LivePlot('det', 'motor'))
    d.start()

On the data collection machine, if there is not already a ``Publisher``
running, add one.

.. code-block:: python

    # Assume you have already create a RunEngine, RE.

    from bluesky.callbacks.zmqpub import Publisher
    p = Publisher(RE, 'localhost', 5577)

And now run a demo scan with a simulated motor and detector.

.. code-block:: python

    from bluesky.plans import scan
    from bluesky.examples import motor, det
    motor._fake_sleep = 0.5  # makes motor "move" slowly so we can watch it
    RE(scan([det], motor, 1, 10, 100))

API Documentation
^^^^^^^^^^^^^^^^^

.. autoclass:: bluesky.callbacks.zmqpub.Publisher
.. autoclass:: bluesky.callbacks.zmqsub.RemoteDispatcher
