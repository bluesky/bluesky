Live Visualization and Processing
*********************************

.. ipython:: python
   :suppress:
   :okwarning:

   from bluesky import RunEngine
   RE = RunEngine({})

.. _callbacks:

Overview of Callbacks
---------------------

As the RunEngine executes a plan, it organizes metadata and data into
*Documents,* Python dictionaries organized in a
:doc:`specified but flexible <documents>` way. Each time a new Document is
created, the RunEngine passes it to a list of functions. These functions can do
anything: store the data to disk, print a line of text to the screen, add a
point to a plot, or even transfer the data to a cluster for immediate
processing. These functions are called "callbacks."

We "subscribe" callbacks to the live stream of Documents coming from the
RunEngine. You can think of a callback as a self-addressed stamped envelope: it
tells the RunEngine, "When you create a Document, send it to this function for
processing."

Callback functions are run in a blocking fashion: data acquisition cannot
continue until they return. For light tasks like simple plotting or critical
tasks like sending the data to a long-term storage medium, this behavior is
desirable. It is easy to debug and it guarantees that critical errors will be
noticed immediately. But heavy computational tasks --- anything that takes more
than about 0.2 seconds to finish --- should be executed in a separate process
or server so that they do not hold up data acquisition. Bluesky provides nice
tooling for this use case --- see :ref:`zmq_callback`.

Simplest Working Example
------------------------

This example passes every Document to the ``print`` function, printing
each Document as it is generated during data collection.

.. code-block:: python

    from bluesky.plans import count
    from ophyd.sim import det

    RE(count([det]), print)

The ``print`` function is a blunt instrument; it dumps too much information to
the screen.  See :ref:`LiveTable <livetable>` below for a more refined option.

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

    from ophyd.sim import det, motor
    from bluesky.plans import scan
    from bluesky.callbacks import LiveTable
    dets = [det]
    RE(scan(dets, motor, 1, 5, 5), LiveTable(dets))

A *list* of callbacks --- ``[cb1, cb2]`` --- is also accepted; see
:ref:`filtering`, below, for additional options.

Persistently
++++++++++++

The RunEngine keeps a list of callbacks to apply to *every* plan it executes.
For example, the callback that saves the data to a database is typically
invoked this way. For some callback function ``cb``, the usage is:

.. code-block:: python

    RE.subscribe(cb)

This step is usually performed in a startup file (i.e., IPython profile).

.. automethod:: bluesky.run_engine.RunEngine.subscribe
    :noindex:

.. automethod:: bluesky.run_engine.RunEngine.unsubscribe
    :noindex:

.. _subs_decorator:

Through a plan
++++++++++++++

Use the ``subs_decorator`` :ref:`plan preprocessor <preprocessors>` to attach
callbacks to a plan so that they are subscribed every time it is run.

In this example, we define a new plan, ``plan2``, that adds some callback
``cb`` to some existing plan, ``plan1``.

.. code-block:: python

    from bluesky.preprocessors import subs_decorator

    @subs_decorator(cb)
    def plan2():
        yield from plan1()

or, equivalently,

.. code-block:: python

    plan2 = subs_decorator(cb)(plan1)

For example, to define a variant of ``scan`` that includes a table by default:

.. code-block:: python

    from bluesky.plans import scan
    from bluesky.preprocessors import subs_decorator

    def my_scan(detectors, motor, start, stop, num, *, per_step=None, md=None):
        "This plan takes the same arguments as `scan`."

        table = LiveTable([motor] + list(detectors))

        @subs_decorator(table)
        def inner():
            yield from scan(detectors, motor, start, stop, num,
                            per_step=per_step, md=md)

        yield from inner()

Callbacks for Visualization & Fitting
-------------------------------------

.. _livetable:

LiveTable
+++++++++

As each data point is collected (i.e., as each Event Document is generated) a
row is added to the table. Demo:

.. ipython:: python

    from bluesky.plans import scan
    from ophyd.sim import motor, det
    from bluesky.callbacks import LiveTable

    RE(scan([det], motor, 1, 5, 5), LiveTable([motor, det]))

Pass an empty list of columns to show simply 'time' and 'seq_num' (sequence
number).

.. code-block:: python

    LiveTable([])

In the demo above, we passed in a list of *device(s)*, like so:

.. code-block:: python

    LiveTable([motor])

Internally, ``LiveTable`` obtains the name(s) of the field(s) produced by
reading ``motor``. You can do this yourself too:

.. ipython:: python

    list(motor.describe().keys())

In the general case, a device can produce tens or even hundreds of separate
readings, and it can be useful to spell out specific fields rather than a whole
device.

.. code-block:: python

    # the field 'motor', in quotes, not the device, motor
    LiveTable(['motor'])

In fact, almost all other callbacks (including :ref:`LivePlot`) *require* a
specific field. They will not accept a device because it may have more than one
field.

.. autoclass:: bluesky.callbacks.LiveTable

.. _kickers:

Aside: Making plots update live
+++++++++++++++++++++++++++++++

.. note::

    If you are a user working with a pre-configured setup, you can probably
    skip this. Come back if your plots are not appearing / updating.

    This configuration is typically performed in an IPython profile startup
    script so that is happens automatically at startup time.

To make plots live-update while the RunEngine is executing a plan, you have run
this command once. In an IPython terminal, the command is:

.. code-block:: python

    %matplotlib qt
    from bluesky.utils import install_qt_kicker
    install_qt_kicker()

If you are using a Jupyter notebook, the command is:

.. code-block:: python

    %matplotlib notebook
    from bluesky.utils import install_nb_kicker
    install_nb_kicker()

Why? The RunEngine and matplotlib (technically, matplotlib's Qt backend) both
use an event loop. The RunEngine takes control of the event loop while it is
executing a plan. The kicker function periodically "kicks" the Qt event loop so
that the plots can re-draw while the RunEngine is running.

The ``%matplotlib ...`` command is standard setup, having nothing to do with
bluesky in particular. See
`the relevant section of the IPython documentation <https://ipython.readthedocs.io/en/stable/interactive/magics.html?highlight=matplotlib#magic-matplotlib>`_
for details.

.. autofunction:: bluesky.utils.install_kicker
.. autofunction:: bluesky.utils.install_qt_kicker
.. autofunction:: bluesky.utils.install_nb_kicker

.. _liveplot:

LivePlot (for scalar data)
++++++++++++++++++++++++++

Plot scalars. Example:

.. code-block:: python

    from bluesky.plans import scan
    from ophyd.sim import det, motor
    from bluesky.callbacks import LivePlot

    RE(scan([det], motor, -5, 5, 30), LivePlot('det', 'motor'))

.. plot::

    from bluesky import RunEngine
    from bluesky.plans import scan
    from ophyd.sim import det, motor
    from bluesky.callbacks import LivePlot
    RE = RunEngine({})
    RE(scan([det], motor, -5, 5, 30), LivePlot('det', 'motor'))

To customize style, pass in any
`matplotlib line style keyword argument <http://matplotlib.org/api/lines_api.html#module-matplotlib.lines>`_.
(``LivePlot`` will pass it through to ``Axes.plot``.) Example:

.. code-block:: python

    RE(scan([det], motor, -5, 5, 30),
       LivePlot('det', 'motor', marker='x', markersize=10, color='red'))

.. plot::

    from bluesky import RunEngine
    from bluesky.plans import scan
    from ophyd.sim import det, motor
    from bluesky.callbacks import LivePlot
    RE = RunEngine({})
    RE(scan([det], motor, -5, 5, 30),
       LivePlot('det', 'motor', marker='x', markersize=10, color='red'))

.. autoclass:: bluesky.callbacks.LivePlot

Live Image
++++++++++

.. autoclass:: bluesky.callbacks.broker.LiveImage

.. _liveraster:

LiveGrid (gridded heat map)
+++++++++++++++++++++++++++

Plot a scalar value as a function of two variables on a regular grid. Example:

.. code-block:: python

    from bluesky.plans import grid_scan
    from ophyd.sim import det4, motor1, motor2
    from bluesky.callbacks import LiveGrid

    RE(grid_scan([det4], motor1, -3, 3, 6, motor2, -5, 5, 10, False),
       LiveGrid((6, 10), 'det4'))

.. plot::

    from bluesky import RunEngine
    from bluesky.plans import grid_scan
    from ophyd.sim import det4, motor1, motor2
    from bluesky.callbacks import LiveGrid
    motor1.delay = 0
    motor2.delay = 0
    RE = RunEngine({})
    RE(grid_scan([det4], motor1, -3, 3, 6, motor2, -5, 5, 10, False),
       LiveGrid((6, 10), 'det4'))

.. autoclass:: bluesky.callbacks.LiveGrid

LiveScatter (scattered heat map)
++++++++++++++++++++++++++++++++

Plot a scalar value as a function of two variables. Unlike
:class:`bluesky.callbacks.LiveGrid`, this does not assume a regular grid.
Example:

.. code-block:: python

    from bluesky.plans import grid_scan
    from ophyd.sim import det5, jittery_motor1, jittery_motor2
    from bluesky.callbacks import LiveScatter

    # The 'jittery' example motors won't go exactly where they are told to go.

    RE(grid_scan([det5],
                          jittery_motor1, -3, 3, 6,
                          jittery_motor2, -5, 5, 10, False),
       LiveScatter('jittery_motor1', 'jittery_motor2', 'det5',
                xlim=(-3, 3), ylim=(-5, 5)))

.. plot::

    from bluesky import RunEngine
    from bluesky.plans import grid_scan
    from ophyd.sim import det5, jittery_motor1, jittery_motor2
    from bluesky.callbacks import LiveScatter
    RE = RunEngine({})
    RE(grid_scan([det5],
                          jittery_motor1, -3, 3, 6,
                          jittery_motor2, -5, 5, 10, False),
       LiveScatter('jittery_motor1', 'jittery_motor2', 'det5',
                xlim=(-3, 3), ylim=(-5, 5)))

.. autoclass:: bluesky.callbacks.LiveScatter

LiveFit
+++++++

Perform a nonlinear least squared best fit to the data with a user-defined
model function. The function can depend on any number of independent variables.
We integrate with the package
`lmfit <https://lmfit.github.io/lmfit-py/model.html>`_, which provides a nice
interface for NLS minimization.

In this example, we fit a Gaussian to detector readings as a function of motor
position. First, define a Gaussian function, create an ``lmfit.Model`` from it,
and provide initial guesses for the parameters.

.. code-block:: python

    import numpy as np
    import lmfit

    def gaussian(x, A, sigma, x0):
        return A*np.exp(-(x - x0)**2/(2 * sigma**2))

    model = lmfit.Model(gaussian)
    init_guess = {'A': 2,
                  'sigma': lmfit.Parameter('sigma', 3, min=0),
                  'x0': -0.2}

The guesses can be given as plain numbers or as ``lmfit.Parameter`` objects, as
in the case of 'sigma' above, to specify constraints.

To integrate with the bluesky we need to provide:

* the field with the dependent variable (in this example, ``'noisy_det'``)
* a mapping between the name(s) of independent variable(s) in
  the function (``'x'``) to the corresponding field(s) in the data
  (``'motor'``)
* any initial guesses expected by the model (defined above)

.. code-block:: python

    from bluesky.plans import scan
    from ophyd.sim import motor, noisy_det
    from bluesky.callbacks import LiveFit

    lf = LiveFit(model, 'noisy_det', {'x': 'motor'}, init_guess)

    RE(scan([noisy_det], motor, -1, 1, 100), lf)
    # best-fit values for 'A', 'sigma' and 'x0' are in lf.result.values

The fit results are accessible in the ``result`` attribute of the callback.
For example, the center of the Gaussian is ``lf.result.values['x0']``. This
could be used in a next step, like so:

.. code-block:: python

    x0 = lf.result.values['x0']
    RE(scan([noisy_det], x0 - 1, x0 + 1, 100))

Refer the
`lmfit documentation <https://lmfit.github.io/lmfit-py/model.html#the-modelresult-class>`_
for more about ``result``.

This example uses a model with two independent variables, x and y.

.. code-block:: python

    from ophyd.sim import motor1, motor2, det4

    def gaussian(x, y, A, sigma, x0, y0):
        return A*np.exp(-((x - x0)**2 + (y - y0)**2)/(2 * sigma**2))

    # Specify the names of the independent variables to Model.
    model = lmfit.Model(gaussian, ['x', 'y'])

    init_guess = {'A': 2,
                  'sigma': lmfit.Parameter('sigma', 3, min=0),
                  'x0': -0.2,
                  'y0': 0.3}

    lf = LiveFit(model, 'det4', {'x': 'motor1', 'y': 'motor2'}, init_guess)

    # Scan a 2D mesh.
    RE(grid_scan([det4], motor1, -1, 1, 20, motor2, -1, 1, 20, False),
       lf)

By default, the fit is recomputed every time a new data point is available. See
the API documentation below for other options. Fitting does not commence until
the number of accumulated data points is equal to the number of free parameters
in the model.

.. autoclass:: bluesky.callbacks.LiveFit

LiveFitPlot
+++++++++++

This is a variation on ``LivePlot`` that plots the best fit curve from
``LiveFit``. It applies to 1D model functions only.

Repeating the example from ``LiveFit`` above, adding a plot:

.. code-block:: python

    # same as above...

    import numpy as np
    import lmfit
    from bluesky.plans import scan
    from ophyd.sim import motor, noisy_det
    from bluesky.callbacks import LiveFit

    def gaussian(x, A, sigma, x0):
        return A*np.exp(-(x - x0)**2/(2 * sigma**2))

    model = lmfit.Model(gaussian)
    init_guess = {'A': 2,
                  'sigma': lmfit.Parameter('sigma', 3, min=0),
                  'x0': -0.2}

    lf = LiveFit(model, 'noisy_det', {'x': 'motor'}, init_guess)

    # now add the plot...

    from bluesky.callbacks import LiveFitPlot
    lpf = LiveFitPlot(lf, color='r')

    RE(scan([noisy_det], motor, -1, 1, 100), lfp)

    # Notice that we did'nt need to subscribe lf directly, just lfp.
    # But, as before, the results are in lf.result.

.. plot::

    import numpy as np
    import lmfit
    from bluesky.plans import scan
    from ophyd.sim import motor, noisy_det
    from bluesky.callbacks import LiveFit, LiveFitPlot
    from bluesky import RunEngine

    RE = RunEngine({})

    def gaussian(x, A, sigma, x0):
        return A*np.exp(-(x - x0)**2/(2 * sigma**2))

    model = lmfit.Model(gaussian)
    init_guess = {'A': 2,
                  'sigma': lmfit.Parameter('sigma', 3, min=0),
                  'x0': -0.2}

    lf = LiveFit(model, 'noisy_det', {'x': 'motor'}, init_guess)
    lfp = LiveFitPlot(lf, color='r')

    RE(scan([noisy_det], motor, -1, 1, 100), lfp)

We can use the standard ``LivePlot`` to show the data on the same axes.
Notice that they can styled independently.

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()  # explitly create figure, axes to use below
    lfp = LiveFitPlot(lf, ax=ax, color='r')
    lp = LivePlot('noisy_det', 'motor', ax=ax, marker='o', linestyle='none')

    RE(scan([noisy_det], motor, -1, 1, 100), [lp, lfp])

.. plot::

    import numpy as np
    import lmfit
    from bluesky.plans import scan
    from ophyd.sim import motor, noisy_det
    from bluesky.callbacks import LiveFit, LivePlot, LiveFitPlot
    from bluesky import RunEngine

    RE = RunEngine({})

    def gaussian(x, A, sigma, x0):
        return A*np.exp(-(x - x0)**2/(2 * sigma**2))

    model = lmfit.Model(gaussian)
    init_guess = {'A': 2,
                  'sigma': lmfit.Parameter('sigma', 3, min=0),
                  'x0': -0.2}

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    lf = LiveFit(model, 'noisy_det', {'x': 'motor'}, init_guess)
    lfp = LiveFitPlot(lf, ax=ax, color='r')
    lp = LivePlot('noisy_det', 'motor', ax=ax, marker='o', linestyle='none')

    RE(scan([noisy_det], motor, -1, 1, 50), [lfp, lp])
    plt.draw()

.. autoclass:: bluesky.callbacks.LiveFitPlot

PeakStats
++++++++++

Compute statistics of peak-like data. Example:

.. code-block:: python

    from bluesky.callbacks.fitting import PeakStats
    from ophyd.sim import motor, det
    from bluesky.plans import scan

    ps = PeakStats('motor', 'det')
    RE(scan([det], motor, -5, 5, 10), ps)

Now attributes of ``ps``, documented below, contain various peak statistics.
There is also a convenience function for plotting:

.. code-block:: python

    from bluesky.callbacks.mpl_plotting import plot_peak_stats

    plot_peak_stats(ps)

.. plot::

    from bluesky import RunEngine
    from bluesky.callbacks.fitting import PeakStats
    from bluesky.callbacks.mpl_plotting import plot_peak_stats
    from ophyd.sim import motor, det
    from bluesky.plans import scan

    RE = RunEngine({})
    ps = PeakStats('motor', 'det')
    RE(scan([det], motor, -5, 5, 10), ps)
    plot_peak_stats(ps)

.. autoclass:: bluesky.callbacks.fitting.PeakStats
.. autofunction:: bluesky.callbacks.mpl_plotting.plot_peak_stats

.. _best_effort_callback:

Best-Effort Callback
--------------------

.. warning::

    This is a new, experimental feature. It will likely be changed in future
    releases in a way that is not backward-compatible.

This is meant to be permanently subscribed to the RunEngine like so:

.. code-block:: python

    # one-time configuration
    from bluesky.callbacks.best_effort import BestEffortCallback
    bec = BestEffortCallback()
    RE.subscribe(bec)

It provides best-effort plots and visualization for *any* plan. It uses the
'hints' key provided by the plan, if present. (See the source code of the
plans in :mod:`bluesky.plans` for examples.)

.. ipython:: python
    :suppress:

    from bluesky.callbacks.best_effort import BestEffortCallback
    bec = BestEffortCallback()
    RE.subscribe(bec)

.. ipython:: python

    from ophyd.sim import det1, det2
    from bluesky.plans import scan

    dets = [det1, det2]

    RE(scan(dets, motor, 1, 5, 5))  # automatically prints table, shows plot

.. plot::

    from bluesky import RunEngine
    from bluesky.plans import scan
    from ophyd.sim import det, motor
    from bluesky.callbacks.best_effort import BestEffortCallback
    RE = RunEngine({})
    bec = BestEffortCallback()
    RE.subscribe(bec)
    RE(scan([det], motor, 1, 5, 5))

Use these methods to toggle on or off parts of the functionality.

.. currentmodule:: bluesky.callbacks.best_effort

.. autosummary::
    :toctree: generated

    BestEffortCallback.enable_heading
    BestEffortCallback.disable_heading
    BestEffortCallback.enable_table
    BestEffortCallback.disable_table
    BestEffortCallback.enable_baseline
    BestEffortCallback.disable_baseline
    BestEffortCallback.enable_plots
    BestEffortCallback.disable_plots

Blacklist plotting certain streams using the ``bec.noplot_streams`` attribute,
which is a list of stream names.  The blacklist is set to ``['baseline']`` by
default.

The attribute ``bec.overplot`` can be used to control whether line plots for
subsequent runs are plotted on the same axes. It is ``True`` by default.
Overplotting only occurs if the names of the axes are the same from one plot
to the next.

Peak Stats
++++++++++

For each plot, simple peak-fitting is performed in the background. Of
course, it may or may not be applicable depending on your data, and it is
not shown by default. To view fitting annotations in a plot, click the
plot area and press Shift+P. (Lowercase p is a shortcut for
"panning" the plot.)

To access the peak-fit statistics programmatically, use ``bec.peaks``.

.. _hints:

Hints
+++++

The best-effort callback aims to print and plot useful information without
being overwhelmingly comprehensive. Its usefulness is improved and tuned by the
``hints`` attribute on devices (if available) and ``hints`` metadata injected
by plans (if available). If either or both of these are not available, the
best-effort callback still makes a best effort to display something useful.

The contents of hints *do not at all affect what data is saved*. The content
only affect what is displayed automatically by the best-effort callback and
other tools that opt to look at the hints. Additional callbacks may still be
set up for live or *post-facto* visualization or processing that do more
specific things without relying on hints.

The ``hints`` attribute or property on devices is a dictionary with the key
``'fields'`` mapped to a list of fields.

On movable devices such as motors or temperature controllers, these fields are
expected to comprise the independent axes of the device. A motor that reads
the fields ``['x', 'x_setpoint']`` might provide the hint ``{'fields': ['x']}``
to indicate that it has one independent axis and that the field ``x`` is the best
representation of its value.

A readable device might report many fields like
``['chan1', 'chan2', 'chan3', 'chan4', 'chan5']`` but perhaps only a couple are
usually interesting. A useful hint might narrow them down to
``{'fields': ['chan1', 'chan2']}`` so that a "best-effort" plot does not
display an overwhelming amount of information.

The hints provided by the devices are read by the RunEngine and collated in the
:doc:`Event Descriptor documents <event_descriptors>`.

The plans generally know which devices are being used as dependent and
independent variables (i.e., which are being "scanned" over), and they may
provide this information via a ``'hints'`` metadata key that they inject into
the start document along with the rest of their metadata. Examples:

.. code-block:: python

    # The pattern is
    # {'dimensions': [(fields, stream_name), (fields, stream_name), ...]}

    # a scan over time
    {'dimensions': [(('time',), 'primary')]}

    # a one-dimensional scan
    {'dimensions': [(motor.hints['fields'], 'primary')]}

    # a two-dimensional scan
    {'dimensions': [(x_motor.hints['fields'], 'primary'),
                    (y_motor.hints['fields'], 'primary')]}

    # an N-dimensional scan
    {'dimensions': [(motor.hints['fields'], 'primary') for motor in motors]}

It's possible to adjust hints interactively, but they are generally intended to
be set in a startup file. Err on the side of displaying more information than
you need to see, and you will rarely need to adjust them.

Plans may also hint that their data is sampled on a regular rectangular grid
via the hint ``{'gridding': 'rectilinear'}``. This is useful, for example, for
decided whether to visualize 2D data with LiveGrid or with LiveScatter.

.. _export:

Callback for Export
-------------------

Exporting Image Data as TIFF Files
++++++++++++++++++++++++++++++++++

First, compose a filename template. The template can include metadata or event
data from the scan.

.. code-block:: python

    # a template that includes the scan ID and sequence number in each filename
    template = "output_dir/{start[scan_id]}_{event[seq_num]}.tiff"

    # a template that sorts files into directories based user and scan ID
    template = "output_dir/{start[user]}/{start[scan_id]}/{event[seq_num]}.tiff"

    # a more complex template includes actual measurements in the filenames
    template = ("output_dir/{start[scan_id]}_{start[sample_name]}_"
                "{event[data][temperature]}_{event[seq_num]}.tiff")

Above, we are using a Python language feature called format strings. Notice
that inside the curly brackets we don't use quotes around the key names; it's
``{event[seq_num]}`` not ``{event['seq_num']}``.

If each image data point is actually a stack of 2D image planes, the template
must also include ``{i}``, which will count through the image planes in the
stack.

.. note::

    Most metadata comes from the "start" document, hence ``start.scan_id``
    above.  Review the :doc:`documents` section for details.

Create a callback that exports TIFFs using your template.

.. code-block:: python

    from bluesky.callbacks.broker import LiveTiffExporter

    exporter = LiveTiffExporter('image', template)

Finally, to export all the images from a run when it finishes running, wrap the
exporter in ``post_run`` and subscribe.

.. code-block:: python

    from bluesky.callbacks.broker import post_run

    RE.subscribe(post_run(exporter))

It also possible to write TIFFs live, hence the name ``LiveTiffExporter``, but
there is an important disadvantage to doing this subscription in the same
process: progress of the experiment may be intermittently slowed while data is
written to disk. In some circumstances, this affect on the timing of the
experiment may not be acceptable.

.. code-block:: python

    RE.subscribe(exporter)

There are more configuration options available, as given in detail below. It is
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

    RE.subscribe(suitcase_as_callback, 'stop')

Export Metadata to the Olog
+++++++++++++++++++++++++++

The `Olog <http://olog.github.io/2.2.7-SNAPSHOT/>`_ ("operational log") is an
electronic logbook. We can use a callback to automatically generate log entries
at the beginning of a run. The Python interface to Olog is not straightforward,
so there is some boilerplate:

.. code-block:: python

    from functools import partial
    from pyOlog import SimpleOlogClient
    from bluesky.callbacks.olog import logbook_cb_factory

    # Set up the logbook. This configures bluesky's summaries of
    # data acquisition (scan type, ID, etc.).

    LOGBOOKS = ['Data Acquisition']  # list of logbook names to publish to
    simple_olog_client = SimpleOlogClient()
    generic_logbook_func = simple_olog_client.log
    configured_logbook_func = partial(generic_logbook_func, logbooks=LOGBOOKS)

    cb = logbook_cb_factory(configured_logbook_func)
    RE.subscribe(cb, 'start')

The module ``bluesky.callbacks.olog`` includes some templates that format the
data from the 'start' document into a readable log entry. You can also write
customize templates and pass them to ``logbook_cb_factory``.

You may specify a custom template. Here is a very simple example; see the
`source code <https://github.com/NSLS-II/bluesky/blob/master/bluesky/callbacks/olog.py>`_
for a more complex example (the default template).

.. code-block:: python

    CUSTOM_TEMPLATE = """
    My Log Entry

    {{ start.plan_name }}
    Detectors: {{ start.detectors }}
    """

    # Do same boilerplate above to set up configured_logbook_func. Then:
    cb = logbook_cb_factory(configured_logbook_func,
                            desc_template=CUSTOM_TEMPLATE)

You may also specify a variety of different templates that are suitable for
different kinds of plans. The callback will use the ``'plan_name'`` field to
determine which template to use.

.. code-block:: python

    # a template for a 'count' plan (which has no motors)
    COUNT_TEMPLATE = """
    Plan Name: {{ start.plan_name }}
    Detectors: {{ start.detectors }}
    """

    # a template for any plan with motors
    SCAN_TEMPLATE = """
    Plan Name: {{ start.plan_name }}
    Detectors: {{ start.detectors }}
    Motor(s): {{ start.motors }}
    """

    templates = {'count': COUNT_TEMPLATE,
                 'scan': SCAN_TEMPLATE,
                 'rel_scan': SCAN_TEMPLATE}

    # Do same boilerplate above to set up configured_logbook_func. Then:
    cb = logbook_cb_factory(configured_logbook_func,
                            desc_dispatch=templates)

.. autofunction:: bluesky.callbacks.olog.logbook_cb_factory

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

    RE.subscribe(post_run(verify_files_saved))

.. _debugging_callbacks:

Ignoring Callback Exceptions
----------------------------

If an exception is raised while processing a callback, the error can interrupt
data collection. Usually, this is good: if, for example, the callback that is
saving your data encounters an error, you want to know immediately.

But if a "flaky" callback is causing errors, it is possible to convert errors
to warnings like so.

.. code-block:: python

    RE.ignore_callback_exceptions = False

This is ``False`` by default. In bluesky version 0.6.4 (September 2016) and
earlier, this was ``True`` by default.

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

    from ophyd.sim import det
    from bluesky.plans import count

    RE(count([det]), {'event': print_data})

Each time the RunEngine generates a new Event Document (i.e., data point)
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

.. _zmq_callback:

Subscriptions in Separate Processes or Host with 0MQ
----------------------------------------------------

Because subscriptions are processed during a scan, it's possible that they can
slow down data collection. We mitigate this by making the subscriptions run in
a separate process.

In the main process, where the RunEngine is executing the plan, a ``Publisher``
is created. It subscribes to the RunEngine. It serializes the documents it
receives and it sends them over a socket to a 0MQ proxy which rebroadcasts the
documents to any number of other processes or machines on the network.

These other processes or machines set up a ``RemoteDispatcher`` which connects
to the proxy receives the documents, and then runs callbacks just as they would
be run if they were in the local ``RunEngine`` process.

Multiple Publishers (each with its own RunEngine) can send documents to the
same proxy. RemoteDispatchers can filter the document stream based on host,
process ID, and/or ``id(RE)`` with ``RE`` is a particular instance of
``RunEngine``.

Minimal Example
+++++++++++++++

Start a 0MQ proxy using the CLI packaged with bluesky. It requires two ports as
arguments.

.. code-block:: bash

    bluesky-0MQ-proxy 5577 5578

Alternatively, you can start the proxy using a Python API:

.. code-block:: python

    from bluesky.callbacks.zmq import Proxy
    proxy = Proxy(5577, 5578)
    proxy.start()

Start a callback that will receive documents from the proxy and, in this
simple example, just print them.

.. code-block:: python

    from bluesky.callbacks.zmq import RemoteDispatcher
    d = RemoteDispatcher('localhost:5578')
    d.subscribe(print)

    # when done subscribing things and ready to use:
    d.start()  # runs event loop forever

As `described above <kickers>`_, if you want to use any live-updating plots,
you will need to install a "kicker". It needs to be installed on the same
event loop used by the RemoteDispatcher, like so, and it must be done before
calling ``d.start()``.

.. code-block:: python

    from bluesky.utils import install_qt_kicker
    install_qt_kicker(loop=d.loop)

In a Jupyter notebook, replace ``install_qt_kicker`` with
``install_nb_kicker``.

On the machine/process where you want to collect data, hook up a subscription
to publish documents to the proxy.

.. code-block:: python

    # Create a RunEngine instance (or, of course, use your existing one).
    from bluesky import RunEngine, Msg
    RE = RunEngine({})

    from bluesky.callbacks.zmq import Publisher
    Publisher('localhost:5577', RE)

Finally, execute a plan with the RunEngine. As a result, the callback in the
RemoteDispatcher should print the documents generated by this plan.

Publisher / RemoteDispatcher API
++++++++++++++++++++++++++++++++

.. autoclass:: bluesky.callbacks.zmq.Proxy
.. autoclass:: bluesky.callbacks.zmq.Publisher
.. autoclass:: bluesky.callbacks.zmq.RemoteDispatcher


Secondary Event Stream
----------------------
For certain applications, it may desirable to interpret event documents as
they are created instead of waiting for them to reach offline storage. In order
to keep this information completely quarantined from the raw data, the
:class:`.LiveDispatcher` presents a completely unique stream that can be
subscribed to using the same syntax as the RunEngine.

In the majority of applications of :class:`.LiveDispatcher`, it is expected
that subclasses are created to implement online analysis. This secondary event
stream can be displayed and saved offline using the same callbacks that you
would use to display the raw data.

Below is an example using the `streamz
<https://streamz.readthedocs.io/en/latest>`_ library to average a number of
events together. The callback can be configured by looking at the start
document metadata, or at initialization time. Events are then received and
stored by the ``streamz`` network and a new averaged event is emitted when the
correct number of events are in the cache. The important thing to note here is
that the analysis only handles creating new ``data`` keys, but the descriptors,
sequence numbering and event ids are all handled by the base `LiveDispatcher`
class.

.. code-block:: python

    class AverageStream(LiveDispatcher):
        """Stream that averages data points together"""
        def __init__(self, n=None):
            self.n = n
            self.in_node = None
            self.out_node = None
            self.averager = None
            super().__init__()

        def start(self, doc):
            """
            Create the stream after seeing the start document

            The callback looks for the 'average' key in the start document to
            configure itself.
            """
            # Grab the average key
            self.n = doc.get('average', self.n)
            # Define our nodes
            if not self.in_node:
                self.in_node = streamz.Source(stream_name='Input')

            self.averager = self.in_node.partition(self.n)

            def average_events(cache):
                average_evt = dict()
                desc_id = cache[0]['descriptor']
                # Check that all of our events came from the same configuration
                if not all([desc_id == evt['descriptor'] for evt in cache]):
                    raise Exception('The events in this bundle are from '
                                    'different configurations!')
                # Use the last descriptor to avoid strings and objects
                data_keys = self.raw_descriptors[desc_id]['data_keys']
                for key, info in data_keys.items():
                    # Information from non-number fields is dropped
                    if info['dtype'] in ('number', 'array'):
                        # Average together
                        average_evt[key] = np.mean([evt['data'][key]
                                                    for evt in cache], axis=0)
                return {'data': average_evt, 'descriptor': desc_id}

            self.out_node = self.averager.map(average_events)
            self.out_node.sink(self.process_event)
            super().start(doc)

        def event(self, doc):
            """Send an Event through the stream"""
            self.in_node.emit(doc)

        def stop(self, doc):
            """Delete the stream when run stops"""
            self.in_node = None
            self.out_node = None
            self.averager = None
            super().stop(doc)


LiveDispatcher API
++++++++++++++++++
.. autoclass:: bluesky.callbacks.stream.LiveDispatcher
   :members:
