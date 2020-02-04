=================
 Release History
=================

v1.6.0 (Pre-release)
====================

The most important change in this release is a complete reworking of how
bluesky interacts with the asyncio event loop. This resolves a long-running
issue of bluesky being incompatible with ``tornado >4``, which often tripped up
users in the context of using bluesky from Jupyter notebooks.

There are several other new features and fixes, including new plans and more
helpful error messages, enumerated further below.

Event loop re-factor
--------------------

Previously, the :class:`~bluesky.RunEngine` had been repeatedly starting and
stopping the asyncio event loop in :meth:`~bluesky.RunEngine.__call__`,
:meth:`~bluesky.RunEngine.request_pause`, :meth:`~bluesky.RunEngine.stop`, in
:meth:`~bluesky.RunEngine.abort`, :meth:`~bluesky.RunEngine.halt`, and
:meth:`~bluesky.RunEngine.resume`.  This worked, but is bad practice.  It
complicates attempts to integrate with the event loop with other tools.
Further, because as of tornado 5, tornado reports its self as an asyncio event
loop so attempts to start another asyncio event loop inside of a task fails
which means bluesky will not run in a jupyter notebook.  To fix this we now
continuously run the event loop on a background thread and the
:class:`~bluesky.RunEngine` object manages the interaction with creating tasks
on that event loop.  To first order, users should not notice this change,
however details of how

API Changes
~~~~~~~~~~~

``install_qt_kicker`` deprecated
++++++++++++++++++++++++++++++++

Previously, we were running the asyncio event loop on the main thread
and blocked until it returned.  This meant that if you were using
Matplotlib and Qt for plots they would effectively be "frozen" because
the Qt event loop was not being given a chance to run.  We worked
around this by installing a 'kicker' task onto the asyncio event loop
that would periodically spin the Qt event loop to keep the figures
responsive (both to addition of new data from callbacks and from user
interaction).

Now that we are running the event loop on a background thread this no
longer works because the Qt event loop must be run on the main thread.
Instead we use *during_task* to block the main thread by running the
Qt event loop directly.

*during_task* kwarg to :meth:`RunEngine.__init__`
+++++++++++++++++++++++++++++++++++++++++++++++++

We need to block the main thread in :meth:`~bluesky.RunEngine.__call__` (and
:meth:`~bluesky.RunEngine.resume`) until the user supplied plan is complete.
Previously, we would do this by calling ``self.loop.run_forever()`` to
start the asyncio event loop.  We would then stop the event loop an
the bottom of :meth:`~bluesky.RunEngine._run` and in
:meth:`~bluesky.RunEngine.request_pause` to un-block the main thread and return
control to the user terminal.  Now we must find an alternative way to achieve
this effect.

There is a a :class:`threading.Event` on the :class:`~bluesky.RunEngine` that
will be set when the task for :meth:`~bluesky.RunEngine._run` in completed,
however we can not simple wait on that event as that would again cause the Qt
windows to freeze.  We also do not want to bake a Matplotlib / Qt dependency
directly into the :class:`~bluesky.RunEngine` so we added a hook, set at init
time, for a function that passed the :class:`threading.Event` and is
responsible for blocking until it is set.  This function can do other things
(such as run the Qt event loop) during that time.  The required signature is ::

  def blocking_func(ev: Threading.Event) -> None:
      "Returns when ev is set"


The default hook will handle the case of the Matplotilb Qt backend and
the case of Matplotlib not being imported.


``'wait_for'`` Msg now expects callables rather than futures
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Messages are stashed and re-run when plans are interrupted which would
result in re-using the coroutines passed through.  This has always
been broken, but due to the way were stopping the event loop to pause
the scan it was passing tests.

Instead of directly passing the values passed into :func:`asyncio.wait`, we
now expect that the iterable passed in is callables with the signature::

  def fut_fac() -> awaitable:
      'This must work multiple times'

Features
--------

* Generalized :func:`~bluesky.plans.list_scan` to work on any number of motors,
  not just one. In v1.2.0, :func:`~bluesky.plans.scan` was generalized in the
  same way.
* Added :func:`~bluesky.plans.list_grid_scan`.
* Raise a more helpful error message if the ``num`` parameter given to
  :func:`~bluesky.plans.scan` is not a whole number, as can happen if ``num`` is
  mistaken to mean "step size".
* Report the version of bluesky and (if available) ophyd in the metadata.
* Add a more helpful error message if the value returned from some call to
  ``obj.read()`` returns ``None`` instead of the expected dict.
* If the user tries to start a :class:`~bluesky.callbacks.zmq.RemoteDispatcher`
  after it has been stopped, raise a more helpful error message.

Bug Fixes
---------

* The ``state`` attribute of the ``RunEngine`` is now a read-only property, as
  it should have always been.
* In the Best-Effort Callback, do not assume that the RunStart document
  includes ``'scan_id'``, which is an optional key.

v1.5.4 (2019-08-09)
===================

Support Maplotlib 3.1 and above. (Do not use deprecated and removed aspect
adjustable values.)

v1.5.3 (2019-05-27)
===================

This release removes the dependency on an old version of the ``jsonschema``
library and requires the latest version of the ``event-model`` library.


v1.5.2 (2019-03-11)
===================

This release fixes compatibility with matplotlib 2.x; at least some matplotlib
2.x releases are not compatible with the matplotlib plotting callbacks in
bluesky v1.5.1. This release of bluesky is compatible with all 2.x and 3.x
releases.

v1.5.1 (2019-03-08)
===================

This release contains bug fixes and documentation updates.

Features
--------

* Use the ISO8601 delimiters for date in RE scans.

Bug Fixes
---------

* Pin jsonschema <3 due to its deprecations.
* Stop using deprecated API in Matplotlib.


v1.5.0 (2019-01-03)
===================

This release includes many documentation fixes and handful of new features,
especially around improved logging.

Features
--------

* Logging has been increased and improved.
* A default handler is added to the ``'bluesky'`` logger at import time. A new
  convenience function, :func:`~bluesky.set_handler`, addresses common cases
  such as directing the log output to a file.
* The ``bluesky-0MQ-proxy`` script now supports a ``-v, --verbose`` option,
  which logs every start and stop document received and a ``-vvv`` ("very
  verbose") option, which logs every document of every type.
* The prefix on messages sent by :class:`bluesky.callbacks.zmq.Publisher` can
  be set to arbitrary bytes. (In previous versions, the prefix was hardcoded to
  an encoded combination of the hostname, process ID, and the Python object ID
  of a RunEngine instance.)
* The RunEngine includes a human-readable, not-necessarily-unique ``scan_id``
  key in the RunStart document. The source of the ``scan_id`` is now pluggable
  via a new parameter, ``scan_id_source``. See :doc:`run_engine_api` for
  details.
* The convenience function, :func:`bluesky.utils.ts_msg_hook` accepts new
  parameter ``file`` for directing the output to a file instead of the standard
  out.
* It is possible to use those callbacks that do not require matplotlib without
  importing it.

Bug Fixes
---------

* Fixed BestEffortCallback's handling of integer data in plots.
* Fixed invalid escape sequence that produced a warning in Python 3.6.

Breaking Changes
----------------

* The signature of :class:`bluesky.callbacks.zmq.RemoteDispatcher` has been
  changed in a non-backward-compatible way. The parameters for filtering
  messages by ``hostname``, ``pid``, and ``run_engine_id`` have been replaced
  by one new parameter, ``prefix``.
* The default value of ``RunEngine.verbose`` is now ``True``, meaning that the
  ``RunEngine.log`` is *not* disabled by default.

Deprecations
------------

* The :class:`bluesky.callbacks.zmq.Publisher` accepts an optional RunEngine
  instance, which the Publisher subscribes to automatically. This parameter has
  been deprecated; users are now encouraged to subscribe the publisher to the
  RunEngine manually, in the normal way (``RE.subscribe(publisher)``). The
  parameter may be removed in a future release of bluesky.

v1.4.1 (2018-09-24)
===================

This release fixes a single regression introduced in v1.4.0. We recommend all
users upgrade.

Bug Fixes
---------

* Fix a critical typo that made
  :class:`~bluesky.callbacks.mpl_plotting.LiveGrid` unusable.

Note that the 1.4.x series is not compatible with newer versions of matplotlib;
it needs a version lower than 3.1.0 due to an API change in matplotlib. The
1.5.x series is compatible with matplotlib versions before and after the
change.

v1.4.0 (2018-09-05)
===================

Features
--------

* Added ability to control 'sense' of
  :class:`~bluesky.callbacks.mpl_plotting.LiveGrid` (ex "positive goes
  down and to the right") to match the coordinates in the hutch.
* Learned how to specify the serializer / deserializer for the zmq
  publisher / client.
* Promoted the inner function from :func:`~bluesky.plan_stubs.one_nd_step`
  to a top-level plan :func:`bluesky.plan_stubs.move_per_step`.
* Added flag to :func:`~bluesky.plans.ramp_plan` to control if a
  data point is taken before the ramp starts.

Bug Fixes
---------

* Ensure order stability in :func:`~bluesky.magics.get_labeled_devices`
  on all supported versions of Python.
* Fixed typos, dev requirements, and build details.


v1.3.3 (2018-06-06)
===================

Bug Fixes
---------

* Fixed show-shopping RunEngine bug in flyer asset collection. (The impact of
  this bug is expected to be low, as there *are* no flyers with asset
  collection yet and the bug was discovered while writing the first one.)
* Fixed packaging issue where certain important files (notably
  ``requirements.txt``) were not included in the source tarball.
* Made BestEffortCallback swallow errors related to matplotlib's "tight layout"
  if the occur --- better to show a messy plot than error out.

v1.3.2 (2018-05-24)
===================

Bug Fixes
---------

* Revised behavior of magics that integrate with ophyd's experimental
  "labels" feature. The most important difference is that the ``%wa`` magic now
  traverses the children of labeled devices to find any sub-devices that are
  positioners.

v1.3.1 (2018-05-19)
===================

Bug Fixes
---------

* Fixed race condition where monitored signals could emit an Event document
  before the corresponding Event Descriptor document.
* Addressed incompatibilities with upcoming release of Python, 3.7.

v1.3.0 (2018-05-15)
===================

Features
--------

* When used with ophyd v1.2.0 or later, emit Resource and Datum documents
  through the RunEngine. Previously, ophyd would insert these documents
  directly into a database. This left other consumers with only partial
  information (for example, missing file paths to externally-stored data) and
  no guarantees around synchronization. Now, ophyd need not interact with a
  database directly. All information flows through the RunEngine and out to any
  subscribed consumers in a deterministic order.
* New Msg commands, ``install_suspender`` and ``remove_suspender``, allow plans
  to temporarily add and remove Suspenders.
* The RunEngine's signal handling (i.e. Ctrl+C capturing) is now configurable.
  The RunEngine accepts a list of ``context_managers`` that it will enter and
  exit before and after running. By default, it has one context manager that
  handles Ctrl+C. To disable Ctrl+C handling, pass in an empty list instead.
  This can also be used to inject other custom behavior.
* Add new plans: :func:`~bluesky.plans.x2x_scan`,
  :func:`~bluesky.plans.spiral_square_plan`, and
  :func:`~bluesky.plans.rel_spiral_square_plan`.
* Add convenience methods for reviewing the available commands,
  :meth:`~bluesky.run_engine.RunEngine.commands` and
  :meth:`~bluesky.run_engine.RunEngine.print_command_registry`.
* Add a ``crossings`` attribute to ``PeakStats``.

Bug Fixes
---------

* When resuming after a suspender, call ``resume()`` on all devices (if
  present).
* Fixed BEC LiveGrid plot for a motor with one step.
* A codepath in ``LiveFit`` that should have produced a warning produced an
  error instead.

Breaking Changes
----------------

* User-defined callbacks subscribed to the RunEngine ``'all'`` stream must
  accept documents with names ``'resource'``, ``'datum'`` and ``'bulk_datum'``.
  It does not necessarily have to heed their contents, but it must not fall
  over if it receives one.

Deprecations
------------

* The IPython "magics", always marked as experimental, have been reworked.
  Instead of relying on the singleton lists, ``BlueskyMagics.positioners`` and
  ``BlueskyMagics.detectors``, the magics now scrape the user namespace for
  objects that implement the ``_ophyd_labels_`` interface. See :doc:`magics`
  for the new usage. The magics will revert to their old behavior if the
  singleton lists are non-empty, but they will produce a warning. The old
  behavior will be removed in a future release.

v1.2.0 (2018-02-20)
===================

Features
--------

* Refreshed documentation with a new :doc:`tutorial` section.
* Extend :func:`.scan` and :func:`.rel_scan` to
  handle multiple motors, rendering :func:`.inner_product_scan` and
  :func:`relative_inner_product_scan` redundant.
* A new plan stub, :func:`~bluesky.plan_stubs.repeat`, repeats another plan N
  times with optional interleaved delays --- a kind of customizable version of
  :func:`~bluesky.plans.count`.
* Better validation of user-defined ``per_step`` functions and more informative
  error messages to match.

Bug Fixes
---------

* Fix axes orientation in :class:`.LiveRaster`.
* Make :class:`.BestEffortCallback` display multi-motor scans properly.
* Fix bug in :func:`.ts_msg_hook` where it conflated month and minute. Also,
  include sub-second precision.
* Avoid situation where plan without hints caused the
  :class:`.BestEffortCallback` to error instead of do its best to guess useful
  behavior.
* Skip un-filled externally-stored data in :class:`.LiveTable`. This fixes a
  bug where it is expecting array data but gets UUID (``datum_id``) and errors
  out.

Deprecations
------------

* The :func:`~bluesky.plan_stubs.caching_repeater` plan has been deprecated
  because it is incompatible with some preprocessors. It will be removed in
  a future release of bluesky. It was not documented in any previous releases
  and rarely if ever used, so the impact of this removal is expected to be low.

v1.1.0 (2017-12-19)
===================

This release fixes small bugs in v1.0.0 and introduces one new feature. The
API changes or deprecations are not expected to affect many users.

Features
--------

* Add a new command to the :class:`~bluesky.run_engine.RunEngine`, ``'drop'``,
  which jettisons the currently active event bundle without saving. This is
  useful for workflows that generate many readings that can immediately be
  categorized as not useful by the plan and summarily discarded.
* Add :func:`~bluesky.utils.install_kicker`, which dispatches automatically to
  :func:`~bluesky.utils.install_qt_kicker` or
  :func:`~bluesky.utils.install_nb_kicker` depending on the current matplotlib
  backend.

Bug Fixes
---------

* Fix the hint for :func:`~bluesky.plans.inner_product_scan`, which previously
  used a default hint that was incorrect.

Breaking Changes and Deprecations
---------------------------------

* In :func:`~bluesky.plans.tune_centroid`, change the meaning of the
  ``step_factor`` parameter to be the factor to reduce the range of each
  successive iteration. Enforce bounds on the motion, and determine the
  centroid from each pass separately.
* The :class:`~bluesky.preprocessors.SupplementalData` preprocessor inserts its
  instructions in a more logical order: first baseline readings, then
  monitors, then flyers. Previously, the order was reversed.
* The suspender :class:`~bluesky.suspenders.SuspendInBand` has been renamed to
  :class:`~bluesky.suspenders.SuspendWhenOutsideBand` to make its meaning more
  clear. Its behavior has not changed: it suspends when a value exits a given
  range. The original, confusing name now issues a warning.
* The suspender :class:`~bluesky.suspenders.SuspendOutBand`, which
  counter-intuitively suspends *when a value enters a given range*, has been
  deprecated. (If some application is found for this unusual scenario, the user
  can always implement a custom suspender to handle it.)

v1.0.0 (2017-11-07)
===================

This tag marks an important release for bluesky, signifying the conclusion of
the early development phase. From this point on, we intend that this project
will be co-developed between multiple facilities. The 1.x series is planned to
be a long-term-support release.

Bug Fixes
---------

* :func:`~bluesky.plan_stubs.mv` and :func:`~bluesky.plan_stubs.mvr` now works
  on pseudopositioners.
* :func:`~bluesky.preprocessors.reset_positions_wrapper` now works on
  pseudopositioners.
* Plans given an empty detectors list, such as ``count([])``, no longer break
  the :class:`~bluesky.callbacks.best_effort.BestEffortCallback`.

v0.11.0 (2017-11-01)
====================

This is the last release before 1.0.0. It contains major restructurings and
general clean-up.

Breaking Changes and Deprecations
---------------------------------

* The :mod:`bluesky.plans` module has been split into

    * :mod:`bluesky.plans` --- plans that create a run, such as :func:`count`
      and :func:`scan`
    * :mod:`bluesky.preprocessors` --- plans that take in other plans and
      motify them, such as :func:`baseline_wrapper`
    * :mod:`bluesky.plan_stubs` --- small plans meant as convenient building
      blocks for creating custom plans, such as :func:`trigger_and_read`
    * :mod:`bluesky.object_plans` and :mod:`bluesky.cntx`, containing
      legacy APIs to plans that were deprecated in a previous release and
      will be removed in a future release.

* The RunEngine raises a ``RunEngineInterrupted`` exception when interrupted
  (e.g. paused). The optional argument ``raise_if_interrupted`` has been
  removed.
* The module :mod:`bluesky.callbacks.scientific` has been removed.
* ``PeakStats`` has been moved to :mod:`bluesky.callbacks.fitting`, and
  :func:`plot_peak_stats` has been moved to `bluesky.callbacks.mpl_plotting`.
* The synthetic 'hardware' objects in ``bluesky.examples`` have been relocated
  to ophyd (bluesky's sister package) and aggressively refactored to be more
  closely aligned with the behavior of real hardware. The ``Reader`` and
  ``Mover`` classes have been removed in favor of ``SynSignal``,
  ``SynPeriodicSignal``, ``SynAxis``, ``SynSignalWithRegistry``.

Features
--------

* Add :func:`stub_wrapper` and :func:`stub_decorator` that strips
  open_run/close_run and stage/unstage messages out of a plan, so that it can
  be reused as part of a larger plan that manages the scope of a run manually.
* Add :func:`tune_centroid` plan that iteratively finds the centroid of a
  single peak.
* Allow devices with couple axes to be used in N-dimensional scan plans.
* Add :func:`contingency_wrapper` and :func:`contingency_decorator` for
  richer cleanup specification.
* The number of events in each event stream is recorded in the RunStop document
  under the key 'num_events'.
* Make the message shown when the RunEngine is paused configurable via the
  attribute ``RunEngine.pause_msg``.

Bug Fixes
---------

* Fix ordering of dimensions in :func:`grid_scan` hints.
* Show Figures created internally.
* Support a negative direction for adaptive scans.
* Validate that all descriptors with a given (event stream) name have
  consistent data keys.
* Correctly mark ``exit_status`` field in RunStop metadata based on which
  termination method was called (abort, stop, halt).
* ``LiveFitPlot`` handles updates more carefully.

Internal Changes
----------------

* The :mod:`bluesky.callbacks` package has been split up into more modules.
  Shim imports maintain backward compatibility, except where noted in the
  section on API Changes above.
* Matplotlib is now an optional dependency. If it is not importable,
  plotting-related callbacks will not be available.
* An internal change to the RunEngine supports ophyd's new Status object API
  for adding callbacks.

v0.10.3 (2017-09-12)
====================

Bug Fixes
---------

* Fix critical :func:`baseline_wrapper` bug.
* Make :func:`plan_mutator` more flexible. (See docstring.)

v0.10.2 (2017-09-11)
====================

This is a small release with bug fixes and UI improvements.

Bug Fixes
---------

* Fix bug wherein BestEffortCallback tried to plot strings as floats. The
  intended behavior is to skip them and warn.

Features
--------

* Include a more informative header in BestEffortCallback.
* Include an 'Offset' column in %wa output.

v0.10.1 (2017-09-11)
====================

This release is equivalent to v0.10.2. The number was skipped due to packaging
problems.

v0.10.0 (2017-09-06)
====================

Highlights
----------

* Automatic best-effort visualization and peak-fitting is available for all
  plans, including user-defined ones.
* The "SPEC-like" API has been fully removed, and its most useful features have
  been applied to the library in a self-consistent way. See the next section
  for detailed instructions on migrating.
* Improved tooling for streaming documents over a network for live processing
  and visualization in a different process or on a different machine.

Breaking Changes
----------------

* The modules implementing what was loosely dubbed a "SPEC-like" interface
  (``bluesky.spec_api`` and ``bluesky.global_state``) have been entirely
  removed. This approach was insufficently similar to SPEC to satisfy SPEC
  users and confusingly inconsistent with the rest of bluesky.

  The new approach retains the good things about that interface and makes them
  available for use with *all* plans consistently, including user defined ones.
  Users who have been fully utilitzing these "SPEC-like" plans will notice four
  differences.

  1. No ``gs.DETS``. Just use your own variable for detectors. Instead of:

     .. code-block:: python

         # OLD ALTERNATIVE, NO LONGER SUPPORTED

         from bluesky.global_state import gs
         from bluesky.spec_api import ct

         gs.DETS = # a list of some detectors
         RE(ct())

     do:

     .. code-block:: python

        from bluesky.plans import count

        dets = # a list of some detectors
        RE(count(dets))

     Notice that you can use multiple lists to enable easy task switching.
     Instead of continually updating one global list like this:

     .. code-block:: python

         # OLD ALTERNATIVE, NO LONGER SUPPORTED

         gs.DETS = # some list of detectors
         RE(ct())

         gs.DETS.remove(some_detector)
         gs.DETS.append(some_other_detector)
         RE(ct())

     you can define as many lists as you want and call them whatever you want.

     .. code-block:: python

        d1 = # a list of some detectors
        d2 = # a list of different detectors
        RE(count(d1))
        RE(count(d2))

  2. Automatic baseline readings, concurrent monitoring, and "flying"
     can be set up uniformly for all plans.

     Formerly, a list of devices to read at the beginning and the end of each
     run ("baseline" readings), a list of signals to concurrent monitor, and
     a list of "flyers" to run concurrently were configured like so:

     .. code-block:: python

        # OLD ALTERNATIVE, NO LONGER SUPPORTED

        from bluesky.spec_api import ct

        gs.BASELINE_DEVICES = # a list of devices to read at start and end
        gs.MONTIORS = # a list of signals to monitor concurrently
        gs.FLYERS = # a list of "flyable" devices

        gs.DETS = # a list of detectors

        RE(ct())  # monitoring, flying, and baseline readings are added

     And formerly, those settings only affected the behavior of the "SPEC-like"
     plans, such as ``ct`` and ``ascan``. They were ignored by their
     counterparts ``count`` and ``scan``, as well as user-defined plans. This
     was not desirable!

     This scheme has been replaced by the
     :ref:`supplemental data <supplemental_data>`, which can be
     used to globally modify *all* plans, including user-defined ones.

     .. code-block:: python

        from bluesky.plans import count

        # one-time configuration
        from bluesky import SupplementalData
        sd = SupplementalData()
        RE.preprocessors.append(sd)

        # interactive use
        sd.monitors = # a list of signals to monitor concurrently
        sd.flyers = # a list of "flyable" devices
        sd.baseline = # a list of devices to read at start and end

        dets = # a list of detectors
        RE(count(dets))  # monitoring, flying, and baseline readings are added

  3. Automatic live visualization and peak analysis can be set up uniformly for
     all plans.

     Formerly, the "SPEC-like" plans such as ``ct`` and ``ascan`` automatically
     set up a suitable table and a plot, while their "standard" vanilla
     counterparts, :func:`bluesky.plans.count` and :func:`bluesky.plans.scan`
     required explicit, detailed instructions to do so. Now, a best-effort
     table and plot can be made for *all* plans, including user-defined ones,
     by invoking this simple configuration:

     .. code-block:: python

        from bluesky.plans import count

        # one-time configuration
        from bluesky.callbacks.best_effort import BestEffortCallback
        bec = BestEffortCallback()
        RE.subscribe(bec)

        # interactive use
        dets = # a list of detectors
        RE(count(dets), num=5))  # automatically prints table, shows plot

     Use ``bec.disable()`` and ``bec.enable()`` to temporarily toggle the
     output off and on.

  4. Peak anallysis, now computed automatically by the BestEffortCallback
     above, can be viewed with a keyboard shortcut. The peak statistics,
     formerly encapsulated in ``gs.PS``, are now organized differently.

     For each plot, simple peak-fitting is performed in the background. Of
     course, it may or may not be applicable depending on your data, and it is
     not shown by default. To view fitting annotations in a plot, click the
     plot area and press Shift+P. (Lowercase p is a shortcut for
     "panning" the plot.)

     To access the peak-fit statistics programmatically, use ``bec.peaks``. For
     convenience, you may alias this like:

     .. code-block:: python

        peaks = bec.peaks

     Inside ``peaks``, access various statistics like:

     .. code-block:: python

        peaks.com
        peaks.cen
        peaks.max
        peaks.min

     Each of these is a dictionary with an entry for each field that was fit.
     For example, the 'center of mass' peak statistics for a field named
     ``'ccd_stats1_total'`` would be accessed like
     ``peaks.com['ccd_stats1_total']``.
* The functions and classes in the module ``bluesky.callbacks.broker`` require
  a instance of ``Broker`` to be passed in as an argument. They used to default
  to the 'singleton' instance via ``from databroker import db``, which is now a
  deprecated usage in databroker.
* The plan preprocessors ``configure_count_time_wrapper`` and
  ``configure_count_time_decorator`` were moved to ``bluesky.plans`` from
  ``bluesky.spec_api``, reverting a change made in v0.9.0.
* The 0MQ pubsub integration classes ``Publisher`` and ``RemoteDispatcher``
  have been overhauled. They have been moved from
  :mod:`bluesky.callbacks.zmqpub` and :mod:`bluesky.callbacks.zmqsub` to
  :mod:`bluesky.callbacks.zmq` and their signatures have been changed to match
  similar utilities in the pydata ecosystem. See the Enhancements section for
  more information.
* The module ``bluesky.qt_kicker`` has been removed. Its former contents are
  avaiable in ``bluesky.utils``. The module was originally deprecated in April
  2016, and it has been issuing warnings about this change since.
* The plan ``bluesky.plans.input`` has been renamed
  ``bluesky.plans.input_plan`` to avoid shadowing a builtin if the module is
  bulk-imported. The plan was previously undocumented and rarely used, so the
  impact of this change on users is expected to be small.

Deprecations
------------

* The module :mod:`bluesky.plan_tools` has been renamed
  :mod:`bluesky.simualtors`.  In the new module,
  :func:`bluesky.plan_tools.print_summary`` has been renamed
  :func:`bluesky.simulators.summarize_plan`.
  The old names are supported in this release, with a warning, but will be
  removed in a future release.
* The Object-Orientated plans (``Count``, ``Scan``, etc.) have been deprecated
  and will be removed in a future release. Their documentation has been
  removed.
* The plan context managers (``run_context``, ``stage_context``, etc.) have
  been deprecated and will be removed in a future release. They were never
  documented or widely used.
* The method :meth:`bluesky.Dispatcher.subscribe` (which is encapsulated into
  :class:`bluesky.RunEngine` and inherited by
  :class:`bluesky.callbacks.zmq.RemoteDispatcher`) has a new signature. The
  former signature was ``subscribe(name, func)``. The new signature is
  ``subscribe(func, name='all')``. Because the meaning of the arguments is
  unambigious (they must be a callable and a string, respectively) the old
  order will be supported indefeinitely, with a warning.

Features
--------

* A :doc:`progress bar <progress-bar>` add-on is available.
* As addressed above:
    * The new :ref:`supplemental data <supplemental_data>` feature make it
      easy to set up "baseline" readings and asynchronous acquisition in a way
      that applies automatically to all plans.
    * The new :ref:`best-effort callback <best_effort_callback>` sets up
      automatic terminal output and plots for all plans, including user-defined
      ones.
* ``LivePlot`` now accepts ``x='time'``. It can set t=0 to the UNIX epoch or to
  the start of the run. It also accepts ``x='seq_num'``---a synonym for
  ``x=None``, which remains the default.
* A new simulator :func:`bluesky.simulators.check_limits` verifies that a plan
  will not try to move a movable device outside of its limits.
* A new plan, :func:`bluesky.plan.mvr`, has been added as a relative counterpart
  to :func:`bluesky.plan.mv`.
* The 0MQ pubsub integration classes :class:`bluesky.callbacks.zmq.Publisher``
  and :class:`bluesky.callbacks.zmq.RemoteDispatcher` have been simplified.
  A new class :class:`bluesky.callbacks.zmq.Proxy` and command-line utility
  ``bluesky-0MQ-proxy`` has been added to streamline configuration.
* Metadata recorded by many built-in plans now includes a new item,
  ``'hints'``, which is used by the best-effort callback to produce useful
  visualizations. Additionally, the built-in examples devices have
  :ref:`a new hints attribute <hints>`. Its content may change or expand in
  future releases as this new feature is explored.
* Some :doc:`IPython magics <magics>` mimicing the SPEC API have been added.
  These are experimental and may be altered or removed in the future.

Bug Fixes
---------

* Using the "fake sleep" feature of simulated Movers (motors) caused them to
  break.
* The ``requirements.txt`` failed to declare that bluesky requires matplotlib.

v0.9.0 (2017-05-08)
===================

Breaking Changes
----------------

* Moved ``configure_count_time_wrapper`` and
  ``configure_count_time_decorator`` to ``bluesky.spec_api`` from
  ``bluesky.plans``.
* The metadata reported by step scans that used to be labeled ``num_steps``
  is now renamed ``num_points``, generally considered a less ambiguous name.
  Separately, ``num_interals`` (which one might mistakenly assume is what was
  meant by ``num_steps``) is also stored.


v0.8.0 (2017-01-03)
===================

Features
--------

* If some plan or callback has hung the RunEngine and blocked its normal
  ability to respond to Ctrl+C by pausing, it is not possible to trigger a
  "halt" (emergency stop) by hammering Ctrl+C more than ten times.

Bug Fixes
---------

* Fix bug where failed or canceled movements could cause future executions of
  the RunEngine to error.
* Fix bug in ``plan_mutator`` so that it properly handles return values. One
  effect of this fix is that ``baseline_wrapper`` properly passed run uids
  through.
* Fix bug in ``LiveFit`` that broke multivariate fits.
* Minor fixes to example detectors.

Breaking Changes
----------------

* A ``KeyboardInterrupt`` exception captured during a run used to cause the
  RunEngine to pause. Now it halts instead.

v0.7.0 (2016-11-01)
===================

Features
--------

* Nonlinear least-squares minimization callback ``LiveFit`` with
  ``LiveFitPlot``
* Added ``RunEngine.clear_suspenders()`` convenience method.
* New ``RunEngine.preprocessors`` list that modifies all plans passed to the
  RunEngine.
* Added ``RunEngine.state_hook`` to monitor state changes, akin to ``msg_hook``.
* Added ``pause_for_debug`` options to ``finalize_wrapper`` which allows pauses
  the RunEngine before performing any cleanup, making it easier to debug.
* Added many more examples and make it easier to create simulated devices that
  generate interesting simulated data. They have an interface closer to the
  real devices implemented in ophyd.
* Added ``mv``, a convenient plan for moving multiple devices in parallel.
* Added optional ``RunEngine.max_depth`` to raise an error if the RunEngine
  thinks it is being called from inside a function.

Bug Fixes
---------

* The 'monitor' functionality was completely broken, packing configuration
  into the wrong structure and starting seq_num from 0 instead of 1, which is
  the (regrettable) standard we have settled on.
* The RunEngine coroutines no longer mutate the messages they receive.
* Fix race condition in ``post_run`` callback.
* Fix bugs in several callbacks that caused them not to work on saved documents
  from the databroker. Also, make them call ``super()`` to play better with
  multiple inheritance in user code.


Breaking Changes
----------------

* The flag ``RunEngine.ignore_callback_exceptions`` now defaults to False.
* The plan ``complete``, related to fly scans, previously had ``wait=True`` by
  default, although its documentation indicated that ``False`` was the default.
  The code has been changed to match the documentation. Any calls to
  ``complete`` that are expected to be blocking should be updated with the
  keyword ``wait=True``.
* Completely change the API of ``Reader`` and ``Mover``, the classes for
  definding simulated devices.
* The bluesky interface now expects the ``stop`` method on a device to accept
  an optional ``success`` argument.
* The optional, undocumented ``fig`` argument to ``LivePlot`` has been
  deprecated and will be removed in a future release.  An ``ax`` argument has
  been added. Additionally, the axes used by ``LiveGrid`` and ``LiveScatter`` is
  configurable through a new, optional ``ax`` argument.
* The "shortcut" where mashing Ctrl+C three times quickly ran ``RE.abort()``
  has been removed.
* Change the default stream name for monitors to ``<signal_name>_monitor`` from
  ``signal_name>-monitor`` (underscore vs. dash). The impact of this change is
  minimal because, as noted above, the monitor functionality was completely
  broken in previous releases.

v0.6.4 (2016-09-07)
===================

Features
--------

* Much-expanded and overhauled documentation.
* Add ``aspect`` argument to ``LiveGrid``.
* Add ``install_nb_kicker`` to get live-updating matplotlib figures in the
  notebook while the RunEngine is running.
* Simulated hardware devices ``Reader`` and ``Mover`` can be easily customized
  to mock a wider range of behaviors, for testing and demos.
* Integrate the SPEC API with mew global state attribute ``gs.MONITORS``.
* Callbacks that use the databroker accept an optional ``Broker`` instance
  as an argument.

Bug Fixes
---------

* Minor fix in the tilt computation for spiral scans.
* Expost 'tilt' option through SPEC-like API
* The "infinite count" (``ct`` with ``num=None``) should spawn a LivePlot.
* ``finalize_decorator`` accepts a callable (e.g., generator function)
  and does not accept an iterable (e.g., generator instance)
* Restore ``gs.FLYERS`` integration to the SPEC API (accidentally removed).

Breaking Changes
----------------

* The API for the simulated hardware example devices ``Reader`` and ``Mover``
  has been changed to make them more general.
* Remove ``register_mds`` metadatastore integration.

v0.6.3 (2016-08-16)
===================

Features
--------

* Change how "subscription factories" are handled, making them configurable
  through global state.
* Make PeakStats configurable through global state.
* Add an experimental utility for passing documents over a network and
  processing them on a separate process or host, using 0MQ.
* Add ``monitor_during_wrapper`` and corresponding decorator.
* Add ``stage_wrapper`` and corresponding decorator.
* Built-in plans return the run uid that they generated.
* Add a new ``ramp_plan`` for taking data while polling the status of a
  movement.

Bug Fixes
---------

* Boost performance by removing unneeded "sleep" step in message processing.
* Fix bug related to rewinding in preparation for resuming.

Breaking Changes
----------------

* Remove the ``planify`` decorator and the plan context managers. These were
  experimental and ultimately proved problematic because they make it difficult
  to pass through return values cleanly.
* Remove "lossy" subscriptions feature, rendered unnecessary by the utility for
  processing documents in separate processes (see Enhancements, above).

v0.6.2 (2016-07-26)
===================

Bug Fixes
---------

* Make ``make_decorator`` return proper decorators. The original implementation
  returned functions that could not actually be used as decorators.

v0.6.1 (2016-07-25)
===================

This release contained only a minor UX fix involving more informative error
reporting related to Area Detector plugin port configuration.

v0.6.0 (2016-07-25)
===================

Features
--------

* Address the situation where plan "rewinding" after a pause or suspension
  interacted badly with some devices. There are now three ways to temporarily
  turn off rewinding: a Msg with a new 'rewindable' command; a special
  attribute on the device that the ``trigger_and_read`` plan looks for;
  and a special exception that devices can raise when their ``pause`` method
  is called. All three of these features should be considered experimental.
  They will likely be consolidated in the future once their usage is tested
  in the wild.
* Add new plan wrappers and decorators: ``inject_md_wrapper``, ``run_wrapper``,
  ``rewindable_wrapper``.

Bug Fixes
---------

* Fix bug where RunEngine was put in the "running" state, encountered an
  error before starting the ``_run`` coroutine, and thus never switch back to
  "idle."
* Ensure that plans are closed correctly and that, if they fail to close
  themselves, a warning is printed.
* Allow plan to run its cleanup messages (``finalize``) when the RunEngine is
  stopped or aborted.
* When an exception is raised, give each plan in the plan stack an opportunity
  to handle it. If it is handled, carry on.
* The SPEC-style ``tw`` was not passing its parameters through to the
  underlying ``tweak`` plan.
* Silenced un-needed suspenders warnings
* Fix bug in separating devices

Internal Changes
----------------

* Reduce unneeded usage of ``bluesky.plans.single_gen``.
* Don't emit create/save messages with no reads in between.
* Re-work exception handling in main run engine event loop.

v0.5.3 (2016-06-06)
===================

Breaking Changes
----------------

* ``LiveTable`` only displays data from one event stream.
* Remove used global state attribute ``gs.COUNT_TIME``.

Bug Fixes
---------

* Fix "infinite count", ``ct(num=None)``.
* Allow the same data keys to be present in different event streams. But, as
  before, a given data key can only appear once per event.
* Make SPEC-style plan ``ct`` implement baseline readings, referring to
  ``gs.BASELINE_DETS``.
* Upon resuming after a deferred pause, clear the deferred pause request.
* Make ``bluesky.utils.register_transform`` character configurable.

v0.5.2 (2016-05-25)
===================

Features
--------

* Plans were reimplemented as simple Python generators instead of custom Python
  classes. The old "object-oriented" plans are maintained for
  back-compatibility. See plans documentation to review new capabilities.

Breaking Changes
----------------

* SPEC-style plans are now proper generators, not bound to the RunEngine.

v0.5.0 (2016-05-11)
===================

Breaking Changes
----------------

* Move ``bluesky.scientific_callbacks`` to ``bluesky.callbacks.scientific``
  and ``bluesky.broker_callbacks`` to ``bluesky.callbacks.broker``.
* Remove ``bluesky.register_mds`` whose usage can be replaced by:
  ``import metadatastore.commands; RE.subscribe_lossless('all', metadatastore.commands.insert)``
* In all occurrences, the argument ``block_group`` has been renamed ``group``
  for consistency. This affects the 'trigger' and 'set' messages.
* The (not widely used) ``Center`` plan has been removed. It may be
  distributed separately in the future.
* Calling a "SPEC-like" plan now returns a generator that must be passed
  to the RunEngine; it does not execute the plan with the global RunEngine in
  gs.RE. There is a convenience wrapper available to restore the old behavior
  as desired. But since that usage renders the plans un-composable, it is
  discouraged.
* The 'time' argument of the SPEC-like plans is a keyword-only argument.
* The following special-case SPEC-like scans have been removed

    * hscan
    * kscan
    * lscan
    * tscan
    * dtscan
    * hklscan
    * hklmesh

  They can be defined in configuration files as desired, and in that location
  they will be easier to customize.
* The ``describe`` method on flyers, which returns an iterable of dicts of
  data keys for one or more descriptors documents, has been renamed to
  ``describe_collect`` to avoid confusion with ``describe`` on other devices,
  which returns a dict of data keys for one descriptor document.
* An obscure feature in ``RunEngine.request_pause`` has been removed, which
  involved removing the optional arguments ``callback`` and ``name``.

v0.4.3 (2016-03-03)
===================

Bug Fixes
---------

* Address serious performance problem in ``LiveTable``.

v0.4.2 (2016-03-02)
===================

Breaking Changes
----------------

* Stage the ultimate parent ("root") when a device is staging its child, making
  it impossible to leave a device in a partially-staged state.

v0.4.1 (2016-02-29)
===================

Features
--------

* Give every event stream a ``name``, using ``'primary'`` by default.
* Record a mapping of device/signal names to ordered data keys in the
  EventDescriptor.
* Let ``LiveRaster`` account for "snaked" trajectories.

Bug Fixes
---------

* ``PeakStats.com`` is a scalar, not a single-element array.
* Restore Python 3.4 compatibility.

v0.4.0 (2016-02-23)
===================

(TO DO)

v0.3.2 (2015-10-28)
===================

(TO DO)

v0.3.1 (2015-10-15)
===================

(TO DO)

v0.3.0 (2015-10-14)
===================

Breaking Changes
----------------

* Removed ``RunEngine.persistent_fields``; all fields in ``RE.md`` persist
  between runs by default.
* No metadata fields are "reserved"; any can be overwritten by the user.
* No metadata fields are absolutely required. The metadata validation function
  is user-customizable. The default validation function behaves the same
  as previous versions of bluesky, but it is no longer manditory.
* The signature of ``RunEngine`` has changed. The ``logbook`` argument is now
  keyword-only, and there is a new keyword-only argument, ``md_validator``.
  See docstring for details.
* The ``configure`` method on readable objects now takes a single optional
  argument, a dictionary that the object can use to configure itself however
  it sees fit. The ``configure`` method always has a new return value, a tuple
  of dicts describing its old and new states:
  ``old, new = obj.configure(state)``
* Removed method ``increment_scan_id``
* `callbacks.broker.post_run` API and docstring brought into agreement.
  The API is change to expect a callable with signature
  ``foo(doc_name, doc)`` rather than

    - a callable which takes a document (as documented)
    - an object with ``start``, ``descriptor``, ``event`` and ``stop``
      methods (as implemented).

  If classes derived from ``CallbackBase`` are being used this will not
  not have any effect on user code.

v0.2.3 (2015-09-29)
===================

(TO DO)

v0.2.2 (2015-09-24)
===================

(TO DO)

v0.2.1 (2015-09-24)
===================

(TO DO)

v0.2.0 (2015-09-22)
===================

(TO DO)

v0.1.0 (2015-06-25)
===================

Initial release
