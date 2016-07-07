API Changes
===========

v0.6.0
------

Enhancements
^^^^^^^^^^^^
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
^^^^^^^^^
* Fix bug where RUnEngine was put in the "running" state, encountered an
  error before starting the ``_run`` coroutine, and thus never switch back to
  "idle."
* Ensure that plans are closed correctly and that, if they fail to close
  themselves, a warning is printed.
* Allow plan to run its cleanup messages (``finalize``) when the RunEngine is
  stopped or aborted.
* When an exception is raised, give each plan in the plan stack an opportunity
  to handle it. If it is handled, carry on.
* Make ``make_decorator`` return proper generators.
* The SPEC-style ``tw`` was not passing its parameters through to the
  underlying ``tweak`` plan.
* Silenced un-needed suspenders warnings
* Fix bug in separating devices

Cleanup
^^^^^^^
* Reduce unneeded usage of ``bluesky.plans.single_gen``.
* Don't emit create/save messages with no reads in between.
* Re-work exception handling in main run engine event loop.

v0.5.3
------

API Changes
^^^^^^^^^^^
* ``LiveTable`` only displays data from one event stream.
* Remove used global state attribute ``gs.COUNT_TIME``.

Bug Fixes
^^^^^^^^^
* Fix "infinite count", ``ct(num=None)``.
* Allow the same data keys to be present in different event streams. But, as
  before, a given data key can only appear once per event.
* Make SPEC-style plan ``ct`` implement baseline readings, referring to
  ``gs.BASELINE_DETS``.
* Upon resuming after a deferred pause, clear the deferred pause request.
* Make ``bluesky.utils.register_transform`` character configurable.

v0.5.2
------
* Plans were completely refactored. The API of the exist plans is supported
  for back-compatibility. See plans documentation to review new capabilities.
* SPEC-style plans are now proper generators, not bound to the RunEngine.


v0.5.0
------

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

v0.3.0
------

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

  If classes derived from `CallbackBase` are being used this will not
  not have any effect on user code.
