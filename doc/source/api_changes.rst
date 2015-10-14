API Changes
===========


v0.4.0
------
* `broker_callbacks.post_run` API and docstring brought into agreement.
  The API is change to expect a callable with signature
  ``foo(doc_name, doc)`` rather than:
    - a callable which takes a document (as documented)
    - an object with ``start``, ``descriptor``, ``event`` and ``stop``
      methods (as implemented).

  If classes derived from `CallbackBase` are being used this will not
  not have any effect on user code.


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
