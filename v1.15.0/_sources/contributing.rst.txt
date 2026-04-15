============
Contributing
============

Getting Started
===============

* Make sure you have a GitHub account
* Submit a ticket for your issue, assuming one does not already exist.

   * Clearly describe the issue including steps to reproduce when it is a bug.
   * Make sure you fill in the earliest version that you know has the issue.
* Fork the repository on GitHub


Making Changes
==============

* Create a topic branch from where you want to base your work.

   * This is usually the master branch.
   * Only target release branches if you are certain your fix must be on that
     branch.
   * To quickly create a topic branch based on master; `git checkout -b
     fix/master/my_contribution master`. Please avoid working directly on the
     `master` branch.
* Make commits of logical units.
* Check for unnecessary whitespace with `git diff --check` before committing.
* Make sure your commit messages are in the proper format (see below)
* Make sure you have added the necessary tests for your changes.
* Run _all_ the tests to assure nothing else was accidentally broken.

Writing the commit message
--------------------------

Commit messages should be clear and follow a few basic rules. Example:

.. code-block::

   ENH: add functionality X to bluesky.<submodule>.

   The first line of the commit message starts with a capitalized acronym
   (options listed below) indicating what type of commit this is.  Then a blank
   line, then more text if needed.  Lines shouldn't be longer than 72
   characters.  If the commit is related to a ticket, indicate that with
   "See #3456", "See ticket 3456", "Closes #3456" or similar.

Describing the motivation for a change, the nature of a bug for bug fixes
or some details on what an enhancement does are also good to include in a
commit message. Messages should be understandable without looking at the code
changes.

Standard acronyms to start the commit message with are:

* API: an (incompatible) API change
* BLD: change related to building the project
* BUG: bug fix
* CI : continuous integration
* DEP: deprecate something, or remove a deprecated object
* DEV: development tool or utility
* DOC: documentation
* ENH: enhancement
* MNT: maintenance commit (refactoring, typos, etc.)
* REV: revert an earlier commit
* STY: style fix (whitespace, PEP8)
* TST: addition or modification of tests
* REL: related to releases

The Pull Request
----------------

* Now push to your fork
* Submit a `pull request <https://help.github.com/articles/using-pull-requests>`_

At this point you're waiting on us. We like to at least comment on pull requests within three business days (and, typically, one business day).
We may suggest some changes or improvements or alternatives.

Hints to make the integration of your changes easy (and happen faster):
* Keep your pull requests small
* Don't forget your unit tests
* All features need documentation, don't forget the .rst file
* Don't take changes requests to change your code personally

Releasing
=========

To release Bluesky:

* Make a pull request adding release notes to the API changes document:
  :file:`docs/source/api_changes.rst`.
  It might be helpful to `review the changes since last release <https://docs.github.com/en/github/administering-a-repository/releasing-projects-on-github/comparing-releases>`_.
* Once the changelog is merged, mint a new release.

   * tag: :code:`v<VERSION>`
   * title: :code:`bluesky v<VERSION>`
   * description: copy content from api_changes
* Bluesky will be released to PyPI via our `github action <https://github.com/bluesky/bluesky/blob/master/.github/workflows/python-publish.yml>`_.
* The conda-forge bot will make a PR to update the `bluesky-feedstock <https://github.com/conda-forge/bluesky-feedstock>`_.
