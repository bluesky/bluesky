Comparison with SPEC
====================

`SPEC <https://www.certif.com/content/spec/>`_ is a popular software package
for instrument control and data acquisition. Many users in the synchrotron
community, from which bluesky originated, know SPEC and ask what differentiates
and motivates bluesky. Answering this question in an informed and unbiased way
is difficult, and we welcome any corrections.

There are many good features of SPEC that have been incorporated into
bluesky, including:

* Simple commands for common experiments, which can be used as building blocks
  for more complex procedures
* Easy hardware configuration
* Interruption handling (Ctrl+C)
* Integration with EPICS (and potentially other instrument control systems)
* "Pseudomotors" presenting virtual axes
* Integration with reciprocal space transformation code

Bluesky has also addressed certain limitations of SPEC. In fairness to SPEC, we
have the benefit of learning from its decades of use, and we are standing on
the shoulders of the modern open-source community.

* Bluesky is free and open-source in all aspects. Macros from the SPEC user
  community are open-source, but the core SPEC C source code is closed and not
  free.
* Bluesky provides more control over the console output and plotting.
* SPEC was designed before large area detectors existed. Ingesting area
  detector data is possible, but *ad hoc*. In bluesky, area detectors and other
  higher-dimensional inputs are integrated naturally.
* SPEC writes to a custom text-based format (a "SPEC file"). Bluesky can
  write---in real time or *post facto*---to any format.
* SPEC has a simulation mode. Bluesky allows users to incorporate much richer
  simulation capabilities (about which more below) but, as of version 0.9.0,
  provides less than SPEC out of the box.

Using Python, a general-purpose programming language, gives several immediate
advantages:

* Python checks syntax automatically.
* Python provides tools for interactive debugging.
* There are many more resources for learning Python.
* The language is more flexible.
* It's easy to integrate with the scientific Python ecosystem.

Bluesky tries to go further than SPEC in some regards:

* Complex custom procedures are easier to express.
* Automated "suspension" (pausing and resuming) is consistent and easier to
  manage.
* The prevailing model in SPEC is to collect data as a step scan. Other types
  of scans---such as fly scans or asynchronous monitoring---can be
  done, but they are *ad hoc*. Bluesky supports several modalities of data
  acquisition with equal ease.
* Bluesky can acquire multiple asynchronous, uncoordinated streams of data and
  represent them in a simple :doc:`event-based data model <documents>`.
* It is easy to build tools that inspect/simulate a procedure before it is run
  to check for safety, estimate time to completion, or visualize its behavior.
* Bluesky is a library that works well interactively but can also be used
  programmatically in scripts or other libraries.
* Users can add arbitrary metadata with rich semantics, including large arrays
  (such as masks) or nested mappings.
* Bluesky is a holistic solution for data acquisition and management. Users can
  push live streaming data directly into their data processing and analysis
  pipelines and/or export it into a file.

On the other hand, one major advantage of SPEC over bluesky is its maturity.
SPEC is battle-hardened from decades of use at many facilities, and it has a
large user community. Bluesky is a young project.

A Remark About Syntax
---------------------

SPEC users immediately notice that simple bluesky commands are more verbose
than their counterparts in SPEC. This is a trade-off we have made in choosing a
more expressive, general-purpose language over a single-purpose command line
interface. That "easy integration with scientific libraries" comes at the cost
of some parentheses and commas. Some of the difference is also due to the
richer abstractions required to capture the complexity of modern hardware.  The
simplest commands are made less terse, but more interesting commands are made
much easier to express. Of course, users can save time by using tab-completion
and by accessing previous commands with the up arrow key.
