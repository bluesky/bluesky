[![CI](https://github.com/bluesky/bluesky/actions/workflows/ci.yml/badge.svg)](https://github.com/bluesky/bluesky/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/bluesky/bluesky/branch/main/graph/badge.svg)](https://codecov.io/gh/bluesky/bluesky)
[![PyPI](https://img.shields.io/pypi/v/bluesky.svg)](https://pypi.org/project/bluesky)


# Bluesky — An Experiment Specification & Orchestration Engine

|    Source     |     <https://github.com/bluesky/bluesky>      |
| :-----------: | :-------------------------------------------: |
|     PyPI      |             `pip install bluesky`             |
| Documentation |      <https://bluesky.github.io/bluesky>      |
|   Releases    | <https://github.com/bluesky/bluesky/releases> |

Bluesky is a library for experiment control and collection of scientific data
and metadata. It emphasizes the following virtues:

* **Live, Streaming Data:** Available for inline visualization and processing.
* **Rich Metadata:** Captured and organized to facilitate reproducibility and
  searchability.
* **Experiment Generality:** Seamlessly reuse a procedure on completely
  different hardware.
* **Interruption Recovery:** Experiments are "rewindable," recovering cleanly
  from interruptions.
* **Automated Suspend/Resume:** Experiments can be run unattended,
  automatically suspending and resuming if needed.
* **Pluggable I/O:** Export data (live) into any desired format or database.
* **Customizability:** Integrate custom experimental procedures and commands,
  and get the I/O and interruption features for free.
* **Integration with Scientific Python:** Interface naturally with numpy and
  Python scientific stack.

[**Bluesky Documentation**](http://blueskyproject.io/bluesky).

The Bluesky Project enables experimental science at the lab-bench or facility scale. It is a collection of Python libraries that are co-developed but independently useful and may be adopted *a la carte*.

[**Bluesky Project Documentation**](http://blueskyproject.io).

<!-- README only content. Anything below this line won't be included in index.md -->

See https://bluesky.github.io/bluesky for more detailed documentation.
