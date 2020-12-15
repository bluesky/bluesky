#!/bin/bash
set -vxeuo pipefail

# These packages are installed in the base environment but may be older
# versions. Explicitly upgrade them because they often create
# installation problems if out of date.
python -m pip install --upgrade pip setuptools wheel numpy
# Versioneer uses the most recent git tag to generate __version__, which appears
# in the published documentation.
git fetch --tags
python -m pip install .
python -m pip list
