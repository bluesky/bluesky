This folder contains source code of vendored packages obtained using the automated `vendoring` tool. Tis allows us to reduce dependency on external packages with unreliable maintenance. For more detail, see https://github.com/pypa/pip/tree/main/src/pip/_vendor.

To vendor additional package, add it to `vendor.txt` (if desired, along with its dependencies) and then run `vendoring sync . -v`; the configuration of `vendoring` is declared in `pyproject.toml`.

The testing code for vendored packages in `tests/_vendor` is copied manually without changes from their corresponding repositories.

The vendored packages retain their original licenses, which are included.