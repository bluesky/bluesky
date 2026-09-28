import sys

import pytest


def test_imports_raise_warnings():
    # Pop from `sys.modules` so the deprecation `warnings.warn(...)` at
    # the top of each shim module re-fires on (re)import.
    sys.modules.pop("bluesky.callbacks.tiled_writer", None)
    with pytest.warns(DeprecationWarning, match="bluesky.callbacks.tiled_writer"):
        import bluesky.callbacks.tiled_writer  # noqa: F401

    sys.modules.pop("bluesky.consolidators", None)
    with pytest.warns(DeprecationWarning, match="bluesky.consolidators"):
        import bluesky.consolidators  # noqa: F401
