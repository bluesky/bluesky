import pytest


def test_imports_raise_warnings():
    with pytest.warns(DeprecationWarning):
        pass

    with pytest.warns(DeprecationWarning):
        pass
