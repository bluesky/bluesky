import collections.abc

import numpy
from numpy.testing import assert_array_equal

from ..utils import PersistentDict
from ..plans import count


def test_persistent_dict(tmp_path):
    d = PersistentDict(tmp_path)
    d['a'] = 1
    d['b'] = (1, 2)
    d['c'] = numpy.zeros((5, 5))
    d['d'] = {'a': 10, 'b': numpy.ones((5, 5))}
    expected = dict(d)
    actual = PersistentDict(tmp_path)
    recursive_assert_equal(actual, expected)

    # Update a value and check again.
    d['a'] = 2
    expected = dict(d)
    recursive_assert_equal(actual, expected)

    # Smoke test the accessor and the __repr__.
    assert d.directory == tmp_path
    d.__repr__()


def test_integration(tmp_path, RE, hw):
    """
    Test integration with RE.

    Not looking for anything *specific* here, just general paranoia in case
    unforseen future changes create a bad interaction between PersistentDict
    and RE, as happened with HistoryDict and RE.
    """
    d = PersistentDict(tmp_path)
    d['a'] = 1
    d['b'] = (1, 2)
    d['c'] = numpy.zeros((5, 5))
    d['d'] = {'a': 10, 'b': numpy.ones((5, 5))}
    expected = dict(d)
    expected['scan_id'] = 1

    RE.md = d
    RE(count([hw.det]))
    recursive_assert_equal(RE.md, expected)

    reloaded = PersistentDict(tmp_path)
    recursive_assert_equal(reloaded, expected)


def recursive_assert_equal(actual, expected):
    assert set(actual.keys()) == set(expected.keys())
    for key in actual:
        if isinstance(actual[key], collections.abc.MutableMapping):
            recursive_assert_equal(actual[key], expected[key])
        else:
            assert_array_equal(actual[key], expected[key])
