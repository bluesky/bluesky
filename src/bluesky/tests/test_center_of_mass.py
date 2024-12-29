import numpy as np
import pytest

from bluesky.callbacks.fitting import center_of_mass


@pytest.mark.parametrize(
    "arr, expected",
    [
        # example from the source code
        [[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]], (2.0, 1.5)],
        # signal is all zero, does not raise ZeroDivisionError or other
        [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  None],
        [[0, 1, 1, 1, 0], (2.0,)],
    ],
)
def test_center_of_mass(arr, labels, index, expected):
    result = center_of_mass(np.array(arr))
    assert result == expected, f"{result=}"
