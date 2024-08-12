import numpy as np
import pytest

from bluesky.callbacks.fitting import center_of_mass


@pytest.mark.parametrize(
    "arr, labels, index, expected",
    [
        # example from the source code
        [[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]], None, None, (2.0, 1.5)],
        # signal is all zero, does not raise ZeroDivisionError or other
        [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], None, None, (0.0, 0.0)],
        # multiple objects in an image
        # # FIXME: TypeError: Field elements must be 2- or 3-tuples, got '1'
        # [
        #     [[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]],
        #     [[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]],
        #     [1, 2],
        #     [(1.0 / 3, 4.0 / 3), (3.5, 2.5)],
        # ],
        # 1-D
        [[0, 1, 1, 1, 0], None, None, (2.0,)],
    ],
)
def test_center_of_mass(arr, labels, index, expected):
    if labels is not None:
        labels = np.array(labels)
    result = center_of_mass(np.array(arr), labels=labels, index=index)
    assert result == expected, f"{result=}"
