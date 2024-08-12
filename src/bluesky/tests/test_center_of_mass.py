import numpy as np
from scipy import ndimage

from bluesky.callbacks.fitting import center_of_mass


def test_center_of_mass():
    assert True

    arr = np.array(([0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]))
    result = ndimage.measurements.center_of_mass(arr)
    assert result == (2.0, 1.5), f"{result=}"

    # Calculation of multiple objects in an image
    b_arr = np.array(([0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]))
    lbl = ndimage.label(b_arr)[0]
    result = ndimage.measurements.center_of_mass(b_arr, lbl, [1, 2])
    assert result == [(0.33333333333333331, 1.3333333333333333), (3.5, 2.5)]
