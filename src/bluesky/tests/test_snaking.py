from bluesky.utils import snake_cyclers
from cycler import cycler


x = cycler('x', [1, 2, 3])
y = cycler('y', [1, 2])
z = cycler('z', [1, 2, 3])


def test_snake_no_snaking():
    actual = list(snake_cyclers([z, y, x], [False, False, False]))
    expected = [
        {'x': 1, 'y': 1, 'z': 1},
        {'x': 2, 'y': 1, 'z': 1},
        {'x': 3, 'y': 1, 'z': 1},
        {'x': 1, 'y': 2, 'z': 1},
        {'x': 2, 'y': 2, 'z': 1},
        {'x': 3, 'y': 2, 'z': 1},
        {'x': 1, 'y': 1, 'z': 2},
        {'x': 2, 'y': 1, 'z': 2},
        {'x': 3, 'y': 1, 'z': 2},
        {'x': 1, 'y': 2, 'z': 2},
        {'x': 2, 'y': 2, 'z': 2},
        {'x': 3, 'y': 2, 'z': 2},
        {'x': 1, 'y': 1, 'z': 3},
        {'x': 2, 'y': 1, 'z': 3},
        {'x': 3, 'y': 1, 'z': 3},
        {'x': 1, 'y': 2, 'z': 3},
        {'x': 2, 'y': 2, 'z': 3},
        {'x': 3, 'y': 2, 'z': 3}]
    assert actual == expected


def test_snake_all_snaking():
    actual = list(snake_cyclers([z, y, x], [False, True, True]))
    expected = [
        {'x': 1, 'y': 1, 'z': 1},
        {'x': 2, 'y': 1, 'z': 1},
        {'x': 3, 'y': 1, 'z': 1},
        {'x': 3, 'y': 2, 'z': 1},
        {'x': 2, 'y': 2, 'z': 1},
        {'x': 1, 'y': 2, 'z': 1},
        {'x': 1, 'y': 2, 'z': 2},
        {'x': 2, 'y': 2, 'z': 2},
        {'x': 3, 'y': 2, 'z': 2},
        {'x': 3, 'y': 1, 'z': 2},
        {'x': 2, 'y': 1, 'z': 2},
        {'x': 1, 'y': 1, 'z': 2},
        {'x': 1, 'y': 1, 'z': 3},
        {'x': 2, 'y': 1, 'z': 3},
        {'x': 3, 'y': 1, 'z': 3},
        {'x': 3, 'y': 2, 'z': 3},
        {'x': 2, 'y': 2, 'z': 3},
        {'x': 1, 'y': 2, 'z': 3}]
    assert actual == expected


def test_snake_some_snaking():
    actual = list(snake_cyclers([z, y, x], [False, True, False]))
    expected = [
        {'x': 1, 'y': 1, 'z': 1},
        {'x': 2, 'y': 1, 'z': 1},
        {'x': 3, 'y': 1, 'z': 1},
        {'x': 1, 'y': 2, 'z': 1},
        {'x': 2, 'y': 2, 'z': 1},
        {'x': 3, 'y': 2, 'z': 1},
        {'x': 1, 'y': 2, 'z': 2},
        {'x': 2, 'y': 2, 'z': 2},
        {'x': 3, 'y': 2, 'z': 2},
        {'x': 1, 'y': 1, 'z': 2},
        {'x': 2, 'y': 1, 'z': 2},
        {'x': 3, 'y': 1, 'z': 2},
        {'x': 1, 'y': 1, 'z': 3},
        {'x': 2, 'y': 1, 'z': 3},
        {'x': 3, 'y': 1, 'z': 3},
        {'x': 1, 'y': 2, 'z': 3},
        {'x': 2, 'y': 2, 'z': 3},
        {'x': 3, 'y': 2, 'z': 3}]
    assert actual == expected
