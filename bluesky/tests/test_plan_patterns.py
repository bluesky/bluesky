import pytest
import numpy as np
import numpy.testing as npt
import itertools
from bluesky.plan_patterns import (
    chunk_outer_product_args, outer_product,
    OuterProductArgsPattern, classify_outer_product_args_pattern)


@pytest.mark.parametrize("args, pattern", [
    (("motor1", 1, 5, 10),
     OuterProductArgsPattern.PATTERN_1),
    (("motor1", 1, 5, 10, "motor2", 2, 10, 20),
     OuterProductArgsPattern.PATTERN_1),
    (("motor1", 1, 5, 10, "motor2", 2, 10, 20, "motor3", 3, 20, 40),
     OuterProductArgsPattern.PATTERN_1),
    (("motor1", 1, 5, 10, "motor2", 2, 10, 20, True),
     OuterProductArgsPattern.PATTERN_2),
    (("motor1", 1, 5, 10, "motor2", 2, 10, 20, True, "motor3", 3, 20, 40, False),
     OuterProductArgsPattern.PATTERN_2),
    # Ambiguous cases (24 elements in 'args')
    (("motor", 1, 7, 24,
      "motor1", 1, 5, 10,
      "motor2", 2, 10, 20,
      "motor3", 3, 5, 15,
      "jittery_motor1", 4, 6, 50,
      "jittery_motor2", 6, 7, 30,
      "motor", 6, 7, 30),
     OuterProductArgsPattern.PATTERN_1),
    (("motor", 1, 7, 24,
      "motor1", 1, 5, 10, False,
      "motor2", 2, 10, 20, True,
      "motor3", 3, 5, 15, False,
      "jittery_motor1", 4, 6, 50, True,
      "jittery_motor2", 6, 7, 30, False),
     OuterProductArgsPattern.PATTERN_2),
])
def test_classify_outer_product_args_pattern(hw, args, pattern):
    """Basic functionality of 'classify_outer_product_args_pattern"""

    # Convert motor names to actual motors in the argument list using fixture 'hw'
    args = tuple([getattr(hw, _) if isinstance(_, str) else _ for _ in args])

    p = classify_outer_product_args_pattern(args)
    assert p == pattern, "Pattern is incorrectly identified"


def test_classify_outer_product_args_pattern_fail(hw):
    """Failing cases for `classify_outer_product_args_pattern`"""

    # Wrong number of elements (doesn't fit any pattern)
    args = (hw.motor1, 1, 5, 10, hw.motor2, 2, 10, 20, hw.motor3, 3, 20)
    with pytest.raises(ValueError, match="Wrong number of elements in 'args'"):
        classify_outer_product_args_pattern(args)

    # Missing or extra motors in the 'args' list
    args_list = [
        (hw.motor1, 1, 5, 10, 50, 2, 10, 20, hw.motor3, 3, 20, 40),
        (hw.motor1, 1, 5, 10, hw.motor2, hw.motor, 10, 20, hw.motor3, 3, 20, 40),
        (hw.motor1, 1, 5, 10, 50, 2, 10, 20, True, hw.motor3, 3, 20, 40, False),
        (hw.motor1, 1, 5, 10, hw.motor2, hw.motor, 10, 20, True, hw.motor3, 3, 20, 40, False),
    ]
    for args in args_list:
        with pytest.raises(ValueError, match="Incorrect order of elements in the argument list 'args'"):
            classify_outer_product_args_pattern(args)


@pytest.mark.parametrize("args, chunked_args", [
    (("motor1", 1, 5, 10), [("motor1", 1, 5, 10, False)]),
    (("motor1", 1, 5, 10, "motor2", 2, 10, 20),
     [("motor1", 1, 5, 10, False), ("motor2", 2, 10, 20, False)]),
    (("motor1", 1, 5, 10, "motor2", 2, 10, 20, "motor3", 3, 5, 15),
     [("motor1", 1, 5, 10, False), ("motor2", 2, 10, 20, False), ("motor3", 3, 5, 15, False)])
])
def test_chunk_outer_product_args_1(hw, args, chunked_args):
    """Check if the new pattern (Pattern 1) works"""

    # Convert motor names to actual motors in the argument list using fixture 'hw'
    args = tuple([getattr(hw, _) if isinstance(_, str) else _ for _ in args])
    for n, a in enumerate(chunked_args):
        chunked_args[n] = tuple([getattr(hw, _) if isinstance(_, str) else _ for _ in a])

    assert list(chunk_outer_product_args(args)) == chunked_args, \
        "Argument list was split into chunks incorrectly"


@pytest.mark.parametrize("args, chunked_args", [
    (("motor1", 1, 5, 10), [("motor1", 1, 5, 10, False)]),
    (("motor1", 1, 5, 10, "motor2", 2, 10, 20, True),
     [("motor1", 1, 5, 10, False), ("motor2", 2, 10, 20, True)]),
    (("motor1", 1, 5, 10, "motor2", 2, 10, 20, True, "motor3", 3, 5, 15, False),
     [("motor1", 1, 5, 10, False), ("motor2", 2, 10, 20, True), ("motor3", 3, 5, 15, False)])
])
def test_chunk_outer_product_args_2(hw, args, chunked_args):
    """Check if the pattern (Pattern 2) works"""

    # Convert motor names to actual motors in the argument list using fixture 'hw'
    args = tuple([getattr(hw, _) if isinstance(_, str) else _ for _ in args])
    for n, a in enumerate(chunked_args):
        chunked_args[n] = tuple([getattr(hw, _) if isinstance(_, str) else _ for _ in a])

    """Check if deprecated pattern (Pattern 2) is properly supported"""
    assert list(chunk_outer_product_args(args)) == chunked_args, \
        "Argument list was split into chunks incorrectly"


@pytest.mark.parametrize("args, chunked_args", [
    # Pattern 1
    (("motor", 1, 7, 24,
      "motor1", 1, 5, 10,
      "motor2", 2, 10, 20,
      "motor3", 3, 5, 15,
      "jittery_motor1", 4, 6, 50,
      "jittery_motor2", 6, 7, 30,
      "motor", 6, 7, 30),
     [("motor", 1, 7, 24, False),
      ("motor1", 1, 5, 10, False),
      ("motor2", 2, 10, 20, False),
      ("motor3", 3, 5, 15, False),
      ("jittery_motor1", 4, 6, 50, False),
      ("jittery_motor2", 6, 7, 30, False),
      ("motor", 6, 7, 30, False)]),
    # Pattern 2
    (("motor", 1, 7, 24,
      "motor1", 1, 5, 10, False,
      "motor2", 2, 10, 20, True,
      "motor3", 3, 5, 15, False,
      "jittery_motor1", 4, 6, 50, True,
      "jittery_motor2", 6, 7, 30, False),
     [("motor", 1, 7, 24, False),
      ("motor1", 1, 5, 10, False),
      ("motor2", 2, 10, 20, True),
      ("motor3", 3, 5, 15, False),
      ("jittery_motor1", 4, 6, 50, True),
      ("jittery_motor2", 6, 7, 30, False)]),
])
def test_chunk_outer_product_args_3(hw, args, chunked_args):
    """Check the ambiguous case: function is called with 24 arguments,
    the pattern can't be resolved just by counting arguments, so the
    presence of boolean values is checked"""

    # Convert motor names to actual motors in the argument list using fixture 'hw'
    args = tuple([getattr(hw, _) if isinstance(_, str) else _ for _ in args])
    for n, a in enumerate(chunked_args):
        chunked_args[n] = tuple([getattr(hw, _) if isinstance(_, str) else _ for _ in a])

    assert list(chunk_outer_product_args(args)) == chunked_args, \
        "Argument list was split into chunks incorrectly"


@pytest.mark.parametrize("args, chunked_args, pattern", [
    (("motor1", 1, 5, 10), [("motor1", 1, 5, 10, False)],
     OuterProductArgsPattern.PATTERN_1),
    (("motor1", 1, 5, 10, "motor2", 2, 10, 20, "motor3", 3, 5, 15),
     [("motor1", 1, 5, 10, False), ("motor2", 2, 10, 20, False), ("motor3", 3, 5, 15, False)],
     OuterProductArgsPattern.PATTERN_1),
    (("motor1", 1, 5, 10, "motor2", 2, 10, 20, True, "motor3", 3, 5, 15, False),
     [("motor1", 1, 5, 10, False), ("motor2", 2, 10, 20, True), ("motor3", 3, 5, 15, False)],
     OuterProductArgsPattern.PATTERN_2),
])
def test_chunk_outer_product_args_4(hw, args, chunked_args, pattern):
    """Test if `chunk_outer_product_args` works with externally supplied pattern"""

    # Convert motor names to actual motors in the argument list using fixture 'hw'
    args = tuple([getattr(hw, _) if isinstance(_, str) else _ for _ in args])
    for n, a in enumerate(chunked_args):
        chunked_args[n] = tuple([getattr(hw, _) if isinstance(_, str) else _ for _ in a])

    assert list(chunk_outer_product_args(args, pattern)) == chunked_args, \
        "Argument list was split into chunks incorrectly"


def test_chunk_outer_product_args_failing(hw):
    """Failing cases for `chunk_outer_product_args` function """

    # Wrong number of arguments
    args = (hw.motor, 1, 7, 24, True)
    with pytest.raises(ValueError, match="Wrong number of elements in 'args'"):
        list(chunk_outer_product_args(args))

    args = (hw.motor, 1, 7, 24,
            hw.motor1, 1, 5, 10, False,
            hw.motor2, 2, 10, 20)
    with pytest.raises(ValueError, match="Wrong number of elements in 'args'"):
        list(chunk_outer_product_args(args))

    args = (hw.motor, 1, 7, 24,
            hw.motor1, 1, 5, 10,
            "abc", 2, 10, 20,)
    with pytest.raises(ValueError, match="Incorrect order of elements in the argument list"):
        list(chunk_outer_product_args(args))

    args = (hw.motor, 1, 7, 24,
            hw.motor1, hw.motor3, 5, 10,
            hw.motor2, 2, 10, 20)
    with pytest.raises(ValueError, match="Incorrect order of elements in the argument list"):
        list(chunk_outer_product_args(args))


def _gen_outer_product(args):
    """
    Generate expected output for the `outer_product` function.
    The output is generated using completely different method not based
    on Cycler, so it is better suited for testing.

    Parameters
    ----------
    args: list
        same as input of the `outer_product` function

    Returns
    -------
    Dictionary:
        {'motor_name_1': list of positions,
         'motor_name_2': list of positions, ...}
    """

    chunk_args = list(chunk_outer_product_args(args))

    # The number of steps is the multiple of the number of steps along all axes
    n_points = 1
    for chunk in chunk_args:
        n_points *= chunk[3]

    positions = {}

    n_passes = 1
    for chunk in chunk_args:
        motor, start, stop, num, snake = chunk

        # The number of measurement during which the motor has to stay at the same position
        n_steps = int(round(n_points / num / n_passes))

        pts = []
        for n in range(n_passes):
            if (n % 2) and snake:  # Odd pass goes backward if the motor is snaked
                v_start, v_stop = stop, start
            else:
                v_start, v_stop = start, stop
            p = list(np.linspace(v_start, v_stop, num))
            p = list(itertools.chain.from_iterable(itertools.repeat(_, n_steps) for _ in p))
            pts += p

        positions[motor.name] = pts

        n_passes *= num

    return positions


@pytest.mark.parametrize("args", [
    ("motor", 1, 2, 3,
     "motor1", 4, 5, 6, True,
     "motor2", 7, 8, 9, True),
    ("motor", 1, 2, 3,
     "motor1", 4, 5, 6, True,
     "motor2", 7, 8, 9, False),
    ("motor", 1, 2, 3,
     "motor1", 4, 5, 6, False,
     "motor2", 7, 8, 9, True),
    ("motor", 1, 2, 3,
     "motor1", 4, 5, 6, True,
     "motor2", 7, 8, 9, False,
     "motor3", 7, 8, 9, True)
])
def test_outer_product(hw, args):
    """Basic test of the functionality"""

    # Convert motor names to actual motors in the argument list using fixture 'hw'
    args = [getattr(hw, _) if isinstance(_, str) else _ for _ in args]

    full_cycler = outer_product(args=args)
    event_list = list(full_cycler)

    # The list of motors
    chunk_args = list(chunk_outer_product_args(args))
    motors = [_[0] for _ in chunk_args]
    motor_names = [_.name for _ in motors]

    positions = {k: [] for k in motor_names}
    for event in event_list:
        for m, mn in zip(motors, motor_names):
            positions[mn].append(event[m])

    positions_expected = _gen_outer_product(args)

    assert set(positions.keys()) == set(positions_expected.keys()), \
        "Different set of motors in dictionaries of actual and expected positions"

    for name in positions_expected.keys():
        npt.assert_array_almost_equal(
            positions[name], positions_expected[name],
            err_msg=f"Expected and actual positions for the motor '{name}' don't match")
