from bluesky.plans import scan
from bluesky.simulators import (print_summary, print_summary_wrapper,
                                check_limits, LimitsExceeded)
import pytest


def test_print_summary(motor_det):
    motor, det = motor_det
    print_summary(scan([det], motor, -1, 1, 10))
    list(print_summary_wrapper(scan([det], motor, -1, 1, 10)))

def test_old_module_name(motor_det):
    from bluesky.plan_tools import print_summary, print_summary_wrapper
    motor, det = motor_det
    with pytest.warns(UserWarning):
        print_summary(scan([det], motor, -1, 1, 10))
    with pytest.warns(UserWarning):
        list(print_summary_wrapper(scan([det], motor, -1, 1, 10)))

def test_check_limits(motor_det):
    motor, det = motor_det
    # The motor object does not currently implement limits.
    # Use an assert to help us out if this changes in the future.
    assert not hasattr(motor, 'limits')

    # check_limits should warn if it can't find limits
    with pytest.warns(UserWarning):
        check_limits(scan([det], motor, -1, 1, 3))

    # monkey-patch some limits
    motor.limits = (-2, 2)
    # check_limits should do nothing here
    check_limits(scan([det], motor, -1, 1, 3))

    # check_limits should error if limits are exceeded
    with pytest.raises(LimitsExceeded):
        check_limits(scan([det], motor, -3, 3, 3))
