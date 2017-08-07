from bluesky.plans import scan
from bluesky.simulators import print_summary, print_summary_wrapper
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
