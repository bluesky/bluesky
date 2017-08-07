from bluesky.plans import scan
from bluesky.simulators import print_summary, print_summary_wrapper


def test_print_summary(motor_det):
    motor, det = motor_det
    print_summary(scan([det], motor, -1, 1, 10))
    list(print_summary_wrapper(scan([det], motor, -1, 1, 10)))
