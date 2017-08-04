from bluesky.examples import motor, motor1, motor2, det, det1, det2, MockFlyer
import pytest
import bluesky.plans as bp
from bluesky.callbacks.best_effort import BestEffortCallback


@pytest.mark.parametrize('pln,args,kwargs', [
    # Notice that 'args' does not include [det], which is inserted in the body
    # of the test function below.
    (bp.count, (), {}),
    (bp.count, (), {'num': 3}),
    (bp.scan, (motor, 1, 2, 2), {}),
    (bp.inner_product_scan, (2, motor, 1, 2), {}),
    (bp.relative_inner_product_scan, (2, motor, 1, 2), {}),
    (bp.outer_product_scan, (motor1, 1, 2, 2, motor2, 1, 2, 3, False), {}),
    (bp.spiral, (motor1, motor2, 0.0, 0.0, 0.1, 0.1, 0.05, 1.0), {}),
    (bp.relative_spiral, (motor1, motor2, 0.1, 0.1, 0.05, 1.0), {}),
    (bp.spiral_fermat, (motor1, motor2, 0.0, 0.0, 0.1, 0.1, 0.05, 1.0), {}),
    (bp.relative_spiral_fermat, (motor1, motor2, 0.1, 0.1, 0.05, 1.0), {}),
    ])
def test_plans(fresh_RE, pln, args, kwargs):
    bec = BestEffortCallback()
    fresh_RE.subscribe(bec)
    fresh_RE(pln([det], *args, **kwargs))
