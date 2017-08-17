from bluesky.examples import (motor, motor1, motor2, det, det1, det2,
                              MockFlyer, NullStatus)

import pytest
import bluesky.plans as bp
from bluesky.callbacks.best_effort import BestEffortCallback


class MotorNoHints:
    def __init__(self, name):
        self.name = name
        self.parent = None
        self.position = 0

    def read(self):
        return {}

    def read_configuration(self):
        return {}

    def describe(self):
        return {}

    def describe_configuration(self):
        return {}

    def set(self, val):
        return NullStatus()


class MotorEmptyHints:
    def __init__(self, name):
        self.name = name
        self.parent = None
        self.position = 0
        self.hints = {}
    def read(self):
        return {}

    def read_configuration(self):
        return {}

    def describe(self):
        return {}

    def describe_configuration(self):
        return {}

    def set(self, val):
        return NullStatus()



def checker(name, doc):
    if name == 'start':
        assert 'hints' in doc
        assert 'dimensions' in doc['hints']
        assert 'uid' in doc


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
    fresh_RE.subscribe(checker)
    fresh_RE(pln([det], *args, **kwargs))


_motor = MotorNoHints('motor')
_motor1 = MotorNoHints('motor1')
_motor2 = MotorNoHints('motor2')


@pytest.mark.parametrize('pln,args,kwargs', [
    # repeat with motor objects that do not have hints
    (bp.scan, (_motor, 1, 2, 2), {}),
    (bp.inner_product_scan, (2, _motor, 1, 2), {}),
    (bp.relative_inner_product_scan, (2, _motor, 1, 2), {}),
    (bp.outer_product_scan, (_motor1, 1, 2, 2, _motor2, 1, 2, 3, False), {}),
    (bp.spiral, (_motor1, _motor2, 0.0, 0.0, 0.1, 0.1, 0.05, 1.0), {}),
    (bp.relative_spiral, (_motor1, _motor2, 0.1, 0.1, 0.05, 1.0), {}),
    (bp.spiral_fermat, (_motor1, _motor2, 0.0, 0.0, 0.1, 0.1, 0.05, 1.0), {}),
    (bp.relative_spiral_fermat, (_motor1, _motor2, 0.1, 0.1, 0.05, 1.0), {}),
    ])
def test_plans_motors_no_hints(fresh_RE, pln, args, kwargs):
    bec = BestEffortCallback()
    fresh_RE.subscribe(bec)
    fresh_RE(pln([det], *args, **kwargs))


_motor = MotorEmptyHints('motor')
_motor1 = MotorEmptyHints('motor1')
_motor2 = MotorEmptyHints('motor2')


@pytest.mark.parametrize('pln,args,kwargs', [
    # repeat with motor objects that do not have hints
    (bp.scan, (_motor, 1, 2, 2), {}),
    (bp.inner_product_scan, (2, _motor, 1, 2), {}),
    (bp.relative_inner_product_scan, (2, _motor, 1, 2), {}),
    (bp.outer_product_scan, (_motor1, 1, 2, 2, _motor2, 1, 2, 3, False), {}),
    (bp.spiral, (_motor1, _motor2, 0.0, 0.0, 0.1, 0.1, 0.05, 1.0), {}),
    (bp.relative_spiral, (_motor1, _motor2, 0.1, 0.1, 0.05, 1.0), {}),
    (bp.spiral_fermat, (_motor1, _motor2, 0.0, 0.0, 0.1, 0.1, 0.05, 1.0), {}),
    (bp.relative_spiral_fermat, (_motor1, _motor2, 0.1, 0.1, 0.05, 1.0), {}),
    ])
def test_plans_motor_empty_hints(fresh_RE, pln, args, kwargs):
    bec = BestEffortCallback()
    fresh_RE.subscribe(bec)
    fresh_RE(pln([det], *args, **kwargs))
