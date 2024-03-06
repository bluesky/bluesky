import pytest
import bluesky.plans as bp
from bluesky.callbacks.best_effort import BestEffortCallback


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
    (bp.scan, ('motor', 1, 2, 2), {}),
    (bp.scan, ('motor', 1, 2), {'num': 2}),
    (bp.scan, ('motor1', 1, 2, 'motor2', 1, 2, 2), {}),
    (bp.scan, ('motor1', 1, 2, 'motor2', 1, 2), {'num': 2}),
    (bp.rel_scan, ('motor', 1, 2, 2), {}),
    (bp.rel_scan, ('motor', 1, 2), {'num': 2}),
    (bp.rel_scan, ('motor1', 1, 2, 'motor2', 1, 2, 2), {}),
    (bp.rel_scan, ('motor1', 1, 2, 'motor2', 1, 2), {'num': 2}),
    (bp.inner_product_scan, (2, 'motor', 1, 2), {}),
    (bp.relative_inner_product_scan, (2, 'motor', 1, 2), {}),
    (bp.grid_scan, ('motor1', 1, 2, 2, 'motor2', 1, 2, 3, False), {}),
    (bp.spiral, ('motor1', 'motor2', 0.0, 0.0, 0.3, 0.3, 0.05, 3), {}),
    (bp.rel_spiral, ('motor1', 'motor2', 0.3, 0.3, 0.05, 3), {}),
    (bp.spiral_fermat, ('motor1', 'motor2', 0.0, 0.0, 0.3, 0.3, 0.05, 3), {}),
    (bp.rel_spiral_fermat, ('motor1', 'motor2', 0.3, 0.3, 0.05, 3), {}),
    ])
def test_plans(RE, pln, args, kwargs, hw):
    args = tuple(getattr(hw, v, v) if isinstance(v, str) else v
                 for v in args)
    det = hw.det
    bec = BestEffortCallback()
    RE.subscribe(bec)
    RE.subscribe(checker)
    RE(pln([det], *args, **kwargs))


@pytest.mark.parametrize('pln,args,kwargs', [
    # repeat with motor objects that do not have hints
    (bp.scan, ('motor_no_hints1', 1, 2, 2), {}),
    (bp.inner_product_scan, (2, 'motor_no_hints1', 1, 2), {}),
    (bp.relative_inner_product_scan, (2, 'motor_no_hints1', 1, 2), {}),
    (bp.grid_scan,
     ('motor_no_hints1', 1, 2, 2, 'motor_no_hints2', 1, 2, 3, False), {}),
    (bp.spiral,
     ('motor_no_hints1', 'motor_no_hints2', 0.0, 0.0, 0.3, 0.3, 0.05, 3), {}),
    (bp.rel_spiral,
     ('motor_no_hints1', 'motor_no_hints2', 0.3, 0.3, 0.05, 3), {}),
    (bp.spiral_fermat,
     ('motor_no_hints1', 'motor_no_hints2', 0.0, 0.0, 0.3, 0.3, 0.05, 3), {}),
    (bp.rel_spiral_fermat,
     ('motor_no_hints1', 'motor_no_hints2', 0.3, 0.3, 0.05, 3), {}),
    ])
@pytest.mark.xfail
def test_plans_motors_no_hints(RE, pln, args, kwargs, hw):
    args = tuple(getattr(hw, v, v) if isinstance(v, str) else v
                 for v in args)
    det = hw.det
    for v in args:
        assert not hasattr(v, 'hints')
    bec = BestEffortCallback()
    RE.subscribe(bec)
    RE(pln([det], *args, **kwargs))


@pytest.mark.parametrize('pln,args,kwargs', [
    # repeat with motor objects that have empty hints
    (bp.scan, ('motor1', 1, 2, 2), {}),
    (bp.inner_product_scan, (2, 'motor1', 1, 2), {}),
    (bp.relative_inner_product_scan, (2, 'motor1', 1, 2), {}),
    (bp.grid_scan, ('motor1', 1, 2, 2, 'motor2', 1, 2, 3, False), {}),
    (bp.spiral, ('motor1', 'motor2', 0.0, 0.0, 0.3, 0.3, 0.05, 3), {}),
    (bp.rel_spiral, ('motor1', 'motor2', 0.3, 0.3, 0.05, 3), {}),
    (bp.spiral_fermat, ('motor1', 'motor2', 0.0, 0.0, 0.3, 0.3, 0.05, 3), {}),
    (bp.rel_spiral_fermat, ('motor1', 'motor2', 0.3, 0.3, 0.05, 3), {}),
    ])
@pytest.mark.xfail
def test_plans_motor_empty_hints(RE, pln, args, kwargs, hw):
    args = tuple(getattr(hw, v, v) if isinstance(v, str) else v
                 for v in args)
    for v in args:
        if hasattr(v, 'hints'):
            v.hints = {}
            assert v.hints == {}
    det = hw.det
    bec = BestEffortCallback()
    RE.subscribe(bec)
    RE(pln([det], *args, **kwargs))
