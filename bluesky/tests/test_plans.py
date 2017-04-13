import pytest

from bluesky.tests.utils import DocCollector
import bluesky.plans as bp
from bluesky.examples import motor, motor1, motor2, det


def _validate_start(start, expected_values):
    '''Basic metadata validtion'''

    plan_md_key = [
        'plan_pattern_module',
        'plan_pattern_args',
        'plan_type',
        'plan_pattern',
        'plan_name',
        'num_points',
        'plan_args',
        'detectors']

    for k in plan_md_key:
        assert k in start
    for k, v in expected_values.items():
        assert start[k] == v


def _make_plan_marker():
    args = []
    ids = []

    ##
    args.append((bp.outer_product_scan([det],
                                       motor, 1, 2, 3,
                                       motor1, 4, 5, 6, True,
                                       motor2, 7, 8, 9, True),
                 {'motors': ('motor', 'motor1', 'motor2'),
                  'extents': ([1, 2], [4, 5], [7, 8]),
                  'shape': (3, 6, 9),
                  'snaking': (False, True, True),
                  'plan_pattern_module': 'bluesky.plan_patterns',
                  'plan_pattern': 'outer_product',
                  'plan_name': 'outer_product_scan'}))
    ids.append('outer_product_scan')

    ##
    args.append((bp.inner_product_scan([det], 9,
                                       motor, 1, 2,
                                       motor1, 4, 5,
                                       motor2, 7, 8),
                {'motors': ('motor', 'motor1', 'motor2')}))
    ids.append('inner_product_scan')

    return pytest.mark.parametrize('plan,target',
                                   args,
                                   ids=ids)


@_make_plan_marker()
def test_plan_header(fresh_RE, plan, target):
    RE = fresh_RE
    c = DocCollector()
    RE(plan, c.insert)
    for s in c.start:
        _validate_start(s, target)
