import pytest

from bluesky.tests.utils import DocCollector
import bluesky.plans as bp
from bluesky.examples import motor, motor1, motor2, det
import numpy as np
import pandas as pd


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
    args.append((bp.grid_scan([det],
                              motor, 1, 2, 3,
                              motor1, 4, 5, 6, True,
                              motor2, 7, 8, 9, True),
                 {'motors': ('motor', 'motor1', 'motor2'),
                  'extents': ([1, 2], [4, 5], [7, 8]),
                  'shape': (3, 6, 9),
                  'snaking': (False, True, True),
                  'plan_pattern_module': 'bluesky.plan_patterns',
                  'plan_pattern': 'outer_product',
                  'plan_name': 'grid_scan'}))
    ids.append('grid_scan')

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


def test_ops_dimension_hints(fresh_RE):
    RE = fresh_RE
    c = DocCollector()
    RE.subscribe(c.insert)
    rs, = RE(bp.grid_scan([det],
                          motor, -1, 1, 7,
                          motor1, 0, 2, 3, False))

    st = c.start[0]

    assert 'dimensions' in st['hints']

    assert st['hints']['dimensions'] == [
        (m.hints['fields'], 'primary') for m in (motor, motor1)]


def test_mesh_pseudo(hw, fresh_RE):
    p3x3 = hw.pseudo3x3
    sig = hw.sig
    RE = fresh_RE
    d = DocCollector()

    RE.subscribe(d.insert)
    rs, = RE(bp.outer_product_scan([sig],
                                   p3x3.pseudo1, 0, 3, 5,
                                   p3x3.pseudo2, 7, 10, 7, False))
    df = pd.DataFrame([_['data']
                       for _ in d.event[d.descriptor[rs][0]['uid']]])

    for k in p3x3.describe():
        assert k in df

    for k in sig.describe():
        assert k in df

    assert all(df[sig.name] == 0)
    assert all(df[p3x3.pseudo3.name] == 0)


def test_rmesh_pseudo(hw, fresh_RE):
    p3x3 = hw.pseudo3x3
    p3x3.set(1, -2, 100)
    init_pos = p3x3.position
    sig = hw.sig
    RE = fresh_RE
    d = DocCollector()

    RE.subscribe(d.insert)
    rs, = RE(bp.relative_outer_product_scan(
        [sig],
        p3x3.pseudo1, 0, 3, 5,
        p3x3.pseudo2, 7, 10, 7, False))
    df = pd.DataFrame([_['data']
                       for _ in d.event[d.descriptor[rs][0]['uid']]])

    for k in p3x3.describe():
        assert k in df

    for k in sig.describe():
        assert k in df

    assert all(df[sig.name] == 0)
    assert all(df[p3x3.pseudo3.name] == 100)
    assert len(df) == 35
    assert min(df[p3x3.pseudo1.name]) == 1
    assert init_pos == p3x3.position


def test_relative_pseudo(hw, fresh_RE, db):
    RE = fresh_RE
    RE.subscribe(db.insert)
    p = hw.pseudo3x3
    p.set(1, 1, 1)
    base_pos = p.position

    # this triggers the merging code path
    rs, = RE(bp.relative_inner_product_scan([p],
                                            5,
                                            p.pseudo1, -1, 1,
                                            p.pseudo2, -2, -1))
    tb1 = db[rs].table().drop('time', 1)
    assert p.position == base_pos

    # this triggers this does not
    rs, = RE(bp.relative_inner_product_scan([p],
                                            5,
                                            p.real1, 1, -1,
                                            p.real2, 2, 1))
    tb2 = db[rs].table().drop('time', 1)
    assert p.position == base_pos

    # same columns
    assert set(tb1) == set(tb2)
    # same number of points
    assert len(tb1) == len(tb2)

    def get_hint(c):
        return c.hints['fields'][0]

    for c in list(p.pseudo_positioners) + list(p.real_positioners):
        col = get_hint(c)
        print(col)
        assert (tb1[col] == tb2[col]).all()

    assert (tb1[get_hint(p.pseudo1)] == np.linspace(0, 2, 5)).all()
