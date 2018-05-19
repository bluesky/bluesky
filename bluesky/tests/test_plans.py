import pytest
from bluesky.tests.utils import DocCollector
import bluesky.plans as bp
import bluesky.preprocessors as bpp
import bluesky.plan_stubs as bps
import numpy as np
import pandas as pd
from bluesky.tests.utils import MsgCollector


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


def test_plan_header(RE, hw):
    args = []

    ##
    args.append((bp.grid_scan([hw.det],
                              hw.motor, 1, 2, 3,
                              hw.motor1, 4, 5, 6, True,
                              hw.motor2, 7, 8, 9, True),
                 {'motors': ('motor', 'motor1', 'motor2'),
                  'extents': ([1, 2], [4, 5], [7, 8]),
                  'shape': (3, 6, 9),
                  'snaking': (False, True, True),
                  'plan_pattern_module': 'bluesky.plan_patterns',
                  'plan_pattern': 'outer_product',
                  'plan_name': 'grid_scan'}))

    ##
    args.append((bp.inner_product_scan([hw.det], 9,
                                       hw.motor, 1, 2,
                                       hw.motor1, 4, 5,
                                       hw.motor2, 7, 8),
                {'motors': ('motor', 'motor1', 'motor2')}))

    for plan, target in args:
        c = DocCollector()
        RE(plan, c.insert)
        for s in c.start:
            _validate_start(s, target)


def test_ops_dimension_hints(RE, hw):
    det = hw.det
    motor = hw.motor
    motor1 = hw.motor1
    c = DocCollector()
    RE.subscribe(c.insert)
    rs, = RE(bp.grid_scan([det],
                          motor, -1, 1, 7,
                          motor1, 0, 2, 3, False))

    st = c.start[0]

    assert 'dimensions' in st['hints']

    assert st['hints']['dimensions'] == [
        (m.hints['fields'], 'primary') for m in (motor, motor1)]


def test_mesh_pseudo(hw, RE):
    p3x3 = hw.pseudo3x3
    sig = hw.sig
    d = DocCollector()

    RE.subscribe(d.insert)
    rs, = RE(bp.grid_scan([sig],
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


def test_rmesh_pseudo(hw, RE):
    p3x3 = hw.pseudo3x3
    p3x3.set(1, -2, 100)
    init_pos = p3x3.position
    sig = hw.sig
    d = DocCollector()

    RE.subscribe(d.insert)
    rs, = RE(bp.rel_grid_scan(
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


def test_relative_pseudo(hw, RE, db):
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
        h = c.hints['fields']
        return h[0] if h else c.name

    for c in list(p.pseudo_positioners) + list(p.real_positioners):
        col = get_hint(c)
        print(col)
        assert (tb1[col] == tb2[col]).all()

    assert (tb1[get_hint(p.pseudo1)] == np.linspace(0, 2, 5)).all()


def test_reset_wrapper(hw, RE):
    p = hw.pseudo3x3
    m_col = MsgCollector()
    RE.msg_hook = m_col

    RE(bp.relative_inner_product_scan([], 1,
                                      p.pseudo1, 0, 1,
                                      p.pseudo2, 0, 1))
    expecte_objs = [p, None, None,
                    p, None, p,
                    None, None, p,
                    None, None, p,
                    p, None]
    assert len(m_col.msgs) == 14
    assert [m.obj for m in m_col.msgs] == expecte_objs


@pytest.mark.parametrize('pln', [bps.mv, bps.mvr])
def test_pseudo_mv(hw, RE, pln):
    p = hw.pseudo3x3
    m_col = MsgCollector()
    RE.msg_hook = m_col

    RE(pln(p.pseudo1, 1,
           p.pseudo2, 1))
    expecte_objs = [p, None]
    assert len(m_col.msgs) == 2
    assert [m.obj for m in m_col.msgs] == expecte_objs
