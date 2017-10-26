import pytest

from functools import reduce
import operator

from bluesky.utils import ensure_generator, Msg, merge_cycler
from cycler import cycler


def test_single_msg_to_gen():
    m = Msg('set', None, 0)

    m_list = [m for m in ensure_generator(m)]

    assert len(m_list) == 1
    assert m_list[0] == m


@pytest.mark.parametrize('traj',
                         ({'pseudo1': [10, 11, 12],
                           'pseudo2': [20, 21, 22]},
                          {'pseudo1': [10, 11, 12],
                           'pseudo2': [20, 21, 22],
                           'pseudo3': [30, 31, 32]},
                          )
                         )
def test_cycler_merge_pseudo(hw, traj):
    p3x3 = hw.pseudo3x3
    sig = hw.sig
    keys = traj.keys()
    tlen = len(next(iter(traj.values())))
    expected_merge = [{k: traj[k][j] for k in keys}
                      for j in range(tlen)]

    cyc = reduce(operator.add, (cycler(getattr(p3x3, k), v)
                                for k, v in traj.items()))

    mcyc = merge_cycler(cyc + cycler(sig, range(tlen)))

    assert mcyc.keys == {p3x3, sig}
    assert mcyc.by_key()[p3x3] == expected_merge
    assert mcyc.by_key()[sig] == list(range(tlen))


@pytest.mark.parametrize('children',
                         (
                             ['pseudo1', 'real1'],
                             ['pseudo1', 'pseudo2', 'real1']))
def test_cycler_merge_pseudo_real_clash(hw, children):
    p3x3 = hw.pseudo3x3
    cyc = reduce(operator.add, (cycler(getattr(p3x3, k), range(5))
                                for k in children))

    with pytest.raises(ValueError):
        merge_cycler(cyc)


@pytest.mark.parametrize('children',
                         (['pseudo1', ],
                          ['pseudo1', 'pseudo2'],
                          ['real1'],
                          ['real1', 'real2']))
def test_cycler_parent_and_parts_fail(hw, children):
    p3x3 = hw.pseudo3x3
    cyc = reduce(operator.add, (cycler(getattr(p3x3, k), range(5))
                                for k in children))
    cyc += cycler(p3x3, range(5))

    with pytest.raises(ValueError):
        merge_cycler(cyc)


@pytest.mark.parametrize('children',
                         (['sig', ],))
def test_cycler_parent_and_parts_succed(hw, children):
    p3x3 = hw.pseudo3x3
    cyc = reduce(operator.add, (cycler(getattr(p3x3, k), range(5))
                                for k in children))
    cyc += cycler(p3x3, range(5))
    mcyc = merge_cycler(cyc)

    assert mcyc.keys == cyc.keys
    assert mcyc.by_key() == cyc.by_key()


@pytest.mark.parametrize('children',
                         (
                             ['pseudo1'],
                             ['pseudo2', 'sig'],
                             ['real1'],
                             ['real1', 'real2'],
                             ['real1', 'real2', 'sig'],
                         ))
def test_cycler_merge_mixed(hw, children):
    p3x3 = hw.pseudo3x3
    cyc = reduce(operator.add, (cycler(getattr(p3x3, k), range(5))
                                for k in children))

    mcyc = merge_cycler(cyc)

    assert mcyc.keys == cyc.keys
    assert mcyc.by_key() == cyc.by_key()
