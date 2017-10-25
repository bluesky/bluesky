import pytest
from types import SimpleNamespace
from functools import reduce
import operator

from bluesky.utils import ensure_generator, Msg, merge_cycler
from cycler import cycler
from ophyd.pseudopos import (PseudoPositioner, PseudoSingle,
                             real_position_argument, pseudo_position_argument)
from ophyd.positioner import SoftPositioner
from ophyd.signal import Signal
from ophyd import (Component as C)


def test_single_msg_to_gen():
    m = Msg('set', None, 0)

    m_list = [m for m in ensure_generator(m)]

    assert len(m_list) == 1
    assert m_list[0] == m


@pytest.fixture()
def hw():
    class SPseudo3x3(PseudoPositioner):
        pseudo1 = C(PseudoSingle, limits=(-10, 10), egu='a')
        pseudo2 = C(PseudoSingle, limits=(-10, 10), egu='b')
        pseudo3 = C(PseudoSingle, limits=None, egu='c')
        real1 = C(SoftPositioner, init_pos=0)
        real2 = C(SoftPositioner, init_pos=0)
        real3 = C(SoftPositioner, init_pos=0)

        sig = C(Signal, value=0)

        @pseudo_position_argument
        def forward(self, pseudo_pos):
            pseudo_pos = self.PseudoPosition(*pseudo_pos)
            # logger.debug('forward %s', pseudo_pos)
            return self.RealPosition(real1=-pseudo_pos.pseudo1,
                                     real2=-pseudo_pos.pseudo2,
                                     real3=-pseudo_pos.pseudo3)

        @real_position_argument
        def inverse(self, real_pos):
            real_pos = self.RealPosition(*real_pos)
            # logger.debug('inverse %s', real_pos)
            return self.PseudoPosition(pseudo1=-real_pos.real1,
                                       pseudo2=-real_pos.real2,
                                       pseudo3=-real_pos.real3)

    class SPseudo1x3(PseudoPositioner):
        pseudo1 = C(PseudoSingle, limits=(-10, 10))
        real1 = C(SoftPositioner, init_pos=0)
        real2 = C(SoftPositioner, init_pos=0)
        real3 = C(SoftPositioner, init_pos=0)

        @pseudo_position_argument
        def forward(self, pseudo_pos):
            pseudo_pos = self.PseudoPosition(*pseudo_pos)
            # logger.debug('forward %s', pseudo_pos)
            return self.RealPosition(real1=-pseudo_pos.pseudo1,
                                     real2=-pseudo_pos.pseudo1,
                                     real3=-pseudo_pos.pseudo1)

        @real_position_argument
        def inverse(self, real_pos):
            real_pos = self.RealPosition(*real_pos)
            # logger.debug('inverse %s', real_pos)
            return self.PseudoPosition(pseudo1=-real_pos.real1)

    return SimpleNamespace(pseudo3x3=SPseudo3x3(name='pseudo3x3'),
                           pseudo1x3=SPseudo1x3(name='pseudo1x3'),
                           sig=Signal(name='sig', value=0))


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
