import pytest
from types import SimpleNamespace

from bluesky.utils import ensure_generator, Msg, merge_cycler
from cycler import cycler
from ophyd.pseudopos import (PseudoPositioner, PseudoSingle,
                             real_position_argument, pseudo_position_argument)
from ophyd.positioner import SoftPositioner
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
                           pseudo1x3=SPseudo1x3(name='pseudo1x3'))


def test_cycler_merge_2pseudo(hw):
    p3x3 = hw.pseudo3x3
    # put the real positions at 1, 2, 3
    p3x3.set(-1, -2, -3)
    r1_vec = [10, 11, 12]
    r2_vec = [20, 21, 22]
    cyc = cycler(p3x3.real1, r1_vec) + cycler(p3x3.real2, r2_vec)

    mcyc = merge_cycler(cyc)
    expected_merge = [{'real1': r1, 'real2': r2}
                      for r1, r2 in zip(r1_vec, r2_vec)]
    assert len(mcyc.keys) == 1
    assert mcyc.keys == {p3x3}
    assert mcyc.by_key()[p3x3] == expected_merge
