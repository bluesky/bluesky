import asyncio
from bluesky.run_engine import RunEngine
from bluesky.examples import Mover, SynGauss
import pytest
from types import SimpleNamespace
from ophyd.pseudopos import (PseudoPositioner, PseudoSingle,
                             real_position_argument, pseudo_position_argument)
from ophyd.positioner import SoftPositioner
from ophyd.signal import Signal
from ophyd import (Component as C)


@pytest.fixture(scope='function')
def fresh_RE(request):
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    RE = RunEngine({}, loop=loop)
    RE.ignore_callback_exceptions = False

    def clean_event_loop():
        if RE.state != 'idle':
            RE.halt()
        ev = asyncio.Event(loop=loop)
        ev.set()
        loop.run_until_complete(ev.wait())

    request.addfinalizer(clean_event_loop)
    return RE


RE = fresh_RE


@pytest.fixture(scope='function')
def motor_det(request):
    motor = Mover('motor', {'motor': lambda x: x}, {'x': 0})
    det = SynGauss('det', motor, 'motor', center=0, Imax=1,
                   sigma=1, exposure_time=0)
    return motor, det


@pytest.fixture(scope='function')
def db(request):
    """Return a data broker
    """
    from databroker.tests.utils import build_sqlite_backed_broker
    db = build_sqlite_backed_broker(request)
    return db


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
