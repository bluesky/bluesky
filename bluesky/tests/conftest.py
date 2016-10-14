import asyncio
from bluesky.run_engine import RunEngine
from bluesky.examples import Mover, SynGauss
import pytest


@pytest.fixture(scope='function')
def fresh_RE(request):
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    RE = RunEngine({}, loop=loop)
    RE.ignore_callback_exceptions = False

    def clean_al():
        if RE.state != 'idle':
            RE.halt()
        ev = asyncio.Event(loop=loop)
        ev.set()
        loop.run_until_complete(ev.wait())

    request.addfinalizer(clean_al)
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
    from portable_mds.sqlite.mds import MDS
    from databroker import Broker
    import tempfile
    import shutil
    td = tempfile.mkdtemp()

    def delete_tmpdir():
        shutil.rmtree(td)

    request.addfinalizer(delete_tmpdir)

    return Broker(MDS({'directory': td, 'timezone': 'US/Eastern'}), None)
