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
    from databroker import temp_config, Broker
    db = Broker.from_config(temp_config())
    fs = db.fs
    mds = db.mds
    fs_test_conf = fs.config
    md_test_conf = mds.config

    def delete_fs_mds():
        print("DROPPING DB")
        fs._connection.drop_database(fs_test_conf['database'])
        mds._connection.drop_database(md_test_conf['database'])

    request.addfinalizer(delete_fs_mds)
    return db
