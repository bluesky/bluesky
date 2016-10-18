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
    from portable_mds.sqlite.mds import MDS
    from filestore.utils import install_sentinels
    import filestore.fs
    from databroker import Broker
    import tempfile
    import shutil
    from uuid import uuid4
    td = tempfile.mkdtemp()
    db_name = "fs_testing_v1_disposable_{}".format(str(uuid4()))
    test_conf = dict(database=db_name, host='localhost',
                     port=27017)
    install_sentinels(test_conf, 1)
    fs = filestore.fs.FileStoreMoving(test_conf,
                                      version=1)

    def delete_dm():
        print("DROPPING DB")
        fs._connection.drop_database(db_name)

    request.addfinalizer(delete_dm)

    def delete_tmpdir():
        shutil.rmtree(td)

    request.addfinalizer(delete_tmpdir)

    return Broker(MDS({'directory': td, 'timezone': 'US/Eastern'}), fs)
