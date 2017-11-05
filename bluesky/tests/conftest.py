import asyncio
from bluesky.run_engine import RunEngine
import numpy as np
import os
import pytest


@pytest.fixture(scope='function')
def RE(request):
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    RE = RunEngine({}, loop=loop)

    def clean_event_loop():
        if RE.state != 'idle':
            RE.halt()
        ev = asyncio.Event(loop=loop)
        ev.set()
        loop.run_until_complete(ev.wait())

    request.addfinalizer(clean_event_loop)
    return RE


@pytest.fixture(scope='function')
def hw(request):
    from ophyd.sim import hw
    return hw()


# vendored from ophyd.sim
class NumpySeqHandler:
    specs = {'NPY_SEQ'}

    def __init__(self, filename, root=''):
        self._name = os.path.join(root, filename)

    def __call__(self, index):
        return np.load('{}_{}.npy'.format(self._name, index))

    def get_file_list(self, datum_kwarg_gen):
        "This method is optional. It is not needed for access, but for export."
        return ['{name}_{index}.npy'.format(name=self._name, **kwargs)
                for kwargs in datum_kwarg_gen]


@pytest.fixture(scope='function')
def db(request):
    """Return a data broker
    """
    from databroker.tests.utils import build_sqlite_backed_broker
    db = build_sqlite_backed_broker(request)
    db.reg.register_handler('NPY_SEQ', NumpySeqHandler)
    return db


@pytest.fixture(autouse=True)
def cleanup_any_figures(request):
    import matplotlib.pyplot as plt
    "Close any matplotlib figures that were opened during a test."
    plt.close('all')
