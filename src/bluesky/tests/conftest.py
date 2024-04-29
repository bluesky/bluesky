import asyncio
import os

import numpy as np
import packaging
import pytest

from bluesky.run_engine import RunEngine, TransitionError


@pytest.fixture(scope="function", params=[False, True])
def RE(request):
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    RE = RunEngine({}, call_returns_result=request.param, loop=loop)

    def clean_event_loop():
        if RE.state not in ("idle", "panicked"):
            try:
                RE.halt()
            except TransitionError:
                pass
        loop.call_soon_threadsafe(loop.stop)
        RE._th.join()
        loop.close()

    request.addfinalizer(clean_event_loop)
    return RE


@pytest.fixture(scope="function")
def hw(tmp_path):
    import ophyd
    from ophyd.sim import hw

    # ophyd 1.4.0 added support for customizing the directory used by simulated
    # hardware that generates files
    if packaging.version.Version(ophyd.__version__) >= packaging.version.Version("1.4.0"):
        return hw(str(tmp_path))
    else:
        return hw()


# vendored from ophyd.sim
class NumpySeqHandler:
    specs = {"NPY_SEQ"}

    def __init__(self, filename, root=""):
        self._name = os.path.join(root, filename)

    def __call__(self, index):
        return np.load(f"{self._name}_{index}.npy")

    def get_file_list(self, datum_kwarg_gen):
        "This method is optional. It is not needed for access, but for export."
        return ["{name}_{index}.npy".format(name=self._name, **kwargs) for kwargs in datum_kwarg_gen]


@pytest.fixture(scope="function")
def db(request):
    """Return a data broker"""
    try:
        from databroker import temp

        db = temp()
        return db
    except ImportError:
        pytest.skip("Databroker v2 still missing temp.")
    except ValueError:
        pytest.skip("Intake is failing for unknown reasons.")


@pytest.fixture(autouse=True)
def cleanup_any_figures(request):
    import matplotlib.pyplot as plt

    "Close any matplotlib figures that were opened during a test."
    plt.close("all")
