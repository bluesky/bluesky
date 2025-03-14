import os
from tempfile import mkdtemp

import matplotlib
import matplotlib.pyplot as plt
import pytest
from ophyd_async import sim
from ophyd_async.core import StaticPathProvider, UUIDFilenameProvider, init_devices

import bluesky.plans as bp
from bluesky.callbacks.core import CollectLiveStream
from bluesky.callbacks.mpl_plotting import LiveStreamPlot

if not os.getenv("GITHUB_ACTIONS"):
    matplotlib.use("QtAgg")
    plt.ion()


@pytest.fixture(autouse=True, params=["1", None])
def _strict_debug(monkeypatch, request):
    if request.param is not None:
        monkeypatch.setenv("BLUESKY_DEBUG_CALLBACKS", request.param)
    else:
        monkeypatch.delenv("BLUESKY_DEBUG_CALLBACKS", raising=False)


def test_hdf5_plotting_1d(RE):
    pattern_generator = sim.PatternGenerator()
    path_provider = StaticPathProvider(UUIDFilenameProvider(), mkdtemp())
    with init_devices():
        bdet = sim.SimBlobDetector(path_provider, pattern_generator)

    cl = CollectLiveStream()
    pl = LiveStreamPlot(cl, data_key="bdet-sum")
    RE.subscribe(cl)
    RE.subscribe(pl)

    RE(bp.count([bdet], num=5))


def test_hdf5_plotting_2d(RE):
    pattern_generator = sim.PatternGenerator()
    path_provider = StaticPathProvider(UUIDFilenameProvider(), mkdtemp())
    with init_devices():
        bdet = sim.SimBlobDetector(path_provider, pattern_generator)

    cl = CollectLiveStream()
    pl = LiveStreamPlot(cl, data_key="PATTERN1")
    RE.subscribe(pl)
    RE.subscribe(cl)
    RE(bp.count([bdet], num=5))
