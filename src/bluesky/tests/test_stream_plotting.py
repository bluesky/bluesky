import os

import matplotlib
import matplotlib.pyplot as plt
import pytest
from ophyd_async.sim.demo import PatternDetector

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


def test_hdf5_plotting_1d(RE, tmp_path):
    cl = CollectLiveStream()
    pl = LiveStreamPlot(cl, data_key="PATTERN1-sum")
    RE.subscribe(cl)
    RE.subscribe(pl)
    det = PatternDetector(name="PATTERN1-sum", path=tmp_path)
    RE(bp.count([det], num=15))


def test_hdf5_plotting_2d(RE, tmp_path):
    cl = CollectLiveStream()
    pl = LiveStreamPlot(cl, data_key="PATTERN1")
    RE.subscribe(pl)
    RE.subscribe(cl)
    det = PatternDetector(name="PATTERN1", path=tmp_path)
    RE(bp.count([det], num=15))
