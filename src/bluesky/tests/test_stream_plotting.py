import os

import matplotlib
import matplotlib.pyplot as plt
from ophyd_async.sim.demo import PatternDetector

import bluesky.plans as bp
from bluesky.callbacks.core import CollectLiveStream
from bluesky.callbacks.mpl_plotting import LiveStreamPlot

if not os.getenv("GITHUB_ACTIONS"):
    matplotlib.use("QtAgg")
    plt.ion()


def test_hdf5_plotting_1d(RE, tmp_path):
    cl = CollectLiveStream()
    pl = LiveStreamPlot(cl, data_key="PATTERN1-sum")
    RE.subscribe(pl)
    det = PatternDetector(name="PATTERN1-sum", path=tmp_path)
    RE(bp.count([det], num=15), cl)


def test_hdf5_plotting_2d(RE, tmp_path):
    cl = CollectLiveStream()
    pl = LiveStreamPlot(cl, data_key="PATTERN1")
    RE.subscribe(pl)
    det = PatternDetector(name="PATTERN1", path=tmp_path)
    RE(bp.count([det], num=15), cl)
