import matplotlib.pyplot as plt

from bluesky import RunEngine
from bluesky.scans import AdaptiveAscan
from bluesky.examples import Mover, SynGauss
from bluesky.callbacks import LivePlot, LiveTable
from bluesky.tests.utils import setup_test_run_engine


#plt.ion()
RE = setup_test_run_engine()
motor = Mover('motor', ['pos'])
det = SynGauss('det', motor, 'pos', center=0, Imax=1, sigma=1)


#fig, ax = plt.subplots()
#ax.set_xlim([-15, 5])
#ax.set_ylim([0, 2])
# Point the function to our axes above, and specify what to plot.
#my_plotter = LivePlot('det', 'pos')
table = LiveTable(['det', 'pos'])
ad_scan = AdaptiveAscan(motor, [det], 'det', -15, 5, .01, 1, .05, True)
RE(ad_scan, subscriptions={'all': [table]}) #, my_plotter})
