import numpy as np
import bluesky.callbacks
import matplotlib.pyplot as plt
from cycler import cycler

xct = 10
yct = 15
ict = xct * yct

cb = bluesky.callbacks.LiveRaster((yct, xct),  'I', clim=[0, 1])

cb('start', {})
cy = ((cycler('x', np.linspace(0, 1, xct, endpoint=True)) *
       cycler('y', np.linspace(0, 1, yct, endpoint=True))) +
      cycler('I', np.linspace(0, 1, ict, endpoint=True)))
for j, d in enumerate(cy):
    ev = {'data': d, 'seq_num': j}
    cb('event', ev)
    plt.pause(.1)
