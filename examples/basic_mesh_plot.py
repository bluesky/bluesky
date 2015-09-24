import numpy as np
import bluesky.callbacks
import matplotlib.pyplot as plt
from cycler import cycler

xct = 10
yct = 15
ict = xct * yct

cb = bluesky.callbacks.LiveMesh('x', 'y', 'I', xlim=[-.1, 1.1],
                                ylim=[-.1, 1.1], clim=[0, 1])

cb('start', {})
cy = ((cycler('x', np.linspace(0, 1, xct, endpoint=True)) *
       cycler('y', np.linspace(0, 1, yct, endpoint=True))) +
      cycler('I', np.linspace(0, 1, ict, endpoint=True)))
for d in cy:
    jitter = np.random.rand(2) * .05
    ev = {'data': d}
    d['x'] += jitter[0]
    d['y'] += jitter[1]
    cb('event', ev)
    plt.pause(.1)
