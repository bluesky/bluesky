import asyncio
import time as ttime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_qt5
from matplotlib.backends.backend_qt5 import _create_qApp
_create_qApp()

qApp = matplotlib.backends.backend_qt5.qApp

plt.close('all')
fig, ax = plt.subplots()
ln, = ax.plot([], [], marker='o')
ax.set_xlim(0, 50)
ax.set_ylim(0, 50)


loop = asyncio.get_event_loop()


def dummy(start_time, j, timeout):
    if loop.time() > start_time + timeout:
        print("skipping  {}".format(j))
        return
    print("running! {}".format(j))


def plotter(j):
    N = 10000
    for _ in range(N):
        ttime.sleep(.3 / N)
    ln.set_xdata(np.r_[ln.get_xdata(), j])
    ln.set_ydata(np.r_[ln.get_ydata(), j])


def expiring_function(func, *args, **kwargs):
    def dummy(start_time, timeout):
        if loop.time() > start_time + timeout:
            print("skipping")
            return
        print("running!")
        return func(*args, **kwargs)

    return dummy


@asyncio.coroutine
def manager(n):
    tasks = []
    for j in range(n):
        start_time = loop.time()
        dummy = expiring_function(plotter, j)
        t = loop.run_in_executor(None, dummy, start_time, 10)
        tasks.append(t)
        yield from asyncio.sleep(.1)

    yield from asyncio.wait(tasks)


def qt_kicker():
    plt.draw_all()
    qApp.processEvents()
    loop.call_later(.1, qt_kicker)

loop.call_later(.1, qt_kicker)


loop.run_until_complete(manager(50))
