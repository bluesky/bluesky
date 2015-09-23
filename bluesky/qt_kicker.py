"""
A utility that, once imported, kicks the qt event loop at regular
intervals. This lets qt and asyncio peacefully coexist, and it is 
necessary for matplotlib < 1.5.0.
"""
import asyncio
import matplotlib.backends.backend_qt5
from matplotlib.backends.backend_qt5 import _create_qApp


_create_qApp()
qApp = matplotlib.backends.backend_qt5.qApp

from matplotlib._pylab_helpers import Gcf

try:
    _draw_all = Gcf.draw_all  # mpl version >= 1.5
except AttributeError:
    # slower, but backward-compatible
    def _draw_all():
        for f_mgr in Gcf.get_all_fig_managers():
            f_mgr.canvas.draw_idle()


def _qt_kicker():
    # The RunEngine Event Loop interferes with the qt event loop. Here we
    # kick it to keep it going.
    _draw_all()

    qApp.processEvents()
    loop.call_later(0.1, _qt_kicker)


loop = asyncio.get_event_loop()
loop.call_soon(_qt_kicker)
