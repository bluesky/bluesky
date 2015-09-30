import sys

_INSTALLED = None


def install_qt_kicker():
    """Install a periodic callback to integrate qt and asyncio event loops

    If a version of the qt bindings are not already imported, this function
    will do nothing.

    It is safe to call this function multiple times.
    """

    global _INSTALLED
    if _INSTALLED is not None:
        return
    if not any(p in sys.modules for p in ['PyQt4', 'pyside', 'PyQt5']):
        return
    import asyncio
    import matplotlib.backends.backend_qt5
    from matplotlib.backends.backend_qt5 import _create_qApp
    from matplotlib._pylab_helpers import Gcf

    _create_qApp()
    qApp = matplotlib.backends.backend_qt5.qApp

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
        loop.call_later(0.03, _qt_kicker)

    loop = asyncio.get_event_loop()
    _INSTALLED = loop.call_soon(_qt_kicker)
