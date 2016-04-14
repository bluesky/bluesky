import warnings
from .utils import install_qt_kicker as _install_qt_kicker


def install_qt_kicker():
    warnings.warn("The qt_kicker module is deprecated. Instead, use "
                  "`from bluesky.utils import install_qt_kicker`")
    _install_qt_kicker()
