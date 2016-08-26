import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from bluesky.callbacks import CollectThenCompute


class PeakStats(CollectThenCompute):
    """
    Compute peak statsitics after a run finishes.

    Results are stored in the attributes.

    Parameters
    ----------
    x : string
        field name for the x variable (e.g., a motor)
    y : string
        field name for the y variable (e.g., a detector)

    edge_count : int or None, optional
        If not None, number of points at beginning and end to use
        for quick and dirty background subtraction.

    Note
    ----
    It is assumed that the two fields, x and y, are recorded in the same
    Event stream.

    Attributes
    ----------
    com : center of mass
    cen : mid-point between half-max points on each side of the peak
    max : x location of y maximum
    min : x location of y minimum
    """

    def __init__(self, x, y, edge_count=None):
        self.x = x
        self.y = y
        self.com = None
        self.cen = None
        self.max = None
        self.min = None
        self.nlls = None
        self.fwhm = None
        self.lin_bkg = None
        self._edge_count = edge_count
        super().__init__()

    def __getitem__(self, key):
        if key in ['com', 'cen', 'max', 'min']:
            return getattr(self, key)
        else:
            raise KeyError

    def compute(self):
        "This method is called at run-stop time by the superclass."
        # clear all results
        self.com = None
        self.cen = None
        self.max = None
        self.min = None
        self.nlls = None
        self.fwhm = None
        self.lin_bkg = None

        x = []
        y = []
        for event in self._events:
            try:
                _x = event['data'][self.x]
                _y = event['data'][self.y]
            except KeyError:
                pass
            else:
                x.append(_x)
                y.append(_y)
        x = np.array(x)
        y = np.array(y)
        if not len(x):
            # nothing to do
            return
        self.x_data = x
        self.y_data = y
        if self._edge_count is not None:
            left_x = np.mean(x[:self._edge_count])
            left_y = np.mean(y[:self._edge_count])

            right_x = np.mean(x[-self._edge_count:])
            right_y = np.mean(y[-self._edge_count:])

            m = (right_y - left_y) / (right_x - left_x)
            b = left_y - m * left_x
            # don't do this in place to not mess with self.y_data
            y = y - (m * x + b)
            self.lin_bkg = {'m': m, 'b': b}

        # Compute x value at min and max of y
        self.max = x[np.argmax(y)], self.y_data[np.argmax(y)],
        self.min = x[np.argmin(y)], self.y_data[np.argmin(y)],
        self.com, = np.interp(center_of_mass(y), np.arange(len(x)), x)
        mid = (np.max(y) + np.min(y)) / 2
        crossings = np.where(np.diff(y > mid))[0]
        _cen_list = []
        for cr in crossings.ravel():
            _x = x[cr:cr+2]
            _y = y[cr:cr+2] - mid

            dx = np.diff(_x)
            dy = np.diff(_y)
            m = dy / dx
            _cen_list.append((-_y[0] / m) + _x[0])

        if _cen_list:
            self.cen = np.mean(_cen_list)

        if len(_cen_list) == 2:
            self.fwhm = float(_cen_list[1] - _cen_list[0])

        # reset y data
        y = self.y_data
        # insert lmfit


def plot_peak_stats(peak_stats, ax=None):
    """
    Plot data and various peak statistics.

    Parameters
    ----------
    peak_stats : PeakStats
    ax : matplotlib.Axes, optional

    Returns
    -------
    arts : dict
        dictionary of matplotlib Artist objects, for further styling
    """
    arts = {}
    ps = peak_stats  # for brevity
    if ax is None:
        fig, ax = plt.subplots()
    ax.margins(.1)
    # Plot points, vertical lines, and a legend. Collect Artist objs to return.
    points, = ax.plot(ps.x_data, ps.y_data, 'o')
    vlines = []
    styles = iter(cycler('color', 'krgbm'))
    for style, attr in zip(styles, ['cen', 'com']):
        print(style, attr)
        val = getattr(ps, attr)
        if val is None:
            continue
        vlines.append(ax.axvline(val, label=attr, **style))

    for style, attr in zip(styles, ['max', 'min']):
        print(style, attr)
        val = getattr(ps, attr)
        if val is None:
            continue
        vlines.append(ax.axvline(val[0], label=attr, lw=3, **style))
        vlines.append(ax.axhline(val[1], lw=3, **style))

    if ps.lin_bkg:
        lb = ps.lin_bkg
        ln, = ax.plot(ps.x_data, ps.x_data*lb['m'] + lb['b'],
                      ls='--', lw=2, color='k')
        arts['bkg'] = ln

    legend = ax.legend(loc='best')
    arts.update({'points': points, 'vlines': vlines, 'legend': legend})
    return arts
