import numpy as np
from scipy.ndimage import center_of_mass
from bluesky.callbacks import CollectThenCompute


class PeakStats(CollectThenCompute):

    def __init__(self, x, y):
        """
        Compute peak statsitics after a run finishes.

        Results are stored in the attributes.

        Parameters
        ----------
        x : string
            field name for the x variable (e.g., a motor)
        y : string
            field name for the y variable (e.g., a detector)

        Note
        ----
        It is assumed that the two fields, x and y, are recorded in the same
        Event stream.

        Attributes
        ----------
        com : center of mass
        cen : TBD
        max : x location of y maximum
        min : x location of y minimum
        """
        self.x = x
        self.y = y
        self.com = None
        self.cen = None
        self.max = None
        self.min = None
        super().__init__()

    def __getitem__(self, key):
        if key in ['com', 'cen', 'max', 'min']:
            return getattr(self, key)
        else:
            raise KeyError

    def compute(self):
        "This method is called at run-stop time by the superclass."
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
        # Compute x value at min and max of y
        self.max = x[np.argmax(y)]
        self.min = x[np.argmin(y)]
        self.com = np.interp(center_of_mass(y), x, y)
        self.x_data = x
        self.y_data = y


def plot_peak_stats(ax, peak_stats):
    """
    Plot data and various peak statistics.

    Parameters
    ----------
    ax : matplotlib.Axes
    peak_stats : PeakStats

    Returns
    -------
    arts : dict
        dictionary of matplotlib Artist objects, for further styling
    """
    ps = peak_stats  # for brevity
    if ax is None:
        fig, ax = plt.subplots()
    # Plot points, vertical lines, and a legend. Collect Artist objs to return.
    points, = ax.plot(ps.x_data, ps.y_data, 'o')
    vlines = []
    for attr in ['cen', 'com', 'max', 'min']:
        vlines.append(ax.axvline(getattr(ps, attr), label=attr))
    legend = ax.legend(loc='best')
    arts = {'points': points, 'vlines': vlines, 'legend': legend}
    return arts
