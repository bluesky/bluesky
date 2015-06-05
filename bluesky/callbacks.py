"""
Useful callbacks for the Run Engine
"""
from itertools import count


class CallbackCounter:
    "As simple as it sounds: count how many times a callback is called."
    # Wrap itertools.count in something we can use as a callback.
    def __init__(self):
        self.counter = count()
        self(None)  # Start counting at 1.

    def __call__(self, doc):
        self.value = next(self.counter)


def collector(field, output):
    """
    Build a function that appends data to a list.

    This is useful for testing but not advised for general use. (There is
    probably a better way to do whatever you want to do!)

    Parameters
    ----------
    field : str
        the name of a data field in an Event
    output : mutable iterable
        such as a list

    Returns
    -------
    func : function
        expects one argument, an Event dictionary
    """
    def f(doc):
        output.append(doc['data'][field])

    return f


def live_scalar_plotter(ax, y, x):
    """
    Build a function that updates a plot from a stream of Events.

    Parameters
    ----------
    ax : Axes
    y : str
        the name of a data field in an Event
    x : str
        the name of a data field in an Event

    Examples
    --------
    >>> import matplotlib as pyplot
    >>> fig, ax = plt.subplots()
    >>> my_plotter = live_scalar_plotter(ax, 'det1', 'motor1')
    >>> RE(my_scan, subs={'event': my_plotter})
    """
    x_data, y_data = [], []
    line, = ax.plot([], [], 'ro', markersize=10)

    def update_plot(doc):
        # Update with the latest data.
        x_data.append(doc['data'][x])
        y_data.append(doc['data'][y])
        line.set_data(x_data, y_data)
        # Rescale and redraw.
        ax.relim(visible_only=True)
        ax.autoscale_view(tight=True)
        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()

    return update_plot
