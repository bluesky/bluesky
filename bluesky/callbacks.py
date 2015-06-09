"""
Useful callbacks for the Run Engine
"""
import sys
from itertools import count
from prettytable import PrettyTable
from collections import OrderedDict
from .utils import doc_type

import logging
logger = logging.getLogger(__name__)


class CallbackBase(object):
    def __init__(self):
        super(CallbackBase, self).__init__()

    def dispatch(self, doc):
        doc_name = doc_type(doc)
        doc_func = getattr(self, doc_name)
        doc_func(doc)

    def event(self, doc):
        logger.debug("CallbackBase: I'm an event with doc = {}".format(doc))

    def descriptor(self, doc):
        logger.debug("CallbackBase: I'm a descriptor with doc = {}".format(doc))

    def start(self, doc):
        logger.debug("CallbackBase: I'm a start with doc = {}".format(doc))

    def stop(self, doc):
        logger.debug("CallbackBase: I'm a stop with doc = {}".format(doc))


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
    def f(event):
        output.append(event['data'][field])

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

    Returns
    -------
    func : function
        expects one argument, an Event dictionary

    Examples
    --------
    >>> import matplotlib as pyplot
    >>> fig, ax = plt.subplots()
    >>> my_plotter = live_scalar_plotter(ax, 'det1', 'motor1')
    >>> RE(my_scan, subs={'event': my_plotter})
    """
    x_data, y_data = [], []
    line, = ax.plot([], [], 'ro', markersize=10)

    def update_plot(event):
        # Update with the latest data.
        x_data.append(event['data'][x])
        y_data.append(event['data'][y])
        line.set_data(x_data, y_data)
        # Rescale and redraw.
        ax.relim(visible_only=True)
        ax.autoscale_view(tight=True)
        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()

    return update_plot


def format_num(x, max_len=11, pre=5, post=5):
    if x > 10**pre or x < 10**-post:
        x = '%.{}e'.format(post) % x
    else:
        x = '%{}.{}f'.format(pre, post) % x

    return x


class LiveTable(CallbackBase):

    base_fields = ['seq_num']

    def __init__(self, rowwise=True, fields=None):
        super(LiveTable, self).__init__()
        self.rowwise = rowwise
        if fields is None:
            fields = []
        self.fields = self.base_fields + fields

    def event(self, event_document):
        row = [event_document['seq_num']]
        row.extend([format_num(event_document['data'].get(field, ''),
                               max_len=11, pre=5, post=5)
                    for field in self.fields])
        self.table.add_row(row)

        if self.rowwise:
            # Print the last row of data only.
            print(str(self.table).split('\n')[-2])  # [-1] is the bottom border
        else:
            print(self.table)
        sys.stdout.flush()

    def stop(self, stop_document):
        """Print the last row of the table

        Parameters
        ----------
        stop_document : dict
            Not explicitly used in this function, other than to signal that
            the run has been completed
        """
        print(str(self.table).split('\n')[-1])



def live_table(fields, rowwise=True):
    """
    Build a function that prints data from each Event as a row in a table.

    Parameters
    ----------
    fields : list
        names of interesting data keys
    rowwise : bool
        If True, append each row to stdout. If False, reprint the full updated
        table each time. This is useful if other messsages are interspersed.

    Examples
    --------
    >>> RE(stepscan(motor, det), subs={'event': live_table(['motor', 'det'])})
    +-------+-----+----------------------+
    |seq_num|motor|         det          |
    +-------+-----+----------------------+
    |   1   |   -5|3.726653172078671e-06 |
    |   2   |   -4|0.00033546262790251185|
    |   3   |   -3| 0.011108996538242306 |
    |   4   |   -2|  0.1353352832366127  |
    |   5   |   -1|  0.6065306597126334  |
    |   6   |   0 |         1.0          |
    |   7   |   1 |  0.6065306597126334  |
    |   8   |   2 |  0.1353352832366127  |
    +-------+-----+----------------------+
    """
    table = PrettyTable(['seq_num'] + fields)

    table.padding_width = 2
    table.align = 'r'
    if rowwise:
        print(table)


    def update_table(event):
        # todo figure out how to format this correctly
        row = [event['seq_num']]
        row.extend([format_num(event['data'].get(field, ''))
                    for field in fields])
        table.add_row(row)

        if rowwise:
            # Print the last row of data only.
            print(str(table).split('\n')[-2])  # [-1] is the bottom border
        else:
            print(table)
        sys.stdout.flush()
    return update_table
