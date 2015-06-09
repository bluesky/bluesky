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

    def __call__(self, doc):
        """Inspect the document, infer its type, and dispatch it."""
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
    """
    Build a function that prints data from each Event as a row in a table.

    Parameters
    ----------
    fields : list, optional
        names of data fields to include in addition to 'seq_num'
    rowwise : bool
        If True, append each row to stdout. If False, reprint the full updated
        table each time. This is useful if other messsages are interspersed.

    Examples
    --------
    Show a table with motor and detector readings..

    >>> RE(stepscan(motor, det), subs={'all': LiveTable(['motor', 'det'])})
    +------------+----------------+----------------+
    |   seq_num  |         motor  |           det  |
    +------------+----------------+----------------+
    |         1  |  -5.00000e+00  |   3.72665e-06  |
    |         2  |  -4.00000e+00  |       0.00034  |
    |         3  |  -3.00000e+00  |       0.01111  |
    |         4  |  -2.00000e+00  |       0.13534  |
    |         5  |  -1.00000e+00  |       0.60653  |
    |         6  |   0.00000e+00  |       1.00000  |
    |         7  |       1.00000  |       0.60653  |
    |         8  |       2.00000  |       0.13534  |
    |         9  |       3.00000  |       0.01111  |
    |        10  |       4.00000  |       0.00034  |
    +------------+----------------+----------------+
    """
    base_fields = ['seq_num']
    base_field_widths = [8]
    data_field_width = 12
    max_pre_decimal = 5
    max_post_decimal = 5

    def __init__(self, rowwise=True, fields=None):
        super(LiveTable, self).__init__()
        self.rowwise = rowwise
        if fields is None:
            fields = []
        self.fields = fields
        self.table = PrettyTable(field_names=(self.base_fields + self.fields))
        self.table.padding_width = 2
        self.table.align = 'r'

    def start(self, start_document):
        base_field_widths = self.base_field_widths
        if len(self.base_fields) > 1 and len(base_field_widths) == 1:
            base_field_widths = base_field_widths * len(self.base_fields)
        # format the placeholder fields for the base fields so that the
        # heading prints at the correct width
        base_fields = [' '*width for width in base_field_widths]
        # format placeholder fields for the data fields so that the heading
        # prints at the correct width
        data_fields = [' '*self.data_field_width for _ in self.fields]
        self.table.add_row(base_fields + data_fields)
        if self.rowwise:
            print('\n'.join(str(self.table).split('\n')[:-2]))
        sys.stdout.flush()

    def event(self, event_document):
        row = [event_document['seq_num']]
        for field in self.fields:
            val = event_document['data'].get(field, '')
            try:
                val = format_num(val,
                                 max_len=self.data_field_width,
                                 pre=self.max_pre_decimal,
                                 post=self.max_post_decimal)
            except Exception:
                val = str(val)[:self.data_field_width]
            row.append(val)
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
        # remove all data from the table
        self.table.clear_rows()
