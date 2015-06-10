"""
Useful callbacks for the Run Engine
"""
import sys
from itertools import count
from prettytable import PrettyTable
from collections import OrderedDict
from .utils import doc_type
from datetime import datetime
import matplotlib
from xray_vision.backend.mpl.cross_section_2d import CrossSection
import numpy as np
import filestore

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


class ImageCallback(CallbackBase):

    def __init__(self, datakey):
        # wheeee MRO
        super(ImageCallback, self).__init__()
        self.cs = CrossSection(matplotlib.figure.Figure())
        self.cs.show()
        self.datakey = datakey

    def event(self, doc):
        uid = doc['data'][self.datakey]
        data = filestore.retrieve(uid)
        self.cs.update_image(data)



class CallbackCounter:
    "As simple as it sounds: count how many times a callback is called."
    # Wrap itertools.count in something we can use as a callback.
    def __init__(self):
        self.counter = count()
        self(None)  # Start counting at 1.

    def __call__(self, doc):
        self.value = next(self.counter)


def print_metadata(doc):
    "Print all fields except uid and time."
    for field, value in sorted(doc.items()):
        # uid is returned by the RunEngine, and time is self-evident
        if field not in ['time', 'uid']:
            print('{0}: {1}'.format(field, value))


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
    print_header_interval : int
        The number of events to process and print their rows before printing
        the header again

    Examples
    --------
    Show a table with motor and detector readings..

    >>> RE(stepscan(motor, det), subs={'all': LiveTable(['motor', 'det'])})
    +------------+-------------------+----------------+----------------+
    |   seq_num  |             time  |         motor  |           det  |
    +------------+-------------------+----------------+----------------+
    |         1  |  17:22:19.476380  |  -5.00000e+00  |   3.72665e-06  |
    |         2  |  17:22:19.552068  |  -4.00000e+00  |       0.00034  |
    |         3  |  17:22:19.630759  |  -3.00000e+00  |       0.01111  |
    |         4  |  17:22:19.696007  |  -2.00000e+00  |       0.13534  |
    |         5  |  17:22:19.761655  |  -1.00000e+00  |       0.60653  |
    |         6  |  17:22:19.827130  |   0.00000e+00  |       1.00000  |
    |         7  |  17:22:19.893318  |       1.00000  |       0.60653  |
    |         8  |  17:22:19.970275  |       2.00000  |       0.13534  |
    |         9  |  17:22:20.052067  |       3.00000  |       0.01111  |
    |        10  |  17:22:20.119620  |       4.00000  |       0.00034  |
    +------------+-------------------+----------------+----------------+

    """
    base_fields = ['seq_num', 'time']
    base_field_widths = [8, 15]
    data_field_width = 12
    max_pre_decimal = 5
    max_post_decimal = 5

    def __init__(self, fields=None, rowwise=True, print_header_interval=50):
        super(LiveTable, self).__init__()
        self.rowwise = rowwise
        if fields is None:
            fields = []
        self.fields = fields
        self.table = PrettyTable(field_names=(self.base_fields + self.fields))
        self.table.padding_width = 2
        self.table.align = 'r'
        self.num_events_since_last_header = 1
        self.print_header_interval = print_header_interval

    def _print_table_header(self):
        print('\n'.join(str(self.table).split('\n')[:3]))

    ### RunEngine document callbacks

    def start(self, start_document):
        # format the placeholder fields for the base fields so that the
        # heading prints at the correct width
        base_fields = [' '*width for width in self.base_field_widths]
        # format placeholder fields for the data fields so that the heading
        # prints at the correct width
        data_fields = [' '*self.data_field_width for _ in self.fields]
        self.table.add_row(base_fields + data_fields)
        if self.rowwise:
            self._print_table_header()
        sys.stdout.flush()

    def event(self, event_document):
        event_time = str(datetime.fromtimestamp(event_document['time']).time())
        row = [event_document['seq_num'], event_time]
        for field in self.fields:
            val = event_document['data'].get(field, '')
            if isinstance(val, np.ndarray) or isinstance(val, list):
                val = np.sum(np.asarray(val))
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
            # only print header intermittently for rowwise table printing
            if self.num_events_since_last_header >= self.print_header_interval:
                self._print_table_header()
                self.num_events_since_last_header = 0
            self.num_events_since_last_header += 1
        else:
            # print the whole table
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
