"""
Useful callbacks for the Run Engine
"""
import sys
from itertools import count
import asyncio
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from collections import OrderedDict
from .utils import doc_type
from datetime import datetime
import numpy as np

import logging
logger = logging.getLogger(__name__)

import matplotlib.backends.backend_qt5
from matplotlib.backends.backend_qt5 import _create_qApp
_create_qApp()
qApp = matplotlib.backends.backend_qt5.qApp


def _qt_kicker():
    # The RunEngine Event Loop interferes with the qt event loop. Here we
    # kick it to keep it going.
    plt.draw_all()
    qApp.processEvents()
    loop.call_later(0.1, _qt_kicker)


loop = asyncio.get_event_loop()
loop.call_soon(_qt_kicker)


class CallbackBase(object):
    def __init__(self):
        super().__init__()

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


class LivePlot(CallbackBase):
    def __init__(self, y, x=None, **kwargs):
        """
        Build a function that updates a plot from a stream of Events.

        Parameters
        ----------
        y : str
            the name of a data field in an Event
        x : str or None
            the name of a data field in an Event
            If None, use the Event's sequence number.

        All additional keyword arguments are passed through to ``Axes.plot``.

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
        super().__init__()
        fig, ax = plt.subplots()
        self.ax = ax
        self.ax.set_ylabel(y)
        self.ax.set_xlabel(x or 'sequence #')
        self.ax.margins(.1)
        self.y = y
        self.x = x
        self.kwargs = kwargs
        self.lines = []

    def start(self, doc):
        # The doc is not used; we just use the singal that a new run began.
        self.x_data, self.y_data = [], []
        self.current_line, = self.ax.plot([], [], **self.kwargs)
        self.lines.append(self.current_line)

    def event(self, doc):
        "Update line with data from this Event."
        # Do repeated 'self' lookups once, for perf.
        ax = self.ax
        if self.x is not None:
            self.x_data.append(doc['data'][self.x])
        else:
            self.x_data.append(doc['seq_num'])
        self.y_data.append(doc['data'][self.y])
        self.current_line.set_data(self.x_data, self.y_data)
        # Rescale and redraw.
        ax.relim(visible_only=True)
        ax.autoscale_view(tight=True)
        ax.figure.canvas.draw()


def format_num(x, max_len=11, pre=5, post=5):
    if (abs(x) > 10**pre or abs(x) < 10**-post) and x != 0:
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
    |   seq_num  |             time  |         motor  |   sum(det_2d)  |
    +------------+-------------------+----------------+----------------+
    |         1  |  12:46:47.503068  |         -5.00  |        449.77  |
    |         3  |  12:46:47.682788  |         -3.00  |        460.60  |
    |         4  |  12:46:47.792307  |         -2.00  |        584.77  |
    |         5  |  12:46:47.915401  |         -1.00  |       1056.37  |
    |         7  |  12:46:48.120626  |          1.00  |       1056.50  |
    |         8  |  12:46:48.193028  |          2.00  |        583.33  |
    |         9  |  12:46:48.318454  |          3.00  |        460.99  |
    |        10  |  12:46:48.419579  |          4.00  |        451.53  |
    +------------+-------------------+----------------+----------------+


    """
    base_fields = ['seq_num', 'time']
    base_field_widths = [8, 15]
    data_field_width = 12
    max_pre_decimal = 5
    max_post_decimal = 2

    def __init__(self, fields=None, rowwise=True, print_header_interval=50,
                 max_post_decimal=2, max_pre_decimal=5, data_field_width=12,
                 logbook=None):
        self.data_field_width = data_field_width
        self.max_pre_decimal = max_pre_decimal
        self.max_post_decimal = max_post_decimal
        super(LiveTable, self).__init__()
        self.rowwise = rowwise
        if fields is None:
            fields = []
        self.fields = _get_obj_fields(fields)
        self.field_column_names = [field for field in self.fields]
        self.num_events_since_last_header = 0
        self.print_header_interval = print_header_interval
        self.logbook = logbook
        self._filestore_keys = set()
        # self.create_table()

    def write(self, s):
        if self.logbook:
            self.logbook(s, {'run_start_uid': self.run_start_uid})
        else:
            print(s)

    def create_table(self):
        self.table = PrettyTable(field_names=(self.base_fields +
                                              self.field_column_names))
        self.table.padding_width = 2
        self.table.align = 'r'
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

    def _print_table_header(self):
        print('\n'.join(str(self.table).split('\n')[:3]))

    ### RunEngine document callbacks

    def start(self, start_document):
        self.run_start_uid = start_document['uid']
        self.create_table()


    def descriptor(self, descriptor):
        # find all keys that are filestore keys
        for key, datakeydict in descriptor['data_keys'].items():
            data_loc = datakeydict.get('external', '')
            if data_loc == 'FILESTORE:':
                self._filestore_keys.add(key)

        # see if any are being shown in the table
        reprint_header = False
        new_names = []
        for key in self.field_column_names:
            if key in self._filestore_keys:
                reprint_header = True
                print('%s is a non-scalar field. '
                           'Computing the sum instead' % key)
                key = 'sum(%s)' % key
                key = key[:self.data_field_width]
            new_names.append(key)
        self.field_column_names = new_names
        if reprint_header:
            print('\n\n')
            self.create_table()
            # self._print_table_header()

    def event(self, event_document):
        event_time = str(datetime.fromtimestamp(event_document['time']).time())
        row = [event_document['seq_num'], event_time]
        for field in self.fields:
            val = event_document['data'].get(field, '')
            if field in self._filestore_keys:
                val = fsapi.retrieve(val)
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
            # [-1] is the bottom border
            print(str(self.table).split('\n')[-2])
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
        if self.logbook:
            # Finnicky styling....
            my_table = str(self.table).split('\n')
            my_table.pop(3)  # remove blank line
            my_table = '\n'.join(my_table)
            self.logbook(str(my_table), {
                'run_start_uid': stop_document['run_start']})
        print(str(self.table).split('\n')[-1])
        # remove all data from the table
        self.table.clear_rows()
        # reset the filestore keys
        self._filestore_keys = set()

def _get_obj_fields(fields):
    """
    If fields includes any objects, get their field names using obj.describe()

    ['det1', det_obj] -> ['det1, 'det_obj_field1, 'det_obj_field2']"
    """
    string_fields = []
    for field in fields:
        if isinstance(field, str):
            string_fields.append(field)
        else:
            try:
                field_list = sorted(field.describe().keys())
            except AttributeError:
                raise ValueError("Fields must be strings or objects with a "
                                 "'describe' method that return a dict.")
            string_fields.extend(field_list)
    return string_fields
