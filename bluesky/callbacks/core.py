"""
Useful callbacks for the Run Engine
"""
import sys
from itertools import count
from collections import deque
import warnings
from prettytable import PrettyTable
import jinja2

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import numpy as np
import ast
import os
import logging
logger = logging.getLogger(__name__)


class CallbackBase:
    def __call__(self, name, doc):
        "Dispatch to methods expecting particular doc types."
        return getattr(self, name)(doc)

    def event(self, doc):
        pass

    def bulk_events(self, doc):
        pass

    def descriptor(self, doc):
        pass

    def start(self, doc):
        pass

    def stop(self, doc):
        pass


class CallbackCounter:
    "As simple as it sounds: count how many times a callback is called."
    # Wrap itertools.count in something we can use as a callback.
    def __init__(self):
        self.counter = count()
        self(None, {})  # Pass a fake doc to prime the counter (start at 1).

    def __call__(self, name, doc):
        self.value = next(self.counter)


def print_metadata(name, doc):
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
    def f(name, event):
        output.append(event['data'][field])

    return f


class LivePlot(CallbackBase):
    """
    Build a function that updates a plot from a stream of Events.

    Note: If your figure blocks the main thread when you are trying to
    scan with this callback, call `plt.ion()` in your IPython session.

    Parameters
    ----------
    y : str
        the name of a data field in an Event
    x : str, optional
        the name of a data field in an Event
        If None, use the Event's sequence number.
    legend_keys : list, optional
        The list of keys to extract from the RunStart document and format
        in the legend of the plot. The legend will always show the
        scan_id followed by a colon ("1: ").  Each
    xlim : tuple
        passed to Axes.set_xlim
    ylim : tuple
        passed to Axes.set_ylim
    All additional keyword arguments are passed through to ``Axes.plot``.

    Examples
    --------
    >>> my_plotter = LivePlot('det', 'motor', legend_keys=['sample'])
    >>> RE(my_scan, my_plotter)
    """
    def __init__(self, y, x=None, legend_keys=None, xlim=None, ylim=None,
                 fig=None, **kwargs):
        super().__init__()
        if fig is None:
            # overplot (or, if no fig exists, one is made)
            fig = plt.gcf()

        ax = fig.gca()

        if legend_keys is None:
            legend_keys = []
        self.legend_keys = ['scan_id'] + legend_keys
        if x is not None:
            self.x, *others = _get_obj_fields([x])
        else:
            self.x = None
        self.y, *others = _get_obj_fields([y])
        self.fig = fig
        self.ax = ax
        self.ax.set_ylabel(y)
        self.ax.set_xlabel(x or 'sequence #')
        if xlim is not None:
            self.ax.set_xlim(*xlim)
        if ylim is not None:
            self.ax.set_ylim(*ylim)
        self.ax.margins(.1)
        self.kwargs = kwargs
        self.lines = []
        self.legend = None
        self.legend_title = " :: ".join([name for name in self.legend_keys])

    def start(self, doc):
        # The doc is not used; we just use the singal that a new run began.
        self.x_data, self.y_data = [], []
        label = " :: ".join(
            [str(doc.get(name, ' ')) for name in self.legend_keys])
        self.current_line, = self.ax.plot([], [], label=label, **self.kwargs)
        self.lines.append(self.current_line)
        self.legend = self.ax.legend(loc=0, title=self.legend_title).draggable()

    def event(self, doc):
        "Update line with data from this Event."
        # Do repeated 'self' lookups once, for perf.
        ax = self.ax
        try:
            if self.x is not None:
                # this try/except block is needed because multiple event streams
                # will be emitted by the RunEngine and not all event streams will
                # have the keys we want
                new_x = doc['data'][self.x]
            else:
                new_x = doc['seq_num']
            new_y = doc['data'][self.y]
        except KeyError:
            # wrong event stream, skip it
            return
        self.y_data.append(new_y)
        self.x_data.append(new_x)
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

    >>> RE(stepscan(motor, det), LiveTable(['motor', 'det']))
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
    base_field_widths = [8, 10]
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
        # prettytable does not allow nonunique names
        self.fields = sorted(set(_get_obj_fields(fields)))
        self.field_column_names = [field for field in self.fields]
        self.num_events_since_last_header = 0
        self.print_header_interval = print_header_interval
        self.logbook = logbook
        self._filestore_keys = set()
        # self.create_table()

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
        self.scan_id = start_document['scan_id']
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
        event_time = datetime.fromtimestamp(event_document['time']).time()
        rounded_time = str(event_time)[:10]
        row = [event_document['seq_num'], rounded_time]
        for field in self.fields:
            val = event_document['data'].get(field, '')
            if field in self._filestore_keys:
                try:
                    import filestore.api as fsapi
                    val = fsapi.retrieve(val)
                except Exception as exc:
                    warnings.warn(UserWarning, "Attempt to read {0} raised {1}"
                                  "".format(field, exc))
                    val = 'Not Available'
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

        if self.logbook and self.run_start_uid == stop_document['run_start']:
            header = ["Scan {scan_id} (uid='{run_start_uid}')", '']
            # drop the padding row
            self.table.start = 1
            my_table = '\n'.join(header + [str(self.table), ])
            self.logbook(my_table, {
                'run_start_uid': stop_document['run_start'],
                'scan_id': self.scan_id})

        self.table.start = 0
        print(str(self.table).split('\n')[-1])
        sys.stdout.flush()
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


class CollectThenCompute(CallbackBase):

    def __init__(self):
        self._start_doc = None
        self._stop_doc = None
        self._events = deque()
        self._descriptors = deque()

    def start(self, doc):
        self._start_doc = doc

    def descriptor(self, doc):
        self._descriptors.append(doc)

    def event(self, doc):
        self._events.append(doc)

    def stop(self, doc):
        self._stop_doc = doc
        self.compute()

    def reset(self):
        self._start_doc = None
        self._stop_doc = None
        self._events.clear()
        self._descriptors.clear()

    def compute(self):
        raise NotImplementedError("This method must be defined by a subclass.")


class LiveMesh(CallbackBase):
    """Simple callback that fills in values based on a mesh scan

    This simply wraps around a `PathCollection` as generated by scatter

    Parameters
    ----------
    x, y : str
       The fields to use for the x and y data

    I : str
        The field to use for the color of the markers

    xlim, ylim, clim : tuple, optional
       The x, y and color limits respectively

    cmap : str or colormap, optional
       The color map to use
    """
    def __init__(self, x, y, I, *, xlim=None, ylim=None,
                 clim=None, cmap='viridis'):
        fig, ax = plt.subplots()
        self.x = x
        self.y = y
        self.I = I
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_aspect('equal')
        self._sc = []
        self.ax = ax
        ax.margins(.1)
        self.fig = fig
        self._xdata, self._ydata, self._Idata = [], [], []
        self._norm = mcolors.Normalize()

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if clim is not None:
            self._norm.vmin, self._norm.vmax = clim
        self.cmap = cmap

    def start(self, doc):
        self._xdata, self._ydata, self._Idata = [], [], []
        sc = self.ax.scatter(self._xdata, self._ydata, c=self._Idata,
                             norm=self._norm, cmap=self.cmap, edgecolor='face',
                             s=50)
        self._sc.append(sc)
        self.sc = sc

    def event(self, doc):
        self._xdata.append(doc['data'][self.x])
        self._ydata.append(doc['data'][self.y])
        self._Idata.append(doc['data'][self.I])

        offsets = np.vstack([self._xdata, self._ydata]).T
        self.sc.set_offsets(offsets)
        self.sc.set_array(np.asarray(self._Idata))


class LiveRaster(CallbackBase):
    """Simple callback that fills in values based on a raster

    This simply wraps around a `AxesImage`.  seq_num is used to
    determine which pixel to fill in

    Parameters
    ----------
    raster_shap : tuple
        The (row, col) shape of the raster

    I : str
        The field to use for the color of the markers

    clim : tuple, optional
       The color limits

    cmap : str or colormap, optional
       The color map to use
    """
    def __init__(self, raster_shape, I, *,
                 clim=None, cmap='viridis',
                 xlabel='x', ylabel='y', extent=None):
        fig, ax = plt.subplots()
        self.I = I
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        self.ax = ax
        self.fig = fig
        self._Idata = np.ones(raster_shape) * np.nan
        self._norm = mcolors.Normalize()
        if clim is not None:
            self._norm.vmin, self._norm.vmax = clim
        self.clim = clim
        self.cmap = cmap
        self.raster_shape = raster_shape
        self.im = None
        self.extent = extent

    def start(self, doc):
        if self.im is not None:
            raise RuntimeError("Can not re-use LiveRaster")
        self._Idata = np.ones(self.raster_shape) * np.nan
        im = self.ax.imshow(self._Idata, norm=self._norm,
                            cmap=self.cmap, interpolation='none',
                            extent=self.extent)
        self.im = im
        self.ax.set_title('scan {uid} [{sid}]'.format(sid=doc['scan_id'],
                                                      uid=doc['uid'][:6]))
        cb = self.fig.colorbar(im)
        cb.set_label(self.I)

    def event(self, doc):
        seq_num = doc['seq_num'] - 1
        pos = np.unravel_index(seq_num, self.raster_shape)

        self._Idata[pos] = doc['data'][self.I]
        if self.clim is None:
            self.im.set_clim(np.nanmin(self._Idata), np.nanmax(self._Idata))

        self.im.set_array(self._Idata)


env = jinja2.Environment()

_SPEC_HEADER_TEMPLATE = env.from_string("""#F {{ filepath }}
#E {{ unix_time }}
#D {{ readable_time }}
#C {{ owner }}  User = {{ owner }}
#O0 {{ positioners | join(' ') }}""")

_SPEC_1D_COMMAND_TEMPLATE = env.from_string("{{ scan_type }} {{ start }} {{ stop }} {{ stides }} {{ time }}")

_PLAN_TO_SPEC_MAPPING = {'AbsScanPlan': 'ascan',
                         'DeltaScanPlan': 'dscan',
                         'Count': 'count',
                         'Tweak': 'tw'}

_SPEC_START_TEMPLATE = env.from_string("""

#S {{ scan_id }} {{ command }}
#D {{ readable_time }}
#T {{ acq_time }} (Seconds)
#P0 {{ positioner_positions }}""")

_SPEC_DESCRIPTOR_TEMPLATE = env.from_string("""
#N {{ length }}
#L {{ motor_name }} Epoch Seconds {{ data_keys | join(' ') }}\n""")

_SPEC_EVENT_TEMPLATE = env.from_string(
    """{{ motor_position }}  {{ unix_time }} {{ acq_time }} {{ values | join(' ') }}\n""")

class LiveSpecFile(CallbackBase):
    """Callback to export scalar values to a spec file for viewing

    Expect:
        `
    1. a descriptor named 'baseline'
    2. an event for that descriptor
    3. a descriptor named 'main'
    4. events for that descriptor

    Other documents can be issues before, between, and after, but
    these must be issued and in this order.

    Example
    -------
    It is suggested to put this in the ipython profile:
    >>> from bluesky.callbacks import LiveSpecFile
    >>> live_specfile_callback = LiveSpecFile(os.path.expanduser('~/specfiles/test.spec'))
    >>> gs.RE.subscribe('all', live_specfile_callback)
    >>> # Modify the spec file location like this:
    >>> # live_specfile_callback.filepath = '/some/new/filepath.spec'
    """
    def __init__(self, specpath):
        """
        Parameters
        ----------
        specpath : str
            The location on disk where you want the specfile to be written
        """
        self.specpath = specpath
        self.pos_names = ["No", "Positioners", "Were", "Given"]
        self.positions = ["-inf", "-inf", "-inf", "-inf"]

    def _write_spec_header(self, doc):
        """
        Parameters
        ----------
        doc : start document from bluesky

        Returns
        -------
        spec_header : list
            The spec header as a list of lines
        Note
        ----
        Writes a new spec file header that looks like this:
        #F /home/xf11id/specfiles/test.spec
        #E 1449179338.3418093
        #D 2015-12-03 16:48:58.341809
        #C xf11id  User = xf11id
        #O [list of all motors, all on one line]
        """
        content = dict(filepath=self.specpath,
                       unix_time=doc['time'],
                       readable_time=datetime.fromtimestamp(doc['time']),
                       owner=doc['owner'],
                       positioners=self.pos_names)
        with open(self.specpath, 'w') as f:
            f.write(_SPEC_HEADER_TEMPLATE.render(content))

    def start(self, doc):
        if not os.path.exists(self.specpath):
            spec_header = self._write_spec_header(doc)
        # TODO verify that list of positioners is unchanged  by reading file
        # and parsing any existing contents.
        plan_type = doc['plan_type']
        plan_args = doc['plan_args']
        if plan_type in ['AbsScanPlan', 'DeltaScanPlan', 'Count', 'Tweak']:
            # Some of these are used in other methods too -- stash them.
            self._unix_time = doc['time']
            self._acq_time = plan_args.get('time', -1)
            self._motor = doc['motors']
            if len(self._motor) > 1:
                raise NotImplementedError(
                    "Your scan has %s scanning motors. They are %s. SpecCallback"
                    " cannot handle multiple scanning motors. Please request "
                    "this feature at https://github.com/NSLS-II/bluesky/issues" %
                    (len(self._motors), self._motors))
            self._motor, = self._motor
            content = dict(scan_type=_PLAN_TO_SPEC_MAPPING[doc['plan_type']],
                           start=plan_args['start'],
                           stop=plan_args['stop'],
                           strides=int(plan_args['num']) - 1,
                           acq_time=self._acq_time,
                           positioner_positions=self.positions)
            command = _SPEC_1D_COMMAND_TEMPLATE.render(content)
        else:
            err_msg = ("Do not know how to represent %s in SPEC. If "
                       "you would like this feature, request it at "
                       "https://github.com/NSLS-II/bluesky/issues"
                       % plan_type)
            raise NotImplementedError(err_msg)
        # write the new scan entry
        content = dict(command=command,
                       scan_id=doc['scan_id'],
                       readable_time=datetime.fromtimestamp(doc['time']),
                       acq_time=self._acq_time)
        self._start_content = content  # can't write until after we see desc
        self._start_doc = doc

    def descriptor(self, doc):
        """Write the header for the actual scan data"""
        self._read_fields = sorted([k for k, v in doc['data_keys'].items()
                                    if v['object_name'] != self._motor])
        content = dict(motor_name=self._motor,
                       acq_time=self._acq_time,
                       unix_time=self._unix_time,
                       length=3 + len(self._read_fields),
                       data_keys=doc['data_keys'].keys())
        with open(self.specpath, 'a') as f:
            f.write(_SPEC_DESCRIPTOR_TEMPLATE.render(content))
            f.write('\n')

    def event(self, doc):
        """Write each event out"""""
#         print(doc)
        data = doc['data']
        values = [str(data[k]) for k in self._read_fields if k != self._motor]
        content = dict(acq_time=self._acq_time,
                       unix_time=doc['time'],
                       motor_position=data[self._motor],
                       values=values)
        with open(self.specpath, 'a') as f:
            f.write(_SPEC_EVENT_TEMPLATE.render(content))
            f.write('\n')

#     def future_descriptor(self, doc):
#         """Write the header for the actual scan data
#         """
#         if 'name' not in doc:
#             return
#         if doc['name'] == 'baseline':
#             self._baseline_desc_uid = doc['uid']
#             # Now we know all the positioners involved and can write the
#             # spec file.
#             pos_names = sorted([dk['object_name'] for dk in doc['data_keys']])
#             self._write_spec_header(self._start_doc, pos_names)
#             with open(self.specpath, 'a') as f:
#                 f.write(_SPEC_SCAN_TEMPLATE.render(content))
#         if doc['name'] == 'main':
#             self._main_desc_uid = doc['main']
#             self._read_fields = sorted([k for k, v in doc['data_keys'].items()
#                                         if v['object_name'] != self._motor])
#         content = dict(motor_name=self._motor,
#                        acq_time=self._acq_time,
#                        unix_time=self._unix_time,
#                        length=3 + len(self._read_fields))
#         with open(self.specpath, 'a') as f:
#             f.write(_SPEC_SCAN_TEMPLATE.render(content))

#     def future_event(self, doc):
#         """
#         Two cases:
#         1. We have a 'baseline' event; write baseline motor positioners
#            and detector values.
#         2. We have a 'main' event; write one line of data.
#         """
#         data = doc['data']
#         if doc['descriptor'] == self._baseline_desc_uid:
#             # This is a 'baseline' event.
#             if self._wrote_baseline_values:
#                 return
#             baseline = {k: str(data[v]) for k, v in sorted(data.items())}
#             with open(self.specpath, 'a') as f:
#                 # using fmt strings; this operation would be a pain with jinja
#                 for idx, (key, val) in enumerate(baseline):
#                     f.write('#M%s %s %s\n' % (idx, key, val))
#             self._wrote_baseline_values = True
#         elif doc['descriptor'] == self._main_desc_uid:
#             values = [str(data[v]) for k, v in self._read_fields]
#             content = dict(acq_time=self._acq_time,
#                            unix_time=self._unix_time,
#                            motor_name=self._motor,
#                            values=values)
#             with open(self.specpath, 'a') as f:
#                 f.write(_SPEC_EVENT_TEMPLATE.render(content))
