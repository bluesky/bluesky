"""
Useful callbacks for the Run Engine
"""
import sys
from itertools import count
from collections import deque, namedtuple, OrderedDict
import warnings
import jinja2

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import numpy as np
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

        if legend_keys is None:
            legend_keys = []
        self.legend_keys = ['scan_id'] + legend_keys
        if x is not None:
            self.x, *others = _get_obj_fields([x])
        else:
            self.x = None
        self.y, *others = _get_obj_fields([y])
        self.fig = fig
        self.ax = fig.gca()
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
        self.ax.relim(visible_only=True)
        self.ax.autoscale_view(tight=True)
        self.ax.figure.canvas.draw_idle()

    def stop(self, doc):
        if not self.x_data:
            print('LivePlot did not get any data that corresponds to the '
                  'x axis. {}'.format(self.x))
        if not self.y_data:
            print('LivePlot did not get any data that corresponds to the '
                    'y axis. {}'.format(self.y))
        if len(self.y_data) != len(self.x_data):
            print('LivePlot has a different number of elements for x ({}) and'
                  'y ({})'.format(len(self.x_data), len(self.y_data)))


def format_num(x, max_len=11, pre=5, post=5):
    if (abs(x) > 10**pre or abs(x) < 10**-post) and x != 0:
        x = '%.{}e'.format(post) % x
    else:
        x = '%{}.{}f'.format(pre, post) % x

    return x



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
        self.snaking = doc.get('snaking', (False, False))

        cb = self.fig.colorbar(im)
        cb.set_label(self.I)

    def event(self, doc):
        seq_num = doc['seq_num'] - 1
        pos = list(np.unravel_index(seq_num, self.raster_shape))
        if self.snaking[1] and (pos[0] % 2):
            pos[1] = self.raster_shape[1] - pos[1] - 1
        pos = tuple(pos)
        self._Idata[pos] = doc['data'][self.I]
        if self.clim is None:
            self.im.set_clim(np.nanmin(self._Idata), np.nanmax(self._Idata))

        self.im.set_array(self._Idata)


class LiveTable(CallbackBase):
    '''Live updating table

    Parameters
    ----------
    fields : list
         List of fields to add to the table.

    print_header_interval : int, optional
         Reprint the header every this many lines, defaults to 50

    min_width : int, optional
         The minimum width is spaces of the data columns.  Defaults to 12

    default_prec : int, optional
         Precision to use if it can not be found in descriptor, defaults to 3

    extra_pad : int, optional
         Number of extra spaces to put around the printed data, defaults to 1

    logbook : callable, optional
        Must take a sting as the first positional argument

           def logbook(input_str):
                pass

    '''
    _FMTLOOKUP = {'s': '{pad}{{{k}: >{width}.{prec}{dtype}}}{pad}',
                  'f': '{pad}{{{k}: >{width}.{prec}{dtype}}}{pad}',
                  'g': '{pad}{{{k}: >{width}.{prec}{dtype}}}{pad}',
                  'd': '{pad}{{{k}: >{width}{dtype}}}{pad}'}
    _FMT_MAP = {'number': 'f',
                'integer': 'd',
                'string': 's',
                }
    _fm_sty = namedtuple('fm_sty', ['width', 'prec', 'dtype'])
    water_mark = "{st[plan_type]} ['{st[uid]:.6s}'] (scan num: {st[scan_id]})"

    def __init__(self, fields, *, print_header_interval=50,
                 min_width=12, default_prec=3, extra_pad=1,
                 logbook=None):
        super().__init__()
        self._header_interval = print_header_interval
        # expand objects
        self._fields = _get_obj_fields(fields)
        self._start = None
        self._stop = None
        self._descriptors = set()
        self._pad_len = extra_pad
        self._extra_pad = ' ' * extra_pad
        self._min_width = min_width
        self._default_prec = default_prec
        self._format_info = OrderedDict([
            ('seq_num', self._fm_sty(10 + self._pad_len, '', 'd')),
            ('time', self._fm_sty(10 + 2 * extra_pad, 10, 's'))
        ])
        self._rows = []
        self.logbook = logbook
        self._sep_format = None

    def descriptor(self, doc):
        def patch_up_precision(p):
            try:
                return int(p)
            except (TypeError, ValueError):
                return self._default_prec

        dk = doc['data_keys']
        for k in self._fields:
            width = max(self._min_width,
                        len(k) + 2,
                        self._default_prec + 1 + 2 * self._pad_len)
            try:
                dk_entry = dk[k]
            except KeyError:
                # this descriptor does not know about this key
                continue

            if dk_entry['dtype'] not in self._FMT_MAP:
                continue

            prec = patch_up_precision(dk_entry.get('precision',
                                                   self._default_prec))
            fmt = self._fm_sty(width=width,
                               prec=prec,
                               dtype=self._FMT_MAP[dk_entry['dtype']])

            self._format_info[k] = fmt

        self._sep_format = ('+' +
                            '+'.join('-'*f.width
                                     for f in self._format_info.values()) +
                            '+')
        self._main_fmnt = '|'.join(
            '{{: >{w}}}{pad}'.format(w=f.width-self._pad_len,
                                     pad=' '*self._pad_len)
            for f in self._format_info.values())
        self._header = ('|' +
                        self._main_fmnt.format(*list(self._format_info)) +
                        '|'
                        )
        self._data_formats = OrderedDict(
            (k, self._FMTLOOKUP[f.dtype].format(k=k,
                                                width=f.width-2*self._pad_len,
                                                prec=f.prec, dtype=f.dtype,
                                                pad=self._extra_pad))
            for k, f in self._format_info.items())

        self._count = 0

        self._print(self._sep_format)
        self._print(self._header)
        self._print(self._sep_format)

    def event(self, doc):
        # shallow copy so we can mutate
        data = dict(doc['data'])
        self._count += 1
        if not self._count % self._header_interval:
            self._print(self._sep_format)
            self._print(self._header)
            self._print(self._sep_format)
        fmt_time = str(datetime.fromtimestamp(doc['time']).time())
        data['time'] = fmt_time
        data['seq_num'] = doc['seq_num']
        cols = [f.format(**{k: data[k]})
                if k in data else ' '*self._format_info[k].width
                for k, f in self._data_formats.items()]
        self._print('|' + '|'.join(cols) + '|')

    def stop(self, doc):
        if doc['run_start'] != self._start['uid']:
            return
        if self._sep_format is not None:
            self._print(self._sep_format)
        self._stop = doc

        wm = self.water_mark.format(st=self._start)
        print(wm)
        if self.logbook:
            self.logbook('\n'.join([wm] + self._rows))

    def start(self, doc):
        self._rows = []
        self._start = doc
        self._stop = None
        self._sep_format = None

    def _print(self, out_str):
        self._rows.append(out_str)
        print(out_str)

env = jinja2.Environment()

_SPEC_HEADER_TEMPLATE = env.from_string("""#F {{ filepath }}
#E {{ unix_time }}
#D {{ readable_time }}
#C {{ owner }}  User = {{ owner }}
#O0 {{ positioners | join(' ') }}""")

_SPEC_1D_COMMAND_TEMPLATE = env.from_string("{{ scan_type }} {{ scan_motor }} {{ start }} {{ stop }} {{ strides }} {{ time }}")

_PLAN_TO_SPEC_MAPPING = {'AbsScanPlan': 'ascan',
                         'DeltaScanPlan': 'dscan',
                         'Count': 'ct',
                         'Tweak': 'tw'}

_SPEC_START_TEMPLATE = env.from_string("""

#S {{ scan_id }} {{ command }}
#D {{ readable_time }}
#T {{ acq_time }} (Seconds)
#P0 {{ positioner_positions | join(' ')}}""")

# It is critical that the spacing on the #L line remain exactly like this!
_SPEC_DESCRIPTOR_TEMPLATE = env.from_string("""
#N {{ length }}
#L {{ motor_name }}    Epoch  Seconds  {{ data_keys | join('  ') }}\n""")

_SPEC_EVENT_TEMPLATE = env.from_string(
    """{{ motor_position }}  {{ unix_time }} {{ acq_time }} {{ values | join(' ') }}\n""")

class LiveSpecFile(CallbackBase):
    """Callback to export scalar values to a spec file for viewing

    Expect:
    1. a descriptor named 'baseline'
    2. an event for that descriptor
    3. a descriptor named 'main'
    4. events for that descriptor

    Other documents can be issues before, between, and after, but
    these must be issued and in this order.

    Notes
    -----
    `Reference <https://github.com/certified-spec/specPy/blob/master/doc/specformat.rst>`_
    for the spec file format.

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
        if plan_type not in _PLAN_TO_SPEC_MAPPING.keys():
            err_msg = ("Do not know how to represent %s in SPEC. If "
                       "you would like this feature, request it at "
                       "https://github.com/NSLS-II/bluesky/issues"
                       % plan_type)
            raise NotImplementedError(err_msg)

        # Some of these are used in other methods too -- stash them.
        self._unix_time = doc['time']
        self._acq_time = plan_args.get('time', -1)
        content = dict(scan_type=_PLAN_TO_SPEC_MAPPING[doc['plan_type']],
                       acq_time=self._acq_time)
        if plan_type == 'Count':
            # count has no motor. Have to fake one.
            self._motor = 'Count'
        else:
            content['start'] = plan_args['start']
            content['stop'] = plan_args['stop']
            content['strides'] = int(plan_args['num']) - 1,
            try:
                # We only support a single scanning motor right now.
                self._motor, = doc['motors']
            except ValueError:
                raise NotImplementedError(
                    "Your scan has %s scanning motors. They are %s. SpecCallback"
                    " cannot handle multiple scanning motors. Please request "
                    "this feature at https://github.com/NSLS-II/bluesky/issues" %
                    (len(self._motor), self._motor))
        content['scan_motor'] = self._motor
        command = _SPEC_1D_COMMAND_TEMPLATE.render(content)
        # Can't write the entry until we see the descriptor, so stash it until
        # we get the descriptor
        self._start_content = dict(
            command=command,
            scan_id=doc['scan_id'],
            readable_time=datetime.fromtimestamp(doc['time']),
            acq_time=self._acq_time,
            positioner_positions=self.positions)

    def descriptor(self, doc):
        """Write the header for the actual scan data"""
        if self._motor not in list(doc['data_keys'].keys()) + ['Count']:
            # see if we can just append _user_readback to the motor
            self._motor += '_user_readback'
            if self._motor not in doc['data_keys']:
                # give up and use the event sequence number as the motor.
                # We are still throwing all the motor information into the
                # spec file, but the user will have to manually choose the
                print("We are unable to guess the motor name. Please set the"
                      "user_readback value to be the same as the motor name "
                      "so that we are able to correctly guess your scanning "
                      "motor. Your 'motor' is actually the sequence number of"
                      "the event")
                self._motor = 'seq_num'

        # List all scalar fields, excluding the motor (x variable).
        self._read_fields = sorted([k for k, v in doc['data_keys'].items()
                                    if k != self._motor and not v['shape']])
        # Remove the motor key. It should only be in the list once!
        content = dict(motor_name=self._motor,
                       length=3 + len(self._read_fields),
                       data_keys=self._read_fields)
        with open(self.specpath, 'a') as f:
            f.write(_SPEC_START_TEMPLATE.render(self._start_content))
            f.write(_SPEC_DESCRIPTOR_TEMPLATE.render(content))
            f.write('\n')

    def event(self, doc):
        """Write each event out"""""
        data = doc['data']
        values = [str(data[k]) for k in self._read_fields]
        if self._motor == "Count":
            motor_position = -1
        elif self._motor == "seq_num":
            motor_position = doc['seq_num']
        else:
            motor_position = data[self._motor]
        content = dict(acq_time=self._acq_time,
                       unix_time=doc['time'],
                       motor_position=motor_position,
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
