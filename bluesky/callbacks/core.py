"""
Useful callbacks for the Run Engine
"""
from itertools import count
import warnings
from collections import deque, namedtuple, OrderedDict, ChainMap
import time as ttime

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import numpy as np
import logging
from ..utils import ensure_uid
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
    xlim : tuple, optional
        passed to Axes.set_xlim
    ylim : tuple, optional
        passed to Axes.set_ylim
    ax : Axes, optional
        matplotib Axes; if none specified, new figure and axes are made.
    fig : Figure
        deprecated: use ax instead
    All additional keyword arguments are passed through to ``Axes.plot``.

    Examples
    --------
    >>> my_plotter = LivePlot('det', 'motor', legend_keys=['sample'])
    >>> RE(my_scan, my_plotter)
    """
    def __init__(self, y, x=None, *, legend_keys=None, xlim=None, ylim=None,
                 ax=None, fig=None, **kwargs):
        super().__init__()
        if fig is not None:
            if ax is not None:
                raise ValueError("Values were given for both `fig` and `ax`. "
                                 "Only one can be used; prefer ax.")
            warnings.warn("The `fig` keyword arugment of LivePlot is "
                          "deprecated and will be removed in the future. "
                          "Instead, use the new keyword argument `ax` to "
                          "provide specific Axes to plot on.")
            ax = fig.gca()
        if ax is None:
            fig, ax = plt.subplots()
        self.ax = ax

        if legend_keys is None:
            legend_keys = []
        self.legend_keys = ['scan_id'] + legend_keys
        if x is not None:
            self.x, *others = _get_obj_fields([x])
        else:
            self.x = None
        self.y, *others = _get_obj_fields([y])
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
            [str(doc.get(name, name)) for name in self.legend_keys])
        kwargs = ChainMap(self.kwargs, {'label': label})
        self.current_line, = self.ax.plot([], [], **kwargs)
        self.lines.append(self.current_line)
        self.legend = self.ax.legend(
            loc=0, title=self.legend_title).draggable()
        super().start(doc)

    def event(self, doc):
        "Unpack data from the event and call self.update()."
        try:
            if self.x is not None:
                # this try/except block is needed because multiple event
                # streams will be emitted by the RunEngine and not all event
                # streams will have the keys we want
                new_x = doc['data'][self.x]
            else:
                new_x = doc['seq_num']
            new_y = doc['data'][self.y]
        except KeyError:
            # wrong event stream, skip it
            return
        self.update_caches(new_x, new_y)
        self.update_plot()
        super().event(doc)

    def update_caches(self, x, y):
        self.y_data.append(y)
        self.x_data.append(x)

    def update_plot(self):
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
        super().stop(doc)


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
        super().start(doc)

    def descriptor(self, doc):
        self._descriptors.append(doc)
        super().descriptor(doc)

    def event(self, doc):
        self._events.append(doc)
        super().event(doc)

    def stop(self, doc):
        self._stop_doc = doc
        self.compute()
        super().stop(doc)

    def reset(self):
        self._start_doc = None
        self._stop_doc = None
        self._events.clear()
        self._descriptors.clear()

    def compute(self):
        raise NotImplementedError("This method must be defined by a subclass.")


class LiveMesh(CallbackBase):
    """Plot scattered 2D data in a "heat map".

    Alternatively, if the data is placed on a regular grid, you can use
    :func:`bluesky.callbacks.LiveRaster`.

    This simply wraps around a `PathCollection` as generated by scatter.

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

    ax : Axes, optional
        matplotib Axes; if none specified, new figure and axes are made.

    See Also
    --------
    :class:`bluesky.callbacks.LiveRaster`.
    """
    def __init__(self, x, y, I, *, xlim=None, ylim=None,
                 clim=None, cmap='viridis', ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.cla()
        self.x = x
        self.y = y
        self.I = I
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_aspect('equal')
        self._sc = []
        self.ax = ax
        ax.margins(.1)
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
        self._xdata.clear()
        self._ydata.clear()
        self._Idata.clear()
        sc = self.ax.scatter(self._xdata, self._ydata, c=self._Idata,
                             norm=self._norm, cmap=self.cmap, edgecolor='face',
                             s=50)
        self._sc.append(sc)
        self.sc = sc
        super().start(doc)

    def event(self, doc):
        x = doc['data'][self.x]
        y = doc['data'][self.y]
        I = doc['data'][self.I]
        self.update(x, y, I)
        super().event(doc)

    def update(self, x, y, I):
        self._xdata.append(x)
        self._ydata.append(y)
        self._Idata.append(I)
        offsets = np.vstack([self._xdata, self._ydata]).T
        self.sc.set_offsets(offsets)
        self.sc.set_array(np.asarray(self._Idata))


class LiveRaster(CallbackBase):
    """Plot gridded 2D data in a "heat map".

    This assumes that readings are placed on a regular grid and can be placed
    into an image by sequence number. The seq_num is used to determine which
    pixel to fill in.

    For non-gridded data with arbitrary placement, use
    :func:`bluesky.callbacks.LiveMesh`.

    This simply wraps around a `AxesImage`.

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

    xlabel, ylabel : str, optional
       Labels for the x and y axis

    extent : scalars (left, right, bottom, top), optional
       Passed through to `Axes.imshow`

    aspect : str or float, optional
       Passed through to `Axes.imshow`

    ax : Axes, optional
        matplotib Axes; if none specified, new figure and axes are made.

    See Also
    --------
    :class:`bluesky.callbacks.LiveMesh`.
    """
    def __init__(self, raster_shape, I, *,
                 clim=None, cmap='viridis',
                 xlabel='x', ylabel='y', extent=None, aspect='equal',
                 ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.cla()
        self.I = I
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        self.ax = ax
        self._Idata = np.ones(raster_shape) * np.nan
        self._norm = mcolors.Normalize()
        if clim is not None:
            self._norm.vmin, self._norm.vmax = clim
        self.clim = clim
        self.cmap = cmap
        self.raster_shape = raster_shape
        self.im = None
        self.extent = extent
        self.aspect = aspect

    def start(self, doc):
        if self.im is not None:
            raise RuntimeError("Can not re-use LiveRaster")
        self._Idata = np.ones(self.raster_shape) * np.nan
        im = self.ax.imshow(self._Idata, norm=self._norm,
                            cmap=self.cmap, interpolation='none',
                            extent=self.extent, aspect=self.aspect)

        self.im = im
        self.ax.set_title('scan {uid} [{sid}]'.format(sid=doc['scan_id'],
                                                      uid=doc['uid'][:6]))
        self.snaking = doc.get('snaking', (False, False))

        cb = self.ax.figure.colorbar(im)
        cb.set_label(self.I)
        super().start(doc)

    def event(self, doc):
        if self.I not in doc['data']:
            return

        seq_num = doc['seq_num'] - 1
        pos = list(np.unravel_index(seq_num, self.raster_shape))
        if self.snaking[1] and (pos[0] % 2):
            pos[1] = self.raster_shape[1] - pos[1] - 1
        pos = tuple(pos)
        I = doc['data'][self.I]
        self.update(pos, I)
        super().event(doc)

    def update(self, pos, I):
        self._Idata[pos] = I
        if self.clim is None:
            self.im.set_clim(np.nanmin(self._Idata), np.nanmax(self._Idata))

        self.im.set_array(self._Idata)


class LiveTable(CallbackBase):
    '''Live updating table

    Parameters
    ----------
    fields : list
         List of fields to add to the table.

    stream_name : str, optional
         The event stream to watch for

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
    water_mark = ("{st[plan_type]} {st[plan_name]} ['{st[uid]:.6s}'] "
                  "(scan num: {st[scan_id]})")
    ev_time_key = 'SUPERLONG_EV_TIMEKEY_THAT_I_REALLY_HOPE_NEVER_CLASHES'

    def __init__(self, fields, *, stream_name='primary',
                 print_header_interval=50,
                 min_width=12, default_prec=3, extra_pad=1,
                 logbook=None):
        super().__init__()
        self._header_interval = print_header_interval
        # expand objects
        self._fields = _get_obj_fields(fields)
        self._stream = stream_name
        self._start = None
        self._stop = None
        self._descriptors = set()
        self._pad_len = extra_pad
        self._extra_pad = ' ' * extra_pad
        self._min_width = min_width
        self._default_prec = default_prec
        self._format_info = OrderedDict([
            ('seq_num', self._fm_sty(10 + self._pad_len, '', 'd')),
            (self.ev_time_key, self._fm_sty(10 + 2 * extra_pad, 10, 's'))
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

        if doc['name'] != self._stream:
            return

        self._descriptors.add(doc['uid'])

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
                warnings.warn("The key {} will be skipped because LiveTable "
                              "does not know how to display the dtype {}"
                              "".format(k, dk_entry['dtype']))
                continue

            prec = patch_up_precision(dk_entry.get('precision',
                                                   self._default_prec))
            fmt = self._fm_sty(width=width,
                               prec=prec,
                               dtype=self._FMT_MAP[dk_entry['dtype']])

            self._format_info[k] = fmt

        self._sep_format = ('+' +
                            '+'.join('-' * f.width
                                     for f in self._format_info.values()) +
                            '+')
        self._main_fmnt = '|'.join(
            '{{: >{w}}}{pad}'.format(w=f.width - self._pad_len,
                                     pad=' ' * self._pad_len)
            for f in self._format_info.values())
        headings = [k if k != self.ev_time_key else 'time'
                    for k in self._format_info]
        self._header = ('|' +
                        self._main_fmnt.format(*headings) +
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
        super().descriptor(doc)

    def event(self, doc):
        # shallow copy so we can mutate
        if ensure_uid(doc['descriptor']) not in self._descriptors:
            return
        data = dict(doc['data'])
        self._count += 1
        if not self._count % self._header_interval:
            self._print(self._sep_format)
            self._print(self._header)
            self._print(self._sep_format)
        fmt_time = str(datetime.fromtimestamp(doc['time']).time())
        data[self.ev_time_key] = fmt_time
        data['seq_num'] = doc['seq_num']
        cols = [f.format(**{k: data[k]})
                if k in data else ' ' * self._format_info[k].width
                for k, f in self._data_formats.items()]
        self._print('|' + '|'.join(cols) + '|')
        super().event(doc)

    def stop(self, doc):
        if ensure_uid(doc['run_start']) != self._start['uid']:
            return

        # This sleep is just cosmetic. It improves the odds that the bottom
        # border is not printed until all the rows from events are printed,
        # avoiding this ugly scenario:
        #
        # |         4 | 22:08:56.7 |      0.000 |
        # +-----------+------------+------------+
        # generator scan ['6d3f71'] (scan num: 1)
        # Out[2]: |         5 | 22:08:56.8 |      0.000 |
        ttime.sleep(0.1)

        if self._sep_format is not None:
            self._print(self._sep_format)
        self._stop = doc

        wm = self.water_mark.format(st=self._start)
        print(wm)
        if self.logbook:
            self.logbook('\n'.join([wm] + self._rows))
        super().stop(doc)

    def start(self, doc):
        self._rows = []
        self._start = doc
        self._stop = None
        self._sep_format = None
        super().start(doc)

    def _print(self, out_str):
        self._rows.append(out_str)
        print(out_str)


class LiveFit(CallbackBase):
    """
    Fit a model to data using nonlinear least-squares minimization.

    Parameters
    ----------
    model : lmfit.Model
    y : string
        name of the field in the Event document that is the dependent variable
    independent_vars : dict
        map the independent variable name(s) in the model to the field(s)
        in the Event document; e.g., ``{'x': 'motor'}``
    init_guess : dict, optional
        initial guesses for other values, if expected by model;
        e.g., ``{'sigma': 1}``
    update_every : int or None, optional
        How often to recompute the fit. If `None`, do not compute until the
        end. Default is 1 (recompute after each new point).

    Attributes
    ----------
    result : lmfit.ModelResult
    """
    def __init__(self, model, y, independent_vars, init_guess=None, *,
                 update_every=1):
        self.ydata = []
        self.independent_vars_data = {}
        self.__stale = False
        self.result = None
        self._model = model
        self.y = y
        self.independent_vars = independent_vars
        if init_guess is None:
            init_guess = {}
        self.init_guess = init_guess
        self.update_every = update_every

    @property
    def model(self):
        # Make this a property so it can't be updated.
        return self._model

    @property
    def independent_vars(self):
        return self._independent_vars

    @independent_vars.setter
    def independent_vars(self, val):
        if set(val) != set(self.model.independent_vars):
            raise ValueError("keys {} must match the independent variables in "
                             "the model "
                             "{}".format(set(val),
                                         set(self.model.independent_vars)))
        self._independent_vars = val
        self.independent_vars_data.clear()
        self.independent_vars_data.update({k: [] for k in val})
        self._reset()

    def _reset(self):
        self.result = None
        self.__stale = False
        self.ydata.clear()
        for v in self.independent_vars_data.values():
            v.clear()

    def start(self, doc):
        self._reset()
        super().start(doc)

    def event(self, doc):
        if self.y not in doc['data']:
            return
        y = doc['data'][self.y]
        idv = {k: doc['data'][v] for k, v in self.independent_vars.items()}

        # Always stash the data for the next time the fit is updated.
        self.update_caches(y, idv)
        self.__stale = True

        # Maybe update the fit or maybe wait.
        if self.update_every is not None:
            i = doc['seq_num']
            N = len(self.model.param_names)
            if i < N:
                # not enough points to fit yet
                pass
            elif (i == N) or ((i - 1) % self.update_every == 0):
                self.update_fit()
        super().event(doc)

    def stop(self, doc):
        # Update the fit if it was not updated by the last event.
        if self.__stale:
            self.update_fit()
        super().stop(doc)

    def update_caches(self, y, independent_vars):
        self.ydata.append(y)
        for k, v in self.independent_vars_data.items():
            v.append(independent_vars[k])

    def update_fit(self):
        N = len(self.model.param_names)
        if len(self.ydata) < N:
            raise RuntimeError("cannot update fit until there are least {} "
                               "data points".format(N))
        kwargs = {}
        kwargs.update(self.independent_vars_data)
        kwargs.update(self.init_guess)
        self.result = self.model.fit(self.ydata, **kwargs)
        self.__stale = False


class LiveFitPlot(LivePlot):
    """
    Add a plot to an instance of LiveFit.

    Note: If your figure blocks the main thread when you are trying to
    scan with this callback, call `plt.ion()` in your IPython session.

    Parameters
    ----------
    livefit : LiveFit
        an instance of ``LiveFit``
    num_points : int, optional
        number of points to sample when evaluating the model; default 100
    legend_keys : list, optional
        The list of keys to extract from the RunStart document and format
        in the legend of the plot. The legend will always show the
        scan_id followed by a colon ("1: ").  Each
    xlim : tuple, optional
        passed to Axes.set_xlim
    ylim : tuple, optional
        passed to Axes.set_ylim
    ax : Axes, optional
        matplotib Axes; if none specified, new figure and axes are made.
    All additional keyword arguments are passed through to ``Axes.plot``.
    """
    def __init__(self, livefit, *, num_points=100, legend_keys=None, xlim=None,
                 ylim=None, ax=None, **kwargs):
        if len(livefit.independent_vars) != 1:
            raise NotImplementedError("LiveFitPlot supports models with one "
                                      "independent variable only.")
        self.__x_key, = livefit.independent_vars.keys()  # this never changes
        x, = livefit.independent_vars.values()  # this may change
        super().__init__(livefit.y, x, legend_keys=legend_keys,
                         xlim=xlim, ylim=xlim, ax=ax, **kwargs)
        self.num_points = num_points
        self._livefit = livefit
        self._xlim = xlim

    @property
    def livefit(self):
        return self._livefit

    def start(self, doc):
        self.livefit.start(doc)
        self.x, = self.livefit.independent_vars.keys()  # in case it changed
        super().start(doc)
        # Put fit above other lines (default 2) but below text (default 3).
        self.current_line.set_zorder(2.5)

    def event(self, doc):
        self.livefit.event(doc)
        if self.livefit.result is not None:
            # Evaluate the model function at equally-spaced points.
            # To determine the domain of x, use xlim if availabe. Otherwise,
            # use the range of x points measured up to this point.
            if self._xlim is None:
                x_data = self.livefit.independent_vars_data[self.__x_key]
                xmin, xmax = np.min(x_data), np.max(x_data)
            else:
                xmin, xmax = self._xlim
            x_points = np.linspace(xmin, xmax, self.num_points)
            kwargs = {self.__x_key: x_points}
            kwargs.update(self.livefit.result.values)
            self.y_data = self.livefit.result.model.eval(**kwargs)
            self.x_data = x_points
            self.update_plot()
        # Intentionally override LivePlot.event. Do not call super().

    def descriptor(self, doc):
        self.livefit.descriptor(doc)
        super().descriptor(doc)

    def stop(self, doc):
        self.livefit.stop(doc)
        # Intentionally override LivePlot.stop. Do not call super().
