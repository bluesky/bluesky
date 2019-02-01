from collections import ChainMap
from cycler import cycler
import numpy as np
import re
import warnings
import weakref

from .core import CallbackBase, get_obj_fields
from .utils import hinted_fields
from .fitting import PeakStats
from event_model import DocumentRouter


class RunFigureManager:
    def __init__(self, doc):
        self.noplot_streams = []  # Do we want to keep this?
        self.overplot = True  # Do we want to keep this?
        self.peaks = dict()
        self.omit_single_point_plot = True

        self._start_doc = doc
        self._descriptors = {}
        self._cleanup_motor_heuristic = False
        self.plan_hints = doc.get('hints', {})

        # Prepare a guess about the dimensions (independent variables) in case
        # we need it.
        motors = self._start_doc.get('motors')
        if motors is not None:
            GUESS = [([motor], 'primary') for motor in motors]
        else:
            GUESS = [(['time'], 'primary')]

        # Ues the guess if there is not hint about dimensions.
        dimensions = self.plan_hints.get('dimensions')
        if dimensions is None:
            self._cleanup_motor_heuristic = True
            dimensions = GUESS

        # We can only cope with all the dimensions belonging to the same
        # stream unless we resample. We are not doing to handle that yet.
        if len(set(d[1] for d in dimensions)) != 1:
            self._cleanup_motor_heuristic = True
            dimensions = GUESS  # Fall back on our GUESS.
            warn("We are ignoring the dimensions hinted because we cannot "
                "combine streams.")

        # for each dimension, choose one field only
        # the plan can supply a list of fields. It's assumed the first
        # of the list is always the one plotted against
        self.dim_fields = [fields[0]
                        for fields, stream_name in dimensions]

        # make distinction between flattened fields and plotted fields
        # motivation for this is that when plotting, we find dependent variable
        # by finding elements that are not independent variables
        self.all_dim_fields = [field
                            for fields, stream_name in dimensions
                            for field in fields]

        _, self.dim_stream = dimensions[0]

    def __call__(self, name, doc):
        # Decide which plots we want for this stream.

        import matplotlib.pyplot as plt

        self._descriptors[doc['uid']] = doc
        stream_name = doc.get('name', 'primary')  # fall back for old docs
        columns = hinted_fields(doc)

        # ## This deals with old documents. ## #

        if stream_name == 'primary' and self._cleanup_motor_heuristic:
            # We stashed object names in self.dim_fields, which we now need to
            # look up the actual fields for.
            self._cleanup_motor_heuristic = False
            fixed_dim_fields = []
            for obj_name in self.dim_fields:
                # Special case: 'time' can be a dim_field, but it's not an
                # object name. Just add it directly to the list of fields.
                if obj_name == 'time':
                    fixed_dim_fields.append('time')
                    continue
                try:
                    fields = doc.get('hints', {}).get(obj_name, {})['fields']
                except KeyError:
                    fields = doc['object_keys'][obj_name]
                fixed_dim_fields.extend(fields)
            self.dim_fields = fixed_dim_fields

        # ## DECIDE WHICH KIND OF PLOT CAN BE USED ## #

        if stream_name in self.noplot_streams:
            return []
        if not columns:
            return []
        if ((self._start_doc.get('num_points') == 1) and
                (stream_name == self.dim_stream) and
                self.omit_single_point_plot):
            return []

        # This is a heuristic approach until we think of how to hint this in a
        # generalizable way.
        if stream_name == self.dim_stream:
            dim_fields = self.dim_fields
        else:
            dim_fields = ['time']  # 'time' once LivePlot can do that

        # Create a figure or reuse an existing one.

        fig_name = '{} vs {}'.format(' '.join(sorted(columns)),
                                     ' '.join(sorted(dim_fields)))
        if self.overplot and len(dim_fields) == 1:
            # If any open figure matches 'figname {number}', use it. If there
            # are multiple, the most recently touched one will be used.
            pat1 = re.compile('^' + fig_name + '$')
            pat2 = re.compile('^' + fig_name + r' \d+$')
            for label in plt.get_figlabels():
                if pat1.match(label) or pat2.match(label):
                    fig_name = label
                    break
        else:
            if plt.fignum_exists(fig_name):
                # Generate a unique name by appending a number.
                for number in itertools.count(2):
                    new_name = '{} {}'.format(fig_name, number)
                    if not plt.fignum_exists(new_name):
                        fig_name = new_name
                        break
        ndims = len(dim_fields)
        if not 0 < ndims < 3:
            # we need 1 or 2 dims to do anything, do not make empty figures
            return

        fig = plt.figure(fig_name)
        if not fig.axes:
            # This is apparently a fresh figure. Make axes.
            # The complexity here is due to making a shared x axis. This can be
            # simplified when Figure supports the `subplots` method in a future
            # release of matplotlib.
            fig.set_size_inches(6.4, min(950, len(columns) * 400) / fig.dpi)
            for i in range(len(columns)):
                if i == 0:
                    ax = fig.add_subplot(len(columns), 1, 1 + i)
                    if ndims == 1:
                        share_kwargs = {'sharex': ax}
                    elif ndims == 2:
                        share_kwargs = {'sharex': ax, 'sharey': ax}
                    else:
                        raise NotImplementedError("we now support 3D?!")
                else:
                    ax = fig.add_subplot(len(columns), 1, 1 + i,
                                         **share_kwargs)
        axes = fig.axes
        callbacks = []

        # ## LIVE PLOT AND PEAK ANALYSIS ## #

        if ndims == 1:
            x_key, = dim_fields
            for y_key, ax in zip(columns, axes):
                dtype = doc['data_keys'][y_key]['dtype']
                if dtype not in ('number', 'integer'):
                    warn("Omitting {} from plot because dtype is {}"
                         "".format(y_key, dtype))
                    continue

                def func(event_page):
                    """
                    Extract x points and y points to plot out of an EventPage.

                    This will be passed to LineWithPeaks.
                    """
                    y_data = event_page['data'][y_key]
                    if x_key == 'time':
                        t0 = self._start_doc['time']
                        x_data = np.asarray(event_page['time']) - t0
                    elif x_key == 'seq_num':
                        x_data = event_page['seq_num']
                    else:
                        x_data = event_page['data'][x_key]
                    return x_data, y_data

                # Create instances of LineWithPeaks and PeakStats.
                live_plot = LineWithPeaks(func, ax=ax, peak_results=self.peaks)
                live_plot('start', self._start_doc)
                live_plot('descriptor', doc)
                peak_stats = PeakStats(x=x_key, y=y_key)
                peak_stats('start', self._start_doc)
                peak_stats('descriptor', doc)
                callbacks.extend([live_plot, peak_stats])

            for ax in axes[:-1]:
                ax.set_xlabel('')
        elif ndims == 2:
            # Decide whether to use LiveGrid or LiveScatter. LiveScatter is the
            # safer one to use, so it is the fallback..
            gridding = self._start_doc.get('hints', {}).get('gridding')
            if gridding == 'rectilinear':
                slow, fast = dim_fields
                try:
                    extents = self._start_doc['extents']
                    shape = self._start_doc['shape']
                except KeyError:
                    warn("Need both 'shape' and 'extents' in plan metadata to "
                         "create LiveGrid.")
                else:
                    data_range = np.array([float(np.diff(e)) for e in extents])
                    y_step, x_step = data_range / [max(1, s - 1) for s in shape]
                    adjusted_extent = [extents[1][0] - x_step / 2,
                                       extents[1][1] + x_step / 2,
                                       extents[0][0] - y_step / 2,
                                       extents[0][1] + y_step / 2]
                    for I_key, ax in zip(columns, axes):
                        # MAGIC NUMBERS based on what tacaswell thinks looks OK
                        data_aspect_ratio = np.abs(data_range[1]/data_range[0])
                        MAR = 2
                        if (1/MAR < data_aspect_ratio < MAR):
                            aspect = 'equal'
                            ax.set_aspect(aspect, adjustable='box-forced')
                        else:
                            aspect = 'auto'
                            ax.set_aspect(aspect, adjustable='datalim')

                        live_grid = LiveGrid(shape, I_key,
                                             xlabel=fast, ylabel=slow,
                                             extent=adjusted_extent,
                                             aspect=aspect,
                                             ax=ax)

                        live_grid('start', self._start_doc)
                        live_grid('descriptor', doc)
                        callbacks.append(live_grid)
            else:
                x_key, y_key = dim_fields
                for I_key, ax in zip(columns, axes):
                    try:
                        extents = self._start_doc['extents']
                    except KeyError:
                        xlim = ylim = None
                    else:
                        xlim, ylim = extents
                    live_scatter = LiveScatter(x_key, y_key, I_key,
                                               xlim=xlim, ylim=ylim,
                                               # Let clim autoscale.
                                               ax=ax)
                    live_scatter('start', self._start_doc)
                    live_scatter('descriptor', doc)
                    callbacks.append(live_scatter)
        else:
            raise NotImplementedError("we do not support 3D+ in BEC yet "
                                      "(and it should have bailed above)")
        try:
            fig.tight_layout()
        except ValueError:
            pass
        return callbacks


class FigureManager:
    def __init__(self):
        self.noplot_streams = []  # Do we want to keep this?
        ...

    def __call__(self, name, doc):
        run_figure_manager = RunFigureManager(doc)
        return [], [run_figure_manager]


class Line(DocumentRouter):
    """
    Draw a matplotlib Line Arist update it for each Event.

    Parameters
    ----------
    func : callable
        This must accept an EventPage and return two lists of floats
        (x points and y points). The two lists must contain an equal number of
        items, but that number is arbitrary. That is, a given document may add
        one new point to the plot, no new points, or multiple new points.
    ax : matplotlib Axes, optional
        If None, a new Figure and Axes are created.
    **kwargs
        Passed through to :meth:`Axes.plot` to style Line object.
    """
    def __init__(self, func, *, legend_keys=('scan_id',), ax=None, **kwargs):
        self.func = func
        if ax is None:
            _, ax = plt.subplots()
        self.ax = ax
        self.line, = ax.plot([], [], **kwargs)
        self.x_data = []
        self.y_data = []
        self.legend_keys = legend_keys
        self.label = kwargs.get('label')

    def start(self, doc):
        if self.label is None:
            label = ' :: '.join([f'{key!s} {doc.get(key)!r}'
                               for key in self.legend_keys])
            self.line.set_label(label)

    def event_page(self, doc):
        x, y = self.func(doc)
        self._update(x, y)

    def _update(self, x, y):
        """
        Takes in new x and y points and redraws plot if they are not empty.
        """
        if not len(x) == len(y):
            raise ValueError("User function is expected to provide the same "
                             "number of x and y points. Got {0} x points and "
                             "{1} y points.".format(len(x), len(y)))
        if not x:
            # No new data. Short-circuit.
            return
        self.x_data.extend(x)
        self.y_data.extend(y)
        self.line.set_data(self.x_data, self.y_data)
        self.ax.relim(visible_only=True)
        self.ax.autoscale_view(tight=True)
        self.ax.figure.canvas.draw_idle()


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
        the name of a data field in an Event, or 'seq_num' or 'time'
        If None, use the Event's sequence number.
        Special case: If the Event's data includes a key named 'seq_num' or
        'time', that takes precedence over the standard 'seq_num' and 'time'
        recorded in every Event.
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
    fig : Figure, optional
        deprecated: use ax instead
    epoch : {'run', 'unix'}, optional
        If 'run' t=0 is the time recorded in the RunStart document. If 'unix',
        t=0 is 1 Jan 1970 ("the UNIX epoch"). Default is 'run'.
    All additional keyword arguments are passed through to ``Axes.plot``.

    Examples
    --------
    >>> my_plotter = LivePlot('det', 'motor', legend_keys=['sample'])
    >>> RE(my_scan, my_plotter)
    """
    def __init__(self, y, x=None, *, legend_keys=None, xlim=None, ylim=None,
                 ax=None, fig=None, epoch='run', **kwargs):
        import matplotlib.pyplot as plt
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
            self.x, *others = get_obj_fields([x])
        else:
            self.x = 'seq_num'
        self.y, *others = get_obj_fields([y])
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
        self._epoch_offset = None  # used if x == 'time'
        self._epoch = epoch

    def start(self, doc):
        # The doc is not used; we just use the signal that a new run began.
        self._epoch_offset = doc['time']  # used if self.x == 'time'
        self.x_data, self.y_data = [], []
        label = " :: ".join(
            [str(doc.get(name, name)) for name in self.legend_keys])
        kwargs = ChainMap(self.kwargs, {'label': label})
        self.current_line, = self.ax.plot([], [], **kwargs)
        self.lines.append(self.current_line)
        self.legend = self.ax.legend(
            loc=0, title=self.legend_title).set_draggable(True)
        super().start(doc)

    def event(self, doc):
        "Unpack data from the event and call self.update()."
        # This outer try/except block is needed because multiple event
        # streams will be emitted by the RunEngine and not all event
        # streams will have the keys we want.
        try:
            # This inner try/except block handles seq_num and time, which could
            # be keys in the data or accessing the standard entries in every
            # event.
            try:
                new_x = doc['data'][self.x]
            except KeyError:
                if self.x in ('time', 'seq_num'):
                    new_x = doc[self.x]
                else:
                    raise
            new_y = doc['data'][self.y]
        except KeyError:
            # wrong event stream, skip it
            return

        # Special-case 'time' to plot against against experiment epoch, not
        # UNIX epoch.
        if self.x == 'time' and self._epoch == 'run':
            new_x -= self._epoch_offset

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


class LiveScatter(CallbackBase):
    """Plot scattered 2D data in a "heat map".

    Alternatively, if the data is placed on a regular grid, you can use
    :func:`bluesky.callbacks.mpl_plotting.LiveGrid`.

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

    All additional keyword arguments are passed through to ``Axes.scatter``.

    See Also
    --------
    :class:`bluesky.callbacks.mpl_plotting.LiveGrid`.
    """
    def __init__(self, x, y, I, *, xlim=None, ylim=None,
                 clim=None, cmap='viridis', ax=None, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        if ax is None:
            fig, ax = plt.subplots()
            fig.show()
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
        self._minx, self._maxx, self._miny, self._maxy = (None,)*4

        self.xlim = xlim
        self.ylim = ylim
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if clim is not None:
            self._norm.vmin, self._norm.vmax = clim
        self.clim = clim
        self.cmap = cmap
        self.kwargs = kwargs
        self.kwargs.setdefault('edgecolor', 'face')
        self.kwargs.setdefault('s', 50)

    def start(self, doc):
        self._xdata.clear()
        self._ydata.clear()
        self._Idata.clear()
        sc = self.ax.scatter(self._xdata, self._ydata, c=self._Idata,
                             norm=self._norm, cmap=self.cmap, **self.kwargs)
        self._sc.append(sc)
        self.sc = sc
        cb = self.ax.figure.colorbar(sc, ax=self.ax)
        cb.set_label(self.I)
        super().start(doc)

    def event(self, doc):
        x = doc['data'][self.x]
        y = doc['data'][self.y]
        I = doc['data'][self.I]
        self.update(x, y, I)
        super().event(doc)

    def update(self, x, y, I):
        # if one is None all are
        if self._minx is None:
            self._minx = x
            self._maxx = x
            self._miny = y
            self._maxy = y

        self._xdata.append(x)
        self._ydata.append(y)
        self._Idata.append(I)
        offsets = np.vstack([self._xdata, self._ydata]).T
        self.sc.set_offsets(offsets)
        self.sc.set_array(np.asarray(self._Idata))

        if self.xlim is None:
            minx, maxx = np.minimum(x, self._minx), np.maximum(x, self._maxx)
            self.ax.set_xlim(minx, maxx)

        if self.ylim is None:
            miny, maxy = np.minimum(y, self._miny), np.maximum(y, self._maxy)
            self.ax.set_ylim(miny, maxy)

        if self.clim is None:
            clim = np.nanmin(self._Idata), np.nanmax(self._Idata)
            self.sc.set_clim(*clim)


class LiveMesh(LiveScatter):
    __doc__ = LiveScatter.__doc__

    def __init__(self, *args, **kwargs):
        warnings.warn("LiveMesh has been renamed to LiveScatter. The name "
                      "LiveMesh will eventually be removed. Use LiveScatter.")
        super().__init__(*args, **kwargs)


class Grid(CallbackBase):
    '''Draw a matplotlib AxesImage artist and update it for each event.
    The purposes of this callback is to create (on initialization) of a
    matplotlib grid image and then update it with new data for every `event`.
    NOTE: Some important parameters are fed in through **kwargs like `extent`
    which defines the axes min and max and `origin` which defines if the grid
    co-ordinates start in the bottom left or top left of the plot. For more
    info see https://matplotlib.org/tutorials/intermediate/imshow_extent.html
    or https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.imshow.html#matplotlib.axes.Axes.imshow

    Parameters
    ----------
    func : callable
        This must accept a BulkEvent and return three lists of floats (x
        grid co-ordinates, y grid co-ordinates and grid position intensity
        values). The three lists must contain an equal number of items, but
        that number is arbitrary. That is, a given document may add one new
        point, no new points or multiple new points to the plot.
    shape : tuple
        The (row, col) shape of the grid.
    ax : matplotlib Axes, optional.
        if ``None``, a new Figure and Axes are created.
    **kwargs
        Passed through to :meth:`Axes.imshow` to style the AxesImage object.
    '''

    def __init__(self, func, shape, *, ax=None, **kwargs):
        self.func = func
        self.shape = shape
        if ax is None:
            _, ax = plt.subplots()
        self.ax = ax
        self.grid_data = np.full(self.shape, np.nan)
        self.image, = ax.imshow(self.grid_data, **kwargs)

    def bulk_events(self, doc):
        '''
        Takes in a bulk_events document and updates grid_data with the values
        returned from self.func(doc)

        Parameters
        ----------
        doc : dict
            The bulk event dictionary that contains the 'data' and 'timestamps'
            associated with the bulk event.

        Returns
        -------
        x_coords, y_coords, I_vals : Lists
            These are lists of x co-ordinate, y co-ordinate and intensity
            values arising from the bulk event.
        '''
        x_coords, y_coords, I_vals = self.func(doc)
        self._update(x_coords, y_coords, I_vals)

    def event(self, doc):
        '''
        Takes in a event documents and updates grid_data with the values
        returned from self.single_func(doc) or, if it is `None`, self.func(doc)

        Parameters
        ----------
        doc : dict
            The bulk event dictionary that contains the 'data' and 'timestamps'
            associated with the event.

        Returns
        -------
        x_coords, y_coords, I_vals : Lists
            These are lists of x co-ordinate, y co-ordinate and intensity
            values arising from the event.
        '''

        if self.single_func is not None:
            x_coords, y_coords, I_vals = self.single_func(doc)
        else:
            bulk_doc = event2bulk_event(doc)
            x_coords, y_coords, Ivals = self.func(bulk_doc)

        self._update(x_coords, y_coords, I_vals)

    def _update(self, x_coords, y_coords, I_vals):
        '''
        Updates self.grid_data with the values from the lists x_coords,
        y_coords, I_vals.

        Parameters
        ----------
        x_coords, y_coords, I_vals : Lists
            These are lists of x co-ordinate, y co-ordinate and intensity
            values arising from the event. The length of all three lists must
            be the same.
        '''

        if not len(x_coords) == len(y_coords) == len(I_vals):
            raise ValueError("User function is expected to provide the same "
                             "number of x, y and I points. Got {0} x points, "
                             "{1} y points and {2} I values."
                             "".format(len(x_coords), len(y_coords),
                                       len(I_vals)))

        if not x_coords:
            # No new data, Short-circuit.
            return

        # Update grid_data and the plot.
        self.grid_data[x_coords, y_coords] = I_vals
        self.image.set_array(self.grid_data)


class LiveGrid(CallbackBase):
    """Plot gridded 2D data in a "heat map".

    This assumes that readings are placed on a regular grid and can be placed
    into an image by sequence number. The seq_num is used to determine which
    pixel to fill in.

    For non-gridded data with arbitrary placement, use
    :func:`bluesky.callbacks.mpl_plotting.LiveScatter`.

    This simply wraps around a `AxesImage`.

    Parameters
    ----------
    raster_shape : tuple
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
       Passed through to :meth:`matplotlib.axes.Axes.imshow`

    aspect : str or float, optional
       Passed through to :meth:`matplotlib.axes.Axes.imshow`

    ax : Axes, optional
        matplotib Axes; if none specified, new figure and axes are made.

    x_positive: string, optional
        Defines the positive direction of the x axis, takes the values 'right'
        (default) or 'left'.

    y_positive: string, optional
        Defines the positive direction of the y axis, takes the values 'up'
        (default) or 'down'.

    See Also
    --------
    :class:`bluesky.callbacks.mpl_plotting.LiveScatter`.
    """
    def __init__(self, raster_shape, I, *,
                 clim=None, cmap='viridis',
                 xlabel='x', ylabel='y', extent=None, aspect='equal',
                 ax=None, x_positive='right', y_positive='up'):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        if ax is None:
            fig, ax = plt.subplots()
        ax.cla()
        self.I = I
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect(aspect)
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
        self.x_positive = x_positive
        self.y_positive = y_positive

    def start(self, doc):
        if self.im is not None:
            raise RuntimeError("Can not re-use LiveGrid")
        self._Idata = np.ones(self.raster_shape) * np.nan
        # The user can control origin by specific 'extent'.
        extent = self.extent
        # origin must be 'lower' for the plot to fill in correctly
        # (the first voxel filled must be closest to what mpl thinks
        # is the 'lower left' of the image)
        im = self.ax.imshow(self._Idata, norm=self._norm,
                            cmap=self.cmap, interpolation='none',
                            extent=extent, aspect=self.aspect,
                            origin='lower')

        # make sure the 'positive direction' of the axes matches what is defined in
        #axes_positive
        xmin, xmax = self.ax.get_xlim()
        if ((xmin > xmax and self.x_positive == 'right') or
                (xmax > xmin and self.x_positive == 'left')):
            self.ax.set_xlim(xmax, xmin)
        elif ((xmax >= xmin and self.x_positive == 'right') or
                (xmin >= xmax and self.x_positive == 'left')):
            self.ax.set_xlim(xmin, xmax)
        else:
            raise ValueError('x_positive must be either "right" or "left"')

        ymin, ymax = self.ax.get_ylim()
        if ((ymin > ymax and self.y_positive == 'up') or
                (ymax > ymin and self.y_positive == 'down')):
            self.ax.set_ylim(ymax, ymin)
        elif ((ymax >= ymin and self.y_positive == 'up') or
                (ymin >= ymax and self.y_positive == 'down')):
            self.ax.set_ylim(ymin, ymax)
        else:
            raise ValueError('y_positive must be either "up" or "down"')

        self.im = im
        self.ax.set_title('scan {uid} [{sid}]'.format(sid=doc['scan_id'],
                                                      uid=doc['uid'][:6]))
        self.snaking = doc.get('snaking', (False, False))

        cb = self.ax.figure.colorbar(im, ax=self.ax)
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


class LiveRaster(LiveGrid):
    __doc__ = LiveGrid.__doc__

    def __init__(self, *args, **kwargs):
        warnings.warn("LiveRaster has been renamed to LiveGrid. The name "
                      "LiveRaster will eventually be removed. Use LiveGrid.")
        super().__init__(*args, **kwargs)


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
        self._has_been_run = False

    @property
    def livefit(self):
        return self._livefit

    def start(self, doc):
        self.livefit.start(doc)
        self.x, = self.livefit.independent_vars.keys()  # in case it changed
        if self._has_been_run:
            label = '_nolegend_'
        else:
            label = 'init guess'
        self._has_been_run = True
        self.init_guess_line, = self.ax.plot([], [], color='grey', label=label)
        self.lines.append(self.init_guess_line)
        super().start(doc)
        # Put fit above other lines (default 2) but below text (default 3).
        [line.set_zorder(2.5) for line in self.lines]

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
            # update kwargs to inital guess
            kwargs.update(self.livefit.result.init_values)
            self.y_guess = self.livefit.result.model.eval(**kwargs)
            self.update_plot()
        # Intentionally override LivePlot.event. Do not call super().

    def update_plot(self):
        self.current_line.set_data(self.x_data, self.y_data)
        self.init_guess_line.set_data(self.x_data, self.y_guess)
        # Rescale and redraw.
        self.ax.relim(visible_only=True)
        self.ax.autoscale_view(tight=True)
        self.ax.figure.canvas.draw_idle()

    def descriptor(self, doc):
        self.livefit.descriptor(doc)
        super().descriptor(doc)

    def stop(self, doc):
        self.livefit.stop(doc)
        # Intentionally override LivePlot.stop. Do not call super().


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
    import matplotlib.pyplot as plt
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


class Trajectory(CallbackBase):
    '''Draw a matplotlib Line2D artist and update it for each event.
    The purposes of this callback is to create (on initialization) a
    matplotlib plot indicating the trajectory that a scan will take. During
    the scan it should also indicate when a point has been taken by removing
    the point from the trajectory. A second Line2D artist is also included that
    indicates the actual points that the trajectory took and then update it
    with new data for every `event`.

    Parameters
    ----------
    func : callable
        This must accept a BulkEvent and return two lists of floats (x
        values and y values). The two lists must contain an equal number of
        items, but that number is arbitrary. That is, a given document may add
        one new point, no new points or multiple new points to the 'completed'
        plot and remove 0, 1 or more points from the 'future' path.
    x_trajectory, y_trajectory : Lists
        Two lists ( `'x_vals'` and `'y_vals'`) which are a list of succesive
        x_vals or y_vals indicating the trajectory to be taken. The length of
        the two lists should be identical.
    single_func : callback, optional
        This parameter is available as a perfomrance operation. For most uses,
        ``func`` is sufficient. This is like ``func``, but it gets an Event
        instead of a BulkEvent. If ``None`` is given, Events are up-cast into
        BulkEvents and handed to ``func``.
    ax : matplotlib Axes, optional.
        if ``None``, a new Figure and Axes are created.
    **kwargs
        Passed through to :meth:`Axes.imshow` to style the AxesImage object.
    '''

    def __init__(self, func, x_trajectory, y_trajectory, *,
                 single_func=None, ax=None, **kwargs):
        self.func = func
        self.x_trajectory = x_trajectory
        self.y_trajectory = y_trajectory
        self.single_func = single_func
        if ax is None:
            _, ax = plt.subplots()
        self.ax = ax
        self.x_past = []
        self.y_past = []
        self.trajectory = self.ax.plot(self.x_trajectory, self.y_trajectory,
                                       **kwargs)
        self.past = self.ax.plot(self.x_past, self.y_past, **kwargs)

    def bulk_events(self, doc):
        '''Takes in a bulk_events document and updates x_past and y_past with
        the values returned from self.func(doc).

        Parameters
        ----------
        doc : dict
            The bulk event dictionary that contains the 'data' and 'timestamps'
            associated with the bulk event.

        Returns
        -------
        x_vals, y_vals : Lists
            These are lists of x values and y values arising from the bulk
            event.
        '''

        x_vals, y_vals = self.func(doc)
        self._update(x_vals, y_vals)

    def event(self, doc):
        '''
        Takes in a event documents and returns the values returned from
        self.single_func(doc) or, if it is `None`, self.func(doc).

        Parameters
        ----------
        doc : dict
            The bulk event dictionary that contains the 'data' and 'timestamps'
            associated with the event.

        Returns
        -------
        x_vals, y_vals : Lists
            These are lists of x values and y values arising
            from the event.
        '''

        if self.single_func is not None:
            x_vals, y_vals = self.single_func(doc)
        else:
            bulk_doc = event2bulk_event(doc)
            x_vals, y_vals = self.func(bulk_doc)
        self._update(x_vals, y_vals)

    def _update(self, x_vals, y_vals):
        '''
        Updates self.x_past and self.y_past with the values from the lists
        x_coords, y_coords.

        Parameters
        ----------
        x_vals, y_vals : Lists
            These are lists of x co-ordinate and y co-ordinate values arising
            from the event. The length of all three lists must be the same.
        '''

        if not len(x_vals) == len(y_vals):
            raise ValueError("User function is expected to provide the same "
                             "number of x and y points. Got {0} x points "
                             "and {1} y points."
                             "".format(len(x_vals), len(y_vals)))

        if not x_vals:
            # No new data, Short-circuit.
            return

        # Update grid_data and the plot.
        for x_val, y_val in zip(x_vals, y_vals):
            # add the new values to the past lists
            self.x_past.extend(x_val)
            self.y_past.extend(y_val)
            # remove the first item from the path lists
            del self.x_path.pop[0]
            del self.y_path.pop[0]

        self.path.set_data(self.x_path, self.y_path)
        self.past.set_data(self.x_past, self.y_past)


def event2bulk_event(doc):
    '''Make a BulkEvent from this Event.

    Parameters
    ----------
    doc : dict
        The event dictionary that contains the 'data' and 'timestamps'
        associated with the bulk event.

    Returns
    -------
    bulk_event : dict
        The bulk event dictionary that contains the 'data' and 'timestamp'
        associated with the event.
    '''

    bulk_event = doc.copy()
    bulk_event['data'] = {k: np.expand_dims(v, 0)
                          for k, v in doc['data'].items()}

    bulk_event['timestamps'] = {k: np.expand_dims(v, 0)
                                for k, v in doc['timestamps'].items()}

    return bulk_event


class LineWithPeaks(Line):
    # Track state of axes, which may share instances of LineWithPeaks.
    __labeled = weakref.WeakKeyDictionary()  # map ax to True/False
    __visible = weakref.WeakKeyDictionary()  # map ax to True/False
    __instances = weakref.WeakKeyDictionary()  # map ax to list of instances

    def __init__(self, *args, peak_results, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak_results = peak_results

        ax = self.ax  # for brevity
        if ax not in self.__visible:
            # This is the first instance of LineWithPeaks on these axes.
            # Set up matplotlib event handling.

            self.__visible[ax] = False

            def toggle(event):
                if event.key == 'P':
                    self.__visible[ax] = ~self.__visible[ax]
                    for instance in self.__instances[ax]:
                        instance.check_visibility()

            ax.figure.canvas.mpl_connect('key_press_event', toggle)

        if ax not in self.__instances:
            self.__instances[ax] = []
        self.__instances[ax].append(self)
        self.__arts = None

    def check_visibility(self):
        if self.__visible[self.ax]:
            if self.__arts is None:
                self.plot_annotations()
            else:
                for artist in self.__arts:
                    artist.set_visible(True)
        elif self.__arts is not None:
                for artist in self.__arts:
                    artist.set_visible(False)
        self.ax.legend(loc='best')
        self.ax.figure.canvas.draw_idle()

    def plot_annotations(self):
        styles = iter(cycler('color', 'kr'))
        vlines = []
        for style, attr in zip(styles, ['cen', 'com']):
            val = self.peak_results[attr][self.y]
            # Only put labels in this legend once per axis.
            if self.ax in self.__labeled:
                label = '_no_legend_'
            else:
                label = attr
            vlines.append(self.ax.axvline(val, label=label, **style))
        self.__labeled[self.ax] = None
        self.__arts = vlines

    def stop(self, doc):
        self.check_visibility()
        super().stop(doc)
