from collections import ChainMap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cycler import cycler
import numpy as np
import warnings

from .core import CallbackBase, get_obj_fields


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
            loc=0, title=self.legend_title).draggable()
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
    :func:`bluesky.callbacks.LiveGrid`.

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
    :class:`bluesky.callbacks.LiveGrid`.
    """
    def __init__(self, x, y, I, *, xlim=None, ylim=None,
                 clim=None, cmap='viridis', ax=None, **kwargs):
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


class LiveGrid(CallbackBase):
    """Plot gridded 2D data in a "heat map".

    This assumes that readings are placed on a regular grid and can be placed
    into an image by sequence number. The seq_num is used to determine which
    pixel to fill in.

    For non-gridded data with arbitrary placement, use
    :func:`bluesky.callbacks.LiveScatter`.

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
    :class:`bluesky.callbacks.LiveScatter`.
    """
    def __init__(self, raster_shape, I, *,
                 clim=None, cmap='viridis',
                 xlabel='x', ylabel='y', extent=None, aspect='equal',
                 ax=None, x_positive='right', y_positive='up'):
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
