'''
    Best Effort Callback.
    For instructions on how to test in a simulated environment please see:
        tests/interactive/best_effort_cb.py
'''
from cycler import cycler
from datetime import datetime
from io import StringIO
import itertools
import numpy as np
import matplotlib.pyplot as plt
from pprint import pformat
import re
import sys
import time
from warnings import warn
import weakref

from .core import CallbackBase, Callback, LiveTable
from .mpl_plotting import LivePlot, LiveGrid, LiveScatter
from .fitting import PeakStats


class BestEffortCallback(CallbackBase):
    def __init__(self, *, fig_factory=None, table_enabled=True):
        # internal state
        self._start_doc = None
        self._descriptors = {}
        self._table = None
        self._heading_enabled = True
        self._table_enabled = table_enabled
        self._baseline_enabled = True
        self._plots_enabled = True
        # axes supplied from outside
        self._fig_factory = fig_factory
        # maps descriptor uid to dict which maps data key to LivePlot instance
        self._live_plots = {}
        self._live_grids = {}
        self._live_scatters = {}
        self._peak_stats = {}  # same structure as live_plots
        self._cleanup_motor_heuristic = False
        self._stream_names_seen = set()

        # public options
        self.overplot = True
        self.noplot_streams = ['baseline']
        self.omit_single_point_plot = True

        # public data
        self.peaks = PeakResults()

        # hack to handle the bottom border of the table
        self._buffer = StringIO()
        self._baseline_toggle = True

    def enable_heading(self):
        "Print timestamp and IDs at the top of a run."
        self._heading_enabled = True

    def disable_heading(self):
        "Opposite of enable_heading()"
        self._heading_enabled = False

    def enable_table(self):
        "Print hinted readings from the 'primary' stream in a LiveTable."
        self._table_enabled = True

    def disable_table(self):
        "Opposite of enable_table()"
        self._table_enabled = False

    def enable_baseline(self):
        "Print hinted fields from the 'baseline' stream."
        self._baseline_enabled = True

    def disable_baseline(self):
        "Opposite of enable_baseline()"
        self._baseline_enabled = False

    def enable_plots(self):
        "Plot hinted fields from all streams not in ``noplot_streams``."
        self._plots_enabled = True

    def disable_plots(self):
        "Do not plot anything."
        self._plots_enabled = False

    def __call__(self, name, doc):
        if not (self._table_enabled or self._baseline_enabled or
                self._plots_enabled):
            return

        super().__call__(name, doc)

    def start(self, doc):
        self.clear()
        self._start_doc = doc
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

        # Print heading.
        tt = datetime.fromtimestamp(self._start_doc['time']).utctimetuple()
        if self._heading_enabled:
            print("Transient Scan ID: {0}     Time: {1}".format(
                self._start_doc['scan_id'],
                time.strftime("%Y/%m/%d %H:%M:%S", tt)))
            print("Persistent Unique Scan ID: '{0}'".format(
                self._start_doc['uid']))

    def descriptor(self, doc):
        self._descriptors[doc['uid']] = doc
        stream_name = doc.get('name', 'primary')  # fall back for old docs

        if stream_name not in self._stream_names_seen:
            self._stream_names_seen.add(stream_name)
            if self._table_enabled:
                print("New stream: {!r}".format(stream_name))

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

        # ## TABLE ## #

        if stream_name == self.dim_stream:
            # Ensure that no independent variables ('dimensions') are
            # duplicated here.
            columns = [c for c in columns if c not in self.all_dim_fields]

            if self._table_enabled:
                # plot everything, independent or dependent variables
                self._table = LiveTable(list(self.all_dim_fields) + columns)
                self._table('start', self._start_doc)
                self._table('descriptor', doc)

        # ## DECIDE WHICH KIND OF PLOT CAN BE USED ## #

        if not self._plots_enabled:
            return
        if stream_name in self.noplot_streams:
            return
        if not columns:
            return
        if ((self._start_doc.get('num_points') == 1) and
                (stream_name == self.dim_stream) and
                self.omit_single_point_plot):
            return

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

        if self._fig_factory:
            fig = self._fig_factory(fig_name)
        else:
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

        # ## LIVE PLOT AND PEAK ANALYSIS ## #

        if ndims == 1:
            self._live_plots[doc['uid']] = {}
            self._peak_stats[doc['uid']] = {}
            x_key, = dim_fields
            for y_key, ax in zip(columns, axes):
                dtype = doc['data_keys'][y_key]['dtype']
                if dtype not in ('number', 'integer'):
                    warn("Omitting {} from plot because dtype is {}"
                         "".format(y_key, dtype))
                    continue
                # Create an instance of LivePlot and an instance of PeakStats.
                live_plot = LivePlotPlusPeaks(y=y_key, x=x_key, ax=ax,
                                              peak_results=self.peaks)
                live_plot('start', self._start_doc)
                live_plot('descriptor', doc)
                peak_stats = PeakStats(x=x_key, y=y_key)
                peak_stats('start', self._start_doc)
                peak_stats('descriptor', doc)

                # Stash them in state.
                self._live_plots[doc['uid']][y_key] = live_plot
                self._peak_stats[doc['uid']][y_key] = peak_stats

            for ax in axes[:-1]:
                ax.set_xlabel('')
        elif ndims == 2:
            # Decide whether to use LiveGrid or LiveScatter. LiveScatter is the
            # safer one to use, so it is the fallback..
            gridding = self._start_doc.get('hints', {}).get('gridding')
            if gridding == 'rectilinear':
                self._live_grids[doc['uid']] = {}
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
                        self._live_grids[doc['uid']][I_key] = live_grid
            else:
                self._live_scatters[doc['uid']] = {}
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
                    self._live_scatters[doc['uid']][I_key] = live_scatter
        else:
            raise NotImplementedError("we do not support 3D+ in BEC yet "
                                      "(and it should have bailed above)")
        try:
            fig.tight_layout()
        except ValueError:
            pass

    def event(self, doc):
        descriptor = self._descriptors[doc['descriptor']]
        if descriptor.get('name') == 'primary':
            if self._table is not None:
                self._table('event', doc)

        # Show the baseline readings.
        if descriptor.get('name') == 'baseline':
            columns = hinted_fields(descriptor)
            self._baseline_toggle = not self._baseline_toggle
            if self._baseline_toggle:
                file = self._buffer
                subject = 'End-of-run'
            else:
                file = sys.stdout
                subject = 'Start-of-run'
            if self._baseline_enabled:
                print('{} baseline readings:'.format(subject), file=file)
                border = '+' + '-' * 32 + '+' + '-' * 32 + '+'
                print(border, file=file)
                for k, v in doc['data'].items():
                    if k not in columns:
                        continue
                    print('| {:>30} | {:<30} |'.format(k, v), file=file)
                print(border, file=file)

        for y_key in doc['data']:
            live_plot = self._live_plots.get(doc['descriptor'], {}).get(y_key)
            if live_plot is not None:
                live_plot('event', doc)
            live_grid = self._live_grids.get(doc['descriptor'], {}).get(y_key)
            if live_grid is not None:
                live_grid('event', doc)
            live_sc = self._live_scatters.get(doc['descriptor'], {}).get(y_key)
            if live_sc is not None:
                live_sc('event', doc)
            peak_stats = self._peak_stats.get(doc['descriptor'], {}).get(y_key)
            if peak_stats is not None:
                peak_stats('event', doc)

    def stop(self, doc):
        if self._table is not None:
            self._table('stop', doc)

        # Compute peak stats and build results container.
        ps_by_key = {}  # map y_key to PeakStats instance
        for peak_stats in self._peak_stats.values():
            for y_key, ps in peak_stats.items():
                ps('stop', doc)
                ps_by_key[y_key] = ps
        self.peaks.update(ps_by_key)

        for live_plots in self._live_plots.values():
            for live_plot in live_plots.values():
                live_plot('stop', doc)

        for live_grids in self._live_grids.values():
            for live_grid in live_grids.values():
                live_grid('stop', doc)

        for live_scatters in self._live_scatters.values():
            for live_scatter in live_scatters.values():
                live_scatter('stop', doc)

        if self._baseline_enabled:
            # Print baseline below bottom border of table.
            self._buffer.seek(0)
            print(self._buffer.read())
            print('\n')

    def clear(self):
        self._start_doc = None
        self._descriptors.clear()
        self._stream_names_seen.clear()
        self._table = None
        self._live_plots.clear()
        self._peak_stats.clear()
        self._live_grids.clear()
        self._live_scatters.clear()
        self.peaks.clear()
        self._buffer = StringIO()
        self._baseline_toggle = True


class PeakResults:
    ATTRS = ('com', 'cen', 'max', 'min', 'fwhm', 'nlls')

    def __init__(self):
        for attr in self.ATTRS:
            setattr(self, attr, {})

    def clear(self):
        for attr in self.ATTRS:
            getattr(self, attr).clear()

    def update(self, ps_dict):
        for y_key, ps in ps_dict.items():
            for attr in self.ATTRS:
                getattr(self, attr)[y_key] = getattr(ps, attr)

    def __getitem__(self, key):
        if key in self.ATTRS:
            return getattr(self, key)
        raise KeyError("Keys are: {}".format(self.ATTRS))

    def __repr__(self):
        # This is a proper eval-able repr, but with some manually-tweaked
        # whitespace to make it easier to prase.
        lines = []
        lines.append('{')
        for attr in self.ATTRS:
            lines.append("'{}':".format(attr))
            for line in pformat(getattr(self, attr), width=1).split('\n'):
                lines.append("    {}".format(line))
            lines.append(',')
        lines.append('}')
        return '\n'.join(lines)


class LivePlotPlusPeaks(LivePlot):
    # Track state of axes, which may share instances of LivePlotPlusPeaks.
    __labeled = weakref.WeakKeyDictionary()  # map ax to True/False
    __visible = weakref.WeakKeyDictionary()  # map ax to True/False
    __instances = weakref.WeakKeyDictionary()  # map ax to list of instances

    def __init__(self, *args, peak_results, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak_results = peak_results

        ax = self.ax  # for brevity
        if ax not in self.__visible:
            # This is the first instance of LivePlotPlusPeaks on these axes.
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

class HeadingPrinter(Callback):
    def __init__(self, start_doc):
        self._start_doc = start_doc
        # Print heading.
        tt = datetime.fromtimestamp(self._start_doc['time']).utctimetuple()
        print("Transient Scan ID: {0}     Time: {1}".format(
            self._start_doc['scan_id'],
            time.strftime("%Y/%m/%d %H:%M:%S", tt)))
        print("Persistent Unique Scan ID: '{0}'".format(
            self._start_doc['uid']))


class BaselinePrinter(Callback):
    def __init__(self, start_doc, file=sys.stdout):
        # We accept a start_doc but discard it.
        ...
        self._descriptors = {}
        self._baseline_toggle = True
        self._file = file
    def descriptor(self, doc):
        # TODO Check the stream name of this descriptor using doc['name']
        # If the stream name is 'baseline', record doc['uid'] in an instance
        # variable for future reference.
        ...
        if doc['name'] == 'baseline':
            self._descriptors[doc['uid']] = doc
    def event(self, doc):
        # Check the descriptor uid of this Event using doc['descriptor'].
        # if this matches the uid of the descriptor we have stashed in the 
        # method above, extract the baseline readings and print them, the way
        # we do in BestEffortCallback currently.
        ...
        # Show the baseline readings.
        if doc['descriptor'] in self._descriptors:
            descriptor = self._descriptors[doc['descriptor']]
            columns = hinted_fields(descriptor)
            self._baseline_toggle = not self._baseline_toggle
            if self._baseline_toggle:
                subject = 'End-of-run'
            else:
                subject = 'Start-of-run'
            print('{} baseline readings:'.format(subject), file=self._file)
            border = '+' + '-' * 32 + '+' + '-' * 32 + '+'
            print(border, file=self._file)
            for k, v in doc['data'].items():
                if k not in columns:
                    continue
                print('| {:>30} | {:<30} |'.format(k, v), file=self._file)
            print(border, file=self._file)


def hinted_fields(descriptor):
    # Figure out which columns to put in the table.
    obj_names = list(descriptor['object_keys'])
    # We will see if these objects hint at whether
    # a subset of their data keys ('fields') are interesting. If they
    # did, we'll use those. If these didn't, we know that the RunEngine
    # *always* records their complete list of fields, so we can use
    # them all unselectively.
    columns = []
    for obj_name in obj_names:
        try:
            fields = descriptor.get('hints', {}).get(obj_name, {})['fields']
        except KeyError:
            fields = descriptor['object_keys'][obj_name]
        columns.extend(fields)
    return columns

def get_ps(x, y, shift=0.5):
    """ peak status calculation from CHX algorithm.
    """
    ps = {}
    x=np.array(x)
    y=np.array(y)

    PEAK=x[np.argmax(y)]
    PEAK_y=np.max(y)
    COM=np.sum(x * y) / np.sum(y)
    ps['com'] = COM
    ### from Maksim: assume this is a peak profile:
    def is_positive(num):
        return True if num > 0 else False

    # Normalize values first:
    ym = (y - np.min(y)) / (np.max(y) - np.min(y)) - shift  # roots are at Y=0

    positive = is_positive(ym[0])
    list_of_roots = []
    for i in range(len(y)):
        current_positive = is_positive(ym[i])
        if current_positive != positive:
            list_of_roots.append(x[i - 1] + (x[i] - x[i - 1]) / (abs(ym[i]) + abs(ym[i - 1])) * abs(ym[i - 1]))
            positive = not positive
    if len(list_of_roots) >= 2:
        FWHM=abs(list_of_roots[-1] - list_of_roots[0])
        CEN=list_of_roots[0]+0.5*(list_of_roots[1]-list_of_roots[0])

        ps['fwhm'] = FWHM
        ps['cen'] = CEN

    else:    # ok, maybe it's a step function..
        print('no peak...trying step function...')
        ym = ym + shift
        def err_func(x, x0, k=2, A=1,  base=0 ):     #### erf fit from Yugang
            return base - A * erf(k*(x-x0))
        mod = Model(  err_func )
        ### estimate starting values:
        x0=np.mean(x)
        #k=0.1*(np.max(x)-np.min(x))
        pars  = mod.make_params( x0=x0, k=2,  A = 1., base = 0. )
        result = mod.fit(ym, pars, x = x )
        CEN=result.best_values['x0']
        FWHM = result.best_values['k']
        ps['fwhm'] = FWHM
        ps['cen'] = CEN

    return ps



def center_of_mass(input, labels=None, index=None):
    """
    Calculate the center of mass of the values of an array at labels.
    Parameters
    ----------
    input : ndarray
        Data from which to calculate center-of-mass.
    labels : ndarray, optional
        Labels for objects in `input`, as generated by `ndimage.label`.
        Only used with `index`.  Dimensions must be the same as `input`.
    index : int or sequence of ints, optional
        Labels for which to calculate centers-of-mass. If not specified,
        all labels greater than zero are used.  Only used with `labels`.
    Returns
    -------
    center_of_mass : tuple, or list of tuples
        Coordinates of centers-of-mass.
    Examples
    --------
    >>> a = np.array(([0,0,0,0],
                      [0,1,1,0],
                      [0,1,1,0],
                      [0,1,1,0]))
    >>> from scipy import ndimage
    >>> ndimage.measurements.center_of_mass(a)
    (2.0, 1.5)
    Calculation of multiple objects in an image
    >>> b = np.array(([0,1,1,0],
                      [0,1,0,0],
                      [0,0,0,0],
                      [0,0,1,1],
                      [0,0,1,1]))
    >>> lbl = ndimage.label(b)[0]
    >>> ndimage.measurements.center_of_mass(b, lbl, [1,2])
    [(0.33333333333333331, 1.3333333333333333), (3.5, 2.5)]
    """
    normalizer = np.sum(input, labels, index)
    grids = np.ogrid[[slice(0, i) for i in input.shape]]

    results = [
        np.sum(input * grids[dir].astype(float), labels, index) / normalizer
        for dir in range(input.ndim)]

    if np.isscalar(results[0]):
        return tuple(results)

    return [tuple(v) for v in np.array(results).T]

class CollectThenCompute(Callback):

    def __init__(self, start_doc):
        self._start_doc = start_doc
        self._stop_doc = None
        self._events = deque()
        self._descriptors = deque()

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

class PeakStats(CollectThenCompute):
    """
    Compute peak statsitics after a run finishes.
    Results are stored in the attributes.
    Parameters
    ----------
    x : string
        field name for the x variable (e.g., a motor)
    y : string
        field name for the y variable (e.g., a detector)
    edge_count : int or None, optional
        If not None, number of points at beginning and end to use
        for quick and dirty background subtraction.
    Note
    ----
    It is assumed that the two fields, x and y, are recorded in the same
    Event stream.
    Attributes
    ----------
    com : center of mass
    cen : mid-point between half-max points on each side of the peak
    max : x location of y maximum
    min : x location of y minimum
    crossings : crosses between y and middle line, which is
          ((np.max(y) + np.min(y)) / 2). Users can estimate FWHM based
          on those info.
    fwhm : the computed full width half maximum (fwhm) of a peak.
           The distance between the first and last crossing is taken to
           be the fwhm.
    """

    def __init__(self, x, y, edge_count=None):
        self.x = x
        self.y = y
        self.com = None
        self.cen = None
        self.max = None
        self.min = None
        self.nlls = None
        self.crossings = None
        self.fwhm = None
        self.lin_bkg = None
        self._edge_count = edge_count
        super().__init__()

    def __getitem__(self, key):
        if key in ['com', 'cen', 'max', 'min']:
            return getattr(self, key)
        else:
            raise KeyError

    def compute(self):
        "This method is called at run-stop time by the superclass."
        # clear all results
        self.com = None
        self.cen = None
        self.max = None
        self.min = None
        self.nlls = None
        self.crossings = None
        self.fwhm = None
        self.lin_bkg = None

        x = []
        y = []
        for event in self._events:
            try:
                _x = event['data'][self.x]
                _y = event['data'][self.y]
            except KeyError:
                pass
            else:
                x.append(_x)
                y.append(_y)
        x = np.array(x)
        y = np.array(y)
        if not len(x):
            # nothing to do
            return
        self.x_data = x
        self.y_data = y
        if self._edge_count is not None:
            left_x = np.mean(x[:self._edge_count])
            left_y = np.mean(y[:self._edge_count])

            right_x = np.mean(x[-self._edge_count:])
            right_y = np.mean(y[-self._edge_count:])

            m = (right_y - left_y) / (right_x - left_x)
            b = left_y - m * left_x
            # don't do this in place to not mess with self.y_data
            y = y - (m * x + b)
            self.lin_bkg = {'m': m, 'b': b}

        # Compute x value at min and max of y
        self.max = x[np.argmax(y)], self.y_data[np.argmax(y)],
        self.min = x[np.argmin(y)], self.y_data[np.argmin(y)],
        self.com, = np.interp(center_of_mass(y), np.arange(len(x)), x)
        mid = (np.max(y) + np.min(y)) / 2
        crossings = np.where(np.diff((y > mid).astype(np.int)))[0]
        _cen_list = []
        for cr in crossings.ravel():
            _x = x[cr:cr+2]
            _y = y[cr:cr+2] - mid

            dx = np.diff(_x)[0]
            dy = np.diff(_y)[0]
            m = dy / dx
            _cen_list.append((-_y[0] / m) + _x[0])

        if _cen_list:
            self.cen = np.mean(_cen_list)
            self.crossings = np.array(_cen_list)
            if len(_cen_list) >= 2:
                self.fwhm = np.abs(self.crossings[-1] - self.crossings[0],
                                   dtype=float)
        # reset y data
        y = self.y_data


