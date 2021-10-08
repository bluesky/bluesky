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
import threading
import time
from warnings import warn
import weakref

from .core import LiveTable, make_class_safe
from .mpl_plotting import LivePlot, LiveGrid, LiveScatter, QtAwareCallback
from .fitting import PeakStats
import logging

logger = logging.getLogger(__name__)


@make_class_safe(logger=logger)
class BestEffortCallback(QtAwareCallback):
    def __init__(self, *, fig_factory=None, table_enabled=True,
                 calc_derivative_and_stats=False, **kwargs):
        super().__init__(**kwargs)
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
        self._calc_derivative_and_stats = calc_derivative_and_stats
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

    def __call__(self, name, doc, *args, **kwargs):
        if not (self._table_enabled or self._baseline_enabled or
                self._plots_enabled):
            return
        super().__call__(name, doc, *args, **kwargs)

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
            print("\n\nTransient Scan ID: {0}     Time: {1}".format(
                self._start_doc.get('scan_id', ''),
                time.strftime("%Y-%m-%d %H:%M:%S", tt)))
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

        # Ensure that no independent variables ('dimensions') are
        # duplicated here.
        columns = [c for c in columns if c not in self.all_dim_fields]

        # ## DECIDE WHICH KIND OF PLOT CAN BE USED ## #

        plot_data = True

        if not self._plots_enabled:
            plot_data = False
        if stream_name in self.noplot_streams:
            plot_data = False
        if not columns:
            plot_data = False
        if ((self._start_doc.get('num_points') == 1) and
                (stream_name == self.dim_stream) and
                self.omit_single_point_plot):
            plot_data = False

        if plot_data:

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
                if len(columns) < 5:
                    layout = (len(columns), 1)
                else:
                    nrows = ncols = int(np.ceil(np.sqrt(len(columns))))
                    while (nrows - 1) * ncols > len(columns):
                        nrows -= 1
                    layout = (nrows, ncols)
                if ndims == 1:
                    share_kwargs = {'sharex': True}
                elif ndims == 2:
                    share_kwargs = {'sharex': True, 'sharey': True}
                else:
                    raise NotImplementedError("we now support 3D?!")

                fig_size = np.array(layout[::-1]) * 5
                fig.set_size_inches(*fig_size)
                fig.subplots(*map(int, layout), **share_kwargs)
                for ax in fig.axes[len(columns):]:
                    ax.set_visible(False)

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
                    peak_stats = PeakStats(
                        x=x_key, y=y_key,
                        calc_derivative_and_stats=self._calc_derivative_and_stats
                    )
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
                                ax.set_aspect(aspect, adjustable='box')
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

        # ## TABLE ## #

        if stream_name == self.dim_stream:
            if self._table_enabled:
                # plot everything, independent or dependent variables
                self._table = LiveTable(list(self.all_dim_fields) + columns, separator_lines=False)
                self._table('start', self._start_doc)
                self._table('descriptor', doc)

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

    def plot_prune_fifo(self, num_lines, x_signal, y_signal):
        """
        Find the plot with axes x_signal and y_signal.  Replot with only the last *num_lines* lines.

        Example to remove all scans but the last:
        >>> bec.plot_prune_fifo(1, m1, noisy)

        Parameters
        ----------
        num_lines: int
            number of lines (plotted scans) to keep, must be >= 0
        x_signal: object
            instance of ophyd.Signal (or subclass),
            independent (x) axis
        y_signal: object
            instance of ophyd.Signal (or subclass),
            dependent (y) axis
        """
        if num_lines < 0:
            emsg = (f"Argument 'num_lines' (given as {num_lines})"
                    " must be >= 0.")
            raise ValueError(emsg)
        for liveplot in self._live_plots.values():
            lp = liveplot.get(y_signal.name)
            if lp is None or lp.x != x_signal.name or lp.y != y_signal.name:
                continue

            # pick out only the lines that contain plot data,
            # skipping the lines that show peak centers
            lines = [
                tr
                for tr in lp.ax.lines
                if len(tr._x) != 2
                or len(tr._y) != 2
                or (len(tr._x) == 2
                    and tr._x[0] != tr._x[1])
            ]
            if len(lines) > num_lines:
                keepers = lines[-num_lines:]
                for tr in lines:
                    if tr not in keepers:
                        tr.remove()
                lp.ax.legend()
                if num_lines > 0:
                    lp.update_plot()


class PeakResults:
    ATTRS = ('com', 'cen', 'max', 'min', 'fwhm')

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
        # whitespace to make it easier to parse.
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
        self.__setup_lock = threading.Lock()
        self.__setup_event = threading.Event()
        super().__init__(*args, **kwargs)
        self.peak_results = peak_results

        def setup():
            # Run this code in start() so that it runs on the correct thread.
            with self.__setup_lock:
                if self.__setup_event.is_set():
                    return
                self.__setup_event.set()
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

        self.__setup = setup

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

    def start(self, doc):
        super().start(doc)
        self.__setup()

    def stop(self, doc):
        self.check_visibility()
        super().stop(doc)


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
