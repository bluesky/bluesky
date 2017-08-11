from bluesky.callbacks import CallbackBase, LiveTable, LivePlot, LiveGrid
from bluesky.callbacks.scientific import PeakStats
from cycler import cycler
import itertools
from itertools import chain
import matplotlib.pyplot as plt
import re
from pprint import pformat
from warnings import warn
import weakref


class BestEffortCallback(CallbackBase):
    def __init__(self):
        # internal state
        self._start_doc = None
        self._descriptors = {}
        self._table = None
        # maps descriptor uid to dict which maps data key to LivePlot instance
        self._live_plots = {}
        self._live_grids = {}
        self._peak_stats = {}  # same structure as live_plots
        self._cleanup_motor_heuristic = False
        self._stream_names = set()

        # public options
        self.enabled = True
        self.overplot = True
        self.truncate_table = False 
        # TODO custom width
        self.noplot_streams = ['baseline']

        # public data
        self.peaks = PeakResults()

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def __call__(self, name, doc):
        if not self.enabled:
            return

        super().__call__(name, doc)
    
    def start(self, doc):
        self.clear()
        self._start_doc = doc
        self.plan_hints = doc.get('hints', {})

        # Prepare a guess about the dimensions (independent variables) in case
        # we need it.
        motors = self._start_doc.get('motors') or None
        if motors is not None:
            GUESS = [('primary', [motor]) for motor in motors]
        else:
            GUESS = [('primary', ['time'])]

        # Ues the guess if there is not hint about dimensions.
        dimensions = self.plan_hints.get('dimensions')
        if dimensions is None:
            self._cleanup_motor_heuristic = True
            dimensions = GUESS

        # We can only cope with all the dimensions belonging to the same
        # stream unless we resample. We are not doing to handle that yet.
        if len(set(d[0] for d in dimensions)) != 1:
            self._cleanup_motor_heuristic = True
            dimensions = GUESS  # Fall back on our GUESS.
            warn("We are ignoring the dimensions hinted because we cannot "
                 "combine streams.")
        self.dim_fields = [f
                           for stream_name, field in dimensions
                               for f in field]
        self.dim_stream, _  = dimensions[0]
    
    def descriptor(self, doc):
        self._descriptors[doc['uid']] = doc
        stream_name = doc.get('name', 'primary')  # fall back for old docs

        if stream_name not in self._stream_names:
            self._stream_names.add(stream_name)
            print("New stream: {!r}".format(stream_name))

        columns = hinted_fields(doc)

        ### This deals with old documents. ### 

        if stream_name == 'primary' and self._cleanup_motor_heuristic:
            # We stashed object names in self.dim_fields, which we now need to
            # look up the actual fields for.
            self._cleanup_motor_heuristic = False
            fixed_dim_fields = []
            for obj_name in self.dim_fields:
                try:
                    fields = doc.get('hints', {}).get(obj_name, {})['fields']
                except KeyError:
                    fields = doc['object_keys'][obj_name]
                fixed_dim_fields.extend(fields)
            self.dim_fields = fixed_dim_fields

        ### TABLE ###
        
        if stream_name == self.dim_stream:
            # Ensure that no independent variables ('dimensions') are
            # duplicated here.
            columns = [c for c in columns if c not in self.dim_fields]
            
            self._table = LiveTable(list(self.dim_fields) + columns)
            self._table('start', self._start_doc)
            self._table('descriptor', doc)

        ### DECIDE WHICH KIND OF PLOT CAN BE USED ###

        if stream_name in self.noplot_streams:
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
        if self.overplot:
            # If any open figure matches 'figname {number}', use it. If there
            # are multiple, the most recently touched one will be used.
            pat1 = re.compile('^' + fig_name + '$')
            pat2 = re.compile('^' + fig_name + ' \d+$')
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
        fig = plt.figure(fig_name)

        if not fig.axes:
            # This is apparently a fresh figure. Make axes.
            # The complexity here is due to making a shared x axis. This can be
            # simplified when Figure supports the `subplots` method in a future
            # release of matplotlib.
            for i in range(len(columns)):
                if i == 0:
                    ax = fig.add_subplot(len(columns), 1, 1 + i)
                else:
                    ax = fig.add_subplot(len(columns), 1, 1 + i, sharex=ax)
            fig.subplots_adjust()
        axes = fig.axes

        ### LIVE PLOT AND PEAK ANALYSIS ###

        if len(dim_fields) == 1:
            self._live_plots[doc['uid']] = {}
            self._peak_stats[doc['uid']] = {}
            x_key, = dim_fields
            for y_key, ax in zip(columns, axes):
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
        elif len(dim_fields) == 2:
            self._live_grids[doc['uid']] = {}
            x_key, y_key = dim_fields
            try:
                extents = self._start_doc['extents']
                shape = self._start_doc['shape']
            except KeyError:
                warn("Need both 'shape' and 'extents' in plan metadata to "
                     "create LiveGrid.")
            else:
                for I_key, ax in zip(columns, axes):
                    live_grid = LiveGrid(shape, I_key,
                                         xlabel=x_key, ylabel=y_key,
                                         extent=list(chain(*extents[::-1])),
                                         ax=ax)
                    live_grid('start', self._start_doc)
                    live_grid('descriptor', doc)
                    self._live_grids[doc['uid']][I_key] = live_grid

        fig.tight_layout()

    def event(self, doc):
        if self._descriptors[doc['descriptor']].get('name') == 'primary':
            self._table('event', doc)

        # Show the baseline readings.
        if self._descriptors[doc['descriptor']].get('name') == 'baseline':
            for k, v in doc['data'].items():
                print('Baseline', k, ':', v)

        for y_key in doc['data']:
            live_plot = self._live_plots.get(doc['descriptor'], {}).get(y_key)
            if live_plot is not None:
                live_plot('event', doc)
            live_grid = self._live_grids.get(doc['descriptor'], {}).get(y_key)
            if live_grid is not None:
                live_grid('event', doc)
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

    def clear(self):
        self._start_doc = None
        self._descriptors.clear()
        self._table = None
        self._live_plots.clear()
        self._peak_stats.clear()
        self._live_grids.clear()
        self.peaks.clear()


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
