from cycler import cycler
from pprint import pformat
from bluesky.callbacks import CallbackBase, LiveTable, LivePlot
from bluesky.callbacks.scientific import PeakStats
import matplotlib.pyplot as plt
from warnings import warn


class BestEffortCallback(CallbackBase):
    def __init__(self):
        # internal state
        self.table = None
        self.figures = {}  # maps descriptor uid to (fig, axes)
        self.live_plots = {}  # maps descriptor uid to dict which maps data key to LivePlot instance
        self.peak_stats = {}  # same structure as live_plots
        self.peaks = PeakResults()
        self.start_doc = None
        self.descriptors = {}
        self._cleanup_motor_heuristic = False

        # public options
        self.enabled = True
        self.overplot = True
        self.truncate_table = False 
        # TODO custom width
        self.noplot_streams = ['baseline']

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def __call__(self, name, doc):
        if not self.enabled:
            return

        super().__call__(name, doc)
    
    def start(self, doc):
        self.start_doc = doc
        self.plan_hints = doc.get('hints', {})

        # Prepare a guess about the dimensions (independent variables) in case
        # we need it.
        motors = self.start_doc.get('motors') or None
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
        self.descriptors[doc['uid']] = doc
        stream_name = doc.get('name', 'primary')  # fall back for old docs
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
            
            self.table = LiveTable(list(self.dim_fields) + columns)
            self.table('start', self.start_doc)
            self.table('descriptor', doc)

        ### PLOT AND PEAK ANALYSIS ###

        if stream_name in self.noplot_streams:
            return
        fig_name = ' '.join(sorted(columns))
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
            self.figures[doc['uid']] = (fig, axes)
        else:
            # Overplot on existing axes.
            axes = fig.axes
        self.live_plots[doc['uid']] = {}
        self.peak_stats[doc['uid']] = {}
        for y_key, ax in zip(columns, axes):
            # Are we plotting against a motor or against time?
            if len(self.dim_fields) == 1:
                x_key, = self.dim_fields
            else:
                x_key = None  # causes LivePlot to plot against time

            # Create an instance of LivePlot and an instance of PeakStats.
            live_plot = LivePlotPlusPeaks(y=y_key, x=x_key, ax=ax,
                                          peak_results=self.peaks)
            live_plot('start', self.start_doc)
            live_plot('descriptor', doc)
            peak_stats = PeakStats(x=x_key, y=y_key)
            peak_stats('start', self.start_doc)
            peak_stats('descriptor', doc)

            # Stash them in state.
            self.live_plots[doc['uid']][y_key] = live_plot
            self.peak_stats[doc['uid']][y_key] = peak_stats

    def event(self, doc):
        if self.descriptors[doc['descriptor']].get('name') == 'primary':
            self.table('event', doc)

        # Show the baseline readings.
        if self.descriptors[doc['descriptor']].get('name') == 'baseline':
            for k, v in doc['data'].items():
                print('Baseline', k, ':', v)

        for y_key in doc['data']:
            live_plot = self.live_plots.get(doc['descriptor'], {}).get(y_key)
            if live_plot is not None:
                live_plot('event', doc)
            peak_stats = self.peak_stats.get(doc['descriptor'], {}).get(y_key)
            if peak_stats is not None:
                peak_stats('event', doc)

    def stop(self, doc):
        if self.table is not None:
            self.table('stop', doc)

        for live_plots in self.live_plots.values():
            for live_plot in live_plots.values():
                live_plot('stop', doc)

        # Compute peak stats and build results container.
        ps_by_key = {}  # map y_key to PeakStats instance
        for peak_stats in self.peak_stats.values():
            for y_key, ps in peak_stats.items():
                ps('stop', doc)
                ps_by_key[y_key] = ps
        self.peaks.update(ps_by_key)

    def clear(self):
        self.table = None
        self.descriptors.clear()
        self.live_plots.clear()
        self.peak_stats.clear()
        self.figures.clear()
        self.start_doc = None


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
        for attr in self.ATTRS:
            lines.append("{" + "'{}':".format(attr))
            for line in pformat(getattr(self, attr), width=1).split('\n'):
                lines.append("    {}".format(line))
            lines.append('}')
        return '\n'.join(lines)


class LivePlotPlusPeaks(LivePlot):
    def __init__(self, *args, peak_results, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak_results = peak_results
        self.ax.figure.canvas.mpl_connect('key_press_event', self.on_key)
        self.__arts = None
        self.__visible = False

    def on_key(self, event):
        if event.key == 'P':
            self.toggle_annotations()

    def toggle_annotations(self):
        self.__visible = ~self.__visible
        if self.__visible:
            if self.__arts is None:
                self.plot_annotations()
                self.ax.figure.canvas.draw_idle()
            else:
                for artist in self.__arts:
                    artist.set_visible(True)
                self.ax.figure.canvas.draw_idle()
        else:
            for artist in self.__arts:
                artist.set_visible(False)
            self.ax.figure.canvas.draw_idle()

    def plot_annotations(self):
        styles = iter(cycler('color', 'kr'))
        vlines = []
        for style, attr in zip(styles, ['cen', 'com']):
            val = self.peak_results[attr][self.y]
            vlines.append(self.ax.axvline(val, label=attr, **style))

        self.ax.legend(loc='best')  # re-render legend to include new labels

        self.__arts = vlines


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
