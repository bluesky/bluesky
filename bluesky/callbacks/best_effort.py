# This is an experimental new approach to subscriptions (visualization, table, etc.)

# Old way:
# gs.DETS = [qem07]
# gs.PLOT_Y = 'gc_diag_grid'
# gs.TABLE_COLS = [...]
# RE(ct())

# New way:
# d = [qem07]
# RE(count(d))
# ... and table and plot configure themselves automatically


from bluesky.callbacks import CallbackBase, LiveTable, LivePlot
import matplotlib.pyplot as plt

# This class will someday go int bluesky itself.

class BestEffortCallback(CallbackBase):
    def __init__(self):
        # internal state
        self.table = None
        self.figures = {}  # maps descriptor uid to (fig, axes)
        self.live_plots = {}  # maps descriptor uid to dict which maps data key to LivePlot instance
        self.start_doc = None
        self.descriptors = {}

        # public options
        self.overplot = True
        self.truncate_table = False 
    
    def start(self, doc):
        print('You are running a', doc['plan_name'])
        self.start_doc = doc
    
    def descriptor(self, doc):
        self.descriptors[doc['uid']] = doc
        columns = list(doc['data_keys'])

        # Show the 'primary' stream in a LiveTable.
        if doc.get('name') == 'primary':
            self.table = LiveTable(columns)
            self.table('start', self.start_doc)
            self.table('descriptor', doc)

        fig_name = ' '.join(sorted(columns))
        fig = plt.figure(fig_name)
        if not fig.axes:
            # This is a apparently a fresh figure. Make axes.
            # The complexity here is due to making a shared x axis.
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
        for y_key, ax in zip(columns, axes):
            # Are we plotting against a motor or against time?
            motors = self.start_doc.get('motors') or None
            x_key = None
            if motors:
                x_key = motors[0]
            live_plot = LivePlot(y=y_key, x=x_key, ax=ax)
            live_plot('start', doc)
            self.live_plots[doc['uid']][y_key] = live_plot

    def event(self, doc):
        if self.descriptors[doc['descriptor']].get('name') == 'primary':
            self.table('event', doc)

        # Show the baseline readings.
        if self.descriptors[doc['descriptor']].get('name') == 'baseline':
            for k, v in doc['data'].items():
                print('Baseline', k, ':', v)

        for y_key in doc['data']:
            live_plot = self.live_plots[doc['descriptor']][y_key]
            live_plot('event', doc)

    def stop(self, doc):
        self.table('stop', doc)
        for live_plots in self.live_plots.values():
            for live_plot in live_plots.values():
                live_plot('stop', doc)

    def clear(self):
        self.table = None
        self.descriptors.clear()
        self.live_plots.clear()
        self.figures.clear()
        self.start_doc = None


 
try:
    RE.unsubscribe(token)
except NameError:
    pass
B = BestEffortCallback()
token = RE.subscribe('all', B)


from bluesky.plans import baseline_wrapper, monitor_during_wrapper, fly_during_wrapper
class DiagnosticPreprocessor:
    def __init__(self, *, baseline=None, monitors=None, flyers=None):
        """
        A configurable preprocessor for diagnostic measurements
        This is a plan preprocessor that applies:
        * baseline_wrapper
        * monitor_during_wrapper
        * flyer_during_wrapper
        Parameters
        ----------
        baseline : list
            Devices to be read at the beginning and end of each run
        monitors : list
            Signals (not multi-signal Devices) to be monitored during each run,
            generating readings asynchronously
        flyers : list
            "Flyable" Devices to be kicked off before each run and collected
            at the end of each run
        Example
        -------
        >>> D = DiagnosticPreprocessor(baseline=[some_motor, some_detector]),
        ...                            monitors=[some_signal],
        ...                            flyers=[some_flyer])
        >>> RE = RunEngine({})
        >>> RE.preprocessors.append(D)
        """
        if baseline is None:
            baseline = []
        if monitors is None:
            monitors = []
        if flyers is None:
            flyers = []
        self.baseline = baseline
        self.monitors = monitors
        self.flyers = flyers

    def __call__(self, plan):
        plan = baseline_wrapper(plan, self.baseline)
        plan = monitor_during_wrapper(plan, self.monitors)
        plan = fly_during_wrapper(plan, self.flyers)
        return (yield from plan)


from ophyd import EpicsSignal
gc_pressure = EpicsSignalRO('XF:02IDC-VA{BT:16-TCG:16_1}P-I', name='gc_pressure')

# TODO - this needs to be streamlined so that the wrapper is inherent. i.e. acts more like gs.MONITORS.append()

## BELOW WORKS but makes a pandas table that is 3 columns by 6 rows

#D = DiagnosticPreprocessor(baseline=[gc_pressure])
#Cff = DiagnosticPreprocessor(baseline=[pgm.cff.user_readback]) 
#ExtSltVG = DiagnosticPreprocessor(baseline=[extslt.vg.user_readback]) 


#RE.preprocessors.clear()
#RE.preprocessors.append(D)
#RE.preprocessors.append(Cff)
#RE.preprocessors.append(ExtSltVG)

## BELOW is test replacement of above to get 3 colums by 2 rows (hopefully)

#baseline=[gc_pressure,pgm.cff.user_readback,extslt.vg.user_readback]
D = DiagnosticPreprocessor(baseline=[gc_pressure,pgm.cff.user_readback,extslt.vg.user_readback])
RE.preprocessors.clear()
RE.preprocessors.append(D)





def get_config(header, device_name):
    return header.descriptors[0]['configuration'][device_name]['data']
