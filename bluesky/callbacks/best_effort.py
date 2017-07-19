from bluesky.callbacks import CallbackBase, LiveTable, LivePlot
import matplotlib.pyplot as plt


class BestEffortCallback(CallbackBase):
    def __init__(self):
        # internal state
        self.table = None
        self.figures = {}  # maps descriptor uid to (fig, axes)
        self.live_plots = {}  # maps descriptor uid to dict which maps data key to LivePlot instance
        self.start_doc = None
        self.descriptors = {}

        # public options
        self.enabled = True
        self.overplot = True
        self.truncate_table = False 

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def __call__(self, name, doc):
        if not self.enabled:
            return

        super().__call__(name, doc)
    
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
