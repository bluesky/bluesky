import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


class Waterfall:
    """class holds data and generate watefall plot

    Parameters
    ----------
    fig : matplotlib.Figure
        fig this waterfall plot will be drawn on
    canvas : matplotlib.Canvas
        canvas this waterfall plot will be drawn on
    key_list : list, optional
        list of key names. default to None
    int_data_list : list, optional
        list of 1D reduced data. expect each element to be in (x,y)
        format. default to None
    unit : tuple, optional
        a tuple containing strings of x and y labels
    label_size : int, optional
        size of x-, y-label. default is 16
    tick_size : int, optional
        size of x-, y-tick. default is 14
    kwargs :
        keyword arguments for plotting
    """

    def __init__(self, fig=None, canvas=None,
                 key_list=None, int_data_list=None,
                 *, unit=None, label_size=16, tick_size=14,
                 **kwargs):
        if int_data_list is None:
            int_data_list = []
        if key_list is None:
            key_list = []
        if not fig:
            fig = plt.figure()
        self.fig = fig
        if not canvas:
            canvas = self.fig.canvas
        self.canvas = canvas
        self.kwargs = kwargs

        # callback for showing legend
        self.canvas.mpl_connect('pick_event', self.on_plot_hover)
        self.key_list = key_list
        self.int_data_list = int_data_list
        self.ax = self.fig.add_subplot(111)
        self.unit = unit
        self.label_size = label_size
        self.tick_size = tick_size
        # flag to prevent update
        self.halt = False
        # add sliders, which store information
        y_offset_slider_ax = self.fig.add_axes([0.15, 0.95, 0.3, 0.035])
        self.y_offset_slider = Slider(y_offset_slider_ax,
                                      'y-offset', 0.0, 1.0,
                                      valinit=0.1, valfmt='%1.2f')
        self.y_offset_slider.on_changed(self.update_y_offset)

        x_offset_slider_ax = self.fig.add_axes([0.6, 0.95, 0.3, 0.035])
        self.x_offset_slider = Slider(x_offset_slider_ax,
                                      'x-offset', 0.0, 1.0,
                                      valinit=0.1, valfmt='%1.2f')
        self.x_offset_slider.on_changed(self.update_x_offset)
        # init
        self.update(self.key_list, self.int_data_list, refresh=True)

    def update(self, key_list=None, int_data_list=None, refresh=False):
        """top method to update information carried by class and plot

        Parameters
        ----------
        key_list : list, optional
            list of keys. default to None.
        int_data_list : list, optional
            list of 1D data. default to None.
        refresh : bool, optional
            option to set refresh or not. default to False.
        """
        if not int_data_list:
            print("INFO: no reduced data was fed in, "
                  "waterfall plot can't be updated")
            self.halt = True
            self.no_int_data_plot(self.ax, self.canvas)
            return
        # refresh list
        if refresh:
            self.key_list = []
            self.int_data_list = []
        self.key_list.extend(key_list)
        self.int_data_list.extend(int_data_list)
        self._adapt_data_list(self.int_data_list)
        # generate plot
        self.halt = False
        self._update_plot()  # use current value of x,y offset

    def _adapt_data_list(self, int_data_list):
        """method to return statefull information of 1D data list"""
        x_array_list = []
        y_array_list = []
        # parse
        for x, y in int_data_list:
            x_array_list.append(x)
            y_array_list.append(y)
        self.x_array_list = x_array_list
        self.y_array_list = y_array_list
        # stateful information
        self.y_dist = np.max(y_array_list) - np.min(y_array_list)
        self.x_dist = np.max(x_array_list) - np.min(x_array_list)

    def on_plot_hover(self, event):
        """callback to show legend when click on one of curves"""
        line = event.artist
        name = line.get_label()
        line.axes.legend([name], handlelength=0,
                         handletextpad=0, fancybox=True)
        line.figure.canvas.draw_idle()

    def _update_plot(self, x_offset_val=None, y_offset_val=None):
        """core method to update x-, y-offset sliders"""
        self.ax.set_facecolor('w')
        self.ax.cla()
        # remain current offset
        if not x_offset_val:
            x_offset_val = self.x_offset_slider.val
        if not y_offset_val:
            y_offset_val = self.y_offset_slider.val
        # get stateful info
        for ind, el in enumerate(zip(self.x_array_list,
                                     self.y_array_list)):
            x, y = el
            self.ax.plot(x + self.x_dist * ind * x_offset_val,
                         y + self.y_dist * ind * y_offset_val,
                         label=self.key_list[ind], picker=5,
                         **self.kwargs)
        self.ax.autoscale()
        if self.unit:
            xlabel, ylabel = self.unit
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)

        self.ax.xaxis.label.set_size(self.label_size)
        self.ax.yaxis.label.set_size(self.label_size)

        for tick in self.ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(self.tick_size)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(self.tick_size)
        self.canvas.draw_idle()

    def update_y_offset(self, val):
        if self.halt:
            return
        self._update_plot(None, val)

    def update_x_offset(self, val):
        if self.halt:
            return
        self._update_plot(val, None)

    def no_int_data_plot(self, ax, canvas):
        """method to display instructive text about workflow
        """
        ax.cla()
        canvas.draw_idle()
