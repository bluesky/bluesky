"""
Best Effort Callback.
For instructions on how to test in a simulated environment please see:
    tests/interactive/best_effort_cb.py
"""

import itertools
import logging
import re
import threading
import weakref
from collections.abc import Mapping
from functools import partial
from pprint import pformat
from typing import Any, Literal
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure

from .core import BestEffortTable, make_class_safe
from .core import hinted_fields as hinted_fields
from .fitting import PeakStats
from .mpl_plotting import LiveGrid, LivePlot, LiveScatter, QtAwareCallback

logger = logging.getLogger(__name__)


@make_class_safe(logger=logger)
class BestEffortCallback(BestEffortTable, QtAwareCallback):
    def __init__(self, *, fig_factory=None, table_enabled=True, calc_derivative_and_stats=False, **kwargs):
        super().__init__(table_enabled=table_enabled, **kwargs)
        # internal state
        self._plots_enabled = True
        # axes supplied from outside
        self._fig_factory = fig_factory if fig_factory is not None else partial(plt.figure, layout="constrained")
        # maps descriptor uid to dict which maps data key to LivePlot instance
        self._live_plots = {}
        self._live_grids = {}
        self._live_scatters = {}
        self._peak_stats = {}  # same structure as live_plots
        self._calc_derivative_and_stats = calc_derivative_and_stats

        # public options
        self.overplot = True
        self.noplot_streams = ["baseline"]
        self.omit_single_point_plot = True

        # public data
        self.peaks = PeakResults()

    def enable_plots(self):
        "Plot hinted fields from all streams not in ``noplot_streams``."
        self._plots_enabled = True

    def disable_plots(self):
        "Do not plot anything."
        self._plots_enabled = False

    def __call__(self, name, doc, *args, **kwargs):
        if not (self._table_enabled or self._baseline_enabled or self._plots_enabled):
            return
        super().__call__(name, doc, *args, **kwargs)

    def _set_up_plots(self, doc, stream_name, columns: list[Any]):
        """Using the descriptor doc"""
        plot_data = self._verify_plot_data(stream_name, columns)
        if not plot_data:
            return

        # This is a heuristic approach until we think of how to hint this in a
        # generalizable way.
        # 'time' once LivePlot can do that
        dim_fields: list[str] = self.dim_fields if stream_name == self.dim_stream else ["time"]

        ndims = len(dim_fields)
        if not 0 < ndims < 3:
            # we need 1 or 2 dims to do anything, do not make empty figures
            warn(  # noqa: B028
                "Plots are only made for 1 or 2 dimensions. "
                "Adjust the metadata hints field for BestEffortCallback to produce plots."
            )
            return
        # Create a figure or reuse an existing one.

        figure_name = self._get_figure_name(columns, dim_fields)

        figure = self._fig_factory(figure_name)

        axes = self._generate_axes(columns, ndims, figure)

        # ## LIVE PLOT AND PEAK ANALYSIS ## #

        if ndims == 1:
            self._live_plots[doc["uid"]] = {}
            self._peak_stats[doc["uid"]] = {}
            (x_key,) = dim_fields
            for y_key, ax in zip(columns, axes):
                dtype = doc["data_keys"][y_key]["dtype"]
                if dtype not in ("number", "integer"):
                    warn(
                        f"Omitting {y_key} from plot because dtype is {dtype}",
                        stacklevel=1,
                    )
                    continue
                # Create an instance of LivePlot and an instance of PeakStats.
                live_plot = LivePlotPlusPeaks(y=y_key, x=x_key, ax=ax, peak_results=self.peaks)
                live_plot("start", self._start_doc)
                live_plot("descriptor", doc)
                peak_stats = PeakStats(x=x_key, y=y_key, calc_derivative_and_stats=self._calc_derivative_and_stats)
                peak_stats("start", self._start_doc)
                peak_stats("descriptor", doc)

                # Stash them in state.
                self._live_plots[doc["uid"]][y_key] = live_plot
                self._peak_stats[doc["uid"]][y_key] = peak_stats

            for ax in axes[:-1]:
                ax.set_xlabel("")
        elif ndims == 2:
            # Decide whether to use LiveGrid or LiveScatter. LiveScatter is the
            # safer one to use, so it is the fallback..
            gridding = self._start_doc.get("hints", {}).get("gridding")
            if gridding == "rectilinear":
                self._live_grids[doc["uid"]] = {}
                slow, fast = dim_fields
                try:
                    extents = self._start_doc["extents"]
                    shape = self._start_doc["shape"]
                except KeyError:
                    warn(
                        "Need both 'shape' and 'extents' in plan metadata to create LiveGrid.",
                        stacklevel=1,
                    )
                else:
                    data_range = np.array([float(e[-1] - e[0]) for e in extents])
                    y_step, x_step = data_range / [max(1, s - 1) for s in shape]
                    adjusted_extent = [
                        extents[1][0] - x_step / 2,
                        extents[1][1] + x_step / 2,
                        extents[0][0] - y_step / 2,
                        extents[0][1] + y_step / 2,
                    ]
                    # MAGIC NUMBERS based on what tacaswell thinks looks OK
                    ASPECT_RATIO_REF_VALUE = 2
                    for I_key, ax in zip(columns, axes):
                        data_aspect_ratio = np.abs(data_range[1] / data_range[0])
                        aspect: Literal["auto", "equal"]
                        if 1 / ASPECT_RATIO_REF_VALUE < data_aspect_ratio < ASPECT_RATIO_REF_VALUE:
                            aspect = "equal"
                            ax.set_aspect(aspect, adjustable="box")
                        else:
                            aspect = "auto"
                            ax.set_aspect(aspect, adjustable="datalim")

                        live_grid = LiveGrid(
                            shape,
                            I_key,
                            xlabel=fast,
                            ylabel=slow,
                            extent=adjusted_extent,
                            aspect=aspect,
                            ax=ax,
                        )

                        live_grid("start", self._start_doc)
                        live_grid("descriptor", doc)
                        self._live_grids[doc["uid"]][I_key] = live_grid
            else:
                self._live_scatters[doc["uid"]] = {}
                x_key, y_key = dim_fields
                for I_key, ax in zip(columns, axes):
                    try:
                        extents = self._start_doc["extents"]
                    except KeyError:
                        xlim = ylim = None
                    else:
                        xlim, ylim = extents
                    live_scatter = LiveScatter(
                        x_key,
                        y_key,
                        I_key,
                        xlim=xlim,
                        ylim=ylim,
                        # Let clim autoscale.
                        ax=ax,
                    )
                    live_scatter("start", self._start_doc)
                    live_scatter("descriptor", doc)
                    self._live_scatters[doc["uid"]][I_key] = live_scatter

        else:
            raise NotImplementedError("we do not support 3D+ in BEC yet (and it should have bailed above)")

    def _generate_axes(self, columns, ndims, figure: Figure) -> list[Axes]:
        if not figure.axes:
            if len(columns) < 5:
                layout = (len(columns), 1)
            else:
                nrows = ncols = int(np.ceil(np.sqrt(len(columns))))
                while (nrows - 1) * ncols >= len(columns):
                    nrows -= 1
                layout = (nrows, ncols)
            if ndims == 1:
                share_kwargs = {"sharex": True}
            elif ndims == 2:
                share_kwargs = {"sharex": True, "sharey": True}
            else:
                raise NotImplementedError("we now support 3D?!")

            fig_size = np.array(layout[::-1]) * 5
            figure.set_size_inches(*fig_size)
            axes_grid = figure.subplots(*map(int, layout), **share_kwargs)  # type: ignore
            for ax in figure.axes[len(columns) :]:
                ax.set_visible(False)

            if len(figure.axes) > 1 and len(axes_grid.shape) == 2:
                # Axes go left to right, top to bottom, and will make some labels invisible
                for i in range(int(layout[1])):
                    if axes_grid[-1, i].get_visible() is False:
                        axes_grid[-2, i].tick_params(axis="x", labelbottom=True)

        axes = figure.axes
        return axes

    def _get_figure_name(self, columns, dim_fields: list[str]) -> str:
        fig_name = "{} vs {}".format(" ".join(sorted(columns)), " ".join(sorted(dim_fields)))
        if self.overplot and len(dim_fields) == 1:
            # If any open figure matches 'figname {number}', use it. If there
            # are multiple, the most recently touched one will be used.
            pattern1 = re.compile("^" + fig_name + "$")
            pattern2 = re.compile("^" + fig_name + r" \d+$")
            for label in plt.get_figlabels():
                if pattern1.match(label) or pattern2.match(label):
                    fig_name = label
                    break
        else:
            if plt.fignum_exists(fig_name):
                # Generate a unique name by appending a number.
                for number in itertools.count(2):
                    new_name = f"{fig_name} {number}"
                    if not plt.fignum_exists(new_name):
                        fig_name = new_name
                        break
        return fig_name

    def _verify_plot_data(self, stream_name, columns: list[Any]) -> bool:
        plot_data = True

        if not self._plots_enabled:
            plot_data = False
        if stream_name in self.noplot_streams:
            plot_data = False
        if not columns:
            plot_data = False
        if (
            (self._start_doc.get("num_points") == 1)
            and (stream_name == self.dim_stream)
            and self.omit_single_point_plot
        ):
            plot_data = False
        return plot_data

    def descriptor(self, doc):
        super().descriptor(doc)
        # ## DECIDE WHICH KIND OF PLOT CAN BE USED ## #
        self._set_up_plots(doc, doc.get("name", "primary"), self._columns(doc))

    def event(self, doc):
        super().event(doc)
        for y_key in doc["data"]:
            live_plot = self._live_plots.get(doc["descriptor"], {}).get(y_key)
            if live_plot is not None:
                live_plot("event", doc)
            live_grid = self._live_grids.get(doc["descriptor"], {}).get(y_key)
            if live_grid is not None:
                live_grid("event", doc)
            live_sc = self._live_scatters.get(doc["descriptor"], {}).get(y_key)
            if live_sc is not None:
                live_sc("event", doc)
            peak_stats = self._peak_stats.get(doc["descriptor"], {}).get(y_key)
            if peak_stats is not None:
                peak_stats("event", doc)

    def stop(self, doc):
        self._stop_table(doc)

        # Compute peak stats and build results container.
        ps_by_key = {}  # map y_key to PeakStats instance
        for peak_stats in self._peak_stats.values():
            for y_key, ps in peak_stats.items():
                ps("stop", doc)
                ps_by_key[y_key] = ps
        self.peaks.update(ps_by_key)

        for live_plots in self._live_plots.values():
            for live_plot in live_plots.values():
                live_plot("stop", doc)

        for live_grids in self._live_grids.values():
            for live_grid in live_grids.values():
                live_grid("stop", doc)

        for live_scatters in self._live_scatters.values():
            for live_scatter in live_scatters.values():
                live_scatter("stop", doc)

        self._print_end_baseline()

    def clear(self):
        super().clear()
        self._live_plots.clear()
        self._peak_stats.clear()
        self._live_grids.clear()
        self._live_scatters.clear()
        self.peaks.clear()

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
            emsg = f"Argument 'num_lines' (given as {num_lines}) must be >= 0."
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
                if len(tr._x) != 2 or len(tr._y) != 2 or (len(tr._x) == 2 and tr._x[0] != tr._x[1])
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
    ATTRS = ("com", "cen", "max", "min", "fwhm")

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
        raise KeyError(f"Keys are: {self.ATTRS}")

    def __repr__(self):
        # This is a proper eval-able repr, but with some manually-tweaked
        # whitespace to make it easier to parse.
        lines = []
        lines.append("{")
        for attr in self.ATTRS:
            lines.append(f"'{attr}':")
            for line in pformat(getattr(self, attr), width=1).split("\n"):
                lines.append(f"    {line}")
            lines.append(",")
        lines.append("}")
        return "\n".join(lines)


class LivePlotPlusPeaks(LivePlot):
    # Track state of axes, which may share instances of LivePlotPlusPeaks.
    __labeled: Mapping[Axis, bool] = weakref.WeakKeyDictionary()  # map ax to True/False
    __visible: Mapping[Axis, bool] = weakref.WeakKeyDictionary()  # map ax to True/False
    __instances: Mapping[Axis, list["LivePlotPlusPeaks"]] = (
        weakref.WeakKeyDictionary()
    )  # map ax to list of instances

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
                    if event.key == "P":
                        self.__visible[ax] = not self.__visible[ax]
                        for instance in self.__instances[ax]:
                            instance.check_visibility()

                ax.figure.canvas.mpl_connect("key_press_event", toggle)

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
        self.ax.legend(loc="best")
        self.ax.figure.canvas.draw_idle()

    def plot_annotations(self):
        styles = iter(cycler("color", "kr"))
        vlines = []
        for style, attr in zip(styles, ["cen", "com"]):
            val = self.peak_results[attr][self.y]
            # Only put labels in this legend once per axis.
            if self.ax in self.__labeled:
                label = "_no_legend_"
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
