import hashlib
import sys
import time
import numpy as np

from functools import partial
from typing import Any, List, Optional, TextIO

from IPython.core.display import display, HTML
from ipywidgets import HBox
from tqdm.notebook import tqdm as tqdm_nb
from threading import RLock

from bluesky.utils import ProgressBarBase, ProgressBarManager, _L2norm


class NotebookProgressBar(ProgressBarBase):
    containers: List[HBox]
    status_objs: List[Any]
    fp: TextIO
    creation_time: float
    lock: RLock

    def __init__(self,
                 status_objs: List[Any],
                 delay_draw: float = 0.2):
        """
        Represents status objects with Jupyter Notebook progress bars.

        Parameters
        ----------
        status_objs : list
            Status objects
        """
        self.containers = []
        self.status_objs = []
        self.fp = sys.stdout
        self.creation_time = time.time()
        self.delay_draw = delay_draw
        self.lock = RLock()

        # Create a closure over self.update for each status object that
        # implemets the 'watch' method.
        for st in status_objs:
            with self.lock:
                if hasattr(st, "watch") and not st.done:
                    pos = len(self.containers)
                    container: HBox = tqdm_nb.status_printer(self.fp, 1, None, None)
                    display(container)
                    self.containers.append(container)
                    self.status_objs.append(st)
                    st.watch(partial(self.update, pos))

    def update(
            self,
            pos: Any,
            *,
            name: str = None,
            current: Any = None,
            initial: Any = None,
            target: Any = None,
            unit: str = "units",
            precision: Any = None,
            fraction: Any = None,
            time_elapsed: float = None,
            time_remaining: float = None,
    ):
        if all(x is not None for x in (current, initial, target)):
            # In this case there is enough information to draw a progress bar with
            # current and maximum values
            total = round(_L2norm(target, initial), precision or 3)
            n = np.clip(round(_L2norm(current, initial), precision or 3), 0, total)
            if time_elapsed is None:
                time_elapsed = time.time() - self.creation_time
            meta = tqdm_nb.format_meter(
                n=n,
                total=total,
                elapsed=time_elapsed,
                unit=unit,
                prefix=name,
                ncols=79,
                bar_format="{r_bar}",
            )
            self.draw(
                pos,
                name or "",
                meta,
                color=html_color_from_name(name or ""),
                total=total,
                value=n,
            )
        elif self.status_objs[pos].done:
            # In this case the operation is complete, we just
            # display a finished progress bar
            # and a completion message.
            self.draw(
                pos,
                name or "",
                "[Complete.]",
                color=html_color_from_name(name or ""),
                value=1,
            )
        else:
            # In this case there is not enough information to draw a progress bar so
            # we just display a message.
            self.draw(pos, name or "", "[In progress. No progress bar available.]")

    def draw(
            self,
            pos: int,
            label: str = "",
            meta: str = "",
            color: str = "#97d4e8",
            total: float = 1.0,
            value: Optional[float] = None,
    ) -> None:
        """
        Draws the progress bar or a message if there is
        not enough information

        Parameters
        ----------
        pos : int
            The index of the progress bar in self.containers
        label : str, optional
            The label to go on the left-hand side of the progress bar, by default ""
        meta : str, optional
            Any metadata to go on the right-hand side of the progress bar, by default ""
        color : str, optional
            The color of the progress bar as a valid HTML/CSS string, by default "#97d4e8"
        total : float, optional
            The maximum value of the progress bar, defaults to 1, by default 1.0
        value : Optional[float], optional
            The current value of the progress bar, optional, by default None
        """

        # Do not draw if this was only created recently, otherwise it
        # will just flash up and disappear
        if (time.time() - self.creation_time) < self.delay_draw:
            return

        container = self.containers[pos]
        with self.lock:
            ltext, pbar, rtext = container.children
            ltext.value = label
            pbar.layout.display = "flex"
            rtext.value = meta

            if value is not None:
                pbar.value = value
                pbar.max = total or 1
                pbar.style.bar_color = color
            else:
                pbar.layout.display = "none"

    def clear(self) -> None:
        """
        Clears all progress bars
        """

        with self.lock:
            for container in self.containers:
                container.close()
        hide_leftover_progress_bars()


def html_color_from_name(name: str) -> str:
    """
    Transforms a string into a unique, bright HTML color
    (hue varies, saturation and value are fixed).
    Transformation is pure.

    Parameters
    ----------
    name : str
        The string to convert

    Returns
    -------
    str
        A string describing a color, valid in HTML/CSS.
    """

    hue = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % 360
    return f"hsl({hue}, 90%, 55%)"


def hide_leftover_progress_bars() -> None:
    """
    Sets the style in Jupyterlab to remove leftover whitespace
    from deleted progress bars. Workaround for:
    https://github.com/jupyterlab/jupyterlab/issues/7354
    """

    display(
        HTML("""
            <style>
                .p-Widget.jp-OutputPrompt.jp-OutputArea-prompt:empty {
                    padding: 0;
                    border: 0;
                }
            </style>
        """)
    )


def pbar_manager_for_notebook(delay_draw: float = 0.2) -> ProgressBarManager:
    """
    Helper method for generating managers when using notebooks.

    Returns
    -------
    ProgressBarManager
        A manager that creates progress bars for Jupuyter notebooks
    """

    return ProgressBarManager(partial(NotebookProgressBar, delay_draw=delay_draw))
