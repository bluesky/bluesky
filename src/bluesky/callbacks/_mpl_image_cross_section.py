import logging
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import LinearLocator, NullLocator
from matplotlib.widgets import Cursor
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)


class InterpolationEnum(Enum):
    NONE = ("none",)
    NEAREST = ("nearest",)
    BILINEAR = ("bilinear",)
    BICUBIC = ("bicubic",)
    SPLINE16 = ("spline16",)
    SPLINE36 = ("spline36",)
    HANNING = ("hanning",)
    HAMMING = ("hamming",)
    HERMITE = ("hermite",)
    KAISER = ("kaiser",)
    QUADRIC = ("quadric",)
    CATROM = ("catrom",)
    GAUSSIAN = ("gaussian",)
    BESSEL = ("bessel",)
    MITCHELL = ("mitchell",)
    SINC = ("sinc",)
    LANCZOS = ("lanczos",)


def auto_redraw(func):
    """
    decorator to automatically redraw the figure after a function call
    """

    def inner(self, *args, **kwargs):
        if self._fig.canvas is None:
            return
        force_redraw = kwargs.pop("force_redraw", None)
        if force_redraw is None:
            force_redraw = self._auto_redraw

        ret = func(self, *args, **kwargs)

        if force_redraw:
            self._update_artists()
            self._draw()

        return ret

    inner.__name__ = func.__name__
    inner.__doc__ = func.__doc__

    return inner


class CrossSection:
    """
    Class to manage the axes, artists and properties associated with
    showing a 2D image, a cross-hair cursor and two parasite axes which
    provide horizontal and vertical cross sections of image.

    You will likely need to call `CrossSection.init_artists(init_image)` after
    creating this object.

    Parameters
    ----------

    fig : matplotlib.figure.Figure
        The figure object to build the class on, will clear
        current contents

    cmap : str,  colormap, or None
        color map to use.  Defaults to gray

    norm : Normalize or None
       Normalization function to use

    limit_func : callable, optional
        function that takes in the image and returns clim values
    auto_redraw : bool, optional
    interpolation : str, optional
        Interpolation method to use. List of valid options can be found in
        CrossSection2DView.interpolation
    aspect : str, optional
        Aspect passed into imshow, defaults to equal

    Properties
    ----------
    interpolation : str
        The stringly-typed pixel interpolation. See _INTERPOLATION attribute
        of this cross_section_2d module
    cmap : str
        The colormap to use for rendering the image


    """

    def __init__(
        self,
        figure: Figure,
        colormap: Optional[Colormap] = None,
        norm=None,
        limit_func: Optional[Callable[[Optional[Any]], Callable[[Any], tuple]]] = None,
        auto_redraw=True,
        interpolation: InterpolationEnum = InterpolationEnum.NONE,
        aspect="equal",
    ):
        self._cursor_position_cbs: List[Any] = []
        self._interpolation = interpolation
        # used to determine if setting properties should force a re-draw
        self._auto_redraw = auto_redraw
        # clean defaults
        if limit_func is None:
            # note - cannot import this in the defaults init
            limit_func = percentile_limit_factory([5.0, 95.0])
        # stash the color map
        self._cmap = colormap
        # set the default norm if not passed
        if norm is None:
            norm = Normalize()
        # always set the vmin/vmax as we can not auto-limit with an empty array
        # below.  When we set the data we are going to fully rescale this
        # anyway.
        norm.vmin, norm.vmax = 0, 1
        self._norm = norm
        # save a copy of the limit function, we will need it later
        self._limit_func: Callable[[Optional[Any]], Callable[[Any], tuple]] = limit_func

        # this is used by the widget logic
        self._active = True
        self._dirty = True
        self._cb_dirty = True

        # work on setting up the mpl axes

        self._figure = figure
        # blow away what ever is currently on the figure
        figure.clf()
        # Configure the figure in our own image
        #
        #     	  +----------------------+
        #         |   H cross section    |
        #     	  +----------------------+
        #   +---+ +----------------------+
        #   | V | |                      |
        #   |   | |                      |
        #   | x | |                      |
        #   | s | |      Main Axes       |
        #   | e | |                      |
        #   | c | |                      |
        #   | t | |                      |
        #   | i | |                      |
        #   | o | |                      |
        #   | n | |                      |
        #   +---+ +----------------------+

        # make the main axes
        self._image_axes = figure.add_subplot(1, 1, 1)
        self._image_axes.set_aspect(aspect)
        self._image_axes.xaxis.set_major_locator(NullLocator())
        self._image_axes.yaxis.set_major_locator(NullLocator())
        self._imdata = None
        self._im = self._image_axes.imshow(
            [[]],
            cmap=self._cmap,
            norm=self._norm,
            interpolation=str(self._interpolation.value),
            aspect=aspect,
        )

        # make it dividable
        divider = make_axes_locatable(self._image_axes)

        # set up all the other axes
        # (set up the horizontal and vertical cuts)
        self._ax_h = divider.append_axes("top", 0.5, pad=0.1, sharex=self._image_axes)
        self._ax_h.yaxis.set_major_locator(LinearLocator(numticks=2))
        self._ax_v = divider.append_axes("left", 0.5, pad=0.1, sharey=self._image_axes)
        self._ax_v.xaxis.set_major_locator(LinearLocator(numticks=2))
        self._ax_cb = divider.append_axes("right", 0.2, pad=0.5)
        # add the color bar
        self._cb = figure.colorbar(self._im, cax=self._ax_cb)

        # add the cursor place holder
        self._cursor = None

        # turn off auto-scale for the horizontal cut
        self._ax_h.autoscale(enable=False)

        # turn off auto-scale scale for the vertical cut
        self._ax_v.autoscale(enable=False)

        # create line artists
        (self._ln_v,) = self._ax_v.plot([], [], "k-", animated=True, visible=False)

        (self._ln_h,) = self._ax_h.plot([], [], "k-", animated=True, visible=False)

        # backgrounds for blitting
        self._ax_v_bk = None
        self._ax_h_bk = None

        # stash last-drawn row/col to skip if possible
        self._row = None
        self._col = None

        # make attributes for callback ids
        self._move_cid = None
        self._click_cid = None
        self._clear_cid = None

    def add_cursor_position_cb(self, callback) -> None:
        """Add a callback for the cursor position in the main axes

        Parameters
        ----------
        callback : callable(cc, rr)
            Function that gets called when the cursor position moves to a new
            row or column on main axes
        """
        self._cursor_position_cbs.append(callback)

    # set up the call back for the updating the side axes
    def _move_cb(self, event):
        if not self._active:
            return
        if event is None:
            x = self._col
            y = self._row
            self._col = None
            self._row = None
        else:
            # short circuit on other axes
            if event.inaxes is not self._image_axes:
                return
            x, y = event.xdata, event.ydata
        numrows, numcols = self._imdata.shape
        if x is not None and y is not None:
            self._ln_h.set_visible(True)
            self._ln_v.set_visible(True)
            col = int(x + 0.5)
            row = int(y + 0.5)
            if row != self._row or col != self._col:
                if 0 <= col < numcols and 0 <= row < numrows:
                    self._col = col
                    self._row = row
                    for cb in self._cursor_position_cbs:
                        cb(col, row)
                    for data, ax, bkg, art, set_fun in zip(
                        (self._imdata[row, :], self._imdata[:, col]),
                        (self._ax_h, self._ax_v),
                        (self._ax_h_bk, self._ax_v_bk),
                        (self._ln_h, self._ln_v),
                        (self._ln_h.set_ydata, self._ln_v.set_xdata),
                    ):
                        self._figure.canvas.restore_region(bkg)
                        set_fun(data)
                        ax.draw_artist(art)
                        self._figure.canvas.blit(ax.bbox)

    def _click_cb(self, event):
        if event.inaxes is not self._image_axes:
            return
        self.active = not self.active
        if self.active:
            self._cursor.onmove(event)
            self._move_cb(event)

    @auto_redraw
    def _connect_callbacks(self):
        """
        Connects all of the callbacks for the motion and click events
        """
        self._disconnect_callbacks()
        self._cursor = Cursor(self._image_axes, useblit=True, color="red", linewidth=2)
        self._move_cid = self._figure.canvas.mpl_connect(
            "motion_notify_event", self._move_cb
        )

        self._click_cid = self._figure.canvas.mpl_connect(
            "button_press_event", self._click_cb
        )

        self._clear_cid = self._figure.canvas.mpl_connect("draw_event", self._clear)
        self._figure.tight_layout()
        self._figure.canvas.draw()

    def _disconnect_callbacks(self):
        """
        Disconnects all of the callbacks
        """
        if self._figure.canvas is None:
            # no canvas -> can't do anything about the call backs which
            # should not exist
            self._move_cid = None
            self._clear_cid = None
            self._click_cid = None
            return

        for atr in ("_move_cid", "_clear_cid", "_click_cid"):
            cid = getattr(self, atr, None)
            if cid is not None:
                self._figure.canvas.mpl_disconnect(cid)
                setattr(self, atr, None)

        # clean up the cursor
        if self._cursor is not None:
            self._cursor.disconnect_events()
            del self._cursor
            self._cursor = None

    @auto_redraw
    def _init_artists(self, init_image):
        """
        Update the CrossSection with a new base-image.  This function
        takes care of setting up all of the details about the image size
        in the limits/artist extent of the image and the secondary data
        in the cross-section parasite plots.

        Parameters
        ----------
        init_image : ndarray
           An image to serve as the new 'base' image.
        """

        im_shape = init_image.shape

        # first deal with the image axis
        # update the image, `update_artists` takes care of
        # updating the actual artist
        self._imdata = init_image

        # update the extent of the image artist
        self._im.set_extent([-0.5, im_shape[1] + 0.5, im_shape[0] + 0.5, -0.5])

        # update the limits of the image axes to match the exent
        self._image_axes.set_xlim([-0.05, im_shape[1] + 0.5])
        self._image_axes.set_ylim([im_shape[0] + 0.5, -0.5])

        # update the format coords printer
        numrows, numcols = im_shape

        # note, this is a closure over numrows and numcols
        def format_coord(x: float | int, y: float | int) -> str:
            # adjust xy -> col, row
            col = int(x + 0.5)
            row = int(y + 0.5)
            point_falls_inside_array: bool = (
                col >= 0 and col < numcols and row >= 0 and row < numrows
            )
            if point_falls_inside_array and self._imdata is not None:
                # if it does, grab the value
                z = self._imdata[row, col]
                return f"X: {col:d} Y: {row:d} I: {z:.2f}"
            return f"X: {col:d} Y: {row:d}"

        # replace the current format_coord function
        self._image_axes.format_coord = format_coord

        # net deal with the parasite axes and artist
        self._ln_v.set_data(np.zeros(im_shape[0]), np.arange(im_shape[0]))
        self._ax_v.set_ylim([0, im_shape[0]])

        self._ln_h.set_data(np.arange(im_shape[1]), np.zeros(im_shape[1]))
        self._ax_h.set_xlim([0, im_shape[1]])

        # if we have a cavas, then connect/set up junk
        if self._figure.canvas is not None:
            self._connect_callbacks()
        # mark as dirty
        self._dirty = True

    def _clear(self, event):
        self._ax_v_bk = self._figure.canvas.copy_from_bbox(self._ax_v.bbox)
        self._ax_h_bk = self._figure.canvas.copy_from_bbox(self._ax_h.bbox)
        self._ln_h.set_visible(False)
        self._ln_v.set_visible(False)
        # this involves reaching in and touching the guts of the
        # cursor widget.  The problem is that the mpl widget
        # skips updating it's saved background if the widget is inactive
        if self._cursor:
            self._cursor.background = self._cursor.canvas.copy_from_bbox(
                self._cursor.canvas.figure.bbox
            )

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, val):
        self._active = val
        self._cursor.active = val

    @auto_redraw
    def update_interpolation(self, interpolation):
        """
        Set the interpolation method

        """
        self._dirty = True
        self._im.set_interpolation(interpolation)

    @auto_redraw
    def update_cmap(self, cmap):
        """
        Set the color map used
        """
        # TODO: this should stash new value, not apply it
        self._cmap = cmap
        self._dirty = True

    @auto_redraw
    def update_image(self, image):
        """
        Set the image data

        The input data does not necessarily have to be the same shape as the
        original image
        """
        if self._imdata is None or self._imdata.shape != image.shape:
            self._init_artists(image)
        self._imdata = image
        self._move_cb(None)
        self._dirty = True

    @auto_redraw
    def update_norm(self, norm):
        """
        Update the way that matplotlib normalizes the image
        """
        self._norm = norm
        self._dirty = True
        self._cb_dirty = True

    @auto_redraw
    def update_limit_func(self, limit_func):
        """
        Set the function to use to determine the color scale
        """
        # set the new function to use for computing the color limits
        self._limit_func = limit_func
        self._dirty = True

    def _update_artists(self):
        """
        Updates the figure by re-drawing
        """
        # if the figure is not dirty, short-circuit
        if not (self._dirty or self._cb_dirty):
            return

        # this is a tuple which is the max/min used in the color mapping.
        # these values are also used to set the limits on the value
        # axes of the parasite axes
        # value_limits
        vlim = self._limit_func(self._imdata)
        # set the color bar limits
        self._im.set_clim(vlim)
        self._norm.vmin, self._norm.vmax = vlim
        # set the cross section axes limits
        self._ax_v.set_xlim(*vlim[::-1])
        self._ax_h.set_ylim(*vlim)
        # set the imshow data
        self._im.set_cmap(self._cmap)
        self._im.set_norm(self._norm)
        if self._imdata is None:
            return
        self._im.set_data(self._imdata)
        # TODO if cb_dirty, remake the colorbar, I think this is
        # why changing the norm does not play well
        self._dirty = False
        self._cb_dirty = False

    def _draw(self):
        self._figure.canvas.draw()

    @auto_redraw
    def autoscale_horizontal(self, enable):
        self._ax_h.autoscale(enable=enable)

    @auto_redraw
    def autoscale_vertical(self, enable):
        self._ax_v.autoscale(enable=False)

    def draw_idle(self):
        self._figure.canvas.draw_idle()


# for Python 3.12 - the type to all 3 of the functions could be
# type LimitFactory = Callable[[Any | None], Callable[[Any], Tuple[np.ndarray]]]


def fullrange_limit_factory(limit_args=None):
    """
    Factory for returning full-range limit functions

    limit_args is ignored.
    """

    def _full_range(im):
        """
        Plot the entire range of the image

        Parameters
        ----------
        im : ndarray
           image data, nominally 2D

        limit_args : object
           Ignored, here to match signature with other
           limit functions

        Returns
        -------
        climits : tuple
           length 2 tuple to be passed to `im.clim(...)` to
           set the color limits of a ColorMappable object.
        """
        return (np.nanmin(im), np.nanmax(im))

    return _full_range


def absolute_limit_factory(limit_args):
    """
    Factory for making absolute limit functions
    """

    def _absolute_limit(im):
        """
        Plot the image based on the min/max values in limit_args

        This function is a no-op and just return the input limit_args.

        Parameters
        ----------
        im : ndarray
            image data.  Ignored in this method

        limit_args : array
           (min_value, max_value)  Values are in absolute units
           of the image.

        Returns
        -------
        climits : tuple
           length 2 tuple to be passed to `im.clim(...)` to
           set the color limits of a ColorMappable object.

        """
        return limit_args

    return _absolute_limit


def percentile_limit_factory(limit_args: Tuple[np.ndarray]):
    """
    Factory to return a percentile limit function
    """

    def _percentile_limit(image_data: np.ndarray):
        """
        Sets limits based on percentile.

        Parameters
        ----------
        im : ndarray
            image data

        limit_args : tuple of floats in [0, 100]
            upper and lower percetile values

        Returns
        -------
        climits : tuple
           length 2 tuple to be passed to `im.clim(...)` to
           set the color limits of a ColorMappable object.

        """
        return np.percentile(image_data, limit_args)

    return _percentile_limit
