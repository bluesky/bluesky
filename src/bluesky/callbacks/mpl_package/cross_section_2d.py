# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################

import logging

import numpy as np
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import LinearLocator, NullLocator
from matplotlib.widgets import Cursor
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .. import AbstractDataView2D
from . import AbstractMPLDataView

logger = logging.getLogger(__name__)


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


def percentile_limit_factory(limit_args):
    """
    Factory to return a percentile limit function
    """

    def _percentile_limit(im):
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
        return np.percentile(im, limit_args)

    return _percentile_limit


_INTERPOLATION = [
    "none",
    "nearest",
    "bilinear",
    "bicubic",
    "spline16",
    "spline36",
    "hanning",
    "hamming",
    "hermite",
    "kaiser",
    "quadric",
    "catrom",
    "gaussian",
    "bessel",
    "mitchell",
    "sinc",
    "lanczos",
]


class CrossSection2DView(AbstractDataView2D, AbstractMPLDataView):
    """
    CrossSection2DView docstring

    """

    # list of valid options for the interpolation parameter. The first one is
    # the default value.
    interpolation = _INTERPOLATION

    def __init__(
        self,
        base_figure: Figure,
        data_list,
        key_list,
        cmap: str | Colormap | None = None,
        normalization_function: Normalize | None = None,
        limit_func=None,
        interpolation=None,
        **kwargs,
    ):
        """
        Sets up figure with cross section viewer

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object to build the class on, will clear
            current contents
        clim_percentile : float or None
           percentile away from 0, 100 to put the max/min limits at
           ie, clim_percentile=5 -> vmin=5th percentile vmax=95th percentile
        interpolation : str, optional
            Interpolation method to use. List of valid options can be found in
            CrossSection2DView.interpolation
        """
        # call up the inheritance chain
        super().__init__(
            fig=base_figure, data_list=data_list, key_list=key_list, norm=normalization_function, cmap=cmap
        )
        self._xsection = CrossSection(
            base_figure, cmap=self._cmap, norm=self._norm, limit_func=limit_func, interpolation=interpolation
        )

    def update_cmap(self, cmap):
        self._xsection.update_cmap(cmap)

    def update_image(self, img_idx):
        self._xsection.update_image(self._data_dict[self._key_list[img_idx]])

    def replot(self):
        """
        Update the image displayed by the main axes

        Parameters
        ----------
        new_image : 2D ndarray
           The new image to use
        """
        self._xsection._update_artists()

    def update_norm(self, new_norm):
        """
        Update the way that matplotlib normalizes the image. Default is linear
        """
        self._xsection.update_norm(new_norm)

    def set_limit_func(self, limit_func):
        """
        Set the function to use to determine the color scale

        """
        self._xsection.update_limit_func(limit_func)

    def update_interpolation(self, interpolation):
        """
        Update the way that matplotlib interpolates the image. Default is none

        Parameters
        ----------
        interpolation : str
            Interpolation method to use. List of valid options can be found in
            CrossSection2DView.interpolation
        """
        self._xsection.update_interpolation(interpolation)


def auto_redraw(func):
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
        self, fig, cmap=None, norm=None, limit_func=None, auto_redraw=True, interpolation=None, aspect="equal"
    ):
        self._cursor_position_cbs = []
        if interpolation is None:
            interpolation = _INTERPOLATION[0]
        self._interpolation = interpolation
        # used to determine if setting properties should force a re-draw
        self._auto_redraw = auto_redraw
        # clean defaults
        if limit_func is None:
            limit_func = fullrange_limit_factory()
        if cmap is None:
            cmap = "gray"
        # stash the color map
        self._cmap = cmap
        # set the default norm if not passed
        if norm is None:
            norm = Normalize()
        # always set the vmin/vmax as we can not auto-limit with an empty array
        # below.  When we set the data we are going to fully rescale this
        # anyway.
        norm.vmin, norm.vmax = 0, 1
        self._norm = norm
        # save a copy of the limit function, we will need it later
        self._limit_func = limit_func

        # this is used by the widget logic
        self._active = True
        self._dirty = True
        self._cb_dirty = True

        # work on setting up the mpl axes

        self._fig = fig
        # blow away what ever is currently on the figure
        fig.clf()
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
        self._im_ax = fig.add_subplot(1, 1, 1)
        self._im_ax.set_aspect(aspect)
        self._im_ax.xaxis.set_major_locator(NullLocator())
        self._im_ax.yaxis.set_major_locator(NullLocator())
        self._imdata = None
        self._im = self._im_ax.imshow(
            [[]],
            cmap=self._cmap,
            norm=self._norm,
            interpolation=self._interpolation,
            aspect=aspect,
        )

        # make it dividable
        divider = make_axes_locatable(self._im_ax)

        # set up all the other axes
        # (set up the horizontal and vertical cuts)
        self._ax_h = divider.append_axes("top", 0.5, pad=0.1, sharex=self._im_ax)
        self._ax_h.yaxis.set_major_locator(LinearLocator(numticks=2))
        self._ax_v = divider.append_axes("left", 0.5, pad=0.1, sharey=self._im_ax)
        self._ax_v.xaxis.set_major_locator(LinearLocator(numticks=2))
        self._ax_cb = divider.append_axes("right", 0.2, pad=0.5)
        # add the color bar
        self._cb = fig.colorbar(self._im, cax=self._ax_cb)

        # add the cursor place holder
        self._cur = None

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

    def add_cursor_position_cb(self, callback):
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
            if event.inaxes is not self._im_ax:
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
                        self._fig.canvas.restore_region(bkg)
                        set_fun(data)
                        ax.draw_artist(art)
                        self._fig.canvas.blit(ax.bbox)

    def _click_cb(self, event):
        if event.inaxes is not self._im_ax:
            return
        self.active = not self.active
        if self.active:
            self._cur.onmove(event)
            self._move_cb(event)

    @auto_redraw
    def _connect_callbacks(self):
        """
        Connects all of the callbacks for the motion and click events
        """
        self._disconnect_callbacks()
        self._cur = Cursor(self._im_ax, useblit=True, color="red", linewidth=2)
        self._move_cid = self._fig.canvas.mpl_connect("motion_notify_event", self._move_cb)

        self._click_cid = self._fig.canvas.mpl_connect("button_press_event", self._click_cb)

        self._clear_cid = self._fig.canvas.mpl_connect("draw_event", self._clear)
        self._fig.tight_layout()
        self._fig.canvas.draw()

    def _disconnect_callbacks(self):
        """
        Disconnects all of the callbacks
        """
        if self._fig.canvas is None:
            # no canvas -> can't do anything about the call backs which
            # should not exist
            self._move_cid = None
            self._clear_cid = None
            self._click_cid = None
            return

        for atr in ("_move_cid", "_clear_cid", "_click_cid"):
            cid = getattr(self, atr, None)
            if cid is not None:
                self._fig.canvas.mpl_disconnect(cid)
                setattr(self, atr, None)

        # clean up the cursor
        if self._cur is not None:
            self._cur.disconnect_events()
            del self._cur
            self._cur = None

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
        self._im_ax.set_xlim([-0.05, im_shape[1] + 0.5])
        self._im_ax.set_ylim([im_shape[0] + 0.5, -0.5])

        # update the format coords printer
        numrows, numcols = im_shape

        # note, this is a closure over numrows and numcols
        def format_coord(x, y):
            # adjust xy -> col, row
            col = int(x + 0.5)
            row = int(y + 0.5)
            # make sure the point falls in the array
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                # if it does, grab the value
                z = self._imdata[row, col]
                return f"X: {col:d} Y: {row:d} I: {z:.2f}"
            else:
                return f"X: {col:d} Y: {row:d}"

        # replace the current format_coord function
        self._im_ax.format_coord = format_coord

        # net deal with the parasite axes and artist
        self._ln_v.set_data(np.zeros(im_shape[0]), np.arange(im_shape[0]))
        self._ax_v.set_ylim([0, im_shape[0]])

        self._ln_h.set_data(np.arange(im_shape[1]), np.zeros(im_shape[1]))
        self._ax_h.set_xlim([0, im_shape[1]])

        # if we have a cavas, then connect/set up junk
        if self._fig.canvas is not None:
            self._connect_callbacks()
        # mark as dirty
        self._dirty = True

    def _clear(self, event):
        self._ax_v_bk = self._fig.canvas.copy_from_bbox(self._ax_v.bbox)
        self._ax_h_bk = self._fig.canvas.copy_from_bbox(self._ax_h.bbox)
        self._ln_h.set_visible(False)
        self._ln_v.set_visible(False)
        # this involves reaching in and touching the guts of the
        # cursor widget.  The problem is that the mpl widget
        # skips updating it's saved background if the widget is inactive
        if self._cur:
            self._cur.background = self._cur.canvas.copy_from_bbox(self._cur.canvas.figure.bbox)

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, val):
        self._active = val
        self._cur.active = val

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
        self._fig.canvas.draw()

    @auto_redraw
    def autoscale_horizontal(self, enable):
        self._ax_h.autoscale(enable=enable)

    @auto_redraw
    def autoscale_vertical(self, enable):
        self._ax_v.autoscale(enable=False)
