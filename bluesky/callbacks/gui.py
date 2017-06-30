import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


class StackViewer(object):
    """
    Parameters
    ----------
    viewer : object
        expected to have update_image method and fig attribute
    images : array-like
        must support integer indexing and return a 2D array
    """

    def __init__(self, viewer, images=None, update_upon_add=True):
        self.update_upon_add = update_upon_add
        if images is None:
            images = []
        self.viewer = viewer
        self.images = images
        self.slider_ax = None
        self.slider = None
        length = len(self.images)
        fig = self.viewer._fig
        if length > 0:
            self.create_slider(length)
        fig.show()

    def update(self, val):
        if not isinstance(val, int):
            self.slider.set_val(int(round(val)))
            # sends up through 'update' again
        self.viewer.update_image(self.images[int(val)])

    def create_slider(self, length):
        if self.slider_ax:
            self.slider_ax.remove()
        self.slider_ax = fig.add_axes([0.1, 0.01, 0.8, 0.02])
        self.slider = Slider(self.slider_ax, 'Frame', 0, length - 1, 0,
                             valfmt='%d/{}'.format(length - 1))
        self.slider.on_changed(self.update)

    def add_image(self, img):
        self.images.append(img)
        self.create_slider(len(self.images))
        if self.update_upon_add:
            self.update(len(self.images) - 1)
            self.slider.set_val(len(self.images) - 1)

class CrossSection(object):
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
    """
    def __init__(self, fig, cmap=None, norm=None,
                 limit_func=None, auto_redraw=True, interpolation=None):

        self._cursor_position_cbs = []
        self._interpolation = interpolation
        # used to determine if setting properties should force a re-draw
        self._auto_redraw = auto_redraw
        # clean defaults
        if limit_func is None:
            limit_func = lambda image: (image.min(), image.max())
        if cmap is None:
            cmap = 'gray'
        # stash the color map
        self._cmap = cmap
        # let norm pass through as None, mpl defaults to linear which is fine
        if norm is None:
            norm = Normalize()
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
        #	      |   H cross section    |
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
        self._im_ax.set_aspect('equal')
        self._im_ax.xaxis.set_major_locator(NullLocator())
        self._im_ax.yaxis.set_major_locator(NullLocator())
        self._imdata = None
        self._im = self._im_ax.imshow([[]], cmap=self._cmap, norm=self._norm,
                        interpolation=self._interpolation,
                                      aspect='equal', vmin=0,
                                      vmax=1)

        # make it dividable
        divider = make_axes_locatable(self._im_ax)

        # set up all the other axes
        # (set up the horizontal and vertical cuts)
        self._ax_h = divider.append_axes('top', .5, pad=0.1,
                                         sharex=self._im_ax)
        self._ax_h.yaxis.set_major_locator(LinearLocator(numticks=2))
        self._ax_v = divider.append_axes('left', .5, pad=0.1,
                                         sharey=self._im_ax)
        self._ax_v.xaxis.set_major_locator(LinearLocator(numticks=2))
        self._ax_cb = divider.append_axes('right', .2, pad=.5)
        # add the color bar
        self._cb = fig.colorbar(self._im, cax=self._ax_cb)

        # add the cursor place holder
        self._cur = None

        # turn off auto-scale for the horizontal cut
        self._ax_h.autoscale(enable=False)

        # turn off auto-scale scale for the vertical cut
        self._ax_v.autoscale(enable=False)

        # create line artists
        self._ln_v, = self._ax_v.plot([],
                                      [], 'k-',
                                      animated=True,
                                      visible=False)

        self._ln_h, = self._ax_h.plot([],
                                      [], 'k-',
                                      animated=True,
                                      visible=False)

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
        """ Add a callback for the cursor position in the main axes

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
                            (self._ln_h.set_ydata, self._ln_v.set_xdata)):
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
        self._cur = Cursor(self._im_ax, useblit=True, color='red', linewidth=2)
        self._move_cid = self._fig.canvas.mpl_connect('motion_notify_event',
                                                      self._move_cb)

        self._click_cid = self._fig.canvas.mpl_connect('button_press_event',
                                                       self._click_cb)

        self._clear_cid = self._fig.canvas.mpl_connect('draw_event',
                                                       self._clear)
        self._fig.tight_layout()
        self._fig.canvas.draw_idle()

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

        for atr in ('_move_cid', '_clear_cid', '_click_cid'):
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
        self._im.set_extent([-0.5, im_shape[1] + .5,
                             im_shape[0] + .5, -0.5])

        # update the limits of the image axes to match the exent
        self._im_ax.set_xlim([-.05, im_shape[1] + .5])
        self._im_ax.set_ylim([im_shape[0] + .5, -0.5])

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
                return "X: {x:d} Y: {y:d} I: {i:.2f}".format(x=col, y=row, i=z)
            else:
                return "X: {x:d} Y: {y:d}".format(x=col, y=row)

        # replace the current format_coord function
        self._im_ax.format_coord = format_coord

        # net deal with the parasite axes and artist
        self._ln_v.set_data(np.zeros(im_shape[0]),
                            np.arange(im_shape[0]))
        self._ax_v.set_ylim([0, im_shape[0]])

        self._ln_h.set_data(np.arange(im_shape[1]),
                            np.zeros(im_shape[1]))
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
            self._cur.background = self._cur.canvas.copy_from_bbox(
                self._cur.canvas.figure.bbox)

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
        image = np.asarray(image)
        if self._imdata is None or self._imdata.shape != image.shape:
            self._init_artists(image)
        self._imdata = image
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
        self._fig.canvas.draw_idle()

    @auto_redraw
    def autoscale_horizontal(self, enable):
        self._ax_h.autoscale(enable=enable)

    @auto_redraw
    def autoscale_vertical(self, enable):
        self._ax_v.autoscale(enable=False)


if __name__ == '__main__':
    fig = plt.figure()
    imgs = (np.random.random((10, 10)) for i in range(10))
    v = CrossSection(fig, cmap='viridis')
    a = StackViewer(v)
    input()
    for img in imgs:
        a.add_image(img)
        input()