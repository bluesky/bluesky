import matplotlib.widgets as mwidgets
from bluesky.plans import outer_product_scan
from collections import ChainMap


class ROIPlanCreator:
    def __init__(self, ax):
        '''Base class for interactively creating plans in a 2D bounding box.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to install the widget on
        '''
        self.ax = ax
        self.widget = mwidgets.RectangleSelector(
            self.ax, self._onselect, useblit=True, interactive=True)
        self._pt1 = self._pt2 = None

    def _onselect(self, pt1, pt2):
        self._pt1 = pt1
        self._pt2 = pt2

    @property
    def last_rect(self):
        '''data coordinates of the first and second corners.

        Coordinates are lower left and upper right.

        Returns
        -------
        points : tuple
           ((x1, y1), (x2, y2))
        '''
        # TODO deal with ordering issues
        x1, y1 = self._pt1.xdata, self._pt1.ydata
        x2, y2 = self._pt2.xdata, self._pt2.ydata
        return (x1, y1), (x2, y2)

    @property
    def last_plan(self):
        '''Plan constructed from the last rectangle

        Returns
        -------
        plan : iterable
        '''
        return self._plan(*self.last_rect)

    def _plan(self, pt1, pt2):
        '''Construct a plan given the ROI

        Sub-classes must override this

        Returns
        -------
        plan
        '''
        raise NotImplementedError()

    @property
    def active(self):
        'If the widget should be active'
        return self.widget.get_active()

    @active.setter
    def active(self, active):
        return self.widget.set_active(active)


class OuterProductWidget(ROIPlanCreator):
    def __init__(self, ax, dets, m1, m2, numsteps1, numsteps2=None,
                 snake=True, md=None):
        '''Create a 2D raster scan in the ROI bounding box

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to install the widget on

        dets : list
            List of detectors to read at each point

        m1 : Positioner
            The 'slow' motor

        m2 : Positioner
            The 'fast' motor

        numsteps1 : int
            Number of steps to take along the slow direction

        numsteps2 : int, optional
            Number of steps to take along the fast direction.  Defaults to
            ``numsteps1``.

        snake : bool, optional
            Should the fast direction snake.  Defaults to `True`

        md : dict, optional
            Any extra metadata to pass through to the plan
        '''
        super().__init__(ax)
        self.m1 = m1
        self.m2 = m2
        self.dets = dets
        self.snake = snake
        if numsteps2 is None:
            numsteps2 = numsteps1

        self.numsteps1 = numsteps1
        self.numsteps2 = numsteps2
        if md is None:
            md = {}
        self.md = md

    def _plan(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        local_md = {'interactive': {
            'created_by': self.__class__.__name__,
            'rect_points': {'p1': p1, 'p2': p2}}}

        md = ChainMap(self.md, local_md)

        return outer_product_scan(self.dets,
                                  self.m1, x1, x2, self.numsteps1,
                                  self.m2, y1, y2, self.numsteps2,
                                  self.snake, md=md)
