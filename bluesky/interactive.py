import matplotlib.widgets as mwidgets
from bluesky.plans import grid_scan
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
    def creator_md(self):
        p1, p2 = self.last_rect
        local_md = {'interactive': {
            'widget_class': self.__class__.__name__,
            'rect_points': {'p1': p1, 'p2': p2}}}
        return local_md

    @property
    def active(self):
        'If the widget should be active'
        return self.widget.get_active()

    @active.setter
    def active(self, active):
        return self.widget.set_active(active)


class OuterProductWidget(ROIPlanCreator):
    def __init__(self, ax, dets, motor1, motor2, num1, num2=None,
                 snake=True, md=None):
        '''Create a 2D raster scan in the ROI bounding box

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to install the widget on

        dets : list
            List of detectors to read at each point

        motor1 : Positioner
            The 'slow' motor

        motor2 : Positioner
            The 'fast' motor

        num1 : int
            Number of steps to take along the slow direction

        num2 : int, optional
            Number of steps to take along the fast direction.  Defaults to
            ``num1``.

        snake : bool, optional
            Should the fast direction snake.  Defaults to `True`

        md : dict, optional
            Any extra metadata to pass through to the plan
        '''
        super().__init__(ax)
        self.motor1 = motor1
        self.motor2 = motor2
        self.dets = dets
        self.snake = snake
        if num2 is None:
            num2 = num1

        self.num1 = num1
        self.num2 = num2
        if md is None:
            md = {}
        self._md = md

    @property
    def md(self):
        return ChainMap(self._md, self.creator_md)

    def _plan(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        return grid_scan(self.dets,
                                  self.motor1, x1, x2, self.num1,
                                  self.motor2, y1, y2, self.num2,
                                  self.snake, md=self.md)
