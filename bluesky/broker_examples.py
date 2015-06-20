import numpy as np
from .examples import Reader
from .run_engine import Msg
from filestore.file_writers import save_ndarray
import tempfile
from .examples import *


class SynGauss2D(Reader):
    """
    Evaluate a point on a Gaussian based on the value of a motor.

    Example
    -------
    motor = Mover('motor', ['motor'])
    det = SynGauss('det', motor, 'motor', center=0, Imax=1, sigma=1)
    """
    _klass = 'reader'

    def __init__(self, name, motor, motor_field, center, Imax=1000, sigma=1,
                 nx=250, ny=250, img_sigma=50):
        super(SynGauss2D, self).__init__(name, [name, ])
        self.ready = True
        self._motor = motor
        self._motor_field = motor_field
        self.center = center
        self.Imax = Imax
        self.sigma = sigma
        self.dims = (nx, ny)
        self.img_sigma = img_sigma
        # stash these things in a temp directory. This might cause an
        # exception to be raised if/when the file system cleans its temp files
        self.output_dir = tempfile.gettempdir()

    def trigger(self, *, block_group=True):
        self.ready = False
        m = self._motor._data[self._motor_field]['value']
        v = self.Imax * np.exp(-(m - self.center)**2 / (2 * self.sigma**2))
        arr = self.gauss(self.dims, self.img_sigma) * v + np.random.random(
            self.dims) * .01
        fs_uid = save_ndarray(arr, self.output_dir)
        self._data = {self._name: {'value': fs_uid, 'timestamp': ttime.time()}}
        ttime.sleep(0.05)  # simulate exposure time
        self.ready = True
        return self

    def read(self):
        return self._data

    def _dist(self, dims):
        """
        Create array with pixel value equals to the distance from array center.

        Parameters
        ----------
        dims : list or tuple
            shape of array to create

        Returns
        -------
        arr : np.ndarray
            ND array whose pixels are equal to the distance from the center
            of the array of shape `dims`
        """
        dist_sum = []
        shape = np.ones(len(dims))
        for idx, d in enumerate(dims):
            vec = (np.arange(d) - d // 2) ** 2
            shape[idx] = -1
            vec = vec.reshape(*shape)
            shape[idx] = 1
            dist_sum.append(vec)

        return np.sqrt(np.sum(dist_sum, axis=0))

    def gauss(self, dims, sigma):
        """
        Generate Gaussian function in 2D or 3D.

        Parameters
        ----------
        dims : list or tuple
            shape of the data
        sigma : float
            standard deviation of gaussian function

        Returns
        -------
        Array :
            ND gaussian
        """
        x = self._dist(dims)
        y = np.exp(-(x / sigma)**2 / 2)
        return y / np.sum(y)

    def describe(self):
        return {self._name: {'source': self._name,
                             'dtype': 'array',
                             'shape': list(self.dims),
                             'external': 'FILESTORE:'}}


det_2d = SynGauss2D('det_2d', motor, 'motor', center=0, Imax=1000, sigma=1,
                    nx=300, ny=300)
det1_2d = SynGauss2D('det1_2d', motor1, 'motor1', center=0, Imax=10,
                     sigma=1, nx=100, ny=600)
det2_2d = SynGauss2D('det2_2d', motor2, 'motor2', center=1, Imax=10,
                     sigma=.5, nx=1000, ny=1000)
det3_2d = SynGauss2D('det3_2d', motor3, 'motor3', center=-1, Imax=10,
                     sigma=1.5, nx=500, ny=200)
