import numpy as np
from .examples import Reader, motor, motor1, motor2, motor3
from .run_engine import Msg
from filestore.file_writers import save_ndarray
import tempfile
import time as ttime


class SynGauss2D(Reader):
    """
    Evaluate a point on a Gaussian based on the value of a motor.

    Example
    -------
    motor = Mover('motor', {'motor': lambda x: x}, {'x': 0})
    det = SynGauss2D('det', motor, 'motor', center=0, Imax=1, sigma=1)
    """
    def __init__(self, name, motor, motor_field, center, Imax=1000, sigma=1,
                 nx=250, ny=250, img_sigma=50):
        dims = (nx, ny)
        self.dims = dims
        self.name = name

        def func():
            m = motor.read()[motor_field]['value']
            v = Imax * np.exp(-(m - center)**2 / (2 * sigma**2))
            arr = self.gauss(dims, img_sigma) * v + np.random.random(dims) * .01
            fs_uid = save_ndarray(arr, self.output_dir)
            return fs_uid

        # exception to be raised if/when the file system cleans its temp files
        self.output_dir = tempfile.gettempdir()
        super().__init__(name, {name: func})

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
        return {self.name: {'source': self.name,
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
