from bluesky.examples import Mover, SynGauss, Syn2DGauss
import bluesky.simple_scans as bss
import bluesky.spec_api as bsa
import bluesky.callbacks
from bluesky.standard_config import gs
import bluesky.qt_kicker

# motors
theta = Mover('theta', ['th'])
gamma = Mover('gamma', ['ga'])

# synthetic detectors coupled to one motor
theta_det = SynGauss('theta_det', theta, 'th', center=0, Imax=1, sigma=1)
gamma_det = SynGauss('gamma_det', gamma, 'ga', center=0, Imax=1, sigma=1)

# synthetic detector coupled to two detectors
tgd = Syn2DGauss('theta_gamma_det', theta, 'th', gamma, 'ga',
                 center=(0, 0), Imax=1)

# set up the default detectors
gs.DETS = [theta_det, gamma_det, tgd]

ysteps = 25
xsteps = 20

# hook up the live raster callback
#cb = bluesky.callbacks.LiveRaster((ysteps + 1, xsteps + 1),
#                                  'theta_gamma_det', clim=[0, 1])
mesha = bss.OuterProductAbsScan()
# run a mesh scan
gs.MASTER_DET_FIELD = 'theta_gamma_det'

# bsa.mesh(theta, -2.5, 2.5, ysteps, gamma, -2, 2, xsteps, False)
