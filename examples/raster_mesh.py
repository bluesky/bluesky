from bluesky.examples import Mover, SynGauss, Syn2DGauss
import bluesky.plans as bp
import bluesky.callbacks
from bluesky.global_state import gs
import bluesky.qt_kicker
bluesky.qt_kicker.install_qt_kicker()

# motors
theta = Mover('theta', ['theta'])
gamma = Mover('gamma', ['gamma'])

# synthetic detectors coupled to one motor
theta_det = SynGauss('theta_det', theta, 'theta', center=0, Imax=1, sigma=1)
gamma_det = SynGauss('gamma_det', gamma, 'gamma', center=0, Imax=1, sigma=1)

# synthetic detector coupled to two detectors
tgd = Syn2DGauss('theta_gamma_det', theta, 'theta', gamma, 'gamma',
                 center=(0, 0), Imax=1)

# set up the default detectors
gs.DETS = [theta_det, gamma_det, tgd]

ysteps = 25
xsteps = 20

# hook up the live raster callback
cb = bluesky.callbacks.LiveRaster((ysteps, xsteps),
                                  'theta_gamma_det', clim=[0, 1])
lt = bluesky.callbacks.LiveTable([theta, gamma, tgd])
gs.MASTER_DET_FIELD = 'theta_gamma_det'
mesha = bp.OuterProductAbsScanPlan(gs.DETS,
                                   theta, -2.5, 2.5, ysteps,
                                   gamma, -2, 2, xsteps, True)
gs.RE(mesha, [cb, lt])
