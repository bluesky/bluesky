import numpy as np
from bluesky.plans import scan
from bluesky.examples import motor, noisy_det, det
from bluesky.callbacks.scientific import PeakStats

def test_peak_statistics(fresh_RE):
    """peak statistics calculation on simple gaussian function
    """
    RE = fresh_RE
    x = 'motor'
    y = 'det'
    ps = PeakStats(x, y)
    RE.subscribe(ps)
    RE(scan([det], motor, -5, 5, 100))

    assert ps.cen == 0
    assert ps.com == 0
    fwhm_gauss = 2*np.sqrt(2*np.log(2)) # theoretical value with std=1
    np.allclose(ps.fwhm, fwhm_gauss, atol=1e-2)
