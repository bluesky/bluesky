from ophyd.sim import SynGauss, det, motor, noisy_det

import numpy as np
from bluesky.callbacks.fitting import PeakStats
from bluesky.plans import scan


def get_ps(x, y, shift=0.5):
    """ peak status calculation from CHX algorithm.
    """
    ps = {}
    x=np.array(x)
    y=np.array(y)

    PEAK=x[np.argmax(y)]
    PEAK_y=np.max(y)
    COM=np.sum(x * y) / np.sum(y)
    ps['com'] = COM
    ### from Maksim: assume this is a peak profile:
    def is_positive(num):
        return True if num > 0 else False

    # Normalize values first:
    ym = (y - np.min(y)) / (np.max(y) - np.min(y)) - shift  # roots are at Y=0

    positive = is_positive(ym[0])
    list_of_roots = []
    for i in range(len(y)):
        current_positive = is_positive(ym[i])
        if current_positive != positive:
            list_of_roots.append(x[i - 1] + (x[i] - x[i - 1]) / (abs(ym[i]) + abs(ym[i - 1])) * abs(ym[i - 1]))
            positive = not positive
    if len(list_of_roots) >= 2:
        FWHM=abs(list_of_roots[-1] - list_of_roots[0])
        CEN=list_of_roots[0]+0.5*(list_of_roots[1]-list_of_roots[0])

        ps['fwhm'] = FWHM
        ps['cen'] = CEN

    else:    # ok, maybe it's a step function..
        print('no peak...trying step function...')
        ym = ym + shift
        def err_func(x, x0, k=2, A=1,  base=0 ):     #### erf fit from Yugang
            return base - A * erf(k*(x-x0))
        mod = Model(  err_func )
        ### estimate starting values:
        x0=np.mean(x)
        #k=0.1*(np.max(x)-np.min(x))
        pars  = mod.make_params( x0=x0, k=2,  A = 1., base = 0. )
        result = mod.fit(ym, pars, x = x )
        CEN=result.best_values['x0']
        FWHM = result.best_values['k']
        ps['fwhm'] = FWHM
        ps['cen'] = CEN

    return ps


def test_peak_statistics(RE):
    """peak statistics calculation on simple gaussian function
    """
    x = 'motor'
    y = 'det'
    ps = PeakStats(x, y)
    RE.subscribe(ps)
    RE(scan([det], motor, -5, 5, 100))

    np.allclose(ps.cen, 0, atol=1e-6)
    np.allclose(ps.com, 0, atol=1e-6)
    fwhm_gauss = 2*np.sqrt(2*np.log(2)) # theoretical value with std=1
    np.allclose(ps.fwhm, fwhm_gauss, atol=1e-2)


def test_peak_statistics_compare_chx(RE):
    """This test focuses on gaussian function with noise.
    """
    s = np.random.RandomState(1)
    noisy_det_fix = SynGauss('noisy_det_fix', motor, 'motor', center=0, Imax=1,
                              noise='uniform', sigma=1, noise_multiplier=0.1, random_state=s)
    x = 'motor'
    y = 'noisy_det_fix'
    ps = PeakStats(x, y)
    RE.subscribe(ps)

    RE(scan([noisy_det_fix], motor, -5, 5, 100))
    ps_chx = get_ps(ps.x_data, ps.y_data)

    assert np.allclose(ps.cen, ps_chx['cen'], atol=1e-6)
    assert np.allclose(ps.com, ps_chx['com'], atol=1e-6)
    assert np.allclose(ps.fwhm, ps_chx['fwhm'], atol=1e-6)
