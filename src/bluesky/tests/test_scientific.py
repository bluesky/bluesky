import contextlib

import numpy as np
import pytest
from ophyd.sim import SynGauss, det, motor

from bluesky.callbacks.fitting import PeakStats
from bluesky.plans import scan


def get_ps(x, y, shift=0.5):
    """peak status calculation from CHX algorithm."""
    pytest.importorskip("scipy")
    from scipy.special import erf

    lmfit = pytest.importorskip("lmfit")
    ps = {}
    x = np.array(x)
    y = np.array(y)

    COM = np.sum(x * y) / np.sum(y)
    ps["com"] = COM

    # from Maksim: assume this is a peak profile:
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
        FWHM = abs(list_of_roots[-1] - list_of_roots[0])
        CEN = list_of_roots[0] + 0.5 * (list_of_roots[1] - list_of_roots[0])

        ps["fwhm"] = FWHM
        ps["cen"] = CEN

    else:  # ok, maybe it's a step function..
        print("no peak...trying step function...")
        ym = ym + shift

        def err_func(x, x0, k=2, A=1, base=0):  # erf fit from Yugang
            return base - A * erf(k * (x - x0))

        mod = lmfit.Model(err_func)
        # estimate starting values:
        x0 = np.mean(x)
        # k=0.1*(np.max(x)-np.min(x))
        pars = mod.make_params(x0=x0, k=2, A=1.0, base=0.0)
        result = mod.fit(ym, pars, x=x)
        CEN = result.best_values["x0"]
        FWHM = result.best_values["k"]
        ps["fwhm"] = FWHM
        ps["cen"] = CEN

    return ps


def test_peak_statistics(RE):
    """peak statistics calculation on simple gaussian function"""
    x = "motor"
    y = "det"
    ps = PeakStats(x, y)
    RE.subscribe(ps)
    RE(scan([det], motor, -5, 5, 100))

    fields = ["x", "y", "min", "max", "com", "cen", "crossings", "fwhm", "lin_bkg"]
    for field in fields:
        assert hasattr(ps, field), f"{field} is not an attribute of ps"

    np.allclose(ps.cen, 0, atol=1e-6)
    np.allclose(ps.com, 0, atol=1e-6)
    fwhm_gauss = 2 * np.sqrt(2 * np.log(2))  # theoretical value with std=1
    assert np.allclose(ps.fwhm, fwhm_gauss, atol=1e-2)


def test_peak_statistics_compare_chx(RE):
    """This test focuses on gaussian function with noise."""
    s = np.random.RandomState(1)
    noisy_det_fix = SynGauss(
        "noisy_det_fix",
        motor,
        "motor",
        center=0,
        Imax=1,
        noise="uniform",
        sigma=1,
        noise_multiplier=0.1,
        random_state=s,
    )
    x = "motor"
    y = "noisy_det_fix"
    ps = PeakStats(x, y)
    RE.subscribe(ps)

    RE(scan([noisy_det_fix], motor, -5, 5, 100))
    ps_chx = get_ps(ps.x_data, ps.y_data)

    assert np.allclose(ps.cen, ps_chx["cen"], atol=1e-6)
    assert np.allclose(ps.com, ps_chx["com"], atol=1e-6)
    assert np.allclose(ps.fwhm, ps_chx["fwhm"], atol=1e-6)


def test_peak_statistics_with_derivatives(RE):
    """peak statistics calculation on simple gaussian function with derivatives"""
    x = "motor"
    y = "det"
    num_points = 100
    ps = PeakStats(x, y, calc_derivative_and_stats=True)
    RE.subscribe(ps)
    RE(scan([det], motor, -5, 5, num_points))

    assert hasattr(ps, "derivative_stats")
    der_fields = ["x", "y", "min", "max", "com", "cen", "crossings", "fwhm", "lin_bkg"]
    for field in der_fields:
        assert hasattr(ps.derivative_stats, field), f"{field} is not an attribute of ps.der"

    assert type(ps.derivative_stats.x) is np.ndarray
    assert type(ps.derivative_stats.y) is np.ndarray
    assert type(ps.derivative_stats.min) is tuple
    assert type(ps.derivative_stats.max) is tuple
    assert type(ps.derivative_stats.com) is np.float64
    assert type(ps.derivative_stats.cen) is np.float64
    assert type(ps.derivative_stats.crossings) is np.ndarray
    if len(ps.derivative_stats.crossings) >= 2:
        assert type(ps.derivative_stats.fwhm) is float  # noqa: E721
    else:
        assert ps.derivative_stats.fwhm is None
    assert len(ps.derivative_stats.x) == num_points - 1
    assert len(ps.derivative_stats.y) == num_points - 1
    assert np.allclose(np.diff(ps.y_data), ps.derivative_stats.y, atol=1e-10)


@pytest.mark.parametrize(
    ("data", "expected_area_under_curve_com", "expected_com", "warns"),
    [
        # Simplest possible case:
        # - Flat, non-zero Y data
        # - Perfectly evenly spaced, monotonically increasing X data
        ([(0, 1), (2, 1), (4, 1), (6, 1), (8, 1), (10, 1)], 5.0, 5.0, False),
        # Simple triangular peak around x=2 over a base of y=0
        ([(0, 0), (1, 1), (2, 2), (3, 1), (4, 0), (5, 0)], 2.0, 2.0, False),
        # Simple triangular peak around x=2 over a base of y=10
        # - area_under_curve_com is explicitly bounded by min(y) at the bottom of the area,
        #   so should be invariant with respect to Y translation.
        # - com uses all data as-is, so is expected to vary with Y translation.
        ([(0, 10), (1, 11), (2, 12), (3, 11), (4, 10), (5, 10)], 2.0, 2.46875, False),
        # No data at all
        ([], None, None, False),
        # Only one point. CoM should be at that one point's x-value.
        ([(5, 0)], 5.0, 5.0, True),
        ([(5, 50)], 5.0, 5.0, False),
        # Two points, flat data, com should be in the middle
        ([(0, 1), (10, 1)], 5.0, 5.0, False),
        # Flat, logarithmically spaced data:
        # - com should be at central x *point*
        # - area_under_curve_com should be in centre of x *range*
        ([(1, 3), (10, 3), (100, 3), (1000, 3), (10000, 3)], 5000.5, 100.0, False),
        # Two measurements:
        # - area_under_curve_com should be com of a triangle under those two points
        #   (i.e. 1/3 along from right angle)
        # - com should be at the larger of the two points (the other point has y=0)
        ([(0, 0), (3, 6)], 2.0, 3.0, False),
        ([(0, 6), (3, 0)], 1.0, 0.0, False),
        # Cases adding extra measurements which don't change the shape of the measured data, and
        # make the first/last points not have equal spacings to each other.
        # - area_under_curve_com: adding extra measurements, which lie along straight lines between
        #   two other existing measurements, should not change the CoM (as the shape of the area under
        #   the curve has not been changed)
        ([(0, 1), (4, 1), (5, 0), (6, 1), (10, 1)], 5.0, 5.0, False),
        ([(0, 1), (0.1, 1), (4, 1), (5, 0), (6, 1), (10, 1)], 5.0, 4.4, False),
        ([(0, 1), (4, 1), (5, 0), (6, 1), (9.9, 1), (10, 1)], 5.0, 5.6, False),
        # As above, but adding extra measurements along sloped sections as opposed to flat sections.
        # Triangular peak around x=2 over a background of y=0. All cases below have the same shape.
        ([(0, 0), (2, 2), (4, 0), (5, 0)], 2.0, 2.0, False),
        ([(0, 0), (1, 1), (2, 2), (4, 0), (5, 0)], 2.0, 1.6666666666666667, False),
        ([(0, 0), (2, 2), (3, 1), (4, 0), (5, 0)], 2.0, 2.333333333333333, False),
        # Two symmetrical triangular peaks next to each other, with different point spacings
        # but the same shapes, over a base of zero.
        # - area_under_curve_com: since the peaks are symmetrical with each other in terms of shapes,
        #   com should lie exactly between the two peaks.
        ([(0, 0), (1, 1), (2, 2), (3, 1), (4, 0), (6, 2), (8, 0), (10, 0)], 4.0, 3.0, False),
        ([(0, 0), (2, 2), (4, 0), (5, 1), (6, 2), (7, 1), (8, 0), (10, 0)], 4.0, 5.0, False),
        # As above, but over a base of y=10.
        ([(0, 10), (1, 11), (2, 12), (3, 11), (4, 10), (6, 12), (8, 10), (10, 10)], 4.0, 3.465116279069767, False),
        ([(0, 10), (2, 12), (4, 10), (5, 11), (6, 12), (7, 11), (8, 10), (10, 10)], 4.0, 5.465116279069767, False),
        # "Narrow" peak at x=5.0, over a base of y=0
        ([(0, 0), (4.999, 0), (5.0, 10), (5.001, 0)], 5.0, 5.0, False),
        # "Narrow" peak as above, at x=5.0, over a base of y=10
        ([(0, 10), (4.999, 10), (5.0, 20), (5.001, 10)], 5.0, 4.9996, False),
        # Non-monotonically increasing x data (e.g. from adaptive scan).
        # Simple triangular peak shape centred at x=2.
        ([(0, 0), (2, 2), (1, 1), (3, 1), (4, 0)], 2.0, 1.25, False),
        # Overscanned data - all measurements duplicated - e.g. there-and-back scan
        ([(0, 0), (1, 1), (2, 0), (2, 0), (1, 1), (0, 0)], 1.0, 2.0, False),
        # Mixed positive/negative Y data.
        # - area_under_curve_com is explicitly calculating area *under* the curve, so should
        #   give CoM of 1 (i.e. the centre of the *positive* peak)
        # - com uses the negative y-values, so should give
        #   a com of 1+2/3 - (i.e. the centre of the *negative* peak)
        ([(0, -1), (1, 0), (2, -1), (3, -1)], 1.0, 1.6666666666666667, False),
        # Y data with a positive peak at x=1 over a base of y=-1, where the Y data happens
        # to sum to zero but never contain zero.
        ([(0, -1), (1, 3), (2, -1), (3, -1)], 1.0, 0.0, True),
        # As above, but the Y data happens to sum to *nearly* zero rather than exactly zero.
        ([(0, -1), (1, 3.000001), (2, -1), (3, -1)], 1.0, 0.0, False),
        ([(0, -1), (1, 2.999999), (2, -1), (3, -1)], 1.0, 3.0, False),
    ],
)
def test_compute_com(data, expected_area_under_curve_com, expected_com, warns):
    ps_default_com = PeakStats("x", "y")
    ps_area_com = PeakStats("x", "y", com_method="area_under_curve")

    for cb in [ps_default_com, ps_area_com]:
        cb.start({})
        for x, y in data:
            cb.event({"data": {"x": x, "y": y}})

    with pytest.warns(RuntimeWarning) if warns else contextlib.nullcontext():
        ps_default_com.stop({})
    ps_area_com.stop({})

    assert ps_area_com["com"] == pytest.approx(expected_area_under_curve_com)
    assert ps_default_com["com"] == pytest.approx(expected_com)
