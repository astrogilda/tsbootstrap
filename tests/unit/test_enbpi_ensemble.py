"""Tests for the EnbPIEnsemble fit/predict object and the calibrator family."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap import AR, IID, ResidualBootstrap
from tsbootstrap.errors import MethodConfigError
from tsbootstrap.uq import (
    EnbPIEnsemble,
    enbpi_intervals,
    sliding_window_halfwidths,
    static_halfwidths,
)


def _regression_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + 0.5 * rng.standard_normal(n)
    return X, y


def _heteroskedastic_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A linear signal with a calm first half and a volatile second half.

    The rows are time-ordered: noise scale is small for the first 150 rows and large for
    the last 150, so a time-local calibrator should produce visibly wider widths in the
    second half than the first.
    """
    rng = np.random.default_rng(seed)
    n = 300
    X = rng.standard_normal((n, 2))
    signal = X @ np.array([1.0, -1.0])
    scale = np.concatenate([np.full(150, 0.2), np.full(150, 3.0)])
    y = signal + scale * rng.standard_normal(n)
    return X, y


class TestFitPredictInSample:
    def test_static_matches_enbpi_intervals(self):
        LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression
        X, y = _regression_data(200, 0)
        ens = EnbPIEnsemble().fit(
            LinearRegression(), X, y, method=IID(), n_bootstraps=60, random_state=0
        )
        lo, hi, point = ens.predict_interval(alpha=0.1, calibrator="static")

        lo_ref, hi_ref, point_ref = enbpi_intervals(
            LinearRegression(), X, y, method=IID(), alpha=0.1, n_bootstraps=60, random_state=0
        )
        np.testing.assert_array_equal(lo, lo_ref)
        np.testing.assert_array_equal(hi, hi_ref)
        np.testing.assert_array_equal(point, point_ref)

    def test_static_width_is_constant(self):
        LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression
        X, y = _regression_data(200, 1)
        ens = EnbPIEnsemble().fit(
            LinearRegression(), X, y, method=IID(), n_bootstraps=60, random_state=0
        )
        lo, hi, _ = ens.predict_interval(calibrator="static")
        width = hi - lo
        assert np.allclose(width, width[0])

    def test_in_sample_coverage_near_nominal(self):
        LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression
        X, y = _regression_data(300, 2)
        ens = EnbPIEnsemble().fit(
            LinearRegression(), X, y, method=IID(), n_bootstraps=80, random_state=0
        )
        lo, hi, _ = ens.predict_interval(alpha=0.1, calibrator="static")
        covered = (y >= lo) & (y <= hi)
        assert 0.80 <= covered.mean() <= 1.0


class TestOutOfSample:
    def test_out_of_sample_predict_interval_works(self):
        LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression
        X, y = _regression_data(250, 3)
        X_train, y_train, X_new, y_new = X[:200], y[:200], X[200:], y[200:]
        ens = EnbPIEnsemble().fit(
            LinearRegression(),
            X_train,
            y_train,
            method=IID(),
            n_bootstraps=60,
            random_state=0,
            store_estimators=True,
        )
        lo, hi, point = ens.predict_interval(X_new, alpha=0.1, calibrator="static")
        assert lo.shape == (X_new.shape[0],)
        assert hi.shape == (X_new.shape[0],)
        assert point.shape == (X_new.shape[0],)
        assert np.all(hi > lo)
        # the held-out point predictions should track the held-out targets
        assert np.corrcoef(point, y_new)[0, 1] > 0.8

    def test_out_of_sample_requires_stored_estimators(self):
        LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression
        X, y = _regression_data(150, 4)
        ens = EnbPIEnsemble().fit(
            LinearRegression(),
            X[:120],
            y[:120],
            method=IID(),
            n_bootstraps=40,
            random_state=0,
            store_estimators=False,
        )
        with pytest.raises(MethodConfigError):
            ens.predict_interval(X[120:], calibrator="static")

    def test_in_sample_still_works_without_stored_estimators(self):
        LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression
        X, y = _regression_data(150, 5)
        ens = EnbPIEnsemble().fit(
            LinearRegression(),
            X,
            y,
            method=IID(),
            n_bootstraps=40,
            random_state=0,
            store_estimators=False,
        )
        lo, hi, _ = ens.predict_interval(calibrator="static")  # X_new is None -> no clones needed
        assert np.all(hi > lo)


class TestSlidingWindowAdaptation:
    def test_sliding_window_widens_in_volatile_region(self):
        LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression
        X, y = _heteroskedastic_data(7)
        ens = EnbPIEnsemble().fit(
            LinearRegression(), X, y, method=IID(), n_bootstraps=80, random_state=0
        )
        lo, hi, _ = ens.predict_interval(alpha=0.1, calibrator="sliding_window", window=40)
        width = hi - lo
        # widths must vary (the whole point of time-local calibration)
        assert width.std() > 0.0
        calm = width[20:140].mean()  # well inside the low-volatility first half
        volatile = width[180:].mean()  # well inside the high-volatility second half
        assert volatile > calm

    def test_sliding_window_default_window(self):
        LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression
        X, y = _regression_data(200, 8)
        ens = EnbPIEnsemble().fit(
            LinearRegression(), X, y, method=IID(), n_bootstraps=60, random_state=0
        )
        lo, hi, _ = ens.predict_interval(calibrator="sliding_window")  # window defaults internally
        assert (hi - lo).shape == lo.shape
        assert np.all(hi >= lo)


class TestCalibratorDelegation:
    def test_aci_requires_test_scores(self):
        LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression
        X, y = _regression_data(150, 9)
        ens = EnbPIEnsemble().fit(
            LinearRegression(), X, y, method=IID(), n_bootstraps=40, random_state=0
        )
        with pytest.raises(MethodConfigError):
            ens.predict_interval(calibrator="aci")  # no test_scores supplied

    def test_aci_in_sample_with_oob_residuals(self):
        LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression
        X, y = _regression_data(200, 10)
        ens = EnbPIEnsemble().fit(
            LinearRegression(), X, y, method=IID(), n_bootstraps=60, random_state=0
        )
        scores = ens.oob_residuals  # one realized score per (in-sample) row
        lo, hi, _ = ens.predict_interval(
            alpha=0.1, calibrator="aci", test_scores=scores, gamma=0.05
        )
        assert lo.shape == scores.shape
        assert np.all(hi >= lo)

    def test_nexcp_calibrator(self):
        LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression
        X, y = _regression_data(200, 11)
        ens = EnbPIEnsemble().fit(
            LinearRegression(), X, y, method=IID(), n_bootstraps=60, random_state=0
        )
        lo, hi, _ = ens.predict_interval(alpha=0.1, calibrator="nexcp", decay=0.9)
        width = hi - lo
        assert np.allclose(width, width[0])  # nexcp emits a single scalar width
        assert np.all(hi >= lo)

    def test_unknown_calibrator_raises(self):
        LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression
        X, y = _regression_data(100, 12)
        ens = EnbPIEnsemble().fit(
            LinearRegression(), X, y, method=IID(), n_bootstraps=30, random_state=0
        )
        with pytest.raises(MethodConfigError):
            ens.predict_interval(calibrator="does_not_exist")


class TestCalibratorPurity:
    def test_static_halfwidths_is_pure(self):
        rng = np.random.default_rng(0)
        residuals = np.abs(rng.standard_normal(200))
        a = static_halfwidths(residuals, 50, alpha=0.1)
        b = static_halfwidths(residuals, 50, alpha=0.1)
        np.testing.assert_array_equal(a, b)
        assert a.shape == (50,)
        assert np.allclose(a, a[0])

    def test_sliding_window_halfwidths_is_pure(self):
        rng = np.random.default_rng(1)
        residuals = np.abs(rng.standard_normal(200))
        a = sliding_window_halfwidths(residuals, 200, alpha=0.1, window=30)
        b = sliding_window_halfwidths(residuals, 200, alpha=0.1, window=30)
        np.testing.assert_array_equal(a, b)
        assert a.shape == (200,)

    def test_static_rejects_empty(self):
        with pytest.raises(ValueError):
            static_halfwidths(np.array([]), 10)

    def test_sliding_window_rejects_empty(self):
        with pytest.raises(ValueError):
            sliding_window_halfwidths(np.array([]), 10)


class TestOOBGuard:
    def test_fit_rejects_recursive_method(self):
        LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression
        X, y = _regression_data(50, 13)
        with pytest.raises(MethodConfigError):
            EnbPIEnsemble().fit(
                LinearRegression(),
                X,
                y,
                method=ResidualBootstrap(model=AR(order=1)),
                n_bootstraps=5,
            )

    def test_predict_before_fit_raises(self):
        with pytest.raises(MethodConfigError):
            EnbPIEnsemble().predict_interval()
