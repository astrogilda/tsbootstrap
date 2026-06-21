"""Tests for the UQ layer: EnbPI (regression) and forecast intervals."""

from __future__ import annotations

import numpy as np
import pytest

from tests._helpers.dgp import ar1
from tsbootstrap import AR, IID, ResidualBootstrap
from tsbootstrap.errors import MethodConfigError
from tsbootstrap.uq import enbpi_intervals, fit_predict_oob, forecast_intervals


def _regression_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + 0.5 * rng.standard_normal(n)
    return X, y


class TestEnbPI:
    def test_fit_predict_oob_recovers_signal(self):
        LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression
        X, y = _regression_data(200, 0)
        oob = fit_predict_oob(
            LinearRegression(), X, y, method=IID(), n_bootstraps=50, random_state=0
        )
        finite = np.isfinite(oob)
        assert finite.mean() > 0.9
        assert np.corrcoef(oob[finite], y[finite])[0, 1] > 0.8

    def test_enbpi_coverage_is_near_nominal_in_sample(self):
        LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression
        X, y = _regression_data(300, 1)
        lo, hi, _ = enbpi_intervals(
            LinearRegression(), X, y, method=IID(), alpha=0.1, n_bootstraps=80, random_state=0
        )
        finite = np.isfinite(lo)
        covered = ((y >= lo) & (y <= hi))[finite]
        assert 0.80 <= covered.mean() <= 1.0

    def test_enbpi_rejects_recursive_method(self):
        LinearRegression = pytest.importorskip("sklearn.linear_model").LinearRegression
        X, y = _regression_data(50, 2)
        with pytest.raises(MethodConfigError):
            enbpi_intervals(
                LinearRegression(),
                X,
                y,
                method=ResidualBootstrap(model=AR(order=1)),
                n_bootstraps=5,
            )


class TestForecastIntervals:
    def test_forecast_intervals_shape_and_determinism(self):
        x = ar1(0.6, 200, 3)
        a = forecast_intervals(x, model=AR(order=1), horizon=10, n_bootstraps=200, random_state=0)
        b = forecast_intervals(x, model=AR(order=1), horizon=10, n_bootstraps=200, random_state=0)
        lo, hi, _ = a
        assert lo.shape == (10,)
        assert hi.shape == (10,)
        assert np.all(hi > lo)
        np.testing.assert_array_equal(a[0], b[0])

    def test_forecast_uncertainty_grows_with_horizon(self):
        x = ar1(0.6, 300, 4)
        lo, hi, _ = forecast_intervals(
            x, model=AR(order=1), horizon=20, n_bootstraps=500, random_state=0
        )
        width = hi - lo
        assert width[-1] > width[0]
