"""Tests for exogenous-covariate support (ARX and VARX residual bootstraps)."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap import AR, ARIMA, VAR, MovingBlock, ResidualBootstrap, bootstrap
from tsbootstrap.errors import MethodConfigError, TSBootstrapError
from tsbootstrap.model.fit import fit_ar, fit_var


def _varx(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    exog = rng.standard_normal((n, 1))
    a = np.array([[0.4, 0.1], [0.05, 0.3]])
    b = np.array([[1.2], [-1.5]])  # (d, k)
    x = np.zeros((n, 2))
    for t in range(1, n):
        x[t] = a @ x[t - 1] + (b @ exog[t]) + rng.standard_normal(2)
    return x, exog


def _arx(n: int, phi: float, beta: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    exog = rng.standard_normal((n, 1))
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + beta * exog[t, 0] + e[t]
    return x, exog


def _arimax(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 1))
    e = rng.standard_normal(n)
    w = np.zeros(n)
    for t in range(1, n):
        w[t] = 0.5 * w[t - 1] + e[t] + 0.3 * e[t - 1]  # ARMA(1,1) on the differenced scale
    y = z @ np.array([2.0]) + np.cumsum(w)  # eta = integrate once (d=1); y = z @ beta + eta
    return y, z


class TestVARX:
    def test_varx_recovers_the_exog_effect(self):
        x, exog = _varx(600, 0)
        spec = ResidualBootstrap(model=VAR(order=1))
        res = bootstrap(x, method=spec, n_bootstraps=100, random_state=0, exog=exog)
        fitted = fit_var(x, 1, exog).exog_coefs
        boot = np.array([fit_var(s, 1, exog).exog_coefs for s in res.values()])
        assert np.abs(boot.mean(axis=0) - fitted).max() < 0.05

    def test_varx_shape_and_determinism(self):
        x, exog = _varx(300, 1)
        spec = ResidualBootstrap(model=VAR(order=1))
        a = bootstrap(x, method=spec, n_bootstraps=8, random_state=7, exog=exog)
        b = bootstrap(x, method=spec, n_bootstraps=8, random_state=7, exog=exog)
        assert a.values().shape == (8, 300, 2)
        np.testing.assert_array_equal(a.values(), b.values())

    def test_varx_rejects_nonzero_burn_in(self):
        x, exog = _varx(200, 2)
        with pytest.raises(MethodConfigError):
            bootstrap(
                x,
                method=ResidualBootstrap(model=VAR(order=1, burn_in=5)),
                n_bootstraps=2,
                exog=exog,
            )


class TestARX:
    def test_arx_recovers_the_exog_effect(self):
        x, exog = _arx(500, 0.5, 2.0, 0)
        spec = ResidualBootstrap(model=AR(order=1))
        res = bootstrap(x, method=spec, n_bootstraps=200, random_state=0, exog=exog)
        fitted_beta = fit_ar(x, 1, exog).exog_coefs[0]
        boot_betas = np.array([fit_ar(s, 1, exog).exog_coefs[0] for s in res.values()])
        assert abs(boot_betas.mean() - fitted_beta) < 0.15

    def test_arx_shape_and_determinism(self):
        x, exog = _arx(300, 0.4, 1.0, 1)
        spec = ResidualBootstrap(model=AR(order=1))
        a = bootstrap(x, method=spec, n_bootstraps=8, random_state=7, exog=exog)
        b = bootstrap(x, method=spec, n_bootstraps=8, random_state=7, exog=exog)
        assert a.values().shape == (8, 300)
        np.testing.assert_array_equal(a.values(), b.values())

    def test_exog_rejected_for_block_method(self):
        x, exog = _arx(100, 0.5, 1.0, 2)
        with pytest.raises(MethodConfigError):
            bootstrap(x, method=MovingBlock(block_length=5), n_bootstraps=2, exog=exog)

    def test_exog_length_mismatch_raises(self):
        x, exog = _arx(120, 0.5, 1.0, 4)
        with pytest.raises(TSBootstrapError):
            bootstrap(
                x, method=ResidualBootstrap(model=AR(order=1)), n_bootstraps=2, exog=exog[:50]
            )


class TestARIMAX:
    def test_arimax_recovers_the_exog_effect(self):
        from tsbootstrap.model.arima import fit_regression_arima_beta

        y, z = _arimax(300, 0)
        spec = ResidualBootstrap(model=ARIMA(order=(1, 1, 1)))
        res = bootstrap(y, method=spec, n_bootstraps=25, random_state=0, exog=z)
        fitted = fit_regression_arima_beta(y, (1, 1, 1), z)[0]
        boot = np.array([fit_regression_arima_beta(s, (1, 1, 1), z)[0] for s in res.values()])
        assert abs(boot.mean() - fitted) < 0.3

    def test_arimax_shape_and_determinism(self):
        y, z = _arimax(250, 1)
        spec = ResidualBootstrap(model=ARIMA(order=(1, 1, 1)))
        a = bootstrap(y, method=spec, n_bootstraps=6, random_state=7, exog=z)
        b = bootstrap(y, method=spec, n_bootstraps=6, random_state=7, exog=z)
        assert a.values().shape == (6, 250)
        np.testing.assert_array_equal(a.values(), b.values())
