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

    def test_varx_accepts_1d_exog(self):
        # A 1D exog array (n,) must be reshaped to (n, 1) and fit identically to the 2D form.
        x, exog = _varx(300, 1)
        np.testing.assert_allclose(
            fit_var(x, 1, exog.ravel()).exog_coefs, fit_var(x, 1, exog).exog_coefs
        )

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

    def test_arx_accepts_1d_exog(self):
        # A 1D exog array (n,) must be treated identically to (n, 1) via the reshape path.
        x, exog = _arx(300, 0.5, 2.0, 0)
        np.testing.assert_allclose(
            fit_ar(x, 1, exog.ravel()).exog_coefs, fit_ar(x, 1, exog).exog_coefs
        )
        res = bootstrap(
            x,
            method=ResidualBootstrap(model=AR(order=1)),
            n_bootstraps=8,
            random_state=0,
            exog=exog.ravel(),
        )
        assert res.values().shape == (8, 300)
        assert np.isfinite(res.values()).all()

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

    def test_arimax_accepts_1d_exog(self):
        # A 1D exog (n,) must reshape to (n, 1) both at the API boundary (bootstrap) and in the
        # ARIMA regression fit itself (the direct call exercises that function's own 1D branch,
        # since coerce_exog already 2D-normalises everything reaching it via bootstrap()).
        from tsbootstrap.model.arima import fit_regression_arima_beta

        y, z = _arimax(250, 1)
        np.testing.assert_allclose(
            fit_regression_arima_beta(y, (1, 1, 1), z.ravel()),
            fit_regression_arima_beta(y, (1, 1, 1), z),
        )
        spec = ResidualBootstrap(model=ARIMA(order=(1, 1, 1)))
        res = bootstrap(y, method=spec, n_bootstraps=6, random_state=0, exog=z.ravel())
        assert res.values().shape == (6, 250)
        assert np.isfinite(res.values()).all()

    def test_arimax_beta_uses_no_trend_for_stationary_order(self):
        """Pin trend="n" so the exog beta is params[:k], not a co-estimated intercept.

        For a stationary order (d=0) on a nonzero-mean series, statsmodels' default
        trend is "c": it prepends a "const" parameter, so res.params[:k] would return
        the intercept instead of the exog coefficient. Passing trend="n" keeps the
        exogenous coefficients first. This pins the exact recovered beta against a
        reference value computed from the real code; trend=None and the dropped-trend
        default both inject the intercept and return ~9.76 instead of ~3.18.
        """
        from tsbootstrap.model.arima import fit_regression_arima_beta

        rng = np.random.default_rng(1)
        n = 150
        z = rng.standard_normal((n, 1))
        e = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + e[t]
        y = (z @ np.array([3.0])).ravel() + y + 10.0  # nonzero-mean stationary AR(1)

        beta = fit_regression_arima_beta(y, (1, 0, 0), z)
        assert beta.shape == (1,)
        np.testing.assert_allclose(beta[0], 3.180011557878669, rtol=0, atol=1e-9)

    def test_arimax_removes_exog_at_correct_sign(self):
        # _prepare_arima fits the ARMA on eta = y - z@beta (exog removed). A sign flip would fit
        # it on ~2x the exog signal: with a dominant integrated exog the correct path leaves a
        # small ARMA residual, while the flip inflates it far past this bound.
        from tsbootstrap.model.recursive import _prepare_arima

        rng = np.random.default_rng(0)
        n = 300
        z = np.cumsum(rng.standard_normal((n, 1)), axis=0)  # strong integrated exog
        small = np.empty(n)  # small stationary AR(1) part
        small[0] = 0.0
        for t in range(1, n):
            small[t] = 0.4 * small[t - 1] + rng.standard_normal()
        y = z[:, 0] * 8.0 + np.cumsum(small)
        ctx = _prepare_arima(y, ARIMA(order=(1, 1, 0)), z)
        assert ctx.arma.residuals.std() < 5.0  # ~0.9 with the correct sign; ~16 if flipped
