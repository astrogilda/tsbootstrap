"""Tests for the recursive ARIMA bootstrap (differenced-scale simulation)."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap.api import bootstrap
from tsbootstrap.methods import ARIMA, ResidualBootstrap
from tsbootstrap.model.arima import difference, integrate


def _arma_series(phi: float, theta: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 50)
    w = np.empty(n + 50)
    w[0] = e[0]
    for t in range(1, n + 50):
        w[t] = phi * w[t - 1] + e[t] + theta * e[t - 1]
    return w[50:]


class TestDifferenceIntegrate:
    @pytest.mark.parametrize("d", [0, 1, 2])
    def test_difference_integrate_roundtrip(self, d):
        rng = np.random.default_rng(0)
        x = np.cumsum(rng.standard_normal(60)) + 0.3 * np.arange(60)
        w, levels = difference(x, d)
        assert w.shape[0] == x.shape[0] - d
        np.testing.assert_allclose(integrate(w, levels), x, atol=1e-8)


class TestARIMAResidualBootstrap:
    def test_arima_shape_and_determinism(self):
        x = np.cumsum(_arma_series(0.5, 0.3, 300, 1))  # integrated once
        spec = ResidualBootstrap(model=ARIMA(order=(1, 1, 1)))
        a = bootstrap(x, method=spec, n_bootstraps=8, random_state=7)
        b = bootstrap(x, method=spec, n_bootstraps=8, random_state=7)
        assert a.values().shape == (8, 300)
        np.testing.assert_array_equal(a.values(), b.values())
        assert a.indices() is None

    def test_arima_initial_levels_are_fixed(self):
        x = np.cumsum(_arma_series(0.4, 0.0, 200, 2))
        res = bootstrap(x, method=ResidualBootstrap(model=ARIMA(order=(1, 1, 0))), n_bootstraps=10, random_state=3)
        # d=1: the first observation is the fixed integration level for every path.
        assert np.allclose(res.values()[:, 0], x[0])

    def test_arma_d0_preserves_dependence(self):
        x = _arma_series(0.7, 0.0, 500, 4)
        orig_acf1 = np.corrcoef(x[:-1], x[1:])[0, 1]
        res = bootstrap(x, method=ResidualBootstrap(model=ARIMA(order=(1, 0, 0))), n_bootstraps=200, random_state=5)
        acf1 = np.array([np.corrcoef(s[:-1], s[1:])[0, 1] for s in res.values()])
        assert abs(acf1.mean() - orig_acf1) < 0.1

    def test_integrated_drift_is_reproduced(self):
        # A differenced series with nonzero mean integrates to a trend; the bootstrap
        # paths must reproduce that drift (not lose or double it).
        n = 400
        increments = 0.5 + _arma_series(0.3, 0.0, n, 6)  # mean increment ~ 0.5
        x = np.cumsum(increments)
        res = bootstrap(x, method=ResidualBootstrap(model=ARIMA(order=(1, 1, 0))), n_bootstraps=300, random_state=7)
        final = res.values()[:, -1]
        # paths start at x[0] and should rise to about x[-1] on average
        assert abs(final.mean() - x[-1]) < 0.25 * abs(x[-1] - x[0])

    def test_arima_engine_perfect_reconstruction(self):
        # DEC-010 regression guard: re-injecting the model's own (lfilter-consistent) residuals
        # reconstructs the fitted series exactly. Catches innovation-definition, initial-condition,
        # and lag-indexing bugs that stochastic tests and the drift gate miss. This invariant
        # originally exposed the Kalman-vs-lfilter residual inconsistency (a 0.49 level error).
        from tsbootstrap.engines.arma_scipy import simulate_arma_batched
        from tsbootstrap.model.arima import fit_arma

        x = _arma_series(0.5, 0.3, 200, 0)
        w, levels = difference(x, 1)
        arma = fit_arma(w, 1, 1)
        wc = simulate_arma_batched(arma.ar_coefs, arma.ma_coefs, arma.residuals.reshape(1, -1))[0] + arma.mean
        np.testing.assert_allclose(integrate(wc, levels), x, atol=1e-9)

    def test_arima_replicates_condition_on_observed_initials(self):
        # DEC-010 part 2: ARIMA now conditions on the observed initial state (the ARMA analogue of
        # AR/VAR's initial="fixed"), so every replicate begins at the observed series rather than a
        # zero-state burn-in draw.
        x = np.cumsum(0.5 + _arma_series(0.3, 0.2, 300, 2))
        res = bootstrap(x, method=ResidualBootstrap(model=ARIMA(order=(1, 1, 1))), n_bootstraps=20, random_state=0)
        v = res.values()
        np.testing.assert_allclose(v[:, :2], np.broadcast_to(x[:2], (20, 2)))
