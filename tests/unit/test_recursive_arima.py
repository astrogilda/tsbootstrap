"""Tests for the recursive ARIMA bootstrap (differenced-scale simulation)."""

from __future__ import annotations

import numpy as np
import pytest

from tests._helpers.dgp import acf1
from tsbootstrap.api import bootstrap
from tsbootstrap.dispatch import PreparationFailed
from tsbootstrap.errors import Codes, ModelStabilityError
from tsbootstrap.methods import ARIMA, ResidualBootstrap
from tsbootstrap.model.arima import difference, integrate
from tsbootstrap.model.recursive import _arima_batched, _ARIMAContext, _prepare_arima
from tsbootstrap.rng import generators_from_seeds, spawn_seed_sequences


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
        res = bootstrap(
            x,
            method=ResidualBootstrap(model=ARIMA(order=(1, 1, 0))),
            n_bootstraps=10,
            random_state=3,
        )
        # d=1: the first observation is the fixed integration level for every path.
        assert np.allclose(res.values()[:, 0], x[0])

    def test_arma_d0_preserves_dependence(self):
        x = _arma_series(0.7, 0.0, 500, 4)
        orig_acf1 = acf1(x)
        res = bootstrap(
            x,
            method=ResidualBootstrap(model=ARIMA(order=(1, 0, 0))),
            n_bootstraps=200,
            random_state=5,
        )
        acf1s = np.array([acf1(s) for s in res.values()])
        assert abs(acf1s.mean() - orig_acf1) < 0.1

    def test_integrated_drift_is_reproduced(self):
        # A differenced series with nonzero mean integrates to a trend; the bootstrap
        # paths must reproduce that drift (not lose or double it).
        n = 400
        increments = 0.5 + _arma_series(0.3, 0.0, n, 6)  # mean increment ~ 0.5
        x = np.cumsum(increments)
        res = bootstrap(
            x,
            method=ResidualBootstrap(model=ARIMA(order=(1, 1, 0))),
            n_bootstraps=300,
            random_state=7,
        )
        final = res.values()[:, -1]
        # paths start at x[0] and should rise to about x[-1] on average
        assert abs(final.mean() - x[-1]) < 0.25 * abs(x[-1] - x[0])

    def test_arima_engine_perfect_reconstruction(self):
        # Regression guard: re-injecting the model's own (lfilter-consistent) residuals
        # reconstructs the fitted series exactly. Catches innovation-definition, initial-condition,
        # and lag-indexing bugs that stochastic tests and the drift gate miss. This invariant
        # originally exposed the Kalman-vs-lfilter residual inconsistency (a 0.49 level error).
        from tsbootstrap.engines.arma_scipy import simulate_arma_batched
        from tsbootstrap.model.arima import fit_arma

        x = _arma_series(0.5, 0.3, 200, 0)
        w, levels = difference(x, 1)
        arma = fit_arma(w, 1, 1)
        wc = (
            simulate_arma_batched(arma.ar_coefs, arma.ma_coefs, arma.residuals.reshape(1, -1))[0]
            + arma.mean
        )
        np.testing.assert_allclose(integrate(wc, levels), x, atol=1e-9)

    def test_arima_replicates_condition_on_observed_initials(self):
        # ARIMA conditions on the observed initial state (the ARMA analogue of
        # AR/VAR's initial="fixed"), so every replicate begins at the observed series rather than a
        # zero-state burn-in draw.
        x = np.cumsum(0.5 + _arma_series(0.3, 0.2, 300, 2))
        res = bootstrap(
            x,
            method=ResidualBootstrap(model=ARIMA(order=(1, 1, 1))),
            n_bootstraps=20,
            random_state=0,
        )
        v = res.values()
        np.testing.assert_allclose(v[:, :2], np.broadcast_to(x[:2], (20, 2)))


class TestPrepareArimaStabilityGuard:
    """Pin _prepare_arima's stability-policy handling for the ARMA AR coefficients.

    The guard is exercised by monkeypatching the stability check the preparer calls so
    the outcome is deterministic regardless of what statsmodels happens to fit. This is
    the only way to drive the unstable branch reliably (a real fit on a stationary DGP
    is stable, and forcing an explosive fit from data is brittle).
    """

    def _raise_unstable(self, _coefs, **_kwargs):
        raise ModelStabilityError(
            "forced unstable for test",
            code=Codes.UNSTABLE_MODEL,
            context={"spectral_radius": 1.5},
        )

    def test_skip_policy_returns_preparation_failed_on_unstable_fit(self, monkeypatch):
        """With stability_policy='skip', an unstable fit yields a PreparationFailed.

        Kills the mutant that drops the stability guard entirely (failed = None), which
        would return a usable _ARIMAContext for an unstable model, and the mutant that
        passes policy=None to the guard, which re-raises instead of skipping.
        """
        monkeypatch.setattr("tsbootstrap.model.recursive.check_ar_stability", self._raise_unstable)
        x = _arma_series(0.5, 0.3, 120, 11)
        model = ARIMA(order=(1, 0, 1), stability_policy="skip")
        result = _prepare_arima(x, model, exog=None)
        assert isinstance(result, PreparationFailed)
        assert not isinstance(result, _ARIMAContext)

    def test_raise_policy_propagates_stability_error(self, monkeypatch):
        """With stability_policy='raise', an unstable fit re-raises ModelStabilityError.

        Confirms the guard is actually invoked with the model's policy (not a hardcoded
        None or a dropped call); under the policy-None mutant this still raises, but the
        skip test above is what distinguishes the two, so both branches are pinned.
        """
        monkeypatch.setattr("tsbootstrap.model.recursive.check_ar_stability", self._raise_unstable)
        x = _arma_series(0.5, 0.3, 120, 12)
        model = ARIMA(order=(1, 0, 1), stability_policy="raise")
        with pytest.raises(ModelStabilityError):
            _prepare_arima(x, model, exog=None)


class TestArimaBatchedDtype:
    """Pin the final dtype cast in _arima_batched.

    The simulation runs in float64; the function must cast the returned samples to the
    requested sim_dtype at the contiguity boundary. Mutants that pass dtype=None (or drop
    the dtype argument) leave the output as float64, which this test catches.
    """

    def _context(self) -> _ARIMAContext:
        x = np.cumsum(_arma_series(0.4, 0.2, 150, 21))  # integrated once -> ARIMA(1,1,1)
        ctx = _prepare_arima(x, ARIMA(order=(1, 1, 1)), exog=None)
        assert isinstance(ctx, _ARIMAContext)
        return ctx

    def test_output_cast_to_requested_sim_dtype(self):
        """Output dtype equals the requested float32, not the float64 simulation dtype.

        Kills both the dtype=None mutant and the dropped-dtype-argument mutant: with the
        cast removed the array stays float64, so asserting float32 fails on the mutants.
        """
        ctx = self._context()
        n = 150
        generators = generators_from_seeds(spawn_seed_sequences(np.random.SeedSequence(3), 5))
        result = _arima_batched(ctx, n, generators, np.dtype(np.float32))
        assert result.dtype == np.float32
        assert result.shape == (5, n, 1)

    def test_float64_request_is_preserved(self):
        """A float64 request returns float64 (sanity anchor for the dtype contract)."""
        ctx = self._context()
        n = 150
        generators = generators_from_seeds(spawn_seed_sequences(np.random.SeedSequence(4), 3))
        result = _arima_batched(ctx, n, generators, np.dtype(np.float64))
        assert result.dtype == np.float64
        assert result.shape == (3, n, 1)
