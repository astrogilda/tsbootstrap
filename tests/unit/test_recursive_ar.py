"""Tests for the recursive AR and sieve bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from tests._helpers.dgp import acf1, ar1
from tsbootstrap.api import bootstrap
from tsbootstrap.engines.arma_scipy import simulate_ar
from tsbootstrap.errors import MethodConfigError, ModelStabilityError, NearUnitRootWarning
from tsbootstrap.methods import AR, ResidualBootstrap, SieveAR
from tsbootstrap.model.fit import fit_ar, select_ar_order
from tsbootstrap.model.stability import ar_spectral_radius, check_ar_stability


def _ar2(n: int, seed: int, phi1: float = 0.6, phi2: float = -0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[:2] = rng.standard_normal(2)
    for t in range(2, n):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + rng.standard_normal()
    return x


def _explosive_ar1(n: int = 60) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = np.empty(n)
    x[0] = 1.0
    for t in range(1, n):
        x[t] = 1.05 * x[t - 1] + 0.1 * rng.standard_normal()
    return x


class TestSimulateAR:
    def test_one_shock_propagates_ar1(self):
        # A single innovation must propagate recursively by the AR coefficient
        # (geometric decay).
        innov = np.zeros(12)
        innov[3] = 1.0
        path = simulate_ar(np.array([0.8]), 0.0, np.array([0.0]), innov)
        gen = path[1:]  # the generated part after the single initial value
        assert np.isclose(gen[3], 1.0)
        assert np.isclose(gen[4], 0.8)
        assert np.isclose(gen[5], 0.64)
        assert np.isclose(gen[6], 0.512)

    def test_one_shock_propagates_ar2(self):
        phi = np.array([0.5, 0.3])
        innov = np.zeros(10)
        innov[2] = 1.0
        gen = simulate_ar(phi, 0.0, np.array([0.0, 0.0]), innov)[2:]
        assert np.isclose(gen[2], 1.0)
        assert np.isclose(gen[3], 0.5)  # 0.5*1
        assert np.isclose(gen[4], 0.5 * 0.5 + 0.3 * 1.0)  # 0.55


class TestARResidualBootstrap:
    def test_residual_bootstrap_shape_and_determinism(self):
        x = ar1(0.6, 300, 1)
        a = bootstrap(
            x, method=ResidualBootstrap(model=AR(order=1)), n_bootstraps=10, random_state=7
        )
        b = bootstrap(
            x, method=ResidualBootstrap(model=AR(order=1)), n_bootstraps=10, random_state=7
        )
        assert a.values().shape == (10, 300)
        np.testing.assert_array_equal(a.values(), b.values())
        assert a.indices() is None  # recursive methods have no observation indices

    def test_recursive_bootstrap_preserves_ar_dependence(self):
        # The regenerated series must keep the autoregressive dependence.
        x = ar1(0.7, 600, 2)
        orig_acf1 = acf1(x)
        res = bootstrap(
            x, method=ResidualBootstrap(model=AR(order=1)), n_bootstraps=300, random_state=3
        )
        acf1s = np.array([acf1(s) for s in res.values()])
        assert abs(acf1s.mean() - orig_acf1) < 0.1

    def test_bootstrap_coefficient_centers_on_fit(self):
        x = ar1(0.7, 500, 4)
        phi_hat = fit_ar(x, 1).ar_coefs[0]
        res = bootstrap(
            x, method=ResidualBootstrap(model=AR(order=1)), n_bootstraps=300, random_state=5
        )
        boot_phi = np.array([fit_ar(s, 1).ar_coefs[0] for s in res.values()])
        assert abs(boot_phi.mean() - phi_hat) < 0.08

    def test_ar_model_rejects_multivariate(self):
        x = np.random.default_rng(0).standard_normal((100, 2))
        with pytest.raises(MethodConfigError):
            bootstrap(x, method=ResidualBootstrap(model=AR(order=1)), n_bootstraps=2)

    def test_unstable_fit_raises(self):
        # An explosive AR(1) (phi > 1) fits an unstable model -> refuse to simulate.
        with pytest.raises(ModelStabilityError):
            bootstrap(_explosive_ar1(), method=ResidualBootstrap(model=AR(order=1)), n_bootstraps=2)

    def test_stability_policy_skip_returns_failed_result(self):
        # With stability_policy="skip", an unstable fit fails the run honestly instead
        # of crashing: an empty result flagged failed, so a pipeline can continue.
        res = bootstrap(
            _explosive_ar1(),
            method=ResidualBootstrap(model=AR(order=1, stability_policy="skip")),
            n_bootstraps=5,
        )
        assert res.metadata.failed is True
        assert res.metadata.failure_reason
        assert len(res) == 0
        assert res.values().shape == (0,)

    def test_random_block_initial_runs(self):
        # initial="random_block" exercises the random-start branch of the init-block draw
        # (the default is "fixed"); the regenerated paths must stay valid and the right length.
        x = ar1(0.6, 300, 1)
        res = bootstrap(
            x,
            method=ResidualBootstrap(model=AR(order=2, initial="random_block")),
            n_bootstraps=8,
            random_state=0,
        )
        assert res.values().shape == (8, 300)
        assert np.isfinite(res.values()).all()

    def test_positive_burn_in_runs_and_preserves_length(self):
        # burn_in > 0 generates extra leading steps that are then discarded; the returned
        # series must still be exactly the original length (pins the burn-in slice arithmetic).
        x = ar1(0.6, 300, 1)
        res = bootstrap(
            x,
            method=ResidualBootstrap(model=AR(order=1, burn_in=25)),
            n_bootstraps=8,
            random_state=0,
        )
        assert res.values().shape == (8, 300)
        assert np.isfinite(res.values()).all()


class TestSieveARBootstrap:
    def test_sieve_runs_and_selects_order(self):
        x = ar1(0.6, 400, 6)
        res = bootstrap(x, method=SieveAR(), n_bootstraps=8, random_state=0)
        assert res.values().shape == (8, 400)


class TestSelectArOrder:
    def test_recovers_ar2_structure_across_criteria(self):
        # An AR(2) process should select order >= 2 under every criterion; a broken penalty
        # or information-criterion formula collapses to order 1 or runs away to the cap.
        x = _ar2(800, 0)
        for criterion in ("aic", "bic", "hqic"):
            k = select_ar_order(x, max_lag=8, criterion=criterion)
            assert 2 <= k <= 6, f"{criterion} selected order {k}"

    def test_heavier_penalty_never_selects_a_larger_order(self):
        # BIC's penalty (log n) is heavier than AIC's (2), so on the same series BIC must not
        # pick a larger order than AIC. Pins the relative penalty magnitudes.
        x = _ar2(600, 2, phi1=0.5, phi2=0.2)
        aic = select_ar_order(x, max_lag=10, criterion="aic")
        bic = select_ar_order(x, max_lag=10, criterion="bic")
        assert bic <= aic

    def test_max_lag_bounds_the_selected_order(self):
        # The search upper bound must clamp the returned order.
        x = _ar2(400, 3)
        assert select_ar_order(x, max_lag=2) <= 2


class TestARStabilityHelpers:
    def test_stability_helpers(self):
        assert ar_spectral_radius(np.array([0.5])) == pytest.approx(0.5)
        check_ar_stability(np.array([0.5]))  # stable, no raise
        with pytest.raises(ModelStabilityError):
            check_ar_stability(np.array([1.0]))  # unit root

    def test_ar2_spectral_radius_uses_companion_subdiagonal(self):
        # For p >= 2 the companion matrix needs its unit sub-diagonal; without it the
        # eigenvalues (and the radius) are wrong. AR(2) [0.5, 0.3] -> radius 0.8521,
        # not |0.5| as a missing sub-diagonal would give.
        assert ar_spectral_radius(np.array([0.5, 0.3])) == pytest.approx(0.85208, abs=1e-4)

    def test_near_unit_root_warns_at_default_threshold(self):
        # A radius in [0.98, 1.0) must warn at the default threshold (no explicit override).
        with pytest.warns(NearUnitRootWarning):
            check_ar_stability(np.array([0.99]))

    def test_ar3_spectral_radius_matches_polynomial_roots(self):
        # For p >= 3 the companion's unit sub-diagonal spans more than one column; an off-by-one
        # there (e.g. only column 0) is invisible at p=2 but wrong at p=3. The companion
        # eigenvalues are exactly the roots of z^p - phi_1 z^(p-1) - ... - phi_p, so compare
        # against an independent polynomial-root computation.
        phi = np.array([0.5, 0.2, 0.1])
        expected = float(np.max(np.abs(np.roots([1.0, -0.5, -0.2, -0.1]))))
        assert ar_spectral_radius(phi) == pytest.approx(expected, abs=1e-6)

    def test_empty_ar_coefs_has_zero_radius(self):
        # An order-0 (empty) coefficient vector has no dynamics -> radius 0 (stable).
        assert ar_spectral_radius(np.array([])) == 0.0
