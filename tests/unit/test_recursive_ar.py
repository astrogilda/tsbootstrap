"""Tests for the recursive AR and sieve bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from tests._helpers.dgp import acf1, ar1
from tsbootstrap.api import bootstrap
from tsbootstrap.engines.arma_scipy import simulate_ar
from tsbootstrap.errors import (
    Codes,
    MethodConfigError,
    ModelStabilityError,
    NearUnitRootWarning,
)
from tsbootstrap.methods import AR, MovingBlock, ResidualBootstrap, SieveAR
from tsbootstrap.model.fit import fit_ar, select_ar_order
from tsbootstrap.model.stability import ar_spectral_radius, check_ar_stability


def _ar2(n: int, seed: int, phi1: float = 0.6, phi2: float = -0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[:2] = rng.standard_normal(2)
    for t in range(2, n):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + rng.standard_normal()
    return x


def _seasonal(n: int, seed: int, noise: float = 0.1) -> np.ndarray:
    """A near-deterministic two-period seasonal series (periods 7 and 3).

    The strong deterministic structure makes the OLS information criterion keep
    rewarding extra lags, so the selected order saturates at the search upper bound.
    That pins the upper-bound arithmetic (which a loose-range test never exercises).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (
        np.sin(2 * np.pi * t / 7) + 0.5 * np.sin(2 * np.pi * t / 3) + noise * rng.standard_normal(n)
    )


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

    def test_initial_state_uses_correctly_oriented_lags(self):
        """The first generated step must combine the initial values with the right lags.

        With AR(2) coefficients [phi1, phi2] = [0.5, 0.3] and an asymmetric initial
        block X_0=2.0, X_1=-1.0 (so X_1 is the most recent), the first generated value is
        X_2 = phi1*X_1 + phi2*X_0 = 0.5*(-1.0) + 0.3*(2.0) = 0.1 and the next is
        X_3 = phi1*X_2 + phi2*X_1 = 0.5*(0.1) + 0.3*(-1.0) = -0.25. The initial block is
        asymmetric on purpose: a missing or wrong-step reversal of the initial values
        feeding the filter state would orient the lags backwards and produce 0.7 (no
        reversal) or -0.5 (a stride-2 reversal) instead of 0.1.
        """
        phi = np.array([0.5, 0.3])
        init = np.array([2.0, -1.0])
        path = simulate_ar(phi, 0.0, init, np.zeros(6))
        # The two initial values are returned verbatim, then the generated recursion.
        assert np.isclose(path[0], 2.0)
        assert np.isclose(path[1], -1.0)
        assert np.isclose(path[2], 0.1)
        assert np.isclose(path[3], -0.25)


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

    def test_stability_skip_failure_reason_reports_the_real_error(self):
        """On a skipped unstable fit the failure reason must be the actual error text.

        The stability guard records ``str(exc)`` of the ModelStabilityError. A mutation
        to ``str(None)`` would record the literal string ``"None"``, which is still
        truthy and so passes a bare ``assert failure_reason`` check. Asserting the
        diagnostic substrings (the code tag and the "non-stationary"/"spectral radius"
        wording) pins the real message and kills the ``str(None)`` mutant.
        """
        res = bootstrap(
            _explosive_ar1(),
            method=ResidualBootstrap(model=AR(order=1, stability_policy="skip")),
            n_bootstraps=5,
        )
        reason = res.metadata.failure_reason
        assert reason != "None"
        assert Codes.UNSTABLE_MODEL in reason
        assert "non-stationary" in reason and "spectral radius" in reason

    def test_non_iid_innovation_rejected_with_code_and_message(self):
        """A non-IID innovation must raise MethodConfigError with the right code+message.

        Recursive residual bootstraps only support IID innovation resampling. Passing a
        block innovation (MovingBlock) must raise MethodConfigError tagged
        ``UNSUPPORTED_MODEL_FEATURE`` with a message naming IID innovations. This pins
        the message text (a blanked-to-``None`` or dropped-message mutant changes/breaks
        it) and the ``code`` keyword (dropping it falls back to the class default
        INVALID_PARAMETER, not UNSUPPORTED_MODEL_FEATURE).
        """
        x = ar1(0.5, 120, 0)
        with pytest.raises(MethodConfigError) as excinfo:
            bootstrap(
                x,
                method=ResidualBootstrap(model=AR(order=1), innovation=MovingBlock()),
                n_bootstraps=3,
            )
        err = excinfo.value
        assert err.code == Codes.UNSUPPORTED_MODEL_FEATURE
        assert "IID" in str(err)

    def test_order_too_large_carries_code_and_context(self):
        """fit_ar must reject an over-large order with the structured code and context.

        Requesting an AR order >= the series length raises MethodConfigError tagged
        ORDER_TOO_LARGE. Pinning the ``context`` (the offending ``order`` and ``n``)
        catches a mutant that drops the ``context={"order": order, "n": n}`` keyword and
        leaves an empty ``{}``.
        """
        x = ar1(0.5, 8, 0)
        with pytest.raises(MethodConfigError) as excinfo:
            fit_ar(x, order=8)
        err = excinfo.value
        assert err.code == Codes.ORDER_TOO_LARGE
        assert err.context.get("order") == 8
        assert err.context.get("n") == 8

    def test_float32_dtype_is_honored_in_output(self):
        """``dtype='float32'`` must produce a float32 result array.

        The recursive simulation runs in float64 and casts to the requested ``sim_dtype``
        at the final contiguity boundary via ``ascontiguousarray(samples, dtype=sim_dtype)``.
        Mutating that to ``dtype=None`` (or dropping the keyword) would leave the output
        as float64. Requesting ``dtype='float32'`` and asserting the output dtype pins the
        cast.
        """
        x = ar1(0.6, 200, 1)
        res = bootstrap(
            x,
            method=ResidualBootstrap(model=AR(order=2)),
            n_bootstraps=4,
            random_state=0,
            dtype="float32",
        )
        assert res.values().dtype == np.float32

    def test_random_block_initial_runs(self):
        # initial="random_block" draws each path's p-length initial block from a random start in
        # [0, n - p]. A short series + many paths reliably probes that start-range bound: an
        # off-by-one that overshoots would draw a too-short block and raise, so a clean run of
        # the full batch pins it (the default "fixed" path never exercises this branch).
        x = ar1(0.6, 40, 1)
        res = bootstrap(
            x,
            method=ResidualBootstrap(model=AR(order=3, initial="random_block")),
            n_bootstraps=60,
            random_state=0,
        )
        assert res.values().shape == (60, 40)
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

    def test_each_criterion_selects_its_exact_order(self):
        """On one series the three criteria select three distinct, exact orders.

        ``_ar2(300, 19)`` is chosen so AIC, BIC and HQIC land on 5, 2 and 4
        respectively. Pinning the exact value (not a range) catches any rerouting of
        the criterion string or any swap of the if/elif branches: e.g. routing the
        ``"aic"`` request to the BIC penalty would return 2, not 5.
        """
        x = _ar2(300, 19)
        assert select_ar_order(x, max_lag=10, criterion="aic") == 5
        assert select_ar_order(x, max_lag=10, criterion="bic") == 2
        assert select_ar_order(x, max_lag=10, criterion="hqic") == 4

    def test_aic_penalty_magnitude_is_exactly_two(self):
        """The AIC penalty coefficient must be exactly 2.0, not larger.

        On ``_ar2(150, 1)`` AIC selects order 8. Inflating the per-parameter penalty
        (e.g. to 3.0) would over-shrink the order down to 2, so the exact selected
        order pins the penalty magnitude.
        """
        x = _ar2(150, 1)
        assert select_ar_order(x, max_lag=10, criterion="aic") == 8

    def test_hqic_penalty_coefficient_is_exactly_two(self):
        """The HQIC penalty must be exactly ``2.0 * log(log(n_eff))``.

        On ``_ar2(300, 1, phi1=0.5, phi2=0.2)`` HQIC selects order 3. Dividing instead
        of multiplying by ``log(log(n_eff))`` (a much smaller penalty) runs the order up
        to 8; doubling the leading coefficient to 3.0 shrinks it to 1. The exact value
        pins both the operator and the coefficient.
        """
        x = _ar2(300, 1, phi1=0.5, phi2=0.2)
        assert select_ar_order(x, max_lag=12, criterion="hqic") == 3

    def test_default_upper_bound_formula_is_exact(self):
        """With ``max_lag=None`` the default upper bound is ``ceil(10*log10(n))``.

        The existing tests always pass ``max_lag`` explicitly, so the default-bound
        formula is otherwise untested. On a saturating seasonal series of length 200 the
        formula gives 24 and the selection saturates there, so the returned order is
        exactly 24. Replacing ``10 * log10`` with ``10 / log10`` collapses the bound to
        5; replacing it with ``11 * log10`` pushes it to 25. Either is caught.
        """
        x = _seasonal(200, 1)
        assert select_ar_order(x, criterion="aic") == 24

    def test_upper_bound_clamped_to_half_sample(self):
        """A very large ``max_lag`` is clamped to ``n // 2 - 1``.

        With ``max_lag=1000`` on a saturating seasonal series of length 80 the binding
        bound is ``n // 2 - 1 = 39`` and the selection saturates there, so the returned
        order is exactly 39. A wrong clamp (``n // 2 + 1``, ``n // 3 - 1`` or
        ``n // 2 - 2``) returns a different order, and a float clamp (``n / 2 - 1``)
        cannot index the series at all.
        """
        x = _seasonal(80, 0)
        assert select_ar_order(x, max_lag=1000, criterion="aic") == 39

    def test_search_includes_the_top_orders(self):
        """The candidate loop and the design matrix must reach the full upper bound.

        On a saturating seasonal series with ``max_lag=5`` the selected order is exactly
        5 (the cap). Truncating the candidate range (``upper - 1``) or dropping the top
        lag columns from the design would return 3 instead, so this pins both the loop
        upper bound and the design-column construction.
        """
        x = _seasonal(120, 0)
        assert select_ar_order(x, max_lag=5, criterion="aic") == 5

    def test_min_lag_is_a_hard_lower_bound(self):
        """The search must start at ``min_lag``; orders below it are never returned.

        On an effective AR(1) series the unconstrained optimum is order 1, but with
        ``min_lag=3`` the returned order must be at least 3 (here exactly 3). A loop
        that starts at 0 instead of ``min_lag`` would let the order fall back to 1.
        """
        x = _ar2(400, 0, phi1=0.5, phi2=0.0)
        assert select_ar_order(x, min_lag=3, max_lag=8, criterion="bic") == 3

    def test_default_min_lag_is_one(self):
        """With no ``min_lag`` argument the default lower bound is 1, not 2.

        On an effective AR(1) series the selection returns order 1 under the default
        arguments. Bumping the default ``min_lag`` to 2 would force the order up to 2.
        """
        x = _ar2(400, 0, phi1=0.5, phi2=0.0)
        assert select_ar_order(x) == 1


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
        assert ar_spectral_radius(np.array([])) == pytest.approx(0.0, abs=1e-12)

    def test_unstable_error_carries_code_and_spectral_radius_context(self):
        """The raised ModelStabilityError must carry the structured code and context.

        The error message, ``code`` and ``context`` are the public failure contract.
        Pinning ``code == UNSTABLE_MODEL``, the exact ``spectral_radius`` context value
        and the message text catches mutations that blank the message (raise ``None``)
        or drop the ``context={"spectral_radius": radius}`` keyword (leaving an empty
        ``{}``). A unit-root AR(1) [1.0] has companion spectral radius exactly 1.0.
        """
        with pytest.raises(ModelStabilityError) as excinfo:
            check_ar_stability(np.array([1.0]))
        err = excinfo.value
        assert err.code == Codes.UNSTABLE_MODEL
        assert err.context.get("spectral_radius") == pytest.approx(1.0)
        assert "non-stationary" in str(err)

    def test_near_unit_warning_carries_code_context_and_message(self):
        """The near-unit-root warning must carry its code, context and message text.

        A radius in [0.98, 1.0) warns. Pinning the warning's ``code``, its
        ``spectral_radius`` context value and message text catches mutations that blank
        the warning message (``None``) or drop the ``context`` keyword (empty ``{}``).
        AR(1) [0.99] has companion spectral radius exactly 0.99.
        """
        with pytest.warns(NearUnitRootWarning) as record:
            check_ar_stability(np.array([0.99]))
        warning = record[0].message
        assert warning.code == Codes.NEAR_UNIT_ROOT
        assert warning.context.get("spectral_radius") == pytest.approx(0.99)
        assert "near a unit root" in str(warning)

    def test_warns_exactly_at_default_threshold_boundary(self):
        """A radius equal to the default threshold (0.98) must warn (``>=``, not ``>``).

        AR(1) [0.98] has companion spectral radius exactly 0.98, the default
        ``near_unit_threshold``. The boundary must be inclusive: flipping ``radius >=
        near_unit_threshold`` to ``radius >`` would suppress the warning at exactly the
        threshold. ``pytest.warns`` fails if no warning is emitted, killing that flip.
        """
        assert ar_spectral_radius(np.array([0.98])) == pytest.approx(0.98)
        with pytest.warns(NearUnitRootWarning):
            check_ar_stability(np.array([0.98]))
