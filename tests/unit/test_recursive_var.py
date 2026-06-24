"""Tests for the recursive VAR bootstrap (multivariate)."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap.api import bootstrap
from tsbootstrap.errors import MethodConfigError, ModelStabilityError
from tsbootstrap.methods import VAR, ResidualBootstrap
from tsbootstrap.model.stability import check_var_stability, var_spectral_radius

_A = np.array([[0.5, 0.1], [0.2, 0.4]])
_COV = np.array([[1.0, 0.6], [0.6, 1.0]])


def _var1(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(_COV)
    x = np.zeros((n, 2))
    for t in range(1, n):
        e = chol @ rng.standard_normal(2)
        x[t] = _A @ x[t - 1] + e
    return x


class TestVARResidualBootstrap:
    def test_var_shape_and_determinism(self):
        x = _var1(300, 1)
        spec = ResidualBootstrap(model=VAR(order=1))
        a = bootstrap(x, method=spec, n_bootstraps=8, random_state=5)
        b = bootstrap(x, method=spec, n_bootstraps=8, random_state=5)
        assert a.values().shape == (8, 300, 2)
        np.testing.assert_array_equal(a.values(), b.values())
        assert a.indices() is None

    def test_var_requires_multivariate(self):
        x = np.random.default_rng(0).standard_normal(100)  # 1-D
        with pytest.raises(MethodConfigError):
            bootstrap(x, method=ResidualBootstrap(model=VAR(order=1)), n_bootstraps=2)

    def test_var_preserves_cross_correlation(self):
        x = _var1(500, 2)
        orig_xcorr = np.corrcoef(x[:, 0], x[:, 1])[0, 1]
        res = bootstrap(
            x, method=ResidualBootstrap(model=VAR(order=1)), n_bootstraps=200, random_state=3
        )
        xcorr = np.array([np.corrcoef(s[:, 0], s[:, 1])[0, 1] for s in res.values()])
        assert abs(xcorr.mean() - orig_xcorr) < 0.12

    def test_var_unstable_fit_raises(self):
        rng = np.random.default_rng(0)
        explosive = np.array([[1.2, 0.0], [0.0, 1.1]])
        x = np.zeros((80, 2))
        x[0] = [1.0, 1.0]
        for t in range(1, 80):
            x[t] = explosive @ x[t - 1] + 0.1 * rng.standard_normal(2)
        with pytest.raises(ModelStabilityError):
            bootstrap(x, method=ResidualBootstrap(model=VAR(order=1)), n_bootstraps=2)

    def test_var_stability_skip_returns_failed_result(self):
        # stability_policy="skip" must fail an unstable VAR run honestly (empty, flagged failed)
        # instead of raising -- pins that the policy is threaded into the stability guard.
        rng = np.random.default_rng(0)
        explosive = np.array([[1.2, 0.0], [0.0, 1.1]])
        x = np.zeros((80, 2))
        x[0] = [1.0, 1.0]
        for t in range(1, 80):
            x[t] = explosive @ x[t - 1] + 0.1 * rng.standard_normal(2)
        res = bootstrap(
            x,
            method=ResidualBootstrap(model=VAR(order=1, stability_policy="skip")),
            n_bootstraps=5,
        )
        assert res.metadata.failed is True
        assert len(res) == 0

    def test_var_positive_burn_in_runs(self):
        # burn_in > 0 generates and discards extra leading steps; the result must keep length n.
        x = _var1(200, 1)
        res = bootstrap(
            x,
            method=ResidualBootstrap(model=VAR(order=1, burn_in=20)),
            n_bootstraps=8,
            random_state=0,
        )
        assert res.values().shape == (8, 200, 2)
        assert np.isfinite(res.values()).all()

    def test_var_order_too_large_raises(self):
        # order * d >= n is an over-parameterised fit and must be rejected.
        from tsbootstrap.model.fit import fit_var

        with pytest.raises(MethodConfigError):
            fit_var(_var1(8, 0), order=5)  # 5 * 2 = 10 >= 8

    def test_var_fixed_initial_uses_observed_block(self):
        # initial defaults to "fixed": every path must begin with the observed first p rows, not a
        # random block. Pins that the configured initial choice is honoured (not silently dropped).
        x = _var1(200, 1)
        res = bootstrap(
            x, method=ResidualBootstrap(model=VAR(order=2)), n_bootstraps=10, random_state=0
        )
        np.testing.assert_allclose(res.values()[:, :2, :], np.broadcast_to(x[:2], (10, 2, 2)))


class TestVARStabilityHelpers:
    def test_var_stability_helpers(self):
        assert var_spectral_radius(_A[None]) == pytest.approx(np.max(np.abs(np.linalg.eigvals(_A))))
        check_var_stability(_A[None])  # stable, no raise
        with pytest.raises(ModelStabilityError):
            check_var_stability((2.0 * np.eye(2))[None])  # explosive

    def test_var2_spectral_radius_uses_companion_subblock(self):
        # VAR(2) with diagonal lag matrices A1 = 0.5 I, A2 = 0.2 I decouples per dimension to
        # the companion [[0.5, 0.2], [1, 0]], whose eigenvalues solve r^2 - 0.5 r - 0.2 = 0.
        # Without the identity sub-block the second lag is ignored and the radius is wrong.
        coefs = np.stack([0.5 * np.eye(2), 0.2 * np.eye(2)])
        expected = (0.5 + np.sqrt(0.25 + 0.8)) / 2
        assert var_spectral_radius(coefs) == pytest.approx(expected, abs=1e-4)

    def test_var_near_unit_root_warns(self):
        from tsbootstrap.errors import NearUnitRootWarning

        with pytest.warns(NearUnitRootWarning):
            check_var_stability((0.99 * np.eye(2))[None])  # radius 0.99 in [0.98, 1.0)

    def test_empty_var_coefs_has_zero_radius(self):
        # An order-0 (empty) coefficient tensor has no dynamics -> radius 0 (stable).
        assert var_spectral_radius(np.zeros((0, 2, 2))) == pytest.approx(0.0, abs=1e-12)

    def test_unit_root_radius_exactly_one_raises(self):
        # Radius exactly 1.0 is non-stationary and must raise: the guard is `>= 1.0`, not `> 1.0`.
        # coefs = I gives a companion equal to the identity, whose spectral radius is exactly 1.0;
        # a `> 1.0` comparison would let this through.
        coefs = np.eye(2)[None]
        assert var_spectral_radius(coefs) == pytest.approx(1.0, abs=1e-12)
        with pytest.raises(ModelStabilityError):
            check_var_stability(coefs)

    def test_radius_exactly_at_threshold_warns(self):
        # Radius exactly equal to near_unit_threshold (default 0.98) must warn: the test is
        # `>= near_unit_threshold`, not `> near_unit_threshold`. coefs = 0.98 I has radius
        # exactly 0.98, so a `>` comparison would miss the warning.
        from tsbootstrap.errors import NearUnitRootWarning

        coefs = 0.98 * np.eye(2)[None]
        assert var_spectral_radius(coefs) == pytest.approx(0.98, abs=1e-12)
        with pytest.warns(NearUnitRootWarning):
            check_var_stability(coefs)

    def test_unstable_error_carries_code_and_context(self):
        # The raised ModelStabilityError must carry the explicit UNSTABLE_MODEL code and a
        # spectral_radius context payload (not None / dropped kwargs), and a descriptive message.
        from tsbootstrap.errors import Codes

        coefs = 2.0 * np.eye(2)[None]
        with pytest.raises(ModelStabilityError, match="non-stationary") as exc:
            check_var_stability(coefs)
        assert exc.value.code == Codes.UNSTABLE_MODEL
        assert exc.value.context == {"spectral_radius": pytest.approx(2.0)}

    def test_near_unit_warning_carries_message_and_context(self):
        # The near-unit-root warning must carry a descriptive message (not None) and a
        # spectral_radius context payload (not None / dropped kwarg).
        import warnings

        from tsbootstrap.errors import NearUnitRootWarning

        coefs = 0.99 * np.eye(2)[None]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_var_stability(coefs)
        near = [w.message for w in caught if isinstance(w.message, NearUnitRootWarning)]
        assert len(near) == 1
        assert "near a unit root" in str(near[0])
        assert near[0].context == {"spectral_radius": pytest.approx(0.99)}


class TestVarRecurrenceNumpy:
    def test_single_step_recurrence_exact(self):
        # Pin the pure-numpy fallback for a single generated step (m=1). The accumulator is
        # built from innovations[:, 0]; an off-by-one to innovations[:, 1] would index past the
        # m=1 axis and raise IndexError. Output must equal c + A @ X0 + e exactly.
        from tsbootstrap.engines.var import _var_recurrence_numpy

        coefs = np.array([[[0.5, 0.1], [0.2, 0.4]]])  # (p=1, d=2, d=2)
        intercept = np.array([1.0, -1.0])
        inits = np.array([[1.0, 2.0], [3.0, 4.0]])  # (B=2, d=2)
        innovations = np.array([[[0.5, 0.5]], [[1.0, 1.0]]])  # (B=2, m=1, d=2)
        path = np.zeros((2, 2, 2))
        path[:, 0] = inits
        _var_recurrence_numpy(coefs, intercept, path, innovations, 1, 1)
        expected = np.stack([intercept + coefs[0] @ inits[b] + innovations[b, 0] for b in range(2)])
        np.testing.assert_allclose(path[:, 1], expected)
