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
