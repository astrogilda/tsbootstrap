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
        res = bootstrap(x, method=ResidualBootstrap(model=VAR(order=1)), n_bootstraps=200, random_state=3)
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


class TestVARStabilityHelpers:
    def test_var_stability_helpers(self):
        assert var_spectral_radius(_A[None]) == pytest.approx(np.max(np.abs(np.linalg.eigvals(_A))))
        check_var_stability(_A[None])  # stable, no raise
        with pytest.raises(ModelStabilityError):
            check_var_stability((2.0 * np.eye(2))[None])  # explosive
