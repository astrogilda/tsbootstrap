"""Tests for diagnose()."""

from __future__ import annotations

import numpy as np

from tsbootstrap import diagnose


def _ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


class TestDiagnoseRecommendations:
    def test_white_noise_recommends_iid(self):
        d = diagnose(np.random.default_rng(0).standard_normal(300))
        assert d.n_obs == 300
        assert d.n_series == 1
        assert not d.dependent
        assert not d.nonstationary
        assert "IID" in d.recommended_methods

    def test_dependent_series_recommends_block(self):
        d = diagnose(_ar1(0.7, 400, 1))
        assert d.dependent
        assert not d.nonstationary
        assert "StationaryBlock" in d.recommended_methods
        assert any("block length" in note for note in d.notes)

    def test_random_walk_flagged_nonstationary(self):
        x = np.cumsum(np.random.default_rng(2).standard_normal(300))
        d = diagnose(x)
        assert d.nonstationary
        assert any("ARIMA" in m for m in d.recommended_methods)

    def test_multivariate_recommends_var(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((200, 2))
        d = diagnose(x)
        assert d.n_series == 2
        assert any("VAR" in m for m in d.recommended_methods)
