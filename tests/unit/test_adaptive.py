"""Tests for adaptive / nonexchangeable conformal calibration (ACI, NexCP)."""

from __future__ import annotations

import numpy as np

from tsbootstrap.uq import aci_halfwidths, nexcp_quantile


class TestACI:
    def test_aci_recovers_coverage_under_distribution_shift(self):
        rng = np.random.default_rng(0)
        calibration = np.abs(rng.standard_normal(500))  # scale-1 residuals
        test = np.abs(3.0 * rng.standard_normal(3000))  # scale-3: variance shift up

        hw_static, _ = aci_halfwidths(calibration, test, alpha=0.1, gamma=0.0)
        hw_aci, _ = aci_halfwidths(calibration, test, alpha=0.1, gamma=0.05)

        cov_static = float((test <= hw_static).mean())
        cov_aci = float((test <= hw_aci).mean())
        assert cov_static < 0.7  # static calibration badly under-covers under the shift
        assert 0.83 <= cov_aci <= 0.97  # ACI adapts back toward the 0.90 target

    def test_aci_gamma_zero_is_static(self):
        cal = np.abs(np.random.default_rng(1).standard_normal(300))
        test = np.abs(np.random.default_rng(2).standard_normal(120))
        hw, alphas = aci_halfwidths(cal, test, alpha=0.1, gamma=0.0)
        assert np.allclose(alphas, 0.1)  # the level never moves
        assert np.allclose(hw, hw[0])  # so the half-width is constant

    def test_aci_shapes(self):
        cal = np.abs(np.random.default_rng(3).standard_normal(200))
        test = np.abs(np.random.default_rng(4).standard_normal(50))
        hw, alphas = aci_halfwidths(cal, test, alpha=0.1, gamma=0.05)
        assert hw.shape == (50,)
        assert alphas.shape == (50,)
        assert np.all(alphas >= 0.0) and np.all(alphas <= 1.0)


class TestNexCP:
    def test_nexcp_decay_one_is_empirical_quantile(self):
        s = np.random.default_rng(5).standard_normal(400) ** 2
        q = nexcp_quantile(s, alpha=0.1, decay=1.0)
        assert np.quantile(s, 0.85) <= q <= np.quantile(s, 0.95)

    def test_nexcp_recency_surfaces_recent_volatility(self):
        # Old scores small, recent scores large: recency weighting must widen the quantile.
        s = np.concatenate([np.full(180, 0.5), np.full(20, 5.0)])
        q_uniform = nexcp_quantile(s, alpha=0.1, decay=1.0)
        q_recent = nexcp_quantile(s, alpha=0.1, decay=0.8)
        assert q_recent > q_uniform
