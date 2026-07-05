"""Tests for adaptive / nonexchangeable conformal calibration (ACI, NexCP, AgACI)."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap.uq import AgACIBounds, aci_halfwidths, agaci_bounds, nexcp_quantile
from tsbootstrap.uq.adaptive import _boa_aggregate


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


def _two_sided_pinball(
    lower_off: np.ndarray, upper_off: np.ndarray, s: np.ndarray, alpha: float
) -> float:
    """Total two-sided pinball loss of interval offsets against signed residuals s.

    Lower endpoint offset uses tau = alpha/2, upper uses tau = 1 - alpha/2, matching the
    two mixture() calls in the AgACI reference (acp_gamma.R:60,62). Pinball base loss is
    ``(1{s < off} - tau) * (off - s)``.
    """
    tau_lo, tau_hi = alpha / 2.0, 1.0 - alpha / 2.0
    loss_lo = ((s < lower_off).astype(np.float64) - tau_lo) * (lower_off - s)
    loss_hi = ((s < upper_off).astype(np.float64) - tau_hi) * (upper_off - s)
    return float(loss_lo.sum() + loss_hi.sum())


class TestBOA:
    def test_boa_round1_uniform_and_first_adaptation_on_distinct_experts(self):
        # Round 1 has eta_inv2 all zero, so the weights fall back to the uniform prior and
        # the first aggregated endpoint is the mean of the (distinct) expert endpoints.
        # A hand-built matrix with distinct row-0 entries is required: at t=0 all ACI
        # experts share alpha_0, so a real expert matrix has an identical first row and
        # cannot exercise the mean genuinely.
        rng = np.random.default_rng(0)
        experts = rng.standard_normal((20, 3))
        targets = rng.standard_normal(20)
        pred, weights = _boa_aggregate(experts, targets, tau=0.5)
        assert np.isclose(pred[0], experts[0].mean())
        assert np.allclose(weights[0], 1.0 / 3.0)
        # By a later step the weights must have adapted off uniform (round >= 2 branch).
        assert not np.allclose(weights[-1], 1.0 / 3.0)

    def test_boa_zero_regret_expert_retains_prior_mass(self):
        # Symmetric triple [c-d, c, c+d]: at every round the middle expert equals the
        # uniform-mean aggregate only at t=0, but at t=0 ALL experts are scored, so it is
        # the round-1/round-2 mask that is under test. At round 2 experts 0 and 2 have
        # accumulated regret while the middle stays zero-regret (masked out of the softmax
        # and retaining its prior). With the active-prior-mass scaling w0[nz].sum() (= 2)
        # the middle normalized weight is exactly 1/3; the incorrect sum(w0) (= 3) scaling
        # would give 1/4. This pins the fidelity choice and the divide-by-zero mask.
        c, d = 1.0, 0.7
        T = 12
        experts = np.tile(np.array([c - d, c, c + d]), (T, 1))
        targets = np.linspace(c - 2.0, c + 2.0, T)  # crosses c so the gradient is exercised
        pred, weights = _boa_aggregate(experts, targets, tau=0.5)
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(weights))
        assert np.allclose(weights[0], 1.0 / 3.0)  # round-1 uniform fallback
        assert np.isclose(pred[0], c)
        assert np.isclose(weights[1, 1], 1.0 / 3.0)  # active-prior-mass, not 1/4


class TestAgACI:
    def test_agaci_reduces_to_single_aci_for_singleton_grid(self):
        # +inf-free fixture (large-scale calibration, short signed stream) so no expert is
        # ever driven to alpha_t <= 0 and the sentinel clip never fires. With K=1 the BOA
        # weight is trivially 1 every round, so AgACI collapses exactly to single-gamma ACI
        # with symmetric bounds.
        rng = np.random.default_rng(1)
        cal = np.abs(rng.standard_normal(300)) * 3.0 + 1.0
        s = rng.standard_normal(30)  # signed, has negatives
        g = 0.005
        bounds = agaci_bounds(cal, s, alpha=0.1, gammas=[g])
        hw, _ = aci_halfwidths(cal, np.abs(s), alpha=0.1, gamma=g)
        assert np.all(np.isfinite(hw))  # fixture really is sentinel-free
        assert np.allclose(bounds.lower, hw)
        assert np.allclose(bounds.upper, hw)

    def test_agaci_regret_bounded_vs_fixed_gamma_experts(self):
        # BOA guarantees O(sqrt(T log K)) regret to the best fixed convex combination of
        # experts, not a small-constant bound to the single best expert. Assert AgACI is no
        # worse than the average expert AND within the regret budget of the best expert.
        rng = np.random.default_rng(2)
        cal = np.abs(rng.standard_normal(300)) * 2.0 + 0.5
        s = rng.standard_normal(120)  # signed
        gammas = [0.0, 0.005, 0.02, 0.05]
        K, T = len(gammas), s.size
        bounds = agaci_bounds(cal, s, alpha=0.1, gammas=gammas)
        agaci_loss = _two_sided_pinball(-bounds.lower, bounds.upper, s, 0.1)
        expert_losses = []
        for g in gammas:
            q, _ = aci_halfwidths(cal, np.abs(s), alpha=0.1, gamma=g)
            assert np.all(np.isfinite(q))  # no sentinel clipping in this fixture
            expert_losses.append(_two_sided_pinball(-q, q, s, 0.1))
        expert_losses = np.asarray(expert_losses)
        assert agaci_loss <= expert_losses.mean() + 1e-9  # safe: no worse than the average
        budget = float(expert_losses.min()) + 4.0 * np.sqrt(T * np.log(K))  # BOA regret budget
        assert agaci_loss <= budget

    def test_agaci_coverage_recovery_empirical_regression(self):
        # AgACI has NO finite-T or asymptotic coverage theorem; this is a seed-pinned smoke
        # test of empirical recovery under a variance shift, not an invariant.
        rng = np.random.default_rng(3)
        cal = np.abs(rng.standard_normal(500))  # scale-1
        s = 3.0 * rng.standard_normal(3000)  # signed, scale-3: variance shift up
        hw0, _ = aci_halfwidths(cal, np.abs(s), alpha=0.1, gamma=0.0)  # static gamma=0 ACI
        cov_static = float(((-hw0 <= s) & (s <= hw0)).mean())
        assert cov_static < 0.75  # static badly under-covers under the shift
        bounds = agaci_bounds(cal, s, alpha=0.1)
        cov_agaci = float(((-bounds.lower <= s) & (s <= bounds.upper)).mean())
        assert 0.82 <= cov_agaci <= 0.97  # AgACI adapts back toward a generous 0.90 band

    def test_agaci_shapes_and_never_crosses(self):
        rng = np.random.default_rng(4)
        cal = np.abs(rng.standard_normal(200))
        s = rng.standard_normal(80)  # signed
        bounds = agaci_bounds(cal, s, alpha=0.1)
        assert isinstance(bounds, AgACIBounds)
        assert bounds.lower.shape == (80,)
        assert bounds.upper.shape == (80,)
        assert bounds.lower.dtype == np.float64
        assert bounds.upper.dtype == np.float64
        assert np.all(bounds.lower >= 0.0)
        assert np.all(bounds.upper >= 0.0)
        assert np.all(np.isfinite(bounds.lower))
        assert np.all(np.isfinite(bounds.upper))
        point = np.zeros(80)  # non-crossing: pred + upper >= pred - lower everywhere
        assert np.all((point + bounds.upper) >= (point - bounds.lower))

    def test_agaci_rejects_absolute_scores_and_nan(self):
        rng = np.random.default_rng(5)
        cal = np.abs(rng.standard_normal(100))
        abs_scores = np.abs(rng.standard_normal(20))  # all non-negative, length >= 8
        with pytest.raises(ValueError):
            agaci_bounds(cal, abs_scores, alpha=0.1)  # absolute-scores footgun detector
        bounds = agaci_bounds(cal, abs_scores, alpha=0.1, require_signed=False)  # opt out
        assert bounds.lower.shape == (20,)
        s = rng.standard_normal(20)
        s[3] = np.nan
        with pytest.raises(ValueError):
            agaci_bounds(cal, s, alpha=0.1)  # finiteness guard

    def test_agaci_invalid_inputs_raise(self):
        cal = np.abs(np.random.default_rng(6).standard_normal(50))
        s = np.random.default_rng(7).standard_normal(20)  # signed
        with pytest.raises(ValueError):
            agaci_bounds(np.array([]), s)
        with pytest.raises(ValueError):
            agaci_bounds(cal, np.array([]))
        with pytest.raises(ValueError):
            agaci_bounds(cal, s, gammas=[])
        with pytest.raises(ValueError):
            agaci_bounds(cal, s, gammas=[0.01, np.inf])
        with pytest.raises(ValueError):
            agaci_bounds(cal, s, gammas=[0.01, -0.01])
        with pytest.raises(ValueError):
            agaci_bounds(cal, s, alpha=0.0)
        with pytest.raises(ValueError):
            agaci_bounds(cal, s, alpha=1.0)
        with pytest.raises(ValueError):
            agaci_bounds(cal, s, boa_regret_constant=0.0)

    def test_agaci_sentinel_deterministic_and_degenerate_data_finite(self):
        # Force a +inf expert: tiny calibration scores + large residuals drive a large-gamma
        # expert to alpha_t <= 0, emitting +inf half-widths that MUST be clipped.
        rng = np.random.default_rng(8)
        cal = np.abs(rng.standard_normal(100)) * 0.1
        s = 5.0 * rng.standard_normal(60)  # large signed residuals -> frequent misses
        g = 0.09
        q, _ = aci_halfwidths(cal, np.abs(s), alpha=0.1, gamma=g)
        assert np.any(~np.isfinite(q))  # this expert really does emit +inf
        bounds = agaci_bounds(cal, s, alpha=0.1, gammas=[g])
        assert np.all(np.isfinite(bounds.lower))
        assert np.all(np.isfinite(bounds.upper))
        # deterministic floored + capped sentinel, pinned
        finite_q = q[np.isfinite(q)]
        range_ref = max(
            float(finite_q.max()) if finite_q.size else 0.0, float(np.abs(s).max()), 1.0
        )
        sentinel = min(max(1.0, 10.0 * range_ref), 1e150)
        q_clipped = np.where(np.isfinite(q), q, sentinel)
        assert np.allclose(bounds.lower, q_clipped)  # K=1: the clipped expert IS the bound
        assert np.allclose(bounds.upper, q_clipped)
        # Degenerate all-equal data must stay finite: no sentinel->0 no-op under-width, no
        # eta_inv2 overflow -> nan weights.
        cal_deg = np.full(50, 2.0)
        s_deg = np.zeros(40)
        b2 = agaci_bounds(cal_deg, s_deg, alpha=0.1, require_signed=False)
        assert np.all(np.isfinite(b2.lower)) and np.all(np.isfinite(b2.upper))

    def test_agaci_sentinel_scales_with_data_no_inversion(self):
        # The +inf "cover everything" expert is clipped to a data-adaptive sentinel, so it
        # stays the WIDEST interval at any data magnitude. A fixed cap below the data scale
        # (the old 1e6) would clip it BELOW a finite expert and invert its meaning. Assert
        # scale-equivariance across a factor that exceeds the old cap: agaci_bounds scales
        # exactly with the data (quantiles and the sentinel both scale), so a fixed cap
        # would break this at large scale.
        rng = np.random.default_rng(0)
        cal = np.abs(rng.standard_normal(200))
        s = rng.standard_normal(60) * 20.0  # large signed residuals -> the 0.5 expert -> +inf
        q, _ = aci_halfwidths(cal, np.abs(s), alpha=0.1, gamma=0.5)
        assert np.isinf(q).any()  # the sentinel path is exercised
        lo1, hi1 = agaci_bounds(cal, s, alpha=0.1, gammas=[0.0, 0.5])
        c = 1e8  # well past the retired 1e6 cap, well below the 1e150 overflow guard
        lo2, hi2 = agaci_bounds(cal * c, s * c, alpha=0.1, gammas=[0.0, 0.5])
        np.testing.assert_allclose(lo2, c * lo1, rtol=1e-9, atol=0.0)
        np.testing.assert_allclose(hi2, c * hi1, rtol=1e-9, atol=0.0)
        assert hi2.max() > 1e6  # not clipped to the old cap
