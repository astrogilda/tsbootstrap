"""Metamorphic and algebraic invariants for the conformal calibrators and PRNG keys.

These assert exact relationships that hold for ALL valid inputs, complementing the
example-based unit tests: ``agaci_bounds`` is scale-equivariant and never crosses, a
singleton grid collapses to plain ACI, and the counter-based key math is injective,
panel-aliasing free, and identical between the numba-free reference
(:mod:`tsbootstrap.prng_keys`) and the njit kernel. Uses Hypothesis; the active settings
profile is selected by ``HYPOTHESIS_PROFILE`` (see ``tests/conftest.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tsbootstrap import prng_keys as pk
from tsbootstrap.uq.adaptive import aci_halfwidths, agaci_bounds, nexcp_quantile

_FINITE = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=64)
_POS = st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False, width=64)
_SCALE = st.floats(min_value=1e-3, max_value=1e5, allow_nan=False, allow_infinity=False, width=64)
_U64 = st.integers(min_value=0, max_value=(1 << 64) - 1)


def _cal(min_n: int = 20, max_n: int = 80):
    return arrays(np.float64, st.integers(min_n, max_n), elements=_POS)


def _resid(min_n: int = 10, max_n: int = 60):
    return arrays(np.float64, st.integers(min_n, max_n), elements=_FINITE)


class TestAgACIInvariants:
    @given(cal=_cal(), s=_resid(), c=_SCALE)
    def test_scale_equivariance(self, cal, s, c):
        # Scaling the calibration scores and residuals by c > 0 scales the interval by c: the
        # ACI quantiles scale linearly and the +inf sentinel is data-adaptive, while the BOA
        # weights depend only on scale-invariant ratios and are unchanged.
        assume(np.ptp(cal) > 1e-6)  # non-degenerate calibration
        gammas = [0.0, 0.05, 0.2]
        lo1, hi1 = agaci_bounds(cal, s, alpha=0.1, gammas=gammas, require_signed=False)
        lo2, hi2 = agaci_bounds(c * cal, c * s, alpha=0.1, gammas=gammas, require_signed=False)
        np.testing.assert_allclose(lo2, c * lo1, rtol=1e-8, atol=1e-9 * c)
        np.testing.assert_allclose(hi2, c * hi1, rtol=1e-8, atol=1e-9 * c)

    @given(cal=_cal(), s=_resid())
    def test_bounds_are_nonnegative_and_finite(self, cal, s):
        # Lower and upper offsets are convex combinations of non-negative (sentinel-clipped)
        # half-widths, so both are >= 0 and finite: the interval never crosses.
        lo, hi = agaci_bounds(cal, s, alpha=0.1, require_signed=False)
        assert np.all(lo >= 0.0) and np.all(np.isfinite(lo))
        assert np.all(hi >= 0.0) and np.all(np.isfinite(hi))

    @given(cal=_cal(), s=_resid(), g=st.floats(0.0, 0.1, allow_nan=False, allow_infinity=False))
    def test_singleton_grid_equals_aci(self, cal, s, g):
        # With one gamma the BOA weight is trivially 1, so AgACI collapses to a symmetric
        # single-gamma ACI pass over the absolute residuals (compared only where the expert is
        # finite; a +inf expert is clipped by agaci_bounds but not by aci_halfwidths).
        assume(np.ptp(cal) > 1e-6)
        hw, _ = aci_halfwidths(cal, np.abs(s), alpha=0.1, gamma=g)
        assume(np.all(np.isfinite(hw)))
        lo, hi = agaci_bounds(cal, s, alpha=0.1, gammas=[g], require_signed=False)
        np.testing.assert_allclose(lo, hw, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(hi, hw, rtol=1e-12, atol=0.0)

    @given(cal=_cal(), test=arrays(np.float64, st.integers(10, 60), elements=_POS), c=_SCALE)
    def test_aci_halfwidths_scale_equivariance(self, cal, test, c):
        # The ACI half-widths are quantiles of the calibration scores, so scaling the inputs
        # scales the finite half-widths by c and leaves the +inf (cover-everything) ones +inf.
        assume(np.ptp(cal) > 1e-6)
        hw1, _ = aci_halfwidths(cal, test, alpha=0.1, gamma=0.05)
        hw2, _ = aci_halfwidths(c * cal, c * test, alpha=0.1, gamma=0.05)
        assert np.array_equal(np.isinf(hw1), np.isinf(hw2))
        finite = np.isfinite(hw1)
        np.testing.assert_allclose(hw2[finite], c * hw1[finite], rtol=1e-8, atol=1e-9 * c)


class TestACIInvariants:
    @given(
        cal=_cal(4, 60),
        alpha=st.floats(0.02, 0.98, allow_nan=False, allow_infinity=False, width=64),
    )
    def test_static_quantile_exactness_at_gamma0(self, cal, alpha):
        # At gamma=0 the level never adapts, so every ACI half-width is the STATIC conformal
        # quantile np.quantile(cal, 1 - alpha, method="linear"), bit-for-bit (the recursion's
        # two-branch lerp reproduces numpy's own _lerp), and every stored level is exactly alpha.
        assume(np.ptp(cal) > 1e-6)  # non-degenerate calibration
        test = np.zeros(4)
        hw, alphas = aci_halfwidths(cal, test, alpha=alpha, gamma=0.0)
        ref = float(np.quantile(cal, 1.0 - alpha, method="linear"))
        assert np.array_equal(hw, np.full(test.shape[0], ref))
        assert np.array_equal(alphas, np.full(test.shape[0], alpha))

    @given(
        cal=_cal(4, 60),
        test=arrays(np.float64, st.integers(10, 60), elements=_FINITE),
        alpha=st.floats(0.05, 0.5, allow_nan=False, allow_infinity=False, width=64),
        gamma=st.floats(1e-3, 0.2, allow_nan=False, allow_infinity=False, width=64),
    )
    def test_level_recursion_sign(self, cal, test, alpha, gamma):
        # A covered step (test[t] <= q[t]) moves the NEXT level up (alpha increases) and a missed
        # step (test[t] > q[t]) moves it down, matching alpha_{t+1} = alpha_t + gamma*(alpha - err)
        # with err = 1{miss}. The stored levels are that raw recursion clipped to [0, 1],
        # reconstructed here from the realized covers/misses; the levels always stay in [0, 1].
        assume(np.ptp(cal) > 1e-6)
        test = np.abs(test)
        hw, alphas = aci_halfwidths(cal, test, alpha=alpha, gamma=gamma)
        assert np.all(alphas >= 0.0) and np.all(alphas <= 1.0)
        raw = np.empty(test.shape[0], dtype=np.float64)
        a = float(alpha)
        for t in range(test.shape[0]):
            raw[t] = a
            err = 1.0 if test[t] > hw[t] else 0.0
            step = gamma * (alpha - err)
            if err == 0.0:
                assert step > 0.0  # covered -> level moves UP
            else:
                assert step < 0.0  # missed -> level moves DOWN
            a = a + step
        np.testing.assert_array_equal(alphas, np.clip(raw, 0.0, 1.0))

    @given(
        cal=_cal(4, 60),
        test=arrays(np.float64, st.integers(10, 60), elements=_POS),
        alpha=st.floats(0.05, 0.5, allow_nan=False, allow_infinity=False, width=64),
        gamma=st.floats(0.0, 0.2, allow_nan=False, allow_infinity=False, width=64),
    )
    def test_interior_finite_and_gamma0_constant(self, cal, test, alpha, gamma):
        # While the level stays strictly interior the half-widths are finite quantiles of the
        # positive calibration scores, so they are non-negative and finite (a level driven to 0
        # would emit the +inf cover-everything sentinel). At gamma=0 the level is frozen, so the
        # half-width vector is constant and every stored level is exactly alpha.
        assume(np.ptp(cal) > 1e-6)
        hw, alphas = aci_halfwidths(cal, test, alpha=alpha, gamma=gamma)
        if np.all(alphas > 0.0):
            assert np.all(np.isfinite(hw))
            assert np.all(hw >= 0.0)
        hw0, a0 = aci_halfwidths(cal, test, alpha=alpha, gamma=0.0)
        assert np.array_equal(hw0, np.full(hw0.shape[0], hw0[0]))
        assert np.array_equal(a0, np.full(a0.shape[0], alpha))


class TestNexCPInvariants:
    @given(
        scores=arrays(np.float64, st.integers(4, 60), elements=_FINITE),
        alpha=st.floats(0.02, 0.98, allow_nan=False, allow_infinity=False, width=64),
    )
    def test_decay1_equals_empirical_quantile(self, scores, alpha):
        # decay=1 removes recency weighting, so the recency-weighted quantile collapses to the
        # function's own type-1 rule: sort ascending, uniform cumulative weights [1..n], take
        # searchsorted(cumweights, (1 - alpha)*n, side="left") into the sorted scores. Recomputed
        # independently here and asserted exactly, pinning the searchsorted/index arithmetic.
        n = scores.shape[0]
        s_sorted = np.sort(scores, kind="stable")
        cum = np.arange(1, n + 1, dtype=np.float64)  # uniform cumulative weights at decay=1
        target = (1.0 - alpha) * n
        idx = int(np.searchsorted(cum, target, side="left"))
        ref = float(s_sorted[min(idx, n - 1)])
        assert nexcp_quantile(scores, alpha=alpha, decay=1.0) == ref

    @given(
        n_old=st.integers(2, 20),
        n_new=st.integers(2, 20),
        small=st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False, width=64),
        gap=st.floats(0.1, 50.0, allow_nan=False, allow_infinity=False, width=64),
        decay=st.floats(0.1, 0.99, allow_nan=False, allow_infinity=False, width=64),
        alpha=st.floats(0.05, 0.5, allow_nan=False, allow_infinity=False, width=64),
    )
    def test_recency_monotonicity(self, n_old, n_new, small, gap, decay, alpha):
        # Old scores small, recent scores large: shortening the memory (decay < 1) can only shift
        # weight onto the larger, more recent scores, so a smaller decay never SHRINKS the
        # quantile relative to the unweighted (decay=1) quantile.
        large = small + gap
        scores = np.concatenate([np.full(n_old, small), np.full(n_new, large)])
        q_full = nexcp_quantile(scores, alpha=alpha, decay=1.0)
        q_decay = nexcp_quantile(scores, alpha=alpha, decay=decay)
        assert q_decay >= q_full


class TestPRNGKeyInvariants:
    @given(root_a=_U64, root_b=_U64, b1=st.integers(0, 10_000), b2=st.integers(0, 10_000))
    def test_replicate_key_injective_in_b(self, root_a, root_b, b1, b2):
        # b -> key is injective for a fixed root (single-series replicate keys are exact-distinct).
        assume(b1 != b2)
        assert pk.replicate_key(root_a, root_b, b1) != pk.replicate_key(root_a, root_b, b2)

    @given(
        root_a=_U64,
        root_b=_U64,
        pairs=st.lists(
            st.tuples(st.integers(0, 500), st.integers(0, 64)),
            min_size=2,
            max_size=12,
            unique=True,
        ),
    )
    def test_no_panel_key_aliasing(self, root_a, root_b, pairs):
        # Distinct (replicate b, series slot s) map to distinct effective Philox keys, so no two
        # panel series ever share a stream: the anti-aliasing invariant behind the two distinct
        # per-axis goldens.
        keys = [pk.fold_in_key(*pk.replicate_key(root_a, root_b, b), s) for (b, s) in pairs]
        assert len(set(keys)) == len(keys)

    def test_series_slot0_is_identity(self):
        # Slot 0 folds to the identity, so a lone-series panel reproduces the single-series key.
        assert pk.hash_series_words(0) == (0, 0)

    def test_philox_reproduces_published_kat(self):
        # The reference round function reproduces the published Random123 zero-vector.
        assert pk.philox4x32_10((0, 0, 0, 0), (0, 0)) == (
            0x6627E8D5,
            0xE169C58D,
            0xBC57AC4C,
            0x9B00DBD8,
        )

    @given(root_a=_U64, root_b=_U64, b=st.integers(0, 100_000))
    def test_njit_kernel_matches_reference(self, root_a, root_b, b):
        # The njit kernel and the numba-free reference compute the identical key for every
        # (root, b), so the fast path cannot silently drift from the reference oracle.
        pytest.importorskip("numba")
        from tsbootstrap.block import _compiled as sk

        kh, kl = sk._replicate_key(np.uint64(root_a), np.uint64(root_b), b)
        assert (int(kh), int(kl)) == pk.replicate_key(root_a, root_b, b)
