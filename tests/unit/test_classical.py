"""Tests for the classical confidence-interval layer (tsbootstrap.uq.classical)."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap.errors import Codes, MethodConfigError
from tsbootstrap.uq import (
    basic_interval,
    block_jackknife_se,
    jackknife_statistics,
    percentile_interval,
    studentized_interval,
)


def _mean_stat(values, _indices):
    return float(np.mean(values))


class TestPercentileBasic:
    def test_percentile_matches_numpy_quantile_1d(self):
        rng = np.random.default_rng(0)
        stats = rng.standard_normal(500)
        lo, hi = percentile_interval(stats, alpha=0.1)
        np.testing.assert_array_equal(lo, np.quantile(stats, 0.05))
        np.testing.assert_array_equal(hi, np.quantile(stats, 0.95))

    def test_percentile_matches_numpy_quantile_2d(self):
        rng = np.random.default_rng(1)
        stats = rng.standard_normal((500, 3))
        lo, hi = percentile_interval(stats, alpha=0.1)
        np.testing.assert_array_equal(lo, np.quantile(stats, 0.05, axis=0))
        np.testing.assert_array_equal(hi, np.quantile(stats, 0.95, axis=0))
        assert lo.shape == (3,)
        assert hi.shape == (3,)

    def test_basic_reflection_hand_pin(self):
        # Sorted [0, 1, 2, 3, 10]; alpha=0.5 -> q_lo = quantile(.25) = 1.0,
        # q_hi = quantile(.75) = 3.0. theta_hat = 5.0, so the reflected bounds are
        # lower = 2*5 - 3 = 7.0 and upper = 2*5 - 1 = 9.0 (asymmetric about theta_hat,
        # so a percentile interval (1.0, 3.0) would fail this pin).
        stats = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
        lo, hi = basic_interval(stats, 5.0, alpha=0.5)
        np.testing.assert_array_equal(lo, 7.0)
        np.testing.assert_array_equal(hi, 9.0)

    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
    def test_percentile_rejects_alpha_out_of_range(self, bad_alpha):
        stats = np.arange(10.0)
        with pytest.raises(MethodConfigError) as exc:
            percentile_interval(stats, alpha=bad_alpha)
        assert exc.value.code == Codes.INVALID_PARAMETER

    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
    def test_basic_rejects_alpha_out_of_range(self, bad_alpha):
        stats = np.arange(10.0)
        with pytest.raises(MethodConfigError) as exc:
            basic_interval(stats, 4.5, alpha=bad_alpha)
        assert exc.value.code == Codes.INVALID_PARAMETER

    def test_percentile_rejects_empty(self):
        with pytest.raises(ValueError):
            percentile_interval(np.empty(0))


class TestJackknife:
    def test_jackknife_statistics_leave_one_out_means(self):
        x = np.array([2.0, 4.0, 6.0, 8.0])
        got = jackknife_statistics(x, _mean_stat)
        # Leave-one-out means: total = 20, theta_(i) = (20 - x_i) / 3.
        expected = np.array([(20.0 - xi) / 3.0 for xi in x])
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_block_length_one_equals_delete_one_closed_form(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(40)
        se = block_jackknife_se(x, _mean_stat, block_length=1)
        expected = float(np.std(x, ddof=1) / np.sqrt(x.shape[0]))
        np.testing.assert_allclose(se, expected, rtol=1e-12)

    def test_block_jackknife_g_minus_1_over_g_hand_pin(self):
        # x=[1..6], L=2 -> g=3 blocks. Deleting each block leaves means 4.5, 3.5, 2.5;
        # their mean is 3.5 so sum of squared deviations is 1 + 0 + 1 = 2, and
        # se^2 = (g-1)/g * 2 = (2/3)*2 = 4/3.
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        se = block_jackknife_se(x, _mean_stat, block_length=2)
        np.testing.assert_allclose(se, np.sqrt(4.0 / 3.0), rtol=1e-12)

    def test_block_se_exceeds_delete_one_under_dependence(self):
        # Positively autocorrelated AR(1); the block jackknife (which respects the
        # dependence) must inflate the SE relative to the iid delete-one estimate.
        rng = np.random.default_rng(0)
        phi = 0.8
        n = 400
        x = np.empty(n)
        x[0] = rng.standard_normal()
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.standard_normal()
        block_se = block_jackknife_se(x, _mean_stat, block_length=20)
        delete_one = block_jackknife_se(x, _mean_stat, block_length=1)
        assert block_se > delete_one

    def test_too_few_groups_raises(self):
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(MethodConfigError) as exc:
            block_jackknife_se(x, _mean_stat, block_length=2)
        assert exc.value.code == Codes.INVALID_PARAMETER


class TestStudentized:
    def test_pivot_orientation_sign_pin(self):
        # se_b == se_hat == 1, theta_hat == 0, alpha=0.5 -> t == statistics, and
        # t_lo = quantile(.25) = 1.0, t_hi = quantile(.75) = 3.0. The correct minus
        # sign puts the UPPER pivot quantile on the LOWER bound:
        #   lower = 0 - 3.0 = -3.0, upper = 0 - 1.0 = -1.0.
        # A '+' mutant would return (3.0, 1.0); a quantile swap would return (-1, -3).
        stats = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
        ses = np.ones_like(stats)
        lo, hi = studentized_interval(stats, ses, 0.0, 1.0, alpha=0.5)
        np.testing.assert_array_equal(lo, -3.0)
        np.testing.assert_array_equal(hi, -1.0)
        assert lo < hi

    def test_zero_replicate_se_raises(self):
        stats = np.array([1.0, 2.0, 3.0])
        ses = np.array([1.0, 0.0, 1.0])
        with pytest.raises(ValueError):
            studentized_interval(stats, ses, 2.0, 1.0)

    def test_zero_point_se_raises(self):
        stats = np.array([1.0, 2.0, 3.0])
        ses = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError):
            studentized_interval(stats, ses, 2.0, 0.0)


class TestTopLevelExports:
    """The classical CI surface is re-exported at the top level and from the uq package."""

    _CLASSICAL_NAMES = (
        "percentile_interval",
        "basic_interval",
        "jackknife_statistics",
        "block_jackknife_se",
        "studentized_interval",
    )

    def test_names_exported_and_in_all(self):
        import tsbootstrap
        import tsbootstrap.uq as uq

        for name in self._CLASSICAL_NAMES:
            assert hasattr(tsbootstrap, name), name
            assert name in tsbootstrap.__all__, name
            assert name in uq.__all__, name

    def test_flat_import_matches_submodule(self):
        import tsbootstrap
        import tsbootstrap.uq as uq

        for name in self._CLASSICAL_NAMES:
            assert getattr(tsbootstrap, name) is getattr(uq, name)
