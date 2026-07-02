"""Tests for the classical confidence-interval layer (tsbootstrap.uq.classical)."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap.errors import Codes, MethodConfigError
from tsbootstrap.uq import basic_interval, percentile_interval


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


class TestTopLevelExports:
    """The classical CI surface is re-exported at the top level and from the uq package."""

    _CLASSICAL_NAMES = (
        "percentile_interval",
        "basic_interval",
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
