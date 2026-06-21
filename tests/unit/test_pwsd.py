"""Tests for automatic block-length selection (tsbootstrap.block.pwsd)."""

from __future__ import annotations

import numpy as np
import pytest

from tests._helpers.dgp import ar1
from tsbootstrap.block.pwsd import optimal_block_length, resolve_block_length
from tsbootstrap.errors import MethodConfigError


class TestOptimalBlockLength:
    def test_white_noise_gives_short_blocks(self):
        x = np.random.default_rng(1).standard_normal(500)
        b = optimal_block_length(x, kind="circular")
        assert 1 <= b <= 6  # essentially no dependence

    def test_strong_dependence_gives_longer_blocks(self):
        white = optimal_block_length(ar1(0.0, 500, 2), kind="circular")
        strong = optimal_block_length(ar1(0.9, 500, 2), kind="circular")
        assert strong > white
        assert strong >= 4

    def test_block_length_increases_with_dependence(self):
        b = [optimal_block_length(ar1(phi, 600, 3), kind="circular") for phi in (0.0, 0.5, 0.8)]
        assert b[0] <= b[1] <= b[2]

    def test_returns_int_in_range(self):
        x = ar1(0.7, 200, 4)
        b = optimal_block_length(x, kind="stationary")
        assert isinstance(b, int)
        assert 1 <= b <= 200

    def test_deterministic(self):
        x = ar1(0.6, 300, 5)
        assert optimal_block_length(x, kind="circular") == optimal_block_length(x, kind="circular")

    def test_multivariate_uses_max_over_columns(self):
        indep = ar1(0.2, 400, 6)
        dep = ar1(0.9, 400, 7)
        arr = np.column_stack([indep, dep])
        b_joint = optimal_block_length(arr, kind="circular")
        b_dep = optimal_block_length(dep, kind="circular")
        assert b_joint == b_dep  # the most-dependent column drives the joint choice

    def test_reference_agreement_with_arch(self):
        arch_bootstrap = pytest.importorskip("arch.bootstrap")
        x = ar1(0.7, 500, 10)
        mine = optimal_block_length(x, kind="circular")
        ref = float(arch_bootstrap.optimal_block_length(x)["circular"].iloc[0])
        # The two implementations differ in tuning details, so require only that both
        # detect the dependence and agree to within a small factor.
        assert mine >= 3 and ref >= 3
        assert 0.4 <= mine / ref <= 2.5


class TestResolveBlockLength:
    def test_resolve_auto_and_explicit(self):
        x = ar1(0.5, 200, 8).reshape(-1, 1)
        assert resolve_block_length("auto", x, kind="circular") >= 1
        assert resolve_block_length(7, x, kind="circular") == 7

    def test_resolve_rejects_block_length_over_n(self):
        x = ar1(0.5, 50, 9).reshape(-1, 1)
        with pytest.raises(MethodConfigError):
            resolve_block_length(60, x, kind="circular")
