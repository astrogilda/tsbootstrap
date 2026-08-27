"""Tests for automatic block-length selection (tsbootstrap.block.pwsd)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests._helpers.dgp import ar1
from tsbootstrap.block.pwsd import optimal_block_length, resolve_block_length
from tsbootstrap.errors import MethodConfigError

# Politis-White selections pinned on one fixed AR(1) draw. The two kinds resolve to
# DIFFERENT lengths, so a selector that ignores its input and returns a constant fails
# here whatever constant it picks. Both sit well clear of a rounding boundary (the
# pre-ceil values are 11.54 and 13.21), so the pins are not fragile across platforms.
PIN_PHI, PIN_N, PIN_SEED = 0.6, 300, 5
PINNED_BLOCK_LENGTH = {"stationary": 12, "circular": 14}

REPO_ROOT = Path(__file__).resolve().parents[2]


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

    @pytest.mark.parametrize(("kind", "expected"), sorted(PINNED_BLOCK_LENGTH.items()))
    def test_matches_pinned_selection(self, kind, expected):
        """Pin the selected length, so drift or a constant return is caught.

        This replaces a test that compared two identical calls to the same function
        with the same argument. That comparison holds for every implementation,
        including one that ignores its input, so it could not fail.
        """
        assert optimal_block_length(ar1(PIN_PHI, PIN_N, PIN_SEED), kind=kind) == expected

    def test_deterministic_under_global_rng_perturbation(self):
        """The selector must not consult the global RNG.

        Politis-White is a closed-form plug-in rule, so re-seeding and burning the
        legacy global stream between two calls must not move the answer. If anyone
        adds sampling or subsampling to the estimator, these two calls diverge.
        """
        x = ar1(PIN_PHI, PIN_N, PIN_SEED)
        np.random.seed(0)  # noqa: NPY002  legacy global stream is the thing under test
        first = optimal_block_length(x, kind="circular")
        np.random.seed(987654321)  # noqa: NPY002
        np.random.random(10_000)  # noqa: NPY002  burn the stream
        second = optimal_block_length(x, kind="circular")
        assert first == second == PINNED_BLOCK_LENGTH["circular"]

    def test_deterministic_across_a_fresh_interpreter(self):
        """A second interpreter, with a different hash seed, must agree exactly.

        Repeating a call inside one process cannot see state that is fixed for a
        process lifetime: hash randomization, import order, or a module-level cache
        populated once. Determinism only means something across that boundary, so
        the check has to spend a subprocess to cross it.
        """
        script = (
            "from tests._helpers.dgp import ar1\n"
            "from tsbootstrap.block.pwsd import optimal_block_length\n"
            f"print(optimal_block_length(ar1({PIN_PHI}, {PIN_N}, {PIN_SEED}), kind='circular'))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONHASHSEED": "12345"},
        )
        assert int(completed.stdout.strip()) == PINNED_BLOCK_LENGTH["circular"]

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
