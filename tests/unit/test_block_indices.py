"""Tests for block index generation (moving/circular/non-overlapping/stationary)."""

from __future__ import annotations

import numpy as np
import pytest

from tests._helpers.dgp import ar1
from tsbootstrap.api import bootstrap
from tsbootstrap.errors import MethodConfigError
from tsbootstrap.methods import (
    CircularBlock,
    MovingBlock,
    NonOverlappingBlock,
    StationaryBlock,
)

N = 60


def _full_blocks(idx: np.ndarray, length: int):
    n_full = len(idx) // length
    for b in range(n_full):
        yield idx[b * length : (b + 1) * length]


class TestMovingBlock:
    def test_moving_block_is_contiguous_without_wrap(self):
        x = np.arange(N, dtype=float)  # values == indices
        res = bootstrap(x, method=MovingBlock(block_length=5), n_bootstraps=8, random_state=0)
        for sample in res:
            idx = sample.values.astype(int)
            for block in _full_blocks(idx, 5):
                assert np.all(np.diff(block) == 1)  # contiguous, no wrap
                assert block[0] <= N - 5  # valid moving start

    def test_block_method_determinism(self):
        x = ar1(0.5, 80, 6)
        a = bootstrap(x, method=MovingBlock(block_length=6), n_bootstraps=8, random_state=7)
        b = bootstrap(x, method=MovingBlock(block_length=6), n_bootstraps=8, random_state=7)
        np.testing.assert_array_equal(a.values(), b.values())

    def test_explicit_block_length_over_n_raises(self):
        x = np.arange(20.0)
        with pytest.raises(MethodConfigError):
            bootstrap(x, method=MovingBlock(block_length=25), n_bootstraps=2)


class TestCircularBlock:
    def test_circular_block_wraps(self):
        x = np.arange(N, dtype=float)
        res = bootstrap(x, method=CircularBlock(block_length=7), n_bootstraps=20, random_state=1)
        wrapped = False
        for sample in res:
            idx = sample.values.astype(int)
            for block in _full_blocks(idx, 7):
                assert np.all(np.diff(block) % N == 1)  # contiguous modulo n
                if np.any(np.diff(block) != 1):
                    wrapped = True
        assert wrapped  # with 20 reps a wrap should occur


class TestNonOverlappingBlock:
    def test_non_overlapping_blocks_start_on_grid(self):
        x = np.arange(N, dtype=float)
        res = bootstrap(
            x, method=NonOverlappingBlock(block_length=6), n_bootstraps=8, random_state=2
        )
        for sample in res:
            idx = sample.values.astype(int)
            for block in _full_blocks(idx, 6):
                assert block[0] % 6 == 0  # aligned to the fixed grid
                assert np.all(np.diff(block) == 1)


class TestStationaryBlock:
    def test_stationary_starts_are_random(self):
        # Stationary-bootstrap restart points must be genuinely random, not deterministic.
        x = np.arange(N, dtype=float)
        res = bootstrap(
            x, method=StationaryBlock(avg_block_length=5), n_bootstraps=30, random_state=3
        )
        first_indices = {int(s.values[0]) for s in res}
        assert len(first_indices) > 5  # restart points are genuinely random

    def test_stationary_indices_valid_and_deterministic(self):
        x = ar1(0.6, N, 4)
        a = bootstrap(
            x, method=StationaryBlock(avg_block_length=4), n_bootstraps=10, random_state=9
        )
        b = bootstrap(
            x, method=StationaryBlock(avg_block_length=4), n_bootstraps=10, random_state=9
        )
        np.testing.assert_array_equal(a.values(), b.values())
        idx = a.indices()
        assert idx.min() >= 0 and idx.max() < N

    def test_block_bootstrap_preserves_mean_in_expectation(self):
        x = ar1(0.7, 300, 8)
        res = bootstrap(
            x, method=StationaryBlock(avg_block_length=10), n_bootstraps=400, random_state=11
        )
        boot_means = res.values().mean(axis=1)
        assert abs(boot_means.mean() - x.mean()) < 0.15 * x.std()


class TestAutoBlockLength:
    def test_auto_block_length_default_runs_for_all_block_methods(self):
        x = ar1(0.7, 200, 5)
        for method in (MovingBlock(), CircularBlock(), StationaryBlock(), NonOverlappingBlock()):
            res = bootstrap(x, method=method, n_bootstraps=4, random_state=0)
            assert res.values().shape == (4, 200)
