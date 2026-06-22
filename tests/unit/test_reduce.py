"""Tests for bootstrap_reduce, the streaming per-replicate statistic API."""

from __future__ import annotations

import numpy as np
import pytest

from tests._helpers.dgp import ar1
from tsbootstrap import (
    AR,
    IID,
    MovingBlock,
    ReducedResult,
    ResidualBootstrap,
    bootstrap,
    bootstrap_reduce,
)


def _explosive(n: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = 1.0
    for t in range(1, n):
        x[t] = 1.3 * x[t - 1] + 0.01 * rng.standard_normal()
    return x


class TestBootstrapReduce:
    def test_reduce_matches_materialized_bootstrap(self):
        x = ar1(0.5, 200, 0)
        red = bootstrap_reduce(
            x,
            method=MovingBlock(block_length=10),
            statistic=lambda v, idx: float(np.mean(v)),
            n_bootstraps=200,
            random_state=0,
        )
        full = bootstrap(x, method=MovingBlock(block_length=10), n_bootstraps=200, random_state=0)
        expected = np.array([float(np.mean(s.values)) for s in full])
        assert isinstance(red, ReducedResult)
        assert red.statistics.shape == (200,)
        np.testing.assert_allclose(red.statistics, expected)

    def test_reduce_vector_statistic(self):
        red = bootstrap_reduce(
            ar1(0.5, 200, 0),
            method=IID(),
            statistic=lambda v, idx: np.array([v.mean(), v.std()]),
            n_bootstraps=50,
            random_state=0,
        )
        assert red.statistics.shape == (50, 2)

    def test_reduce_passes_indices_for_oob(self):
        x = ar1(0.5, 200, 0)
        n = x.shape[0]

        def oob_count(values, indices):
            assert indices is not None
            return float(n - np.unique(indices).size)

        red = bootstrap_reduce(
            x,
            method=MovingBlock(block_length=10),
            statistic=oob_count,
            n_bootstraps=20,
            random_state=0,
        )
        assert red.statistics.shape == (20,)
        assert (red.statistics > 0).all()

    def test_reduce_indices_none_for_recursive(self):
        seen: dict[str, object] = {}

        def stat(values, indices):
            seen["indices"] = indices
            return float(values.mean())

        bootstrap_reduce(
            ar1(0.5, 200, 0),
            method=ResidualBootstrap(model=AR(order=1)),
            statistic=stat,
            n_bootstraps=10,
            random_state=0,
        )
        assert seen["indices"] is None

    def test_reduce_exact_quantile(self):
        red = bootstrap_reduce(
            ar1(0.5, 200, 0),
            method=IID(),
            statistic=lambda v, idx: float(v.mean()),
            n_bootstraps=500,
            random_state=0,
        )
        lo, hi = red.quantile([0.05, 0.95])
        assert lo < hi

    def test_reduce_failed_preparation(self):
        red = bootstrap_reduce(
            _explosive(),
            method=ResidualBootstrap(model=AR(order=1, stability_policy="skip")),
            statistic=lambda v, idx: float(v.mean()),
            n_bootstraps=5,
            random_state=0,
        )
        assert red.failed is True
        assert red.failure_reason
        assert red.statistics is None
        assert len(red) == 0
        with pytest.raises(ValueError):
            red.quantile(0.5)
