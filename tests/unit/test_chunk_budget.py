"""The byte-budgeted chunk loop: every engine's bytes are independent of the chunk size."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap import (
    AR,
    IID,
    VAR,
    CircularBlock,
    MovingBlock,
    ResidualBootstrap,
    SieveAR,
    StationaryBlock,
    TaperedBlock,
    bootstrap,
    bootstrap_iter,
    bootstrap_reduce,
)

METHODS = [
    IID(),
    MovingBlock(block_length=7),
    CircularBlock(block_length=7),
    StationaryBlock(avg_block_length=7),
    TaperedBlock(block_length=7),
    ResidualBootstrap(model=AR(order=2)),
    SieveAR(max_lag=4),
]


def _x(method: object) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal((90, 2)), axis=0) * 0.1 + rng.standard_normal((90, 2))
    return x if isinstance(method, (IID, MovingBlock, CircularBlock)) else x[:, 0]


@pytest.mark.parametrize("method", METHODS, ids=lambda m: type(m).__name__)
@pytest.mark.parametrize("budget_replicates", [1, 3, 10_000])
def test_values_iter_and_reduce_ignore_the_budget(method, budget_replicates, monkeypatch):
    x = _x(method)
    ref = bootstrap(x, method=method, n_bootstraps=11, random_state=4)
    ref_values, ref_idx = ref.values(), ref.indices()
    monkeypatch.setattr("tsbootstrap.dispatch._CHUNK_BYTES", 90 * 20 * budget_replicates)
    got = bootstrap(x, method=method, n_bootstraps=11, random_state=4)
    np.testing.assert_array_equal(got.values(), ref_values)
    if ref_idx is None:
        assert got.indices() is None
    else:
        np.testing.assert_array_equal(got.indices(), ref_idx)
    chunks = list(bootstrap_iter(x, method=method, n_bootstraps=11, random_state=4))
    np.testing.assert_array_equal(np.concatenate([v for v, _ in chunks]), ref_values)
    red = bootstrap_reduce(x, method=method, statistic="mean", n_bootstraps=11, random_state=4)
    np.testing.assert_allclose(
        red.statistics, ref_values.reshape(11, 90, -1).mean(axis=1), rtol=0, atol=1e-12
    )


def test_var_three_series_ignores_the_budget(monkeypatch):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((80, 3))
    spec = ResidualBootstrap(model=VAR(order=1))
    ref = bootstrap(x, method=spec, n_bootstraps=9, random_state=2).values()
    monkeypatch.setattr("tsbootstrap.dispatch._CHUNK_BYTES", 80 * 20 * 2)
    np.testing.assert_array_equal(
        bootstrap(x, method=spec, n_bootstraps=9, random_state=2).values(), ref
    )
