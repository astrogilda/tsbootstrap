"""Peak memory of a materialised numpy run: the output plus one budgeted chunk."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from tsbootstrap import (
    AR,
    IID,
    CircularBlock,
    MovingBlock,
    NonOverlappingBlock,
    ResidualBootstrap,
    SieveAR,
    StationaryBlock,
    TaperedBlock,
    bootstrap,
)

N_OBS = 2000
N_BOOT = 2000
BUDGET = 8 * 2**20  # one chunk: 8 MiB, so the run takes many chunks

METHODS = [
    IID(),
    MovingBlock(block_length=7),
    CircularBlock(block_length=7),
    NonOverlappingBlock(block_length=7),
    StationaryBlock(block_length=7),
    TaperedBlock(block_length=7),
    ResidualBootstrap(model=AR(order=2)),
    SieveAR(max_lag=4),
]


@pytest.mark.parametrize("method", METHODS, ids=lambda m: type(m).__name__)
def test_peak_is_the_output_plus_one_chunk(method, monkeypatch):
    monkeypatch.setattr("tsbootstrap.dispatch._CHUNK_BYTES", BUDGET)
    x = np.random.default_rng(0).standard_normal(N_OBS)  # stationary, so the AR fit is stable
    bootstrap(x, method=method, n_bootstraps=4, random_state=0)  # imports, caches
    tracemalloc.start()
    try:
        result = bootstrap(x, method=method, n_bootstraps=N_BOOT, random_state=0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    held = result.values().nbytes
    idx = result.indices()
    held += 0 if idx is None else idx.nbytes
    extra = (peak - held) / BUDGET
    print(f"MEMPEAK {type(method).__name__}: held={held} peak={peak} extra={extra:.2f} chunks")
    assert peak <= held + BUDGET
