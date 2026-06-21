"""Golden-master and chunking-determinism tests for the batched recursive engines.

The batched lfilter (AR/ARMA) is bit-identical to the per-path recurrence, so its
output is bit-exact regardless of the chunk size. The batched VAR matmul depends on
the BLAS accumulation order, which varies with matrix shape — hence the chunk size is
a fixed constant, and the VAR output is reproducible to within a tight tolerance
across chunk sizes rather than bit-for-bit. These tests pin both.
"""

from __future__ import annotations

import numpy as np

from tsbootstrap import AR, VAR, MovingBlock, ResidualBootstrap, bootstrap


def _ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _var1(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1], [0.2, 0.4]])
    x = np.zeros((n, 2))
    for t in range(1, n):
        x[t] = a @ x[t - 1] + rng.standard_normal(2)
    return x


def test_ar_golden_master():
    # Pins the batched AR output so a future change to the engine is caught.
    x = _ar1(0.6, 120, 0)
    vals = bootstrap(x, method=ResidualBootstrap(model=AR(order=1)), n_bootstraps=5, random_state=42).values()
    np.testing.assert_allclose(
        vals[0, :4], [0.12573022, -0.2835355, -0.70169428, -0.347962], atol=1e-7
    )


def test_ar_chunking_is_bit_exact(monkeypatch):
    x = _ar1(0.6, 100, 0)
    spec = ResidualBootstrap(model=AR(order=1))
    full = bootstrap(x, method=spec, n_bootstraps=10, random_state=0).values()
    monkeypatch.setattr("tsbootstrap.api._CHUNK_SIZE", 3)
    chunked = bootstrap(x, method=spec, n_bootstraps=10, random_state=0).values()
    np.testing.assert_array_equal(full, chunked)


def test_block_chunking_is_bit_exact(monkeypatch):
    x = np.arange(60.0)
    spec = MovingBlock(block_length=5)
    full = bootstrap(x, method=spec, n_bootstraps=10, random_state=0).values()
    monkeypatch.setattr("tsbootstrap.api._CHUNK_SIZE", 4)
    chunked = bootstrap(x, method=spec, n_bootstraps=10, random_state=0).values()
    np.testing.assert_array_equal(full, chunked)


def test_var_chunking_reproducible_within_tolerance(monkeypatch):
    # VAR's batched matmul is shape-sensitive (BLAS), so a different chunk size can
    # shift a few ULPs — which is exactly why the chunk size is a fixed constant.
    x = _var1(150, 1)
    spec = ResidualBootstrap(model=VAR(order=1))
    full = bootstrap(x, method=spec, n_bootstraps=8, random_state=0).values()
    monkeypatch.setattr("tsbootstrap.api._CHUNK_SIZE", 3)
    chunked = bootstrap(x, method=spec, n_bootstraps=8, random_state=0).values()
    np.testing.assert_allclose(full, chunked, rtol=1e-9, atol=1e-9)
