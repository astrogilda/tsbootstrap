"""Golden-master and chunking-determinism tests for the batched recursive engines.

The batched lfilter (AR/ARMA) is bit-identical to the per-path recurrence, so its
output is bit-exact regardless of the chunk size. The batched VAR matmul depends on
the BLAS accumulation order, which varies with matrix shape, hence the chunk size is
a fixed constant, and the VAR output is reproducible to within a tight tolerance
across chunk sizes rather than bit-for-bit. These tests pin both.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._helpers.dgp import ar1
from tsbootstrap import AR, VAR, MovingBlock, ResidualBootstrap, bootstrap


def _var1(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1], [0.2, 0.4]])
    x = np.zeros((n, 2))
    for t in range(1, n):
        x[t] = a @ x[t - 1] + rng.standard_normal(2)
    return x


class TestARBatchedEngine:
    def test_ar_golden_master(self):
        # Pins the batched AR output so a future change to the engine is caught.
        x = ar1(0.6, 120, 0)
        vals = bootstrap(
            x, method=ResidualBootstrap(model=AR(order=1)), n_bootstraps=5, random_state=42
        ).values()
        np.testing.assert_allclose(
            vals[0, :4], [0.12573022, -0.2835355, -0.70169428, -0.347962], atol=1e-7
        )

    def test_ar_chunking_is_bit_exact(self, monkeypatch):
        x = ar1(0.6, 100, 0)
        spec = ResidualBootstrap(model=AR(order=1))
        full = bootstrap(x, method=spec, n_bootstraps=10, random_state=0).values()
        monkeypatch.setattr("tsbootstrap.api._CHUNK_SIZE", 3)
        chunked = bootstrap(x, method=spec, n_bootstraps=10, random_state=0).values()
        np.testing.assert_array_equal(full, chunked)

    def test_block_chunking_is_bit_exact(self, monkeypatch):
        x = np.arange(60.0)
        spec = MovingBlock(block_length=5)
        full = bootstrap(x, method=spec, n_bootstraps=10, random_state=0).values()
        monkeypatch.setattr("tsbootstrap.api._CHUNK_SIZE", 4)
        chunked = bootstrap(x, method=spec, n_bootstraps=10, random_state=0).values()
        np.testing.assert_array_equal(full, chunked)


class TestVARBatchedEngine:
    def test_var_chunking_reproducible_within_tolerance(self, monkeypatch):
        # VAR's batched matmul is shape-sensitive (BLAS), so a different chunk size can
        # shift a few ULPs, which is exactly why the chunk size is a fixed constant.
        x = _var1(150, 1)
        spec = ResidualBootstrap(model=VAR(order=1))
        full = bootstrap(x, method=spec, n_bootstraps=8, random_state=0).values()
        monkeypatch.setattr("tsbootstrap.api._CHUNK_SIZE", 3)
        chunked = bootstrap(x, method=spec, n_bootstraps=8, random_state=0).values()
        np.testing.assert_allclose(full, chunked, rtol=1e-9, atol=1e-9)

    def test_var_numba_and_numpy_backends_agree(self):
        # The compiled [accel] kernel and the pure-numpy fallback must produce the same VAR
        # recursion to tight tolerance; this verifies the fallback even when numba is present.
        import tsbootstrap.engines.var as var_engine

        rng = np.random.default_rng(3)
        B, p, d, m = 50, 2, 3, 400
        coefs = np.stack([0.2 / (j + 1) * np.eye(d) + 0.03 for j in range(p)])
        intercept = rng.standard_normal(d) * 0.1
        inits = rng.standard_normal((B, p, d))
        innov = rng.standard_normal((B, m, d))

        path_numpy = np.empty((B, p + m, d))
        path_numpy[:, :p] = inits
        var_engine._var_recurrence_numpy(coefs, intercept, path_numpy, innov, p, m)

        if var_engine._HAVE_NUMBA:
            var_engine._warm_var_kernel()
            path_numba = np.empty((B, p + m, d))
            path_numba[:, :p] = inits
            var_engine._var_recurrence_numba(
                np.ascontiguousarray(coefs),
                np.ascontiguousarray(intercept),
                path_numba,
                np.ascontiguousarray(innov),
                p,
                m,
            )
            np.testing.assert_allclose(path_numpy, path_numba, rtol=1e-9, atol=1e-9)

    def test_var_numpy_fallback_through_dispatch(self, monkeypatch):
        # Force the pure-numpy VAR backend so the fallback stays exercised end-to-end even when
        # numba is installed (the dispatch would otherwise always pick the compiled kernel).
        import tsbootstrap.engines.var as var_engine

        monkeypatch.setattr(var_engine, "_HAVE_NUMBA", False)
        x = _var1(120, 0)
        res = bootstrap(
            x, method=ResidualBootstrap(model=VAR(order=1)), n_bootstraps=6, random_state=0
        )
        assert res.values().shape == (6, 120, 2)
        assert np.isfinite(res.values()).all()


class TestEngineGuards:
    def test_arma_initial_state_rejects_mismatched_lengths(self):
        from tsbootstrap.model.arima import arma_initial_state

        # k = max(p, q) = 1, so init_w of length 2 is invalid.
        with pytest.raises(ValueError):
            arma_initial_state(
                np.array([0.5]), np.array([0.3]), np.array([1.0, 2.0]), np.array([0.1])
            )

    def test_simulate_arma_batched_requires_paired_init_args(self):
        from tsbootstrap.engines.arma_scipy import simulate_arma_batched

        ar, ma, e = np.array([0.5]), np.array([0.3]), np.zeros((2, 10))
        with pytest.raises(ValueError):
            simulate_arma_batched(ar, ma, e, init_state=np.zeros(1))  # init_values missing
        with pytest.raises(ValueError):
            simulate_arma_batched(ar, ma, e, init_values=np.zeros(1))  # init_state missing

    def test_simulate_arma_batched_conditional_broadcasts_over_replicates(self):
        # The conditional path must broadcast init_state over the REPLICATE axis (B), not the
        # step axis (m). With B != m, confusing the two raises a shape error rather than silently
        # producing a wrong-length path; this pins the correct axis.
        from tsbootstrap.engines.arma_scipy import simulate_arma_batched
        from tsbootstrap.model.arima import arma_initial_state

        ar, ma = np.array([0.5]), np.array([0.3])
        init_w = np.array([0.7])
        init_state = arma_initial_state(ar, ma, init_w, np.array([0.1]))
        b, m = 3, 12  # b != m so the replicate and step axes cannot be confused silently
        out = simulate_arma_batched(
            ar, ma, np.zeros((b, m)), init_state=init_state, init_values=init_w
        )
        assert out.shape == (b, len(init_w) + m)
        assert np.isfinite(out).all()
