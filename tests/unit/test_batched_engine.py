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
from tsbootstrap.rng import generators_from_seeds


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
        monkeypatch.setattr("tsbootstrap.dispatch._CHUNK_SIZE", 3)
        chunked = bootstrap(x, method=spec, n_bootstraps=10, random_state=0).values()
        np.testing.assert_array_equal(full, chunked)

    def test_block_chunking_is_bit_exact(self, monkeypatch):
        x = np.arange(60.0)
        spec = MovingBlock(block_length=5)
        full = bootstrap(x, method=spec, n_bootstraps=10, random_state=0).values()
        monkeypatch.setattr("tsbootstrap.dispatch._CHUNK_SIZE", 4)
        chunked = bootstrap(x, method=spec, n_bootstraps=10, random_state=0).values()
        np.testing.assert_array_equal(full, chunked)


class TestVARBatchedEngine:
    def test_var_chunking_reproducible_within_tolerance(self, monkeypatch):
        # VAR's batched matmul is shape-sensitive (BLAS), so a different chunk size can
        # shift a few ULPs, which is exactly why the chunk size is a fixed constant.
        x = _var1(150, 1)
        spec = ResidualBootstrap(model=VAR(order=1))
        full = bootstrap(x, method=spec, n_bootstraps=8, random_state=0).values()
        monkeypatch.setattr("tsbootstrap.dispatch._CHUNK_SIZE", 3)
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

    def test_simulate_var_single_path(self):
        # The single-path simulate_var must reproduce the VAR recursion
        # X_t = c + sum_j A_j X_{t-j} + e_t exactly. Pin it against a hand-rolled
        # reference loop so the public single-path entry point stays correct.
        from tsbootstrap.engines.var import simulate_var

        rng = np.random.default_rng(7)
        p, d, m = 2, 3, 25
        coefs = np.stack([0.2 / (j + 1) * np.eye(d) + 0.03 for j in range(p)])
        intercept = rng.standard_normal(d) * 0.1
        init = rng.standard_normal((p, d))
        innovations = rng.standard_normal((m, d))

        out = simulate_var(coefs, intercept, init, innovations)

        assert out.shape == (p + m, d)
        np.testing.assert_array_equal(out[:p], init)

        expected = np.empty((p + m, d))
        expected[:p] = init
        for t in range(p, p + m):
            acc = intercept.copy()
            for j in range(p):
                acc = acc + coefs[j] @ expected[t - 1 - j]
            expected[t] = acc + innovations[t - p]
        np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)

    def test_simulate_var_single_path_matches_batched(self):
        # A single-path simulate_var with B=1 must agree with the numpy batched
        # recursion to tight tolerance, tying the two public entry points together.
        import tsbootstrap.engines.var as var_engine
        from tsbootstrap.engines.var import simulate_var

        rng = np.random.default_rng(11)
        p, d, m = 1, 2, 40
        coefs = np.stack([np.array([[0.5, 0.1], [0.2, 0.4]])])
        intercept = rng.standard_normal(d) * 0.1
        init = rng.standard_normal((p, d))
        innovations = rng.standard_normal((m, d))

        single = simulate_var(coefs, intercept, init, innovations)

        path = np.empty((1, p + m, d))
        path[:, :p] = init[None]
        var_engine._var_recurrence_numpy(coefs, intercept, path, innovations[None], p, m)
        np.testing.assert_allclose(single, path[0], rtol=1e-9, atol=1e-9)

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

    def test_simulate_arma_batched_paired_init_message_is_exact(self):
        # Pin the exact paired-precondition message so a blanked, case-mangled, or otherwise
        # corrupted ValueError string is caught (the bare pytest.raises above would pass on any
        # ValueError regardless of its text).
        from tsbootstrap.engines.arma_scipy import simulate_arma_batched

        ar, ma, e = np.array([0.5]), np.array([0.3]), np.zeros((2, 10))
        with pytest.raises(
            ValueError,
            match=r"^init_state and init_values must be provided together \(both or neither\)$",
        ):
            simulate_arma_batched(ar, ma, e, init_state=np.zeros(1))

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


class TestDrawInnovationsAndInits:
    """Pin the exact innovation resample and initial-block draw.

    These draws are deterministic given a seeded generator, so the resampled innovations and
    the selected initial block can be pinned to exact arrays. The draw bounds (low/high of the
    ``integers`` calls) are load-bearing: shifting either end changes which residuals and which
    initial window are selected, so exact-array assertions catch any off-by-one in the bounds.
    """

    @staticmethod
    def _context(initial: str):
        """Build a real AR(2) context over a fixed AR(1) series for the draw helpers."""
        from tsbootstrap.model.fit import fit_ar
        from tsbootstrap.model.recursive import _ARContext

        series = ar1(0.6, 30, 0)
        fit = fit_ar(series, 2, None)
        centered = fit.residuals - fit.residuals.mean()
        return _ARContext(
            series=series,
            fit=fit,
            resampling_innovations=centered,
            burn_in=0,
            initial=initial,
            exog_state=None,
        )

    def test_innovation_draw_is_exact(self):
        # Pins the resampled innovations e_star. The draw is integers(0, n_resid); shifting the low
        # bound to 1 (or otherwise changing the range) selects different residuals, changing e_star.
        from tsbootstrap.model.recursive import _draw_innovations_and_inits

        ctx = self._context("fixed")
        gens = generators_from_seeds([np.random.SeedSequence(123)])
        e_star, inits = _draw_innovations_and_inits(ctx, gens, 5)
        assert e_star.shape == (1, 5)
        np.testing.assert_allclose(
            e_star[0],
            [
                0.7541078251626815,
                1.4546837724969033,
                0.6376414816103215,
                0.06568949759229185,
                -0.7079368610628352,
            ],
            rtol=0,
            atol=1e-12,
        )
        # initial="fixed" takes the first p observed values verbatim.
        np.testing.assert_array_equal(inits[0], ctx.series[:2])

    def test_random_block_init_draw_is_exact(self):
        # Pins the random-block initial window. The start is drawn as integers(0, n_series - p + 1);
        # shifting the low bound to 1 or the high bound to n_series - p - 1 selects a different start
        # and therefore a different initial block.
        from tsbootstrap.model.recursive import _draw_innovations_and_inits

        ctx = self._context("random_block")
        gens = generators_from_seeds([np.random.SeedSequence(123)])
        _e_star, inits = _draw_innovations_and_inits(ctx, gens, 5)
        assert inits.shape == (1, 2)
        np.testing.assert_allclose(
            inits[0],
            [1.4293668995661357, 1.8047011028689237],
            rtol=0,
            atol=1e-12,
        )


class TestAsUnivariateGuard:
    """Pin the error code and message of the multivariate-input guard in _as_univariate."""

    def test_multivariate_input_raises_with_exact_code_and_message(self):
        from tsbootstrap.errors import Codes, MethodConfigError
        from tsbootstrap.model.recursive import _as_univariate

        data = np.zeros((20, 2))
        with pytest.raises(MethodConfigError) as exc_info:
            _as_univariate(data, "ResidualBootstrap with an AR model")
        exc = exc_info.value
        # The machine-readable code is the public contract; a blanked or defaulted code is wrong.
        assert exc.code == Codes.VAR_REQUIRES_MULTIVARIATE
        # The human message must still name the univariate requirement; a blanked message is wrong.
        assert "requires a univariate series" in str(exc)
