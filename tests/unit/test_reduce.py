"""Tests for bootstrap_reduce, the streaming per-replicate statistic API."""

from __future__ import annotations

import numpy as np
import pytest

from tests._helpers.dgp import ar1
from tsbootstrap import (
    AR,
    IID,
    CircularBlock,
    MovingBlock,
    ReducedResult,
    ResidualBootstrap,
    SieveAR,
    StationaryBlock,
    bootstrap,
    bootstrap_iter,
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


class TestVectorizedReduce:
    """Vectorized-mode behaviors beyond the 1-D property invariant.

    The 1-D obs-method equivalence is the property test
    ``test_vectorized_reduce_equals_per_replicate``; here we cover the multivariate and
    recursive byte-identity, the vector-statistic shape, and the error contract.
    """

    def test_vectorized_matches_per_replicate_multivariate(self):
        rng = np.random.default_rng(1)
        xv = rng.standard_normal((300, 3))
        per = bootstrap_reduce(
            xv,
            method=CircularBlock(block_length=10),
            statistic=lambda v, idx: v.mean(axis=0),
            n_bootstraps=900,
            random_state=1,
        )
        vec = bootstrap_reduce(
            xv,
            method=CircularBlock(block_length=10),
            statistic=lambda v, idx: v.mean(axis=1),
            n_bootstraps=900,
            random_state=1,
            vectorized=True,
        )
        assert per.statistics.shape == (900, 3)
        np.testing.assert_array_equal(per.statistics, vec.statistics)

    def test_vectorized_matches_per_replicate_recursive(self):
        x = ar1(0.5, 300, 0)
        per = bootstrap_reduce(
            x,
            method=ResidualBootstrap(model=AR(order=2)),
            statistic=lambda v, idx: v.mean(),
            n_bootstraps=800,
            random_state=2,
        )
        vec = bootstrap_reduce(
            x,
            method=ResidualBootstrap(model=AR(order=2)),
            statistic=lambda v, idx: v.mean(axis=1),
            n_bootstraps=800,
            random_state=2,
            vectorized=True,
        )
        np.testing.assert_array_equal(per.statistics, vec.statistics)

    def test_vectorized_vector_statistic_shape(self):
        x = ar1(0.5, 200, 0)
        vec = bootstrap_reduce(
            x,
            method=IID(),
            statistic=lambda v, idx: np.stack([v.mean(axis=1), v.std(axis=1)], axis=1),
            n_bootstraps=400,
            random_state=0,
            vectorized=True,
        )
        assert vec.statistics.shape == (400, 2)

    def test_vectorized_scalar_statistic_raises(self):
        with pytest.raises(ValueError, match="one row per replicate"):
            bootstrap_reduce(
                ar1(0.5, 100, 0),
                method=IID(),
                statistic=lambda v, idx: v.mean(),
                n_bootstraps=100,
                random_state=0,
                vectorized=True,
            )

    def test_vectorized_wrong_leading_axis_raises(self):
        with pytest.raises(ValueError, match="one row per replicate"):
            bootstrap_reduce(
                ar1(0.5, 100, 0),
                method=IID(),
                statistic=lambda v, idx: v.mean(axis=1)[:3],
                n_bootstraps=100,
                random_state=0,
                vectorized=True,
            )


class TestBootstrapIter:
    """bootstrap_iter path-chunk behaviors beyond the property invariant.

    The concatenation-equals-materialized invariant lives in the property layer
    (``test_bootstrap_iter_equals_materialized``); here we cover multivariate output, the
    index stream, recursive ``None`` indices, the multi-chunk path, and failed preparation.
    """

    def test_iter_concatenation_matches_full_1d(self):
        x = ar1(0.5, 300, 0)
        full = bootstrap(
            x, method=MovingBlock(block_length=10), n_bootstraps=2500, random_state=3
        ).values()
        chunks = [
            v
            for v, _ in bootstrap_iter(
                x, method=MovingBlock(block_length=10), n_bootstraps=2500, random_state=3
            )
        ]
        np.testing.assert_array_equal(full, np.concatenate(chunks, axis=0))
        assert len(chunks) > 1  # B > _CHUNK_SIZE exercises the multi-chunk path

    def test_iter_concatenation_matches_full_multivariate(self):
        rng = np.random.default_rng(4)
        xv = rng.standard_normal((300, 3))
        full = bootstrap(xv, method=IID(), n_bootstraps=2500, random_state=4).values()
        cat = np.concatenate(
            [v for v, _ in bootstrap_iter(xv, method=IID(), n_bootstraps=2500, random_state=4)],
            axis=0,
        )
        np.testing.assert_array_equal(full, cat)
        assert cat.shape == (2500, 300, 3)

    def test_iter_indices_match_full(self):
        x = ar1(0.5, 200, 0)
        full = bootstrap(
            x, method=MovingBlock(block_length=10), n_bootstraps=300, random_state=5
        ).indices()
        cat = np.concatenate(
            [
                idx
                for _, idx in bootstrap_iter(
                    x, method=MovingBlock(block_length=10), n_bootstraps=300, random_state=5
                )
            ],
            axis=0,
        )
        np.testing.assert_array_equal(full, cat)
        assert cat.dtype == np.int32

    def test_iter_recursive_indices_none(self):
        x = ar1(0.5, 200, 0)
        for _, idx in bootstrap_iter(
            x, method=ResidualBootstrap(model=AR(order=1)), n_bootstraps=50, random_state=0
        ):
            assert idx is None

    def test_iter_failed_preparation_yields_nothing(self):
        chunks = list(
            bootstrap_iter(
                _explosive(),
                method=ResidualBootstrap(model=AR(order=1, stability_policy="skip")),
                n_bootstraps=5,
                random_state=0,
            )
        )
        assert chunks == []


class TestNamedReducer:
    """A string ``statistic`` names a built-in reducer on the default numpy backend."""

    def test_named_mean_matches_callable_mean(self):
        x = ar1(0.5, 200, 0)
        named = bootstrap_reduce(
            x,
            method=MovingBlock(block_length=10),
            statistic="mean",
            n_bootstraps=300,
            random_state=0,
        )
        callable_ = bootstrap_reduce(
            x,
            method=MovingBlock(block_length=10),
            statistic=lambda v, idx: v.mean(axis=0),
            n_bootstraps=300,
            random_state=0,
        )
        np.testing.assert_array_equal(named.statistics, callable_.statistics)

    def test_named_mean_multivariate_shape(self):
        xv = np.random.default_rng(0).standard_normal((200, 3))
        red = bootstrap_reduce(xv, method=IID(), statistic="mean", n_bootstraps=100, random_state=0)
        assert red.statistics.shape == (100, 3)

    @pytest.mark.parametrize(
        ("name", "fn"),
        [
            ("var", lambda v, idx: v.var(axis=0)),
            ("std", lambda v, idx: v.std(axis=0)),
        ],
    )
    def test_named_var_std_match_callable(self, name, fn):
        x = ar1(0.5, 200, 0)
        named = bootstrap_reduce(
            x, method=MovingBlock(block_length=10), statistic=name, n_bootstraps=300, random_state=0
        )
        callable_ = bootstrap_reduce(
            x, method=MovingBlock(block_length=10), statistic=fn, n_bootstraps=300, random_state=0
        )
        np.testing.assert_array_equal(named.statistics, callable_.statistics)

    def test_quantile_tuple_matches_callable(self):
        x = ar1(0.5, 200, 0)
        tup = bootstrap_reduce(
            x,
            method=MovingBlock(block_length=10),
            statistic=("quantile", 0.9),
            n_bootstraps=300,
            random_state=0,
        )
        callable_ = bootstrap_reduce(
            x,
            method=MovingBlock(block_length=10),
            statistic=lambda v, idx: np.quantile(v, 0.9, axis=0),
            n_bootstraps=300,
            random_state=0,
        )
        np.testing.assert_array_equal(tup.statistics, callable_.statistics)

    def test_unknown_named_reducer_raises(self):
        with pytest.raises(Exception, match="unknown built-in reducer"):
            bootstrap_reduce(
                ar1(0.5, 100, 0), method=IID(), statistic="median", n_bootstraps=10, random_state=0
            )

    def test_bare_quantile_string_raises(self):
        # "quantile" needs a level, so it is only selectable as the ("quantile", q) tuple.
        with pytest.raises(Exception, match="unknown built-in reducer"):
            bootstrap_reduce(
                ar1(0.5, 100, 0),
                method=IID(),
                statistic="quantile",
                n_bootstraps=10,
                random_state=0,
            )

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            (("median", 0.5), "must be"),
            (("quantile", 1.5), "must lie in"),
            (("quantile", -0.1), "must lie in"),
        ],
    )
    def test_bad_quantile_tuple_raises(self, bad, match):
        with pytest.raises(Exception, match=match):
            bootstrap_reduce(
                ar1(0.5, 100, 0), method=IID(), statistic=bad, n_bootstraps=10, random_state=0
            )

    def test_unknown_backend_raises(self):
        with pytest.raises(Exception, match="backend must be"):
            bootstrap_reduce(
                ar1(0.5, 100, 0),
                method=IID(),
                statistic="mean",
                n_bootstraps=10,
                backend="gpu",  # type: ignore[arg-type]
            )


class TestCompiledBackend:
    """The opt-in compiled fast path: a distinct RNG stream, equal in distribution.

    The kernel's own statistical goldens live in ``tests/unit/test_compiled.py``;
    these cover the public ``bootstrap_reduce(backend="compiled")`` wiring: dispatch, shape,
    metadata, the distinct-but-equivalent stream, and the error contract.
    """

    def test_compiled_requires_named_reducer(self):
        with pytest.raises(Exception, match="requires a built-in reducer"):
            bootstrap_reduce(
                ar1(0.5, 100, 0),
                method=StationaryBlock(avg_block_length=10),
                statistic=lambda v, idx: v.mean(),
                n_bootstraps=10,
                backend="compiled",
            )

    def test_compiled_rejects_unsupported_method(self):
        pytest.importorskip("numba")
        # SieveAR has no compiled reduce kernel; the unified entry must reject it up front.
        # (ResidualBootstrap with an AR model IS supported, covered in test_compiled.py.)
        with pytest.raises(Exception, match="support"):
            bootstrap_reduce(
                ar1(0.5, 100, 0),
                method=SieveAR(),
                statistic="mean",
                n_bootstraps=10,
                backend="compiled",
            )

    def test_compiled_matches_numpy_in_distribution(self):
        pytest.importorskip("numba")
        from scipy import stats

        x = ar1(0.5, 400, 0)
        spec = StationaryBlock(avg_block_length=12)
        compiled = bootstrap_reduce(
            x, method=spec, statistic="mean", n_bootstraps=6000, random_state=0, backend="compiled"
        )
        numpy_ = bootstrap_reduce(
            x, method=spec, statistic="mean", n_bootstraps=6000, random_state=0
        )
        assert compiled.metadata.backend == "compiled"
        assert compiled.statistics.shape == (6000,)
        # Distinct stream, so not byte-identical; the two means-of-means agree to Monte-Carlo
        # tolerance and the KS test does not reject equality of the two sampling distributions.
        _, p_value = stats.ks_2samp(compiled.statistics, numpy_.statistics)
        assert p_value > 0.01, f"compiled vs numpy sampling distributions differ (KS p={p_value})"

    def test_compiled_multivariate_shape_and_dtype(self):
        pytest.importorskip("numba")
        xv = np.random.default_rng(1).standard_normal((300, 3))
        red = bootstrap_reduce(
            xv,
            method=StationaryBlock(avg_block_length=8),
            statistic="mean",
            n_bootstraps=200,
            random_state=1,
            backend="compiled",
            dtype="float32",
        )
        assert red.statistics.shape == (200, 3)
        assert red.statistics.dtype == np.float32

    @pytest.mark.parametrize("name", ["var", "std"])
    def test_compiled_var_std_match_numpy_in_distribution(self, name):
        pytest.importorskip("numba")
        from scipy import stats

        x = ar1(0.5, 400, 0)
        spec = MovingBlock(block_length=10)
        compiled = bootstrap_reduce(
            x, method=spec, statistic=name, n_bootstraps=5000, random_state=0, backend="compiled"
        ).statistics
        numpy_ = bootstrap_reduce(
            x, method=spec, statistic=name, n_bootstraps=5000, random_state=0
        ).statistics
        _, p_value = stats.ks_2samp(compiled, numpy_)
        assert p_value > 0.01, f"compiled vs numpy {name} distributions differ (KS p={p_value})"

    def test_compiled_quantile_matches_numpy_in_distribution(self):
        pytest.importorskip("numba")
        from scipy import stats

        x = ar1(0.5, 400, 0)
        spec = StationaryBlock(avg_block_length=10)
        compiled = bootstrap_reduce(
            x,
            method=spec,
            statistic=("quantile", 0.95),
            n_bootstraps=5000,
            random_state=0,
            backend="compiled",
        ).statistics
        numpy_ = bootstrap_reduce(
            x, method=spec, statistic=("quantile", 0.95), n_bootstraps=5000, random_state=0
        ).statistics
        assert compiled.shape == (5000,)
        _, p_value = stats.ks_2samp(compiled, numpy_)
        assert p_value > 0.01, f"compiled vs numpy quantile distributions differ (KS p={p_value})"

    def test_compiled_is_thread_count_invariant(self):
        numba = pytest.importorskip("numba")
        x = ar1(0.5, 300, 0)
        spec = StationaryBlock(avg_block_length=10)

        def run() -> np.ndarray:
            return bootstrap_reduce(
                x,
                method=spec,
                statistic="mean",
                n_bootstraps=500,
                random_state=7,
                backend="compiled",
            ).statistics

        original = numba.get_num_threads()
        try:
            numba.set_num_threads(1)
            one = run()
            numba.set_num_threads(max(1, original))
            many = run()
        finally:
            numba.set_num_threads(original)
        np.testing.assert_array_equal(one, many)
