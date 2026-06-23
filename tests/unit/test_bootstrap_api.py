"""End-to-end tests for the public bootstrap() entry point (IID baseline)."""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap.api import bootstrap
from tsbootstrap.errors import MethodConfigError
from tsbootstrap.methods import IID, MovingBlock, SieveAR, StationaryBlock
from tsbootstrap.results import BootstrapResult


class TestIIDShape:
    def test_iid_basic_shape_1d(self):
        x = list(range(20))
        result = bootstrap(x, method=IID(), n_bootstraps=5, random_state=0)
        assert isinstance(result, BootstrapResult)
        assert len(result) == 5
        assert result.values().shape == (5, 20)

    def test_iid_basic_shape_2d(self):
        x = np.arange(30).reshape(15, 2).astype(float)
        result = bootstrap(x, method=IID(), n_bootstraps=4, random_state=0)
        assert result.values().shape == (4, 15, 2)
        assert result.metadata.n_series == 2


class TestDeterminism:
    def test_determinism_same_seed(self):
        x = np.arange(50.0)
        a = bootstrap(x, method=IID(), n_bootstraps=10, random_state=123).values()
        b = bootstrap(x, method=IID(), n_bootstraps=10, random_state=123).values()
        np.testing.assert_array_equal(a, b)

    def test_different_seed_differs(self):
        x = np.arange(50.0)
        a = bootstrap(x, method=IID(), n_bootstraps=10, random_state=1).values()
        b = bootstrap(x, method=IID(), n_bootstraps=10, random_state=2).values()
        assert not np.array_equal(a, b)

    def test_more_bootstraps_preserves_the_prefix(self):
        # The load-bearing contract: replicate i is bound to spawn(n)[i], so asking
        # for more replicates leaves the earlier ones bit-identical.
        x = np.arange(40.0)
        small = bootstrap(x, method=IID(), n_bootstraps=4, random_state=99).values()
        large = bootstrap(x, method=IID(), n_bootstraps=16, random_state=99).values()
        np.testing.assert_array_equal(small, large[:4])


class TestBootstrapResultContract:
    def test_oob_and_inbag(self):
        x = np.arange(25.0)
        result = bootstrap(x, method=IID(), n_bootstraps=6, random_state=3)
        idx = result.indices()
        assert idx is not None and idx.shape == (6, 25)
        inbag = result.inbag_counts()
        assert inbag.shape == (6, 25)
        # Each replicate draws n_obs indices with replacement -> counts sum to n_obs.
        assert (inbag.sum(axis=1) == 25).all()
        oob = result.get_oob_mask()
        assert oob.shape == (6, 25)
        assert oob.dtype == np.bool_

    def test_metadata_records_provenance(self):
        x = np.arange(20.0)
        md = bootstrap(x, method=IID(), n_bootstraps=3, random_state=7).metadata
        assert md.method == "iid"
        assert md.method_params == {"kind": "iid"}
        assert md.n_obs == 20
        assert md.random_state_kind == "int"
        assert "numpy" in md.versions
        assert md.references  # non-empty


class TestDispatchValidation:
    def test_unregistered_method_raises_cleanly(self):
        # The dispatch contract: a spec type with no registered executor fails clearly.
        from tsbootstrap.dispatch import get_executor

        class _UnregisteredSpec:
            pass

        with pytest.raises(MethodConfigError):
            get_executor(_UnregisteredSpec())

    def test_invalid_n_bootstraps(self):
        x = np.arange(20.0)
        with pytest.raises(MethodConfigError):
            bootstrap(x, method=IID(), n_bootstraps=0)


class TestCompiledBackend:
    """The opt-in compiled .values() path: a distinct RNG stream, equal in distribution.

    The kernels' own statistical goldens live in ``tests/unit/test_compiled.py``; these cover
    the public ``bootstrap(backend="compiled")`` wiring.
    """

    def test_bad_backend_raises(self):
        with pytest.raises(MethodConfigError, match="backend must be"):
            bootstrap(np.arange(50.0), method=IID(), n_bootstraps=5, backend="gpu")  # type: ignore[arg-type]

    def test_compiled_rejects_recursive_method(self):
        pytest.importorskip("numba")
        # The compiled .values() path has no recursive kernel; an unsupported recursive method
        # is rejected up front. (ResidualBootstrap(AR) has a fused reduce, not a values, kernel,
        # and points the caller to bootstrap_reduce; that path is covered in test_compiled.py.)
        with pytest.raises(MethodConfigError, match="support"):
            bootstrap(np.arange(80.0), method=SieveAR(), n_bootstraps=5, backend="compiled")

    def test_compiled_values_are_gathered_from_data(self):
        pytest.importorskip("numba")
        x = np.arange(100.0) * 1.5 - 7.0
        res = bootstrap(
            x,
            method=StationaryBlock(avg_block_length=8),
            n_bootstraps=64,
            random_state=0,
            backend="compiled",
        )
        vals, idx = res.values(), res.indices()
        assert vals.shape == (64, 100)
        assert idx.shape == (64, 100) and idx.dtype == np.int32
        np.testing.assert_array_equal(vals, x[idx])  # values are an exact gather of the data
        assert res.metadata.backend == "compiled"

    def test_compiled_multivariate_shape(self):
        pytest.importorskip("numba")
        xv = np.random.default_rng(1).standard_normal((120, 3))
        res = bootstrap(
            xv,
            method=MovingBlock(block_length=8),
            n_bootstraps=40,
            random_state=1,
            backend="compiled",
        )
        assert res.values().shape == (40, 120, 3)
        np.testing.assert_array_equal(res.values(), xv[res.indices()])

    def test_compiled_is_thread_count_invariant(self):
        numba = pytest.importorskip("numba")
        x = np.arange(120.0)
        spec = StationaryBlock(avg_block_length=10)

        def run() -> np.ndarray:
            return bootstrap(
                x, method=spec, n_bootstraps=200, random_state=7, backend="compiled"
            ).values()

        original = numba.get_num_threads()
        try:
            numba.set_num_threads(1)
            one = run()
            numba.set_num_threads(max(1, original))
            many = run()
        finally:
            numba.set_num_threads(original)
        np.testing.assert_array_equal(one, many)

    def test_compiled_matches_numpy_in_distribution(self):
        pytest.importorskip("numba")
        from scipy import stats

        x = np.sin(np.arange(400.0) / 5.0)
        spec = StationaryBlock(avg_block_length=12)
        compiled = (
            bootstrap(x, method=spec, n_bootstraps=5000, random_state=0, backend="compiled")
            .values()
            .mean(axis=1)
        )
        numpy_ = bootstrap(x, method=spec, n_bootstraps=5000, random_state=0).values().mean(axis=1)
        _, p_value = stats.ks_2samp(compiled, numpy_)
        assert p_value > 0.01, f"compiled vs numpy .values distributions differ (KS p={p_value})"
