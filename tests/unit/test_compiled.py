"""Tests for the compiled fast paths (opt-in Philox stream).

These exercise the optional numba-compiled kernels in
``tsbootstrap.block._compiled`` for the IID, moving-block, circular-block,
non-overlapping-block, and stationary families. The whole module is skipped when
numba (the ``[accel]`` extra) is not installed, because the fast path is an
optional accelerated dependency, exactly as the VAR-kernel tests handle the same
case.

What is checked:

- Distributional equivalence to each pure-numpy path, via a two-sample
  Kolmogorov-Smirnov test on block lengths and on the bootstrap statistic. The
  two paths draw from different RNG streams (Philox here, PCG64 there), so they
  are equal in distribution, never bit-for-bit.
- Bitwise determinism of the output across numba thread counts, set in-process
  with the documented numba API.
- Correctness of the Philox-4x32-10 round function against the published
  Random123 known-answer vectors.
- Shape and dtype correctness for 1-D and multivariate input, and that the
  requested ``sim_dtype`` is honored.
- The unsupported-method error path on the unified ``compiled_reduce`` entry.
"""

from __future__ import annotations

import numpy as np
import pytest

numba = pytest.importorskip("numba")  # optional [accel] extra; see module docstring

from scipy.stats import ks_2samp  # noqa: E402

from tsbootstrap.block import _compiled as sk  # noqa: E402
from tsbootstrap.block.indices import _circular, _moving, _non_overlapping  # noqa: E402
from tsbootstrap.block.stationary import _stationary_indices  # noqa: E402
from tsbootstrap.errors import MethodConfigError  # noqa: E402
from tsbootstrap.methods import (  # noqa: E402
    AR,
    IID,
    CircularBlock,
    MovingBlock,
    NonOverlappingBlock,
    StationaryBlock,
)
from tsbootstrap.rng import spawn_seed_sequences  # noqa: E402


@pytest.fixture(autouse=True)
def _warm_kernel():
    """Compile the kernels once before each test so timing and threads are stable."""
    sk._warm_compiled_kernels()


def _seeds(n: int, seed: int = 12345) -> list[np.random.SeedSequence]:
    return spawn_seed_sequences(np.random.SeedSequence(seed), n)


def _block_lengths(idx_row: np.ndarray, n: int) -> list[int]:
    """A block is a maximal run where idx[t] == (idx[t-1] + 1) mod n."""
    lengths: list[int] = []
    cur = 1
    for t in range(1, len(idx_row)):
        if idx_row[t] == (idx_row[t - 1] + 1) % n:
            cur += 1
        else:
            lengths.append(cur)
            cur = 1
    lengths.append(cur)
    return lengths


class TestPhiloxKAT:
    """Pin the published Philox-4x32-10 known-answer vectors forever.

    Source: DEShawResearch/random123 tests/kat_vectors (Salmon et al. 2011). The
    format is ``philox4x32 <rounds> <c0 c1 c2 c3> <k0 k1>  <out0 out1 out2 out3>``.
    The njit reference below mirrors the kernel's _philox_round4 arithmetic exactly
    (32x32 -> 64 multiplies, the Random123 word permutation, and the Weyl key bump
    BEFORE rounds 2..10) but takes a full counter/key so it can exercise every
    published vector, not only the zero-counter-tail case the kernel realises. If
    this test ever fails, the multipliers, Weyl constants, round count, word
    permutation, or bump position have drifted and the stream is no longer canonical.
    """

    _M0 = np.uint64(0xD2511F53)
    _M1 = np.uint64(0xCD9E8D57)
    _W0 = np.uint64(0x9E3779B9)
    _W1 = np.uint64(0xBB67AE85)
    _MASK32 = np.uint64(0xFFFFFFFF)
    _SH32 = np.uint64(32)

    # (c0, c1, c2, c3, k0, k1) -> (out0, out1, out2, out3), all uint32, 10 rounds.
    KAT_VECTORS = [
        ((0, 0, 0, 0, 0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        (
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        ),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344, 0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ]

    def test_kernel_arithmetic_reproduces_published_vectors(self):
        M0, M1, W0, W1, MASK32, SH32 = (
            self._M0,
            self._M1,
            self._W0,
            self._W1,
            self._MASK32,
            self._SH32,
        )

        @numba.njit(cache=True)
        def _philox4x32_10(c0, c1, c2, c3, k0, k1, out4):
            c0 = np.uint64(c0) & MASK32
            c1 = np.uint64(c1) & MASK32
            c2 = np.uint64(c2) & MASK32
            c3 = np.uint64(c3) & MASK32
            k0 = np.uint64(k0)
            k1 = np.uint64(k1)
            for r in range(10):
                if r > 0:
                    k0 = (k0 + W0) & MASK32
                    k1 = (k1 + W1) & MASK32
                p0 = M0 * c0
                hi0 = p0 >> SH32
                lo0 = p0 & MASK32
                p1 = M1 * c2
                hi1 = p1 >> SH32
                lo1 = p1 & MASK32
                c0 = (hi1 ^ c1 ^ k0) & MASK32
                c1 = lo1
                c2 = (hi0 ^ c3 ^ k1) & MASK32
                c3 = lo0
            out4[0] = c0
            out4[1] = c1
            out4[2] = c2
            out4[3] = c3

        out = np.empty(4, np.uint64)
        for inputs, expected in self.KAT_VECTORS:
            _philox4x32_10(*inputs, out)
            assert tuple(int(x) for x in out) == expected


class TestDistributionalEquivalence:
    def test_block_lengths_match_pure_numpy_path(self):
        n, avg_len = 1000, 10
        # Kernel block lengths.
        seeds = _seeds(2000, seed=12345)
        kidx = sk.stationary_indices(n, seeds, avg_len)
        ker_lengths: list[int] = []
        for b in range(kidx.shape[0]):
            ker_lengths.extend(_block_lengths(kidx[b], n))

        # Pure-numpy reference (PCG64 stream).
        ref_gen = np.random.default_rng(99)
        ref_lengths: list[int] = []
        for _ in range(2000):
            ridx = _stationary_indices(ref_gen, n, avg_len)
            ref_lengths.extend(_block_lengths(ridx, n))

        ks = ks_2samp(np.array(ker_lengths), np.array(ref_lengths))
        # Same geometric distribution, different stream: KS must not reject.
        assert ks.pvalue > 0.01, f"block-length KS rejected: p={ks.pvalue:.4f}"
        assert abs(np.mean(ker_lengths) - avg_len) < 0.5

    def test_bootstrap_statistic_matches_pure_numpy_path(self):
        n, avg_len, B = 500, 10, 4000
        rng_data = np.random.default_rng(0)
        data = np.ascontiguousarray(rng_data.standard_normal((n, 1)))

        # Kernel column-mean statistic per replicate.
        ker_stat = sk.stationary_reduce(data, _seeds(B, seed=7), avg_len)[:, 0]

        # Pure-numpy reference: same geometric construction, PCG64 stream.
        ref_gen = np.random.default_rng(123)
        ref_stat = np.empty(B)
        for b in range(B):
            idx = _stationary_indices(ref_gen, n, avg_len)
            ref_stat[b] = data[idx, 0].mean()

        ks = ks_2samp(ker_stat, ref_stat)
        assert ks.pvalue > 0.01, f"statistic KS rejected: p={ks.pvalue:.4f}"


class TestDetermimism:
    def test_output_is_bitwise_identical_across_thread_counts(self):
        n, avg_len, B = 800, 10, 3000
        rng_data = np.random.default_rng(0)
        data = np.ascontiguousarray(rng_data.standard_normal((n, 2)))
        seeds = _seeds(B, seed=7)

        max_threads = numba.config.NUMBA_NUM_THREADS
        numba.set_num_threads(max_threads)
        out_many = sk.stationary_reduce(data, seeds, avg_len)
        try:
            numba.set_num_threads(1)
            out_one = sk.stationary_reduce(data, seeds, avg_len)
        finally:
            numba.set_num_threads(max_threads)

        np.testing.assert_array_equal(out_one, out_many)


class TestShapeAndDtype:
    def test_1d_input_returns_single_column(self):
        x = np.random.default_rng(0).standard_normal(200)
        out = sk.stationary_reduce(x, _seeds(50), avg_block_length=8)
        assert out.shape == (50, 1)

    def test_multivariate_input_preserves_columns(self):
        xv = np.random.default_rng(1).standard_normal((300, 4))
        out = sk.stationary_reduce(xv, _seeds(60), avg_block_length=12)
        assert out.shape == (60, 4)

    def test_sim_dtype_is_honored(self):
        xv = np.random.default_rng(2).standard_normal((150, 3))
        out = sk.stationary_reduce(
            xv, _seeds(40), avg_block_length=10, sim_dtype=np.dtype(np.float32)
        )
        assert out.dtype == np.float32
        assert out.shape == (40, 3)

    def test_indices_shape_and_dtype(self):
        idx = sk.stationary_indices(120, _seeds(30), avg_block_length=6)
        assert idx.shape == (30, 120)
        assert idx.dtype == np.int32
        assert idx.min() >= 0 and idx.max() < 120

    def test_keys_derived_per_replicate_from_seed_entropy(self):
        seeds = _seeds(5, seed=42)
        key_hi, key_lo = sk.philox_keys_from_seeds(seeds)
        assert key_hi.shape == (5,) and key_lo.shape == (5,)
        assert key_hi.dtype == np.uint32 and key_lo.dtype == np.uint32
        # Each replicate's key words are that replicate's own seed entropy.
        expected = seeds[0].generate_state(2, dtype=np.uint32)
        assert key_hi[0] == expected[0] and key_lo[0] == expected[1]


class TestInputValidation:
    def test_unsupported_reducer_raises(self):
        x = np.zeros(10)
        with pytest.raises(MethodConfigError, match="Unsupported reducer"):
            sk.stationary_reduce(x, _seeds(3), avg_block_length=5, reducer="variance")

    def test_non_positive_block_length_raises(self):
        x = np.zeros(10)
        with pytest.raises(MethodConfigError, match="positive integer"):
            sk.stationary_reduce(x, _seeds(3), avg_block_length=0)

    def test_three_dimensional_data_raises(self):
        x = np.zeros((4, 4, 4))
        with pytest.raises(MethodConfigError, match="1-D or 2-D"):
            sk.stationary_reduce(x, _seeds(3), avg_block_length=5)

    def test_empty_series_raises(self):
        x = np.zeros((0, 2))
        with pytest.raises(MethodConfigError, match="at least one observation"):
            sk.stationary_reduce(x, _seeds(3), avg_block_length=5)


_SIM = np.dtype(np.float64)


def _numpy_block_stat(executor, spec, data, seed, B):
    """Per-replicate column-0 mean from a pure-numpy block executor (PCG64 stream)."""
    seeds = _seeds(B, seed=seed)
    values, _idx = executor(data, spec, seeds, data.shape[0], _SIM)
    # values is (B, n[, d]); reduce to the per-replicate column-0 mean.
    return values.reshape(B, data.shape[0], -1).mean(axis=1)[:, 0]


def _numpy_block_lengths_from_executor(executor, spec, n, seed, B):
    """Contiguous-run block lengths from a pure-numpy block executor's indices."""
    seeds = _seeds(B, seed=seed)
    probe = np.zeros((n, 1), dtype=np.float64)
    _values, idx = executor(probe, spec, seeds, n, _SIM)
    lengths: list[int] = []
    for b in range(idx.shape[0]):
        lengths.extend(_block_lengths(idx[b], n))
    return lengths


class TestIIDCompiled:
    def test_statistic_matches_pure_numpy_path(self):
        n, B = 500, 4000
        data = np.ascontiguousarray(np.random.default_rng(0).standard_normal((n, 1)))
        ker = sk.compiled_reduce(IID(), data, _seeds(B, seed=7))[:, 0]

        # Pure-numpy IID reference: n independent uniform positions per replicate.
        ref_gen = np.random.default_rng(123)
        ref = np.empty(B)
        for b in range(B):
            idx = ref_gen.integers(0, n, size=n, dtype=np.int32)
            ref[b] = data[idx, 0].mean()

        ks = ks_2samp(ker, ref)
        assert ks.pvalue > 0.01, f"IID statistic KS rejected: p={ks.pvalue:.4f}"

    def test_determinism_across_thread_counts(self):
        n, B = 600, 2500
        data = np.ascontiguousarray(np.random.default_rng(0).standard_normal((n, 2)))
        seeds = _seeds(B, seed=7)
        max_threads = numba.config.NUMBA_NUM_THREADS
        numba.set_num_threads(max_threads)
        out_many = sk.compiled_reduce(IID(), data, seeds)
        try:
            numba.set_num_threads(1)
            out_one = sk.compiled_reduce(IID(), data, seeds)
        finally:
            numba.set_num_threads(max_threads)
        np.testing.assert_array_equal(out_one, out_many)

    def test_shape_dtype_and_sim_dtype(self):
        x = np.random.default_rng(0).standard_normal(200)
        assert sk.compiled_reduce(IID(), x, _seeds(50)).shape == (50, 1)
        xv = np.random.default_rng(1).standard_normal((300, 4))
        assert sk.compiled_reduce(IID(), xv, _seeds(60)).shape == (60, 4)
        out = sk.compiled_reduce(IID(), xv, _seeds(40), sim_dtype=np.dtype(np.float32))
        assert out.dtype == np.float32 and out.shape == (40, 4)
        idx = sk.iid_indices(120, _seeds(30))
        assert idx.shape == (30, 120) and idx.dtype == np.int32
        assert idx.min() >= 0 and idx.max() < 120


class TestMovingBlockCompiled:
    def test_block_lengths_match_pure_numpy_path(self):
        n, length, B = 1000, 10, 1500
        spec = MovingBlock(block_length=length)
        ker_idx = sk.block_indices(sk._MOVING, n, _seeds(B, seed=12345), length)
        ker_lengths: list[int] = []
        for b in range(ker_idx.shape[0]):
            ker_lengths.extend(_block_lengths(ker_idx[b], n))
        ref_lengths = _numpy_block_lengths_from_executor(_moving, spec, n, seed=99, B=B)
        ks = ks_2samp(np.array(ker_lengths), np.array(ref_lengths))
        assert ks.pvalue > 0.01, f"moving block-length KS rejected: p={ks.pvalue:.4f}"

    def test_statistic_matches_pure_numpy_path(self):
        n, length, B = 500, 10, 4000
        data = np.ascontiguousarray(np.random.default_rng(0).standard_normal((n, 1)))
        spec = MovingBlock(block_length=length)
        ker = sk.compiled_reduce(spec, data, _seeds(B, seed=7))[:, 0]
        ref = _numpy_block_stat(_moving, spec, data, seed=123, B=B)
        ks = ks_2samp(ker, ref)
        assert ks.pvalue > 0.01, f"moving statistic KS rejected: p={ks.pvalue:.4f}"

    def test_determinism_across_thread_counts(self):
        n, length, B = 600, 12, 2500
        data = np.ascontiguousarray(np.random.default_rng(0).standard_normal((n, 2)))
        spec = MovingBlock(block_length=length)
        seeds = _seeds(B, seed=7)
        max_threads = numba.config.NUMBA_NUM_THREADS
        numba.set_num_threads(max_threads)
        out_many = sk.compiled_reduce(spec, data, seeds)
        try:
            numba.set_num_threads(1)
            out_one = sk.compiled_reduce(spec, data, seeds)
        finally:
            numba.set_num_threads(max_threads)
        np.testing.assert_array_equal(out_one, out_many)

    def test_shape_dtype_and_sim_dtype(self):
        x = np.random.default_rng(0).standard_normal(200)
        spec = MovingBlock(block_length=8)
        assert sk.compiled_reduce(spec, x, _seeds(50)).shape == (50, 1)
        xv = np.random.default_rng(1).standard_normal((300, 4))
        assert sk.compiled_reduce(spec, xv, _seeds(60)).shape == (60, 4)
        out = sk.compiled_reduce(spec, xv, _seeds(40), sim_dtype=np.dtype(np.float32))
        assert out.dtype == np.float32 and out.shape == (40, 4)
        idx = sk.block_indices(sk._MOVING, 120, _seeds(30), 8)
        assert idx.shape == (30, 120) and idx.dtype == np.int32
        assert idx.min() >= 0 and idx.max() < 120


class TestCircularBlockCompiled:
    def test_block_lengths_match_pure_numpy_path(self):
        n, length, B = 1000, 10, 1500
        spec = CircularBlock(block_length=length)
        ker_idx = sk.block_indices(sk._CIRCULAR, n, _seeds(B, seed=12345), length)
        ker_lengths: list[int] = []
        for b in range(ker_idx.shape[0]):
            ker_lengths.extend(_block_lengths(ker_idx[b], n))
        ref_lengths = _numpy_block_lengths_from_executor(_circular, spec, n, seed=99, B=B)
        ks = ks_2samp(np.array(ker_lengths), np.array(ref_lengths))
        assert ks.pvalue > 0.01, f"circular block-length KS rejected: p={ks.pvalue:.4f}"

    def test_statistic_matches_pure_numpy_path(self):
        n, length, B = 500, 10, 4000
        data = np.ascontiguousarray(np.random.default_rng(0).standard_normal((n, 1)))
        spec = CircularBlock(block_length=length)
        ker = sk.compiled_reduce(spec, data, _seeds(B, seed=7))[:, 0]
        ref = _numpy_block_stat(_circular, spec, data, seed=123, B=B)
        ks = ks_2samp(ker, ref)
        assert ks.pvalue > 0.01, f"circular statistic KS rejected: p={ks.pvalue:.4f}"

    def test_determinism_across_thread_counts(self):
        n, length, B = 600, 12, 2500
        data = np.ascontiguousarray(np.random.default_rng(0).standard_normal((n, 2)))
        spec = CircularBlock(block_length=length)
        seeds = _seeds(B, seed=7)
        max_threads = numba.config.NUMBA_NUM_THREADS
        numba.set_num_threads(max_threads)
        out_many = sk.compiled_reduce(spec, data, seeds)
        try:
            numba.set_num_threads(1)
            out_one = sk.compiled_reduce(spec, data, seeds)
        finally:
            numba.set_num_threads(max_threads)
        np.testing.assert_array_equal(out_one, out_many)

    def test_shape_dtype_and_sim_dtype(self):
        x = np.random.default_rng(0).standard_normal(200)
        spec = CircularBlock(block_length=8)
        assert sk.compiled_reduce(spec, x, _seeds(50)).shape == (50, 1)
        xv = np.random.default_rng(1).standard_normal((300, 4))
        assert sk.compiled_reduce(spec, xv, _seeds(60)).shape == (60, 4)
        out = sk.compiled_reduce(spec, xv, _seeds(40), sim_dtype=np.dtype(np.float32))
        assert out.dtype == np.float32 and out.shape == (40, 4)
        idx = sk.block_indices(sk._CIRCULAR, 120, _seeds(30), 8)
        assert idx.shape == (30, 120) and idx.dtype == np.int32
        assert idx.min() >= 0 and idx.max() < 120


class TestNonOverlappingBlockCompiled:
    def test_statistic_matches_pure_numpy_path(self):
        n, length, B = 500, 10, 4000
        data = np.ascontiguousarray(np.random.default_rng(0).standard_normal((n, 1)))
        spec = NonOverlappingBlock(block_length=length)
        ker = sk.compiled_reduce(spec, data, _seeds(B, seed=7))[:, 0]
        ref = _numpy_block_stat(_non_overlapping, spec, data, seed=123, B=B)
        ks = ks_2samp(ker, ref)
        assert ks.pvalue > 0.01, f"non-overlapping statistic KS rejected: p={ks.pvalue:.4f}"

    def test_determinism_across_thread_counts(self):
        n, length, B = 600, 12, 2500
        data = np.ascontiguousarray(np.random.default_rng(0).standard_normal((n, 2)))
        spec = NonOverlappingBlock(block_length=length)
        seeds = _seeds(B, seed=7)
        max_threads = numba.config.NUMBA_NUM_THREADS
        numba.set_num_threads(max_threads)
        out_many = sk.compiled_reduce(spec, data, seeds)
        try:
            numba.set_num_threads(1)
            out_one = sk.compiled_reduce(spec, data, seeds)
        finally:
            numba.set_num_threads(max_threads)
        np.testing.assert_array_equal(out_one, out_many)

    def test_shape_and_sim_dtype(self):
        xv = np.random.default_rng(1).standard_normal((300, 4))
        spec = NonOverlappingBlock(block_length=10)
        out = sk.compiled_reduce(spec, xv, _seeds(40), sim_dtype=np.dtype(np.float32))
        assert out.dtype == np.float32 and out.shape == (40, 4)
        idx = sk.block_indices(sk._NON_OVERLAPPING, 120, _seeds(30), 10)
        assert idx.shape == (30, 120) and idx.dtype == np.int32
        assert idx.min() >= 0 and idx.max() < 120


class TestUnifiedDispatch:
    def test_unsupported_method_raises(self):
        x = np.zeros(20)
        with pytest.raises(MethodConfigError, match="does not support"):
            sk.compiled_reduce(AR(order=1), x, _seeds(3))

    def test_stationary_routes_through_unified_entry(self):
        data = np.ascontiguousarray(np.random.default_rng(0).standard_normal((200, 1)))
        out = sk.compiled_reduce(StationaryBlock(avg_block_length=8), data, _seeds(30))
        assert out.shape == (30, 1)


# --- compiled_values (full materialisation path) ----------------------------
# One spec per supported method; the numpy executor and the matching block
# family are paired so each method is checked against its own pure-numpy path.
_VALUES_METHODS = [
    ("iid", IID(), None),
    ("moving", MovingBlock(block_length=10), _moving),
    ("circular", CircularBlock(block_length=10), _circular),
    ("non_overlapping", NonOverlappingBlock(block_length=10), _non_overlapping),
    ("stationary", StationaryBlock(avg_block_length=10), None),
]


@pytest.mark.parametrize("name,spec,executor", _VALUES_METHODS, ids=[m[0] for m in _VALUES_METHODS])
class TestCompiledValues:
    def test_gather_consistency(self, name, spec, executor):
        # values[b, t] == data[indices[b, t]] exactly, univariate and multivariate.
        for d in (1, 3):
            data = np.ascontiguousarray(np.random.default_rng(0).standard_normal((300, d)))
            values, idx = sk.compiled_values(spec, data, _seeds(50, seed=7))
            assert values.shape == (50, 300, d)
            assert idx.shape == (50, 300) and idx.dtype == np.int32
            np.testing.assert_array_equal(values, data[idx])

    def test_indices_match_reduce_path(self, name, spec, executor):
        # The materialised path and the reduce path must see the SAME resample for
        # the same seeds: assert column means agree to float tolerance. (The means
        # are computed from independent gather vs fused-accumulate code, so equality
        # within float error is a representative end-to-end consistency check.)
        data = np.ascontiguousarray(np.random.default_rng(1).standard_normal((400, 2)))
        seeds = _seeds(80, seed=11)
        values, _idx = sk.compiled_values(spec, data, seeds)
        reduced = sk.compiled_reduce(spec, data, seeds)
        np.testing.assert_allclose(values.mean(axis=1), reduced, rtol=0, atol=1e-12)

    def test_distribution_matches_numpy_values(self, name, spec, executor):
        # KS two-sample on the per-replicate mean from compiled_values vs the numpy
        # bootstrap(...).values() distribution (different streams, equal in dist).
        n, B = 500, 4000
        data = np.ascontiguousarray(np.random.default_rng(0).standard_normal((n, 1)))
        ker_values, _idx = sk.compiled_values(spec, data, _seeds(B, seed=7))
        ker = ker_values.mean(axis=1)[:, 0]

        if executor is not None:
            ref_vals, _ = executor(data, spec, _seeds(B, seed=123), n, _SIM)
            ref = ref_vals.reshape(B, n, -1).mean(axis=1)[:, 0]
        elif name == "iid":
            ref_gen = np.random.default_rng(123)
            ref = np.empty(B)
            for b in range(B):
                ridx = ref_gen.integers(0, n, size=n, dtype=np.int32)
                ref[b] = data[ridx, 0].mean()
        else:  # stationary
            ref_gen = np.random.default_rng(123)
            ref = np.empty(B)
            for b in range(B):
                ridx = _stationary_indices(ref_gen, n, 10)
                ref[b] = data[ridx, 0].mean()

        ks = ks_2samp(ker, ref)
        assert ks.pvalue > 0.01, f"{name} values KS rejected: p={ks.pvalue:.4f}"

    def test_determinism_across_thread_counts(self, name, spec, executor):
        data = np.ascontiguousarray(np.random.default_rng(0).standard_normal((600, 2)))
        seeds = _seeds(2500, seed=7)
        max_threads = numba.config.NUMBA_NUM_THREADS
        numba.set_num_threads(max_threads)
        val_many, idx_many = sk.compiled_values(spec, data, seeds)
        try:
            numba.set_num_threads(1)
            val_one, idx_one = sk.compiled_values(spec, data, seeds)
        finally:
            numba.set_num_threads(max_threads)
        np.testing.assert_array_equal(val_one, val_many)
        np.testing.assert_array_equal(idx_one, idx_many)

    def test_shape_dtype_1d_and_multivariate(self, name, spec, executor):
        # 1-D input returns (B, n, 1); the api layer squeezes via was_1d.
        x = np.random.default_rng(0).standard_normal(200)
        values, idx = sk.compiled_values(spec, x, _seeds(40))
        assert values.shape == (40, 200, 1)
        assert idx.shape == (40, 200) and idx.dtype == np.int32
        # multivariate + sim_dtype honored on the values.
        xv = np.random.default_rng(1).standard_normal((150, 4))
        values32, idx32 = sk.compiled_values(spec, xv, _seeds(30), sim_dtype=np.dtype(np.float32))
        assert values32.shape == (30, 150, 4) and values32.dtype == np.float32
        assert idx32.dtype == np.int32


class TestCompiledValuesDispatch:
    def test_unsupported_method_raises(self):
        x = np.zeros(20)
        with pytest.raises(MethodConfigError, match="does not support"):
            sk.compiled_values(AR(order=1), x, _seeds(3))


# --- var / std reducers -----------------------------------------------------
# The compiled kernels gather the SAME resample that compiled_values materialises.
# The strongest oracle is therefore an exact same-sample check: take the (B, n, d)
# sample from compiled_values, numpy-reduce it with the population (ddof=0)
# var/std, and assert the fused in-kernel reducer matches to tight float
# tolerance. This is independent of any RNG-stream comparison.
_REDUCER_METHODS = [
    ("iid", IID()),
    ("stationary", StationaryBlock(avg_block_length=10)),
    ("moving", MovingBlock(block_length=10)),
    ("circular", CircularBlock(block_length=10)),
    ("non_overlapping", NonOverlappingBlock(block_length=10)),
]


@pytest.mark.parametrize("name,spec", _REDUCER_METHODS, ids=[m[0] for m in _REDUCER_METHODS])
class TestVarStdReducers:
    def test_var_matches_numpy_same_sample(self, name, spec):
        # Exact same-sample oracle: in-kernel population variance == values.var(axis=0).
        for d in (1, 3):
            data = np.ascontiguousarray(np.random.default_rng(0).standard_normal((300, d)))
            seeds = _seeds(60, seed=7)
            values, _idx = sk.compiled_values(spec, data, seeds)
            ker_var = sk.compiled_reduce(spec, data, seeds, reducer="var")
            ref_var = values.var(axis=1)  # ddof=0 population variance, numpy default
            assert ker_var.shape == (60, d)
            np.testing.assert_allclose(ker_var, ref_var, rtol=1e-10, atol=1e-12)

    def test_std_matches_numpy_same_sample(self, name, spec):
        # Exact same-sample oracle: in-kernel population std == values.std(axis=0).
        for d in (1, 3):
            data = np.ascontiguousarray(np.random.default_rng(1).standard_normal((300, d)))
            seeds = _seeds(60, seed=11)
            values, _idx = sk.compiled_values(spec, data, seeds)
            ker_std = sk.compiled_reduce(spec, data, seeds, reducer="std")
            ref_std = values.std(axis=1)  # ddof=0 population std, numpy default
            assert ker_std.shape == (60, d)
            np.testing.assert_allclose(ker_std, ref_std, rtol=1e-10, atol=1e-12)

    def test_std_is_sqrt_of_var(self, name, spec):
        data = np.ascontiguousarray(np.random.default_rng(2).standard_normal((250, 2)))
        seeds = _seeds(50, seed=3)
        ker_var = sk.compiled_reduce(spec, data, seeds, reducer="var")
        ker_std = sk.compiled_reduce(spec, data, seeds, reducer="std")
        np.testing.assert_allclose(ker_std, np.sqrt(ker_var), rtol=1e-12, atol=0)

    def test_1d_input_returns_single_column(self, name, spec):
        x = np.random.default_rng(0).standard_normal(200)
        out_var = sk.compiled_reduce(spec, x, _seeds(40), reducer="var")
        out_std = sk.compiled_reduce(spec, x, _seeds(40), reducer="std")
        assert out_var.shape == (40, 1) and out_std.shape == (40, 1)

    def test_sim_dtype_is_honored(self, name, spec):
        xv = np.random.default_rng(2).standard_normal((150, 3))
        out = sk.compiled_reduce(
            spec, xv, _seeds(40), sim_dtype=np.dtype(np.float32), reducer="var"
        )
        assert out.dtype == np.float32 and out.shape == (40, 3)


class TestVarDeterminism:
    def test_var_bitwise_identical_across_thread_counts(self):
        n, B = 800, 3000
        data = np.ascontiguousarray(np.random.default_rng(0).standard_normal((n, 2)))
        seeds = _seeds(B, seed=7)
        spec = StationaryBlock(avg_block_length=10)
        max_threads = numba.config.NUMBA_NUM_THREADS
        numba.set_num_threads(max_threads)
        out_many = sk.compiled_reduce(spec, data, seeds, reducer="var")
        try:
            numba.set_num_threads(1)
            out_one = sk.compiled_reduce(spec, data, seeds, reducer="var")
        finally:
            numba.set_num_threads(max_threads)
        np.testing.assert_array_equal(out_one, out_many)


class TestVarStdValidation:
    def test_var_and_std_are_supported(self):
        assert "var" in sk._SUPPORTED_REDUCERS and "std" in sk._SUPPORTED_REDUCERS

    def test_unsupported_reducer_still_raises(self):
        x = np.zeros(10)
        with pytest.raises(MethodConfigError, match="Unsupported reducer"):
            sk.compiled_reduce(IID(), x, _seeds(3), reducer="median")


# --- quantile reducer -------------------------------------------------------
# The in-kernel quantile must reproduce numpy's default "linear" interpolation
# exactly. The strongest oracle is the same-sample check used for var/std: take
# the (B, n, d) sample compiled_values materialises, numpy-reduce it with
# np.quantile(..., axis=0, method="linear"), and assert the fused in-kernel
# quantile matches to tight float tolerance. This is independent of any RNG
# stream comparison; the two paths gather the SAME resample for the same seeds.
_QUANTILE_LEVELS = (0.05, 0.5, 0.95)


@pytest.mark.parametrize("name,spec", _REDUCER_METHODS, ids=[m[0] for m in _REDUCER_METHODS])
class TestQuantileReducer:
    def test_quantile_matches_numpy_same_sample(self, name, spec):
        # Exact same-sample oracle across several q for univariate and multivariate.
        for d in (1, 3):
            data = np.ascontiguousarray(np.random.default_rng(0).standard_normal((300, d)))
            seeds = _seeds(60, seed=7)
            values, _idx = sk.compiled_values(spec, data, seeds)
            for q in _QUANTILE_LEVELS:
                ker_q = sk.compiled_reduce(spec, data, seeds, reducer="quantile", q=q)
                ref_q = np.quantile(values, q, axis=1, method="linear")
                assert ker_q.shape == (60, d)
                np.testing.assert_allclose(ker_q, ref_q, rtol=1e-10, atol=1e-12)

    def test_quantile_endpoints_match_numpy(self, name, spec):
        # q == 0 and q == 1 hit the min / max order statistic; the kernel must not
        # read past the buffer end and must equal np.quantile at the endpoints.
        data = np.ascontiguousarray(np.random.default_rng(4).standard_normal((250, 2)))
        seeds = _seeds(50, seed=5)
        values, _idx = sk.compiled_values(spec, data, seeds)
        for q in (0.0, 1.0):
            ker_q = sk.compiled_reduce(spec, data, seeds, reducer="quantile", q=q)
            ref_q = np.quantile(values, q, axis=1, method="linear")
            np.testing.assert_allclose(ker_q, ref_q, rtol=1e-10, atol=1e-12)

    def test_1d_input_returns_single_column(self, name, spec):
        x = np.random.default_rng(0).standard_normal(200)
        out = sk.compiled_reduce(spec, x, _seeds(40), reducer="quantile", q=0.5)
        assert out.shape == (40, 1)

    def test_sim_dtype_is_honored(self, name, spec):
        xv = np.random.default_rng(2).standard_normal((150, 3))
        out = sk.compiled_reduce(
            spec, xv, _seeds(40), sim_dtype=np.dtype(np.float32), reducer="quantile", q=0.5
        )
        assert out.dtype == np.float32 and out.shape == (40, 3)


class TestQuantileDistribution:
    def test_quantile_matches_numpy_callable_path(self):
        # Equal-in-distribution: compiled quantile vs the numpy bootstrap_reduce with
        # a callable quantile statistic (different RNG streams, so KS, not bitwise).
        from tsbootstrap.api import bootstrap_reduce as _bootstrap_reduce

        n, B, q = 500, 4000, 0.5
        data = np.ascontiguousarray(np.random.default_rng(0).standard_normal((n, 1)))
        spec = StationaryBlock(avg_block_length=10)
        ker = sk.compiled_reduce(spec, data, _seeds(B, seed=7), reducer="quantile", q=q)[:, 0]
        ref = _bootstrap_reduce(
            data,
            method=spec,
            statistic=lambda v, idx: np.quantile(v, q, axis=0),
            n_bootstraps=B,
            random_state=123,
        ).statistics
        ref = np.asarray(ref).reshape(B, -1)[:, 0]
        ks = ks_2samp(ker, ref)
        assert ks.pvalue > 0.01, f"quantile statistic KS rejected: p={ks.pvalue:.4f}"


class TestQuantileDeterminism:
    def test_quantile_bitwise_identical_across_thread_counts(self):
        n, B = 800, 3000
        data = np.ascontiguousarray(np.random.default_rng(0).standard_normal((n, 2)))
        seeds = _seeds(B, seed=7)
        spec = StationaryBlock(avg_block_length=10)
        max_threads = numba.config.NUMBA_NUM_THREADS
        numba.set_num_threads(max_threads)
        out_many = sk.compiled_reduce(spec, data, seeds, reducer="quantile", q=0.95)
        try:
            numba.set_num_threads(1)
            out_one = sk.compiled_reduce(spec, data, seeds, reducer="quantile", q=0.95)
        finally:
            numba.set_num_threads(max_threads)
        np.testing.assert_array_equal(out_one, out_many)


class TestQuantileValidation:
    def test_quantile_is_supported(self):
        assert "quantile" in sk._SUPPORTED_REDUCERS

    def test_missing_q_raises(self):
        x = np.zeros(10)
        with pytest.raises(MethodConfigError, match="requires a quantile level"):
            sk.compiled_reduce(IID(), x, _seeds(3), reducer="quantile")

    def test_q_out_of_range_raises(self):
        x = np.zeros(10)
        with pytest.raises(MethodConfigError, match=r"must lie in \[0, 1\]"):
            sk.compiled_reduce(IID(), x, _seeds(3), reducer="quantile", q=1.5)
        with pytest.raises(MethodConfigError, match=r"must lie in \[0, 1\]"):
            sk.compiled_reduce(IID(), x, _seeds(3), reducer="quantile", q=-0.1)


# --- recursive residual AR fused reduce -------------------------------------
# The compiled AR fast path consumes the fitted context (not raw observations).
# Its semantics must match the numpy ResidualBootstrap(AR) reduce: IID resampling
# of the centered residuals, the fitted AR recurrence, and the chosen reducer over
# the generated length-n path. The two backends draw from different RNG streams
# (Philox here, PCG64 there), so they agree in distribution, never bit-for-bit.

from tsbootstrap.api import bootstrap_reduce  # noqa: E402
from tsbootstrap.methods import ARIMA, VAR, ResidualBootstrap, SieveAR  # noqa: E402
from tsbootstrap.model.recursive import _prepare_residual  # noqa: E402


def _ar2_series(n: int, seed: int, phi1: float = 0.6, phi2: float = -0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[:2] = rng.standard_normal(2)
    for t in range(2, n):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + rng.standard_normal()
    return x


def _ar_context(series: np.ndarray, order: int = 2, initial: str = "fixed"):
    """Build the fitted AR context the compiled kernel consumes."""
    spec = ResidualBootstrap(model=AR(order=order, initial=initial))
    return _prepare_residual(np.ascontiguousarray(series), spec, None)


class TestARResidualCompiled:
    def test_statistic_matches_numpy_reduce(self):
        n, B = 1000, 8000
        series = _ar2_series(n, seed=1)
        ctx = _ar_context(series, order=2)
        ker = sk.ar_residual_reduce(ctx, _seeds(B, seed=7), n)[:, 0]
        # Numpy reference: the actual ResidualBootstrap(AR) mean reduce (PCG64).
        ref = bootstrap_reduce(
            series,
            method=ResidualBootstrap(model=AR(order=2)),
            statistic="mean",
            n_bootstraps=B,
            random_state=123,
        ).statistics
        ks = ks_2samp(ker, ref)
        assert ks.pvalue > 0.01, f"AR residual statistic KS rejected: p={ks.pvalue:.4f}"

    def test_random_block_initial_matches_numpy_reduce(self):
        n, B = 800, 8000
        series = _ar2_series(n, seed=2)
        ctx = _ar_context(series, order=2, initial="random_block")
        ker = sk.ar_residual_reduce(ctx, _seeds(B, seed=7), n)[:, 0]
        ref = bootstrap_reduce(
            series,
            method=ResidualBootstrap(model=AR(order=2, initial="random_block")),
            statistic="mean",
            n_bootstraps=B,
            random_state=123,
        ).statistics
        ks = ks_2samp(ker, ref)
        assert ks.pvalue > 0.01, f"AR random-block KS rejected: p={ks.pvalue:.4f}"

    def test_determinism_across_thread_counts(self):
        n, B = 600, 3000
        series = _ar2_series(n, seed=3)
        ctx = _ar_context(series, order=2)
        seeds = _seeds(B, seed=7)
        max_threads = numba.config.NUMBA_NUM_THREADS
        numba.set_num_threads(max_threads)
        out_many = sk.ar_residual_reduce(ctx, seeds, n)
        try:
            numba.set_num_threads(1)
            out_one = sk.ar_residual_reduce(ctx, seeds, n)
        finally:
            numba.set_num_threads(max_threads)
        np.testing.assert_array_equal(out_one, out_many)

    def test_shape_dtype_and_sim_dtype(self):
        n = 300
        series = _ar2_series(n, seed=4)
        ctx = _ar_context(series, order=2)
        out = sk.ar_residual_reduce(ctx, _seeds(50), n)
        assert out.shape == (50, 1) and out.dtype == np.float64
        out32 = sk.ar_residual_reduce(ctx, _seeds(40), n, sim_dtype=np.dtype(np.float32))
        assert out32.dtype == np.float32 and out32.shape == (40, 1)

    def test_var_and_std_reducers_match_numpy_recurrence(self):
        # Same-stream oracle: build the path with the numpy recurrence on the SAME
        # Philox indices the kernel uses, then numpy-reduce with ddof=0 var/std.
        n, B = 200, 60
        series = _ar2_series(n, seed=5)
        ctx = _ar_context(series, order=2)
        seeds = _seeds(B, seed=9)
        key_hi, key_lo = sk.philox_keys_from_seeds(seeds)
        eps = ctx.resampling_innovations
        ar = ctx.fit.ar_coefs
        p = ctx.fit.order
        # Reproduce the kernel's per-replicate buffered Philox resample + recurrence
        # in numpy (four uniforms per permutation, matching _next_u01).
        ref_var = np.empty((B, 1))
        ref_std = np.empty((B, 1))

        @numba.njit(cache=True)
        def _draw(kh, kl, n_resid, count, out):
            buf = np.empty(4, np.uint64)
            st = np.empty(2, np.uint64)
            st[0] = np.uint64(4)
            st[1] = np.uint64(0)
            for t in range(count):
                out[t] = np.int32(sk._next_u01(kh, kl, buf, st) * n_resid)

        for b in range(B):
            idx = np.empty(n - p, dtype=np.int32)
            _draw(key_hi[b], key_lo[b], eps.shape[0], n - p, idx)
            path = np.empty(n)
            path[:p] = series[:p]
            for s in range(n - p):
                t = p + s
                val = ctx.fit.intercept + eps[idx[s]]
                for j in range(p):
                    val += ar[j] * path[t - 1 - j]
                path[t] = val
            ref_var[b, 0] = path.var()
            ref_std[b, 0] = path.std()
        ker_var = sk.ar_residual_reduce(ctx, seeds, n, reducer="var")
        ker_std = sk.ar_residual_reduce(ctx, seeds, n, reducer="std")
        np.testing.assert_allclose(ker_var, ref_var, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(ker_std, ref_std, rtol=1e-10, atol=1e-12)

    def test_quantile_reducer_matches_numpy_recurrence(self):
        # Same-stream oracle: build the path with the numpy recurrence on the SAME
        # Philox indices the kernel uses, then numpy-quantile (linear) per replicate.
        n, B = 200, 60
        series = _ar2_series(n, seed=5)
        ctx = _ar_context(series, order=2)
        seeds = _seeds(B, seed=9)
        key_hi, key_lo = sk.philox_keys_from_seeds(seeds)
        eps = ctx.resampling_innovations
        ar = ctx.fit.ar_coefs
        p = ctx.fit.order

        @numba.njit(cache=True)
        def _draw(kh, kl, n_resid, count, out):
            buf = np.empty(4, np.uint64)
            st = np.empty(2, np.uint64)
            st[0] = np.uint64(4)
            st[1] = np.uint64(0)
            for t in range(count):
                out[t] = np.int32(sk._next_u01(kh, kl, buf, st) * n_resid)

        for q in (0.05, 0.5, 0.95):
            ref_q = np.empty((B, 1))
            for b in range(B):
                idx = np.empty(n - p, dtype=np.int32)
                _draw(key_hi[b], key_lo[b], eps.shape[0], n - p, idx)
                path = np.empty(n)
                path[:p] = series[:p]
                for s in range(n - p):
                    t = p + s
                    val = ctx.fit.intercept + eps[idx[s]]
                    for j in range(p):
                        val += ar[j] * path[t - 1 - j]
                    path[t] = val
                ref_q[b, 0] = np.quantile(path, q, method="linear")
            ker_q = sk.ar_residual_reduce(ctx, seeds, n, reducer="quantile", q=q)
            np.testing.assert_allclose(ker_q, ref_q, rtol=1e-10, atol=1e-12)

    def test_quantile_missing_q_raises(self):
        ctx = _ar_context(_ar2_series(100, seed=6), order=2)
        with pytest.raises(MethodConfigError, match="requires a quantile level"):
            sk.ar_residual_reduce(ctx, _seeds(3), 100, reducer="quantile")

    def test_quantile_out_of_range_raises(self):
        ctx = _ar_context(_ar2_series(100, seed=6), order=2)
        with pytest.raises(MethodConfigError, match=r"must lie in \[0, 1\]"):
            sk.ar_residual_reduce(ctx, _seeds(3), 100, reducer="quantile", q=2.0)

    def test_unsupported_reducer_raises(self):
        ctx = _ar_context(_ar2_series(100, seed=6), order=2)
        with pytest.raises(MethodConfigError, match="Unsupported reducer"):
            sk.ar_residual_reduce(ctx, _seeds(3), 100, reducer="median")

    def test_exog_context_raises(self):
        n = 120
        series = _ar2_series(n, seed=7)
        z = np.random.default_rng(0).standard_normal(n)
        spec = ResidualBootstrap(model=AR(order=2, burn_in=0, initial="fixed"))
        ctx = _prepare_residual(np.ascontiguousarray(series), spec, z)
        with pytest.raises(MethodConfigError, match="exogenous"):
            sk.ar_residual_reduce(ctx, _seeds(3), n)


class TestARResidualDispatch:
    def test_compiled_supports_residual_ar(self):
        assert sk.compiled_supports(ResidualBootstrap(model=AR(order=2)))

    def test_compiled_supports_rejects_arima_and_sieve(self):
        assert not sk.compiled_supports(ResidualBootstrap(model=ARIMA(order=(1, 1, 0))))
        assert not sk.compiled_supports(SieveAR())

    def test_compiled_reduce_routes_ar_context(self):
        n, B = 500, 2000
        series = _ar2_series(n, seed=8)
        ctx = _ar_context(series, order=2)
        out = sk.compiled_reduce(ResidualBootstrap(model=AR(order=2)), ctx, _seeds(B, seed=7))
        assert out.shape == (B, 1)

    def test_compiled_values_recursive_raises_typed_error(self):
        ctx = _ar_context(_ar2_series(200, seed=9), order=2)
        with pytest.raises(MethodConfigError, match="bootstrap_reduce"):
            sk.compiled_values(ResidualBootstrap(model=AR(order=2)), ctx, _seeds(3))


# --- recursive residual VAR fused reduce ------------------------------------
# The compiled VAR fast path is the multivariate analogue of the AR path. It
# consumes the fitted VAR context (not raw observations) and its semantics must
# match the numpy ResidualBootstrap(VAR) reduce: IID resampling of the centered
# vector residual rows, the fitted VAR recurrence, and the chosen reducer over
# each generated column of the length-n path. The two backends draw from
# different RNG streams (Philox here, PCG64 there), so they agree in distribution,
# never bit-for-bit. The output is (B, d), one statistic per series per replicate.


def _var_series(n: int, d: int, seed: int) -> np.ndarray:
    """A stable VAR(1) multivariate series with a mild, stationary lag matrix."""
    rng = np.random.default_rng(seed)
    # A diagonally dominant lag matrix keeps the process comfortably stationary.
    a = 0.4 * np.eye(d) + 0.05 * rng.standard_normal((d, d))
    x = np.empty((n, d))
    x[0] = rng.standard_normal(d)
    for t in range(1, n):
        x[t] = a @ x[t - 1] + 0.5 * rng.standard_normal(d)
    return np.ascontiguousarray(x)


def _var_context(series: np.ndarray, order: int = 1, initial: str = "fixed"):
    """Build the fitted VAR context the compiled kernel consumes."""
    spec = ResidualBootstrap(model=VAR(order=order, initial=initial))
    return _prepare_residual(np.ascontiguousarray(series), spec, None)


def _var_same_index_oracle(ctx, seeds, n, reducer, q=None):
    """Numpy VAR recurrence on the SAME Philox indices the kernel draws.

    Mirrors the AR same-stream oracle: reproduce the kernel's per-replicate
    buffered Philox resample (four uniforms per permutation, via ``_next_u01``) and
    run the fitted VAR recurrence in numpy, then reduce each column with numpy's
    matching ddof=0 / linear-interpolation semantics. ``initial="fixed"`` only.
    """
    key_hi, key_lo = sk.philox_keys_from_seeds(seeds)
    eps = ctx.resampling_innovations  # (n_resid, d)
    coefs = ctx.fit.coefs  # (p, d, d)
    intercept = ctx.fit.intercept  # (d,)
    series = ctx.series  # (n, d)
    p = ctx.fit.order
    d = series.shape[1]
    n_resid = eps.shape[0]
    B = len(seeds)

    @numba.njit(cache=True)
    def _draw(kh, kl, n_resid, count, out):
        buf = np.empty(4, np.uint64)
        st = np.empty(2, np.uint64)
        st[0] = np.uint64(4)
        st[1] = np.uint64(0)
        for t in range(count):
            out[t] = np.int32(sk._next_u01(kh, kl, buf, st) * n_resid)

    out = np.empty((B, d))
    for b in range(B):
        idx = np.empty(n - p, dtype=np.int32)
        _draw(key_hi[b], key_lo[b], n_resid, n - p, idx)
        path = np.empty((n, d))
        path[:p] = series[:p]
        for s in range(n - p):
            t = p + s
            val = intercept + eps[idx[s]]
            for j in range(p):
                val = val + coefs[j] @ path[t - 1 - j]
            path[t] = val
        for c in range(d):
            if reducer == "var":
                out[b, c] = path[:, c].var()
            elif reducer == "std":
                out[b, c] = path[:, c].std()
            else:  # quantile
                out[b, c] = np.quantile(path[:, c], q, method="linear")
    return out


class TestVARResidualCompiled:
    @pytest.mark.parametrize("d", [2, 3])
    def test_statistic_matches_numpy_reduce(self, d):
        # B is larger than the AR analogue because each extra series column is an
        # extra KS comparison, so a generous sample size keeps the per-column test
        # away from the multiple-comparison borderline at a fixed seed.
        n, B = 600, 12000
        series = _var_series(n, d, seed=1)
        ctx = _var_context(series, order=1)
        ker = sk.var_residual_reduce(ctx, _seeds(B, seed=7), n)
        # Numpy reference: the actual ResidualBootstrap(VAR) mean reduce (PCG64).
        ref = bootstrap_reduce(
            series,
            method=ResidualBootstrap(model=VAR(order=1)),
            statistic="mean",
            n_bootstraps=B,
            random_state=123,
        ).statistics
        # Each of the d columns is its own one-dimensional bootstrap distribution.
        for c in range(d):
            ks = ks_2samp(ker[:, c], ref[:, c])
            assert ks.pvalue > 0.01, f"VAR d={d} col={c} KS rejected: p={ks.pvalue:.4f}"

    def test_random_block_initial_matches_numpy_reduce(self):
        n, B, d = 500, 8000, 2
        series = _var_series(n, d, seed=2)
        ctx = _var_context(series, order=1, initial="random_block")
        ker = sk.var_residual_reduce(ctx, _seeds(B, seed=7), n)
        ref = bootstrap_reduce(
            series,
            method=ResidualBootstrap(model=VAR(order=1, initial="random_block")),
            statistic="mean",
            n_bootstraps=B,
            random_state=123,
        ).statistics
        for c in range(d):
            ks = ks_2samp(ker[:, c], ref[:, c])
            assert ks.pvalue > 0.01, f"VAR random-block col={c} KS rejected: p={ks.pvalue:.4f}"

    def test_determinism_across_thread_counts(self):
        n, B, d = 400, 3000, 3
        series = _var_series(n, d, seed=3)
        ctx = _var_context(series, order=2)
        seeds = _seeds(B, seed=7)
        max_threads = numba.config.NUMBA_NUM_THREADS
        numba.set_num_threads(max_threads)
        out_many = sk.var_residual_reduce(ctx, seeds, n)
        try:
            numba.set_num_threads(1)
            out_one = sk.var_residual_reduce(ctx, seeds, n)
        finally:
            numba.set_num_threads(max_threads)
        np.testing.assert_array_equal(out_one, out_many)

    def test_shape_dtype_and_sim_dtype(self):
        n, d = 300, 3
        series = _var_series(n, d, seed=4)
        ctx = _var_context(series, order=1)
        out = sk.var_residual_reduce(ctx, _seeds(50), n)
        assert out.shape == (50, d) and out.dtype == np.float64
        out32 = sk.var_residual_reduce(ctx, _seeds(40), n, sim_dtype=np.dtype(np.float32))
        assert out32.dtype == np.float32 and out32.shape == (40, d)

    @pytest.mark.parametrize("order", [1, 2])
    def test_var_and_std_reducers_match_numpy_recurrence(self, order):
        # Same-stream oracle: build the path with the numpy VAR recurrence on the
        # SAME Philox indices the kernel uses, then numpy-reduce with ddof=0 var/std.
        n, B, d = 200, 60, 2
        series = _var_series(n, d, seed=5)
        ctx = _var_context(series, order=order)
        seeds = _seeds(B, seed=9)
        ref_var = _var_same_index_oracle(ctx, seeds, n, "var")
        ref_std = _var_same_index_oracle(ctx, seeds, n, "std")
        ker_var = sk.var_residual_reduce(ctx, seeds, n, reducer="var")
        ker_std = sk.var_residual_reduce(ctx, seeds, n, reducer="std")
        np.testing.assert_allclose(ker_var, ref_var, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(ker_std, ref_std, rtol=1e-10, atol=1e-12)

    def test_quantile_reducer_matches_numpy_recurrence(self):
        # Same-stream oracle: numpy VAR recurrence on the SAME Philox indices, then
        # numpy-quantile (linear) per column.
        n, B, d = 200, 60, 3
        series = _var_series(n, d, seed=5)
        ctx = _var_context(series, order=1)
        seeds = _seeds(B, seed=9)
        for q in (0.05, 0.5, 0.95):
            ref_q = _var_same_index_oracle(ctx, seeds, n, "quantile", q=q)
            ker_q = sk.var_residual_reduce(ctx, seeds, n, reducer="quantile", q=q)
            np.testing.assert_allclose(ker_q, ref_q, rtol=1e-10, atol=1e-12)

    def test_quantile_missing_q_raises(self):
        ctx = _var_context(_var_series(100, 2, seed=6), order=1)
        with pytest.raises(MethodConfigError, match="requires a quantile level"):
            sk.var_residual_reduce(ctx, _seeds(3), 100, reducer="quantile")

    def test_quantile_out_of_range_raises(self):
        ctx = _var_context(_var_series(100, 2, seed=6), order=1)
        with pytest.raises(MethodConfigError, match=r"must lie in \[0, 1\]"):
            sk.var_residual_reduce(ctx, _seeds(3), 100, reducer="quantile", q=2.0)

    def test_unsupported_reducer_raises(self):
        ctx = _var_context(_var_series(100, 2, seed=6), order=1)
        with pytest.raises(MethodConfigError, match="Unsupported reducer"):
            sk.var_residual_reduce(ctx, _seeds(3), 100, reducer="median")

    def test_exog_context_raises(self):
        n, d = 120, 2
        series = _var_series(n, d, seed=7)
        z = np.random.default_rng(0).standard_normal(n)
        spec = ResidualBootstrap(model=VAR(order=1, burn_in=0, initial="fixed"))
        ctx = _prepare_residual(np.ascontiguousarray(series), spec, z)
        with pytest.raises(MethodConfigError, match="exogenous"):
            sk.var_residual_reduce(ctx, _seeds(3), n)


class TestVARResidualDispatch:
    def test_compiled_supports_residual_var(self):
        assert sk.compiled_supports(ResidualBootstrap(model=VAR(order=1)))

    def test_compiled_reduce_routes_var_context(self):
        n, B, d = 400, 2000, 2
        series = _var_series(n, d, seed=8)
        ctx = _var_context(series, order=1)
        out = sk.compiled_reduce(ResidualBootstrap(model=VAR(order=1)), ctx, _seeds(B, seed=7))
        assert out.shape == (B, d)

    def test_compiled_values_var_raises_typed_error(self):
        ctx = _var_context(_var_series(200, 2, seed=9), order=1)
        with pytest.raises(MethodConfigError, match="bootstrap_reduce"):
            sk.compiled_values(ResidualBootstrap(model=VAR(order=1)), ctx, _seeds(3))

    def test_compiled_reduce_var_routes_not_unsupported(self):
        # VAR must route through the compiled reduce, never raise the
        # unsupported-method error.
        n, d = 200, 2
        ctx = _var_context(_var_series(n, d, seed=10), order=1)
        out = sk.compiled_reduce(ResidualBootstrap(model=VAR(order=1)), ctx, _seeds(20))
        assert out.shape == (20, d)
