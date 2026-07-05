"""Tests for the ragged-panel compiled fast path (num_series x B in one parallel pass).

These exercise the panel kernels and the public :func:`bootstrap_reduce_panel` entry
for the observation methods (IID, moving / circular / non-overlapping block, and
stationary). The compiled checks are skipped when numba (the ``[accel]`` extra) is not
installed, exactly as the rectangular compiled tests handle that case.

What is checked:

- Raggedness correctness: every series's resample indices stay strictly inside its own
  ``[0, n_s)`` (the CSR ``indptr`` confines every gather to that series's flat slice).
- Standalone == in-panel (the load-bearing guarantee): a series at panel slot ``s`` is
  bitwise-identical whether the panel has ``s + 1`` series or more, for every slot, B,
  and column count.
- num_series == 1 panel == the existing single-series reduce, bitwise (the panel path
  must not perturb the single-series result).
- Compiled panel vs the numpy reference loop: equal in distribution (two-sample KS per
  series), since the two backends draw from different RNG streams.
- Determinism across numba thread counts.
- Typed errors: malformed ``indptr`` (non-monotonic, wrong total, empty series), and a
  recursive method rejected on a panel.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsbootstrap import (
    IID,
    CircularBlock,
    MovingBlock,
    NonOverlappingBlock,
    ResidualBootstrap,
    StationaryBlock,
    bootstrap_reduce_panel,
)
from tsbootstrap.errors import MethodConfigError
from tsbootstrap.methods import AR

numba = pytest.importorskip("numba")  # optional [accel] extra; see module docstring

from scipy.stats import ks_2samp  # noqa: E402

from tsbootstrap.block import _compiled as sk  # noqa: E402


@pytest.fixture(autouse=True)
def _warm_kernel():
    """Compile the kernels once before each test so timing and threads are stable."""
    sk._warm_compiled_kernels()


def _root(seed: int = 12345) -> tuple[int, int]:
    """Packed 128-bit root the compiled panel kernels key from (mirrors _root_key_from).

    Each fused kernel derives replicate b's key from (root, b) in its parallel loop;
    ``n_bootstraps`` sets B.
    """
    words = np.random.SeedSequence(seed).generate_state(4, dtype=np.uint32)
    return (int(words[0]) << 32) | int(words[1]), (int(words[2]) << 32) | int(words[3])


def _ragged(lengths, d=1, seed=0):
    rng = np.random.default_rng(seed)
    indptr = np.zeros(len(lengths) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(lengths)
    flat = rng.standard_normal((int(indptr[-1]), d)).astype(np.float64)
    return flat, indptr


# --- raggedness containment -------------------------------------------------


@pytest.mark.parametrize(
    "indices_fn",
    [
        lambda indptr, root, nb: sk.panel_iid_local_indices(indptr, root, n_bootstraps=nb),
        lambda indptr, root, nb: sk.panel_stationary_local_indices(
            indptr, root, 8, n_bootstraps=nb
        ),
        lambda indptr, root, nb: sk.panel_block_local_indices(
            "moving", indptr, root, 20, n_bootstraps=nb
        ),
        lambda indptr, root, nb: sk.panel_block_local_indices(
            "circular", indptr, root, 20, n_bootstraps=nb
        ),
        lambda indptr, root, nb: sk.panel_block_local_indices(
            "non_overlapping", indptr, root, 20, n_bootstraps=nb
        ),
    ],
)
def test_raggedness_containment(indices_fn):
    """Every series's local resample indices stay strictly in its own [0, n_s)."""
    lengths = [50, 200, 137, 1000]
    _, indptr = _ragged(lengths)
    flat_idx = indices_fn(indptr, _root(), 16)
    for b in range(flat_idx.shape[0]):
        for s in range(len(lengths)):
            lo, hi = int(indptr[s]), int(indptr[s + 1])
            local = flat_idx[b, lo:hi]
            n_s = hi - lo
            assert local.min() >= 0
            assert local.max() < n_s, f"series {s} index escaped [0, {n_s})"


# --- standalone == in-panel (the load-bearing guarantee) --------------------


@pytest.mark.parametrize(
    "indices_fn",
    [
        lambda indptr, root, nb: sk.panel_iid_local_indices(indptr, root, n_bootstraps=nb),
        lambda indptr, root, nb: sk.panel_stationary_local_indices(
            indptr, root, 8, n_bootstraps=nb
        ),
        lambda indptr, root, nb: sk.panel_block_local_indices(
            "moving", indptr, root, 20, n_bootstraps=nb
        ),
        lambda indptr, root, nb: sk.panel_block_local_indices(
            "circular", indptr, root, 20, n_bootstraps=nb
        ),
    ],
)
@pytest.mark.parametrize("num_series", [1, 4, 8])
def test_standalone_equals_in_panel(indices_fn, num_series):
    """A series at slot s is bitwise-identical regardless of how many series follow it."""
    n = 100
    lengths = [n] * num_series
    _, indptr = _ragged(lengths)
    root = _root()
    full = indices_fn(indptr, root, 16)
    for s in range(num_series):
        # A sub-panel containing only slots 0..s (slot s is the last one).
        sub_indptr = np.array([n * k for k in range(s + 2)], dtype=np.int64)
        sub = indices_fn(sub_indptr, root, 16)
        lo, hi = n * s, n * (s + 1)
        assert np.array_equal(sub[:, lo:hi], full[:, lo:hi]), (
            f"slot {s}: standalone (sub-panel) != in-panel"
        )


def test_standalone_equals_in_panel_reduce():
    """The reduce output for a series matches whether it is alone or inside a panel."""
    lengths = [50, 200, 137, 1000]
    flat, indptr = _ragged(lengths, d=2)
    root = _root()
    panel = sk.panel_stationary_reduce(
        flat, indptr, root, 8, np.dtype("float64"), "mean", None, n_bootstraps=24
    )
    for s in range(len(lengths)):
        # The series alone, but at the SAME slot s (a sub-panel of slots 0..s).
        sub_indptr = np.array([int(indptr[k]) for k in range(s + 2)], dtype=np.int64)
        sub = sk.panel_stationary_reduce(
            flat[: int(indptr[s + 1])],
            sub_indptr,
            root,
            8,
            np.dtype("float64"),
            "mean",
            None,
            n_bootstraps=24,
        )
        assert np.array_equal(sub[:, s, :], panel[:, s, :])


# --- per-(replicate, series) stream distinctness incl. the degenerate root --


def _replicate_key_py(root_a: int, root_b: int, b: int) -> tuple[int, int]:
    """Pure-Python mirror of the njit ``sk._replicate_key`` (see test_compiled.py)."""
    mask = (1 << 64) - 1
    golden = 0xD1B54A32D192ED03
    z = (root_a + (b + 1) * golden) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    z = z ^ (z >> 31)
    z = z ^ root_b
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    z = z ^ (z >> 31)
    return (z >> 32) & 0xFFFFFFFF, z & 0xFFFFFFFF


@pytest.mark.parametrize("root_key", [(0, 0), (12345, 67890)])
def test_panel_stream_distinctness_incl_degenerate_root(root_key):
    """Every (replicate, series) key stays distinct, including the all-zero packed root.

    No two (replicate, series) items share a Philox key under the composed
    _replicate_key -> _fold_in_key derivation, INCLUDING the all-zero packed root.
    This is the regression guard for the domain-separation constant: were
    _replicate_key to reuse _hash_series_words' golden, the effective key would be
    symmetric in (b, s) at root == 0 and replicate b / series s would alias replicate
    s / series b. The distinct _REPLICATE_GOLDEN plus the second-round root_b fold
    break that symmetry structurally, so the whole (b, s) grid must stay distinct even
    at the degenerate root.
    """
    root_a, root_b = root_key
    B, S = 16, 16
    grid = set()
    fold = {}
    for b in range(B):
        kh, kl = _replicate_key_py(root_a, root_b, b)
        for s in range(S):
            fh, fl = sk._fold_in_key(np.uint32(kh), np.uint32(kl), s)
            key = (int(fh), int(fl))
            grid.add(key)
            fold[(b, s)] = key
    assert len(grid) == B * S  # every (b, s) key is distinct
    # Explicitly assert the (b, s) <-> (s, b) symmetry is absent at any root.
    for b in range(min(B, S)):
        for s in range(min(B, S)):
            if b != s:
                assert fold[(b, s)] != fold[(s, b)]


# --- num_series == 1 panel == single-series reduce --------------------------


@pytest.mark.parametrize(
    "reducer,q", [("mean", None), ("var", None), ("std", None), ("quantile", 0.7)]
)
@pytest.mark.parametrize("d", [1, 2])
def test_one_series_panel_equals_single_series(reducer, q, d):
    """A num_series==1 panel reproduces the existing single-series reduce, bitwise."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((300, d))
    root = _root()
    B = 32
    indptr = np.array([0, 300], dtype=np.int64)
    dt = np.dtype("float64")

    single = sk.iid_reduce(data, root, dt, reducer, q, n_bootstraps=B)
    panel = sk.panel_iid_reduce(data, indptr, root, dt, reducer, q, n_bootstraps=B)
    assert np.array_equal(panel[:, 0, :], single)

    for fam, fn in (
        (
            "moving",
            lambda: sk.block_reduce("moving", data, root, 20, dt, reducer, q, n_bootstraps=B),
        ),
        (
            "circular",
            lambda: sk.block_reduce("circular", data, root, 20, dt, reducer, q, n_bootstraps=B),
        ),
        (
            "non_overlapping",
            lambda: sk.block_reduce(
                "non_overlapping", data, root, 20, dt, reducer, q, n_bootstraps=B
            ),
        ),
    ):
        s1 = fn()
        p1 = sk.panel_block_reduce(fam, data, indptr, root, 20, dt, reducer, q, n_bootstraps=B)
        assert np.array_equal(p1[:, 0, :], s1), fam

    s1 = sk.stationary_reduce(data, root, 10, dt, reducer, q, n_bootstraps=B)
    p1 = sk.panel_stationary_reduce(data, indptr, root, 10, dt, reducer, q, n_bootstraps=B)
    assert np.array_equal(p1[:, 0, :], s1)


# --- compiled panel vs numpy reference loop: equal in distribution ----------


@pytest.mark.parametrize(
    "method",
    [
        IID(),
        MovingBlock(block_length=15),
        CircularBlock(block_length=15),
        NonOverlappingBlock(block_length=15),
        StationaryBlock(avg_block_length=10),
    ],
)
def test_compiled_panel_equals_numpy_in_distribution(method):
    """Per series, the compiled and numpy panel statistics match in distribution (KS)."""
    lengths = [120, 300, 200]
    rng = np.random.default_rng(5)
    panel = [rng.standard_normal(n) for n in lengths]
    B = 600
    r_np = bootstrap_reduce_panel(
        panel, method=method, statistic="mean", n_bootstraps=B, random_state=7, backend="numpy"
    )
    r_c = bootstrap_reduce_panel(
        panel, method=method, statistic="mean", n_bootstraps=B, random_state=7, backend="compiled"
    )
    assert r_np.statistics.shape == (B, len(lengths))
    assert r_c.statistics.shape == (B, len(lengths))
    for s in range(len(lengths)):
        _, pval = ks_2samp(r_np.statistics[:, s], r_c.statistics[:, s])
        assert pval > 0.01, f"series {s}: KS p={pval} (distributions differ)"


def test_list_and_flat_inputs_agree():
    """The list-of-series and (flat + indptr) input forms produce identical results."""
    lengths = [50, 200, 137, 1000]
    rng = np.random.default_rng(0)
    panel = [rng.standard_normal(n) for n in lengths]
    flat = np.concatenate(panel)
    indptr = np.zeros(len(lengths) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(lengths)
    r_list = bootstrap_reduce_panel(
        panel, method=IID(), statistic="mean", n_bootstraps=64, random_state=42, backend="compiled"
    )
    r_flat = bootstrap_reduce_panel(
        flat,
        indptr=indptr,
        method=IID(),
        statistic="mean",
        n_bootstraps=64,
        random_state=42,
        backend="compiled",
    )
    assert np.array_equal(r_list.statistics, r_flat.statistics)


def test_multivariate_panel_shape():
    """A multivariate panel keeps the trailing theta axis; univariate collapses it."""
    rng = np.random.default_rng(9)
    panel_2d = [rng.standard_normal((n, 3)) for n in (60, 120)]
    r = bootstrap_reduce_panel(
        panel_2d,
        method=IID(),
        statistic="mean",
        n_bootstraps=32,
        random_state=1,
        backend="compiled",
    )
    assert r.statistics.shape == (32, 2, 3)
    panel_1d = [rng.standard_normal(n) for n in (60, 120)]
    r1 = bootstrap_reduce_panel(
        panel_1d,
        method=IID(),
        statistic="mean",
        n_bootstraps=32,
        random_state=1,
        backend="compiled",
    )
    assert r1.statistics.shape == (32, 2)


# --- thread-count determinism -----------------------------------------------


def test_thread_count_determinism():
    """The panel reduce is bitwise-invariant to the numba thread count."""
    lengths = [50, 200, 137, 1000]
    flat, indptr = _ragged(lengths)
    root = _root()
    orig = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        r1 = sk.panel_stationary_reduce(
            flat, indptr, root, 8, np.dtype("float64"), "mean", None, n_bootstraps=128
        )
        numba.set_num_threads(max(2, orig))
        rN = sk.panel_stationary_reduce(
            flat, indptr, root, 8, np.dtype("float64"), "mean", None, n_bootstraps=128
        )
    finally:
        numba.set_num_threads(orig)
    assert np.array_equal(r1, rN)


# --- typed errors at the boundary -------------------------------------------


def test_indptr_non_monotonic_rejected():
    flat = np.zeros((100, 1))
    bad = np.array([0, 60, 40, 100], dtype=np.int64)  # decreases at position 1->2
    with pytest.raises(MethodConfigError):
        bootstrap_reduce_panel(
            flat, indptr=bad, method=IID(), statistic="mean", n_bootstraps=8, backend="compiled"
        )


def test_indptr_wrong_total_rejected():
    flat = np.zeros((100, 1))
    bad = np.array([0, 50, 90], dtype=np.int64)  # ends at 90, not 100
    with pytest.raises(MethodConfigError):
        bootstrap_reduce_panel(
            flat, indptr=bad, method=IID(), statistic="mean", n_bootstraps=8, backend="compiled"
        )


def test_indptr_empty_series_rejected():
    flat = np.zeros((100, 1))
    bad = np.array([0, 50, 50, 100], dtype=np.int64)  # series 1 is empty
    with pytest.raises(MethodConfigError):
        bootstrap_reduce_panel(
            flat, indptr=bad, method=IID(), statistic="mean", n_bootstraps=8, backend="compiled"
        )


def test_indptr_not_starting_at_zero_rejected():
    flat = np.zeros((100, 1))
    bad = np.array([1, 50, 100], dtype=np.int64)
    with pytest.raises(MethodConfigError):
        bootstrap_reduce_panel(
            flat, indptr=bad, method=IID(), statistic="mean", n_bootstraps=8, backend="compiled"
        )


def test_recursive_method_rejected_on_panel():
    panel = [np.random.default_rng(0).standard_normal(100) for _ in range(3)]
    with pytest.raises(MethodConfigError):
        bootstrap_reduce_panel(
            panel,
            method=ResidualBootstrap(model=AR(order=1)),
            statistic="mean",
            n_bootstraps=8,
            backend="compiled",
        )
    # also rejected on the numpy backend (no coherent recursive-panel meaning in v1)
    with pytest.raises(MethodConfigError):
        bootstrap_reduce_panel(
            panel,
            method=ResidualBootstrap(model=AR(order=1)),
            statistic="mean",
            n_bootstraps=8,
            backend="numpy",
        )


def test_compiled_panel_rejects_callable_statistic():
    panel = [np.random.default_rng(0).standard_normal(100) for _ in range(2)]
    with pytest.raises(MethodConfigError):
        bootstrap_reduce_panel(
            panel,
            method=IID(),
            statistic=lambda v, i: v.mean(axis=0),
            n_bootstraps=8,
            backend="compiled",
        )


def test_quantile_panel_reducer():
    """The quantile reducer works on a panel through both backends, equal in distribution."""
    lengths = [150, 250]
    rng = np.random.default_rng(11)
    panel = [rng.standard_normal(n) for n in lengths]
    B = 600
    # The KS seed is stream-dependent: the two backends are equal in distribution, so the
    # per-comparison p-value is Uniform(0, 1) under H0 and a fixed seed is a golden input.
    # random_state=0 is a robust draw for the root-keyed compiled stream (a scan of 40 seeds
    # left only a lone 1-in-100 unlucky draw, matching the H0 uniform rate, confirming no
    # distributional drift).
    r_np = bootstrap_reduce_panel(
        panel,
        method=IID(),
        statistic=("quantile", 0.9),
        n_bootstraps=B,
        random_state=0,
        backend="numpy",
    )
    r_c = bootstrap_reduce_panel(
        panel,
        method=IID(),
        statistic=("quantile", 0.9),
        n_bootstraps=B,
        random_state=0,
        backend="compiled",
    )
    assert r_np.statistics.shape == (B, 2)
    for s in range(2):
        _, pval = ks_2samp(r_np.statistics[:, s], r_c.statistics[:, s])
        assert pval > 0.01
