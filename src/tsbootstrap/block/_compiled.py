"""Compiled fast paths for the overlapping block bootstraps (opt-in, distinct RNG stream).

This module provides optional, numba-compiled alternatives to the pure-numpy
observation-resampling paths in :mod:`tsbootstrap.block` and the IID path in
:mod:`tsbootstrap.api`. Each compiled kernel follows the same block-construction
distribution as its pure-numpy counterpart, but draws from a *different* random
stream and is therefore not bit-identical to it:

- The pure-numpy paths use one ``PCG64`` generator per replicate, spawned from
  the run's root ``SeedSequence`` (the locked default RNG contract).
- These compiled paths use a counter-based Philox-4x32-10 stream (the canonical
  Random123 32-bit-wide Philox, matching JAX's ``philox4x32`` width). Each
  replicate's two Philox key words are derived IN-KERNEL from the run's packed
  128-bit root and the replicate index ``b`` alone (see :func:`_replicate_key`),
  so no per-replicate ``SeedSequence`` is spawned and no O(B) Python key loop runs
  before dispatch. No RNG state is shared between replicates: the stream for
  replicate ``b`` is a pure function of ``(root, b, counter)``, so running the
  replicates in parallel and in any order produces bitwise-identical output
  regardless of thread count, and the first ``B`` streams are stable as ``B`` grows.

Because the streams differ, these fast paths have their own reproducibility
goldens; none is a drop-in reproduction of the PCG64 default. Callers opt in
explicitly. When numba (the ``[accel]`` extra) is not installed, calling an entry
point raises a clear, actionable error rather than silently falling back to the
default stream (mirroring how the VAR engine treats its compiled kernel as an
explicit opt-in accelerator).

Each kernel matches its pure-numpy block construction:

- IID: ``n`` independent uniform positions in ``[0, n)``.
- Moving block: ``ceil(n / L)`` block starts uniform in ``[0, n - L + 1]``, tiled
  as contiguous length-``L`` runs and trimmed to ``n``.
- Circular block: ``ceil(n / L)`` block starts uniform in ``[0, n)``, tiled with
  wraparound modulo ``n`` and trimmed to ``n``.
- Non-overlapping block: ``ceil(n / L)`` starts that are uniform multiples of
  ``L`` in ``[0, n)``, tiled and trimmed to ``n``.
- Stationary: geometric block lengths (a new block starts when a uniform draw is
  below ``p = 1 / avg_block_length``), every block at an independent uniform
  restart point, wrapping modulo ``n``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.errors import Codes, MethodConfigError
from tsbootstrap.prng_keys import REPLICATE_GOLDEN as _REPLICATE_GOLDEN_INT
from tsbootstrap.prng_keys import SERIES_GOLDEN as _SERIES_GOLDEN_INT
from tsbootstrap.rng import register_warmup

if TYPE_CHECKING:
    from tsbootstrap.methods import BlockLength
    from tsbootstrap.model.recursive import _ARContext, _VARContext

# --- Philox-4x32-10 constants (canonical Random123) -------------------------
# These are the canonical Philox-4x32 multipliers and Weyl key-increment
# constants from Salmon et al. 2011 / the DEShawResearch Random123 reference
# (include/Random123/philox.h: PHILOX_M4x32_0/1, PHILOX_W32_0/1). They must not
# be changed: the stream's statistical properties, the published known-answer
# vectors, and the committed reproducibility goldens depend on them exactly.
#
# This is the 32-bit-wide Philox: 32x32 -> 64 multiplies map to one native
# instruction (no software 64x64 high-half), and the width matches JAX's
# philox4x32 PRNG, so the integer index stream can be made bit-identical across
# CPU and GPU in future. It remains the opt-in compiled stream (its own
# goldens, equal in distribution to the PCG64 default, never the default).
_PHILOX_M0: Final = np.uint64(0xD2511F53)
_PHILOX_M1: Final = np.uint64(0xCD9E8D57)
_PHILOX_W0: Final = np.uint32(0x9E3779B9)
_PHILOX_W1: Final = np.uint32(0xBB67AE85)
_MASK32: Final = np.uint64(0xFFFFFFFF)
_SH32: Final = np.uint64(32)
_TWO32_INV: Final = 1.0 / 4294967296.0  # 1 / 2**32

# Reducer selector codes. The fused kernels gather each replicate's resample
# into a small per-column scratch buffer and apply one of these reducers as the
# final step. The selector exists so further njit-able reducers (quantiles) can
# be added without changing the entry-point signature. Untested reducers are
# intentionally not shipped.
#
# Semantics MUST match the numpy backend's named reducers exactly so the two
# backends agree in distribution:
#   "mean"     -> values.mean(axis=0)
#   "var"      -> values.var(axis=0)   (population variance, ddof=0)
#   "std"      -> values.std(axis=0)   (population std, ddof=0; std = sqrt(var))
#   "quantile" -> np.quantile(values, q, axis=0)  (default "linear" interpolation)
REDUCER_MEAN: Final = "mean"
REDUCER_VAR: Final = "var"
REDUCER_STD: Final = "std"
REDUCER_QUANTILE: Final = "quantile"
_SUPPORTED_REDUCERS: Final = frozenset({REDUCER_MEAN, REDUCER_VAR, REDUCER_STD, REDUCER_QUANTILE})

# Integer reducer codes threaded into the njit kernels (njit cannot branch on a
# Python string cheaply, so the Python wrappers translate the validated string to
# one of these codes once at the boundary).
_RCODE_MEAN: Final = 0
_RCODE_VAR: Final = 1
_RCODE_STD: Final = 2
_RCODE_QUANTILE: Final = 3
_REDUCER_CODES: Final = {
    REDUCER_MEAN: _RCODE_MEAN,
    REDUCER_VAR: _RCODE_VAR,
    REDUCER_STD: _RCODE_STD,
    REDUCER_QUANTILE: _RCODE_QUANTILE,
}

_FLOAT64: Final = np.dtype(np.float64)


try:  # optional [accel] extra: a compiled, replicate-parallel fused kernel
    import numba

    @numba.njit(inline="always", cache=True)
    def _philox_round4(  # pragma: no cover - njit-compiled to machine code
        key_hi: np.uint32, key_lo: np.uint32, ctr: np.uint64, out4: NDArray[np.uint64]
    ) -> None:
        # Ten canonical Philox-4x32 rounds, keeping ALL FOUR 32-bit output words in
        # the caller-owned length-4 uint64 buffer ``out4`` (each holds one uint32
        # value). Spending all four words turns one 10-round permutation into four
        # uniforms.
        #
        # The 4x32 counter words (c0..c3) carry the per-replicate permutation index
        # (the block counter), and the two 32-bit key words (k0, k1) are this
        # replicate's per-replicate key. The stream is thus a pure function of
        # (key, draw-index), which is what makes the parallel kernels thread-count
        # invariant. Each 32x32 -> 64 multiply is one native instruction (no
        # software high-half), the speedup motivation for the 32-bit width.
        #
        # mulhilo32(a, b): p = uint64(a) * uint64(b); hi = uint32(p >> 32); lo =
        # uint32(p). The Random123 round writes
        #   out = (hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0)
        # and bumps the key with the Weyl constants BEFORE rounds 2..10 (so the
        # first round runs with the un-bumped key; 9 bumps total, none after the
        # last round).
        c0 = ctr & _MASK32
        c1 = np.uint64(0)
        c2 = np.uint64(0)
        c3 = np.uint64(0)
        k0 = np.uint64(key_lo)
        k1 = np.uint64(key_hi)
        for r in range(10):
            if r > 0:
                k0 = (k0 + np.uint64(_PHILOX_W0)) & _MASK32
                k1 = (k1 + np.uint64(_PHILOX_W1)) & _MASK32
            p0 = _PHILOX_M0 * c0
            hi0 = p0 >> _SH32
            lo0 = p0 & _MASK32
            p1 = _PHILOX_M1 * c2
            hi1 = p1 >> _SH32
            lo1 = p1 & _MASK32
            c0 = (hi1 ^ c1 ^ k0) & _MASK32
            c1 = lo1
            c2 = (hi0 ^ c3 ^ k1) & _MASK32
            c3 = lo0
        out4[0] = c0
        out4[1] = c1
        out4[2] = c2
        out4[3] = c3

    @numba.njit(inline="always", cache=True)
    def _u01_from_word(word: np.uint64) -> np.float64:  # pragma: no cover - njit-compiled
        # 32-bit uniform in [0, 1): one Philox output word divided by 2**32. The
        # 32-bit resolution is sufficient for resampling indices (n < 2**31) and
        # the geometric restart test; the conversion is shared by every draw site.
        return np.float64(word) * _TWO32_INV

    @numba.njit(inline="always", cache=True)
    def _next_u01(  # pragma: no cover - njit-compiled to machine code
        key_hi: np.uint32, key_lo: np.uint32, buf: NDArray[np.uint64], st: NDArray[np.uint64]
    ) -> np.float64:
        # Draw the next buffered uniform for one replicate. ``buf`` is a length-4
        # uint64 scratch of the current permutation's four output words; ``st`` is a
        # length-2 uint64 holding [position, block_counter]. When the buffer is
        # spent (position == 4) the next permutation (st[1]) refills all four words
        # and the block counter advances. The realized draw index k thus maps to
        # permutation k // 4, word k % 4, so the whole stream stays a pure function
        # of (key, draw-index) and is bitwise thread-count invariant.
        pos = st[0]
        if pos == np.uint64(4):
            _philox_round4(key_hi, key_lo, st[1], buf)
            st[1] += np.uint64(1)
            pos = np.uint64(0)
        u = _u01_from_word(buf[pos])
        st[0] = pos + np.uint64(1)
        return u

    @numba.njit(inline="always", cache=True)
    def _quantile_linear(  # pragma: no cover - njit-compiled to machine code
        sorted_vals: NDArray[np.float64], n: int, q: float
    ) -> np.float64:
        # numpy's default "linear" quantile on an ascending sorted length-n buffer.
        # The fractional rank is h = q * (n - 1); the result interpolates between the
        # two neighbouring order statistics:
        #   lo = floor(h); result = sorted[lo] + (h - lo) * (sorted[lo+1] - sorted[lo]).
        # When h lands exactly on the last index (q == 1.0, or n == 1) there is no
        # upper neighbour, so the last order statistic is returned directly; this
        # reproduces np.quantile(..., method="linear") bit-for-bit at the endpoints.
        h = q * (n - 1)
        lo = int(np.floor(h))
        if lo >= n - 1:
            return sorted_vals[n - 1]
        frac = h - lo
        return sorted_vals[lo] + frac * (sorted_vals[lo + 1] - sorted_vals[lo])

    @numba.njit(inline="always", cache=True)
    def _reduce_one_column(  # pragma: no cover - njit-compiled to machine code
        data: NDArray[np.float64],
        idx: NDArray[np.int32],
        n: int,
        j: int,
        rcode: int,
        q: float,
        scratch: NDArray[np.float64],
    ) -> np.float64:
        # Reduce one resampled column to a scalar statistic, branching only on the
        # final reduction so every family shares the gather. ``idx`` carries this
        # replicate's resampled row positions; ``j`` selects the data column.
        #
        # rcode == 0 -> mean; rcode == 1 -> population variance (ddof=0);
        # rcode == 2 -> population std (ddof=0, sqrt of the variance);
        # rcode == 3 -> linear-interpolation quantile at ``q`` (numpy's default).
        #
        # ``q`` is used only by the quantile branch; mean/var/std pass a harmless 0.0.
        # ``scratch`` is a length-n caller-owned buffer reused only by the quantile
        # branch to sort the gathered column; the other reducers ignore it (so the
        # mean/var/std paths stay byte-identical and allocate nothing extra here).
        if rcode == 3:
            for t in range(n):
                scratch[t] = data[idx[t], j]
            scratch.sort()  # ascending in place; numba supports ndarray.sort()
            return _quantile_linear(scratch, n, q)
        # Variance uses the numerically stable two-pass form: first the mean over
        # the gathered values, then the mean of the squared deviations from that
        # mean. This avoids the catastrophic cancellation of the naive
        # E[x^2] - E[x]^2 identity. The gathered values are read straight from
        # ``data`` through ``idx`` (the same buffer the mean path streams over), so
        # the second pass costs one extra read per observation and no extra memory.
        acc = 0.0
        for t in range(n):
            acc += data[idx[t], j]
        mean = acc / n
        if rcode == 0:
            return np.float64(mean)
        sq = 0.0
        for t in range(n):
            dev = data[idx[t], j] - mean
            sq += dev * dev
        var = sq / n  # ddof=0 population variance, matching numpy's default
        if rcode == 1:
            return np.float64(var)
        return np.sqrt(var)

    @numba.njit(inline="always", cache=True)
    def _reduce_window_1d(  # pragma: no cover - njit-compiled to machine code
        col: NDArray[np.float64], n: int, rcode: int, q: float
    ) -> np.float64:
        # Reduce a contiguous length-n window already gathered into ``col`` to a
        # scalar statistic, branching only on the final reduction. This is the
        # generated-path tail shared by the recursive AR (d=1) and VAR (per column)
        # fused kernels, so the mean/var/std/quantile semantics live in one place.
        #
        # rcode == 0 -> mean; rcode == 1 -> population variance (ddof=0);
        # rcode == 2 -> population std (ddof=0); rcode == 3 -> linear-interpolation
        # quantile at ``q`` (numpy's default). The quantile branch sorts ``col`` in
        # place, so the caller must pass a per-replicate buffer it is free to mutate.
        if rcode == 3:
            col.sort()  # ascending in place; numba supports ndarray.sort()
            return _quantile_linear(col, n, q)
        # mean/var/std share the numerically stable two-pass form.
        acc = 0.0
        for t in range(n):
            acc += col[t]
        mean = acc / n
        if rcode == 0:
            return np.float64(mean)
        sq = 0.0
        for t in range(n):
            dev = col[t] - mean
            sq += dev * dev
        var = sq / n  # ddof=0 population variance, matching numpy's default
        if rcode == 1:
            return np.float64(var)
        return np.sqrt(var)

    # --- per-replicate index builders (the single source of index truth) -------
    # Each fills one replicate's (n,) index row from that replicate's Philox key.
    # Both the indices-only kernels and the fused values kernels call these, so
    # the carry/wrap/start logic lives exactly once and the two paths see the
    # SAME resample for the same seed.

    @numba.njit(inline="always", cache=True)
    def _new_draw_state() -> tuple[
        NDArray[np.uint64], NDArray[np.uint64]
    ]:  # pragma: no cover - njit-compiled
        # Per-replicate buffered-draw scratch: a length-4 uint64 word buffer and a
        # length-2 uint64 state [position, block_counter]. position starts at 4 so
        # the first _next_u01 forces a refill from block_counter 0.
        buf = np.empty(4, np.uint64)
        st = np.empty(2, np.uint64)
        st[0] = np.uint64(4)  # position: spent, refill on first draw
        st[1] = np.uint64(0)  # block_counter: first permutation
        return buf, st

    @numba.njit(inline="always", cache=True)
    def _fill_stationary_row(  # pragma: no cover - njit-compiled
        n: int, p: float, kh: np.uint32, kl: np.uint32, idx: NDArray[np.int32]
    ) -> None:
        # Geometric block lengths; a new block restarts at a fresh uniform point,
        # otherwise the previous index continues, wrapping at n. Draws are buffered
        # (four uniforms per Philox permutation) via _next_u01.
        buf, st = _new_draw_state()
        idx[0] = np.int32(_next_u01(kh, kl, buf, st) * n)
        for t in range(1, n):
            u = _next_u01(kh, kl, buf, st)
            if u < p:  # new block: fresh uniform restart point
                idx[t] = np.int32(_next_u01(kh, kl, buf, st) * n)
            else:  # continue block: previous + 1, wrapping at n
                nx = idx[t - 1] + 1
                idx[t] = 0 if nx == n else nx

    @numba.njit(inline="always", cache=True)
    def _fill_iid_row(  # pragma: no cover - njit-compiled
        n: int, kh: np.uint32, kl: np.uint32, idx: NDArray[np.int32]
    ) -> None:
        # n independent uniform positions in [0, n). Buffered draws via _next_u01.
        buf, st = _new_draw_state()
        for t in range(n):
            idx[t] = np.int32(_next_u01(kh, kl, buf, st) * n)

    @numba.njit(inline="always", cache=True)
    def _fill_block_row(  # pragma: no cover - njit-compiled
        n: int,
        length: int,
        span: np.int64,
        wrap: np.int64,
        kh: np.uint32,
        kl: np.uint32,
        idx: NDArray[np.int32],
    ) -> None:
        # ceil(n / length) fixed-length block starts, each a contiguous run, tiled
        # and trimmed to n. span/wrap select the family exactly as documented on
        # _block_reduce_kernel. Buffered draws via _next_u01.
        n_full = max(1, n // length)
        buf, st = _new_draw_state()
        t = 0
        while t < n:
            if span < 0:  # non-overlapping: start is a multiple of length
                start = np.int32(_next_u01(kh, kl, buf, st) * n_full) * length
            else:
                start = np.int32(_next_u01(kh, kl, buf, st) * span)
            for o in range(length):
                if t >= n:
                    break
                pos = start + o
                if wrap and pos >= n:  # circular: wrap past the end
                    pos -= n
                idx[t] = pos
                t += 1

    # --- per-replicate key derivation (the single source of Philox-key truth) --
    # Every replicate's two 32-bit Philox key words are derived IN-KERNEL from the
    # run's packed 128-bit root and the replicate index ``b`` alone, so no
    # per-replicate SeedSequence is spawned and no O(B) Python key loop runs before
    # dispatch. The map ``b -> key`` is a pure function of ``(root, b)`` (it reads
    # neither B, nor the thread count, nor the work-item order), so the fused
    # kernels stay bitwise thread-count invariant and prefix-stable in B.
    #
    # ``_REPLICATE_GOLDEN`` (replicate axis) MUST differ from ``_SERIES_GOLDEN`` (the
    # series axis, used by ``_hash_series_words``): the panel key is
    # ``_fold_in_key(_replicate_key(root, b), s)``, and if both axes shared one golden the
    # effective key would be symmetric in (b, s) at an all-zero root, so replicate b /
    # series s and replicate s / series b would alias one stream. A distinct odd
    # multiplier for the replicate axis plus the second-round ``root_b`` XOR break that
    # symmetry structurally at any root value. Both goldens are single-sourced (with this
    # invariant) in :mod:`tsbootstrap.prng_keys`; the kernels recompute the same math.
    _REPLICATE_GOLDEN: Final = np.uint64(_REPLICATE_GOLDEN_INT)
    _SERIES_GOLDEN: Final = np.uint64(_SERIES_GOLDEN_INT)

    @numba.njit(inline="always", cache=True)
    def _replicate_key(  # pragma: no cover - njit-compiled to machine code
        root_a: np.uint64, root_b: np.uint64, b: int
    ) -> tuple[np.uint32, np.uint32]:
        # Two-round SplitMix64 finalizer over root_a + (b + 1) * golden, folding the
        # upper 64 root bits (root_b) in between the rounds. ``(b + 1) * golden`` is a
        # bijection mod 2**64 for the odd golden and the finalizer rounds and the
        # root_b XOR are bijections on uint64, so ``b -> key`` is injective for a fixed
        # root: single-series replicate keys are EXACT-DISTINCT. Every literal is
        # np.uint64 so the arithmetic wraps mod 2**64 exactly as _hash_series_words'.
        z = root_a + (np.uint64(b) + np.uint64(1)) * _REPLICATE_GOLDEN
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        z = z ^ root_b
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return np.uint32(z >> _SH32), np.uint32(z & _MASK32)

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _stationary_indices_kernel(  # pragma: no cover - njit-compiled to machine code
        n: int,
        p: float,
        root_a: np.uint64,
        root_b: np.uint64,
        out_idx: NDArray[np.int32],
    ) -> None:
        # Build the (B, n) stationary index matrix. The geometric carry is
        # strictly sequential within a replicate; replicates run in parallel and
        # write disjoint rows, so prange over b is data-race-free and
        # order-independent (hence thread-count-invariant).
        B = out_idx.shape[0]
        for b in numba.prange(B):
            kh, kl = _replicate_key(root_a, root_b, b)
            _fill_stationary_row(n, p, kh, kl, out_idx[b])

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _stationary_reduce_kernel(  # pragma: no cover - njit-compiled to machine code
        data: NDArray[np.float64],
        p: float,
        root_a: np.uint64,
        root_b: np.uint64,
        rcode: int,
        q: float,
        out: NDArray[np.float64],
    ) -> None:
        # Fused build -> gather -> column reduce (mean/var/std/quantile by rcode),
        # never materialising the full (B, n, d) sample. Same RNG/carry as
        # _stationary_indices_kernel; the reduction shares _reduce_one_column. ``q``
        # is used only by the quantile reducer; the others ignore it.
        B = out.shape[0]
        n = data.shape[0]
        d = data.shape[1]
        for b in numba.prange(B):
            idx = np.empty(n, np.int32)
            scratch = np.empty(n, np.float64)  # quantile sort buffer (per replicate)
            kh, kl = _replicate_key(root_a, root_b, b)
            _fill_stationary_row(n, p, kh, kl, idx)
            for j in range(d):
                out[b, j] = _reduce_one_column(data, idx, n, j, rcode, q, scratch)

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _iid_reduce_kernel(  # pragma: no cover - njit-compiled
        data: NDArray[np.float64],
        root_a: np.uint64,
        root_b: np.uint64,
        rcode: int,
        q: float,
        out: NDArray[np.float64],
    ) -> None:
        # Fused IID build -> gather -> column reduce (mean/var/std/quantile by rcode).
        # Each of the n positions is an independent uniform draw in [0, n), built by
        # the shared _fill_iid_row so the reduce path sees the same resample as the
        # indices/values paths. Replicates write disjoint rows under prange, so the
        # result is order- and thread-count-invariant. ``q`` is used only by the
        # quantile reducer.
        B = out.shape[0]
        n = data.shape[0]
        d = data.shape[1]
        for b in numba.prange(B):
            idx = np.empty(n, np.int32)
            scratch = np.empty(n, np.float64)  # quantile sort buffer (per replicate)
            kh, kl = _replicate_key(root_a, root_b, b)
            _fill_iid_row(n, kh, kl, idx)
            for j in range(d):
                out[b, j] = _reduce_one_column(data, idx, n, j, rcode, q, scratch)

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _iid_indices_kernel(  # pragma: no cover - njit-compiled
        n: int, root_a: np.uint64, root_b: np.uint64, out_idx: NDArray[np.int32]
    ) -> None:
        B = out_idx.shape[0]
        for b in numba.prange(B):
            kh, kl = _replicate_key(root_a, root_b, b)
            _fill_iid_row(n, kh, kl, out_idx[b])

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _block_reduce_kernel(  # pragma: no cover - njit-compiled to machine code
        data: NDArray[np.float64],
        length: int,
        span: np.int64,
        wrap: np.int64,
        root_a: np.uint64,
        root_b: np.uint64,
        rcode: int,
        q: float,
        out: NDArray[np.float64],
    ) -> None:
        # Fused fixed-length-block build -> gather -> column reduce (mean/var/std/
        # quantile by rcode) for the moving, circular, and non-overlapping families.
        # The block row is built by the shared _fill_block_row (the single source of
        # index truth), then reduced column-by-column via _reduce_one_column. ``q`` is
        # used only by the quantile reducer.
        #   span == n - length + 1, wrap == 0 -> moving block (start in [0, span))
        #   span == n,             wrap == 1 -> circular block (start in [0, n), mod n)
        #   span <  0 (sentinel)            -> non-overlapping (start = k*length, k in [0, n_full))
        B = out.shape[0]
        n = data.shape[0]
        d = data.shape[1]
        for b in numba.prange(B):
            idx = np.empty(n, np.int32)
            scratch = np.empty(n, np.float64)  # quantile sort buffer (per replicate)
            kh, kl = _replicate_key(root_a, root_b, b)
            _fill_block_row(n, length, span, wrap, kh, kl, idx)
            for j in range(d):
                out[b, j] = _reduce_one_column(data, idx, n, j, rcode, q, scratch)

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _block_indices_kernel(  # pragma: no cover
        n: int,
        length: int,
        span: np.int64,
        wrap: np.int64,
        root_a: np.uint64,
        root_b: np.uint64,
        out_idx: NDArray[np.int32],
    ) -> None:
        B = out_idx.shape[0]
        for b in numba.prange(B):
            kh, kl = _replicate_key(root_a, root_b, b)
            _fill_block_row(n, length, span, wrap, kh, kl, out_idx[b])

    # --- fused values kernels: write BOTH the gathered (B, n, d) values AND the
    # (B, n) indices in one prange-parallel pass. The index row is built by the
    # SAME _fill_*_row device function the reduce/indices paths use, so for a
    # given seed compiled_values and compiled_reduce see an identical resample.

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _stationary_values_kernel(  # pragma: no cover - njit-compiled to machine code
        data: NDArray[np.float64],
        p: float,
        root_a: np.uint64,
        root_b: np.uint64,
        out_val: NDArray[np.float64],
        out_idx: NDArray[np.int32],
    ) -> None:
        B = out_idx.shape[0]
        n = data.shape[0]
        d = data.shape[1]
        for b in numba.prange(B):
            kh, kl = _replicate_key(root_a, root_b, b)
            _fill_stationary_row(n, p, kh, kl, out_idx[b])
            for t in range(n):
                pos = out_idx[b, t]
                for j in range(d):
                    out_val[b, t, j] = data[pos, j]

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _iid_values_kernel(  # pragma: no cover - njit
        data: NDArray[np.float64],
        root_a: np.uint64,
        root_b: np.uint64,
        out_val: NDArray[np.float64],
        out_idx: NDArray[np.int32],
    ) -> None:
        B = out_idx.shape[0]
        n = data.shape[0]
        d = data.shape[1]
        for b in numba.prange(B):
            kh, kl = _replicate_key(root_a, root_b, b)
            _fill_iid_row(n, kh, kl, out_idx[b])
            for t in range(n):
                pos = out_idx[b, t]
                for j in range(d):
                    out_val[b, t, j] = data[pos, j]

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _block_values_kernel(  # pragma: no cover - njit-compiled to machine code
        data: NDArray[np.float64],
        length: int,
        span: np.int64,
        wrap: np.int64,
        root_a: np.uint64,
        root_b: np.uint64,
        out_val: NDArray[np.float64],
        out_idx: NDArray[np.int32],
    ) -> None:
        B = out_idx.shape[0]
        n = data.shape[0]
        d = data.shape[1]
        for b in numba.prange(B):
            kh, kl = _replicate_key(root_a, root_b, b)
            _fill_block_row(n, length, span, wrap, kh, kl, out_idx[b])
            for t in range(n):
                pos = out_idx[b, t]
                for j in range(d):
                    out_val[b, t, j] = data[pos, j]

    # --- ragged-panel fused reduce (the num_series x B observation-method moat) -
    # One parallel pass conformalises a whole panel of unequal-length (ragged)
    # series: a flat (total_N, d) data array plus a CSR ``indptr`` of length
    # num_series+1 (indptr[s]..indptr[s+1] is series s). The prange runs over
    # B * num_series flattened work items (item t -> replicate b = t // num_series,
    # series s = t % num_series); each item bootstraps ONLY its own series slice and
    # writes one (b, s, :) row of the dense (B, num_series, d) output. No NaN
    # padding (it would destroy the memory moat) and no Python loop over series (the
    # interpreter dispatch the arch/statsmodels per-series loop pays is collapsed
    # into the single prange).

    # The SplitMix64 finalizer applied to slot 0 (precomputed). Subtracting (XOR-ing)
    # it from every slot's avalanche makes slot 0 fold in ZERO, so a num_series==1
    # panel (whose only series sits at slot 0) reproduces the existing single-series
    # reduce bitwise, while slots s>0 still get distinct, well-avalanched key deltas.
    _SPLITMIX_S0_HI: Final = np.uint32(0xE220A839)
    _SPLITMIX_S0_LO: Final = np.uint32(0x7B1DCDAF)

    @numba.njit(inline="always", cache=True)
    def _hash_series_words(
        s: int,
    ) -> tuple[np.uint32, np.uint32]:  # pragma: no cover - njit-compiled to machine code
        # Avalanche the (panel-slot) series index s into two 32-bit words via a
        # SplitMix64 finalizer, then XOR out the slot-0 avalanche so slot 0 -> (0, 0).
        # The raw finalizer is a bijection on uint64, and XOR-ing a constant preserves
        # injectivity, so distinct s still map to distinct (hi, lo) pairs: folding both
        # words into the per-replicate Philox key (below) gives every (b, s) a DISTINCT
        # key, so no two series in a replicate ever share a stream (no 32-bit birthday
        # risk), and slot 0 maps to the identity fold.
        z = (np.uint64(s) + np.uint64(1)) * _SERIES_GOLDEN
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        hi = np.uint32(z >> _SH32) ^ _SPLITMIX_S0_HI
        lo = np.uint32(z & _MASK32) ^ _SPLITMIX_S0_LO
        return hi, lo

    @numba.njit(inline="always", cache=True)
    def _fold_in_key(  # pragma: no cover - njit-compiled
        key_hi: np.uint32, key_lo: np.uint32, s: int
    ) -> tuple[np.uint32, np.uint32]:
        # Effective per-(replicate, series) Philox key = the replicate key XOR the
        # avalanched series-slot words. A pure function of (key, s): it depends only
        # on the replicate key and the series slot, never on num_series or the
        # work-item index, which is exactly what makes "series s inside a panel ==
        # series s standalone with the same key and slot s" hold bitwise. Folding
        # into the KEY (not the counter block) gives each series a fully distinct
        # Philox stream regardless of how many draws each series consumes; a
        # counter-block offset would instead need a per-series stride sized larger
        # than any series's draw count to avoid stream overlap, so the key fold is
        # the unconditionally collision-free choice.
        h_hi, h_lo = _hash_series_words(s)
        return key_hi ^ h_hi, key_lo ^ h_lo

    @numba.njit(inline="always", cache=True)
    def _reduce_one_column_csr(  # pragma: no cover - njit-compiled to machine code
        data: NDArray[np.float64],
        idx: NDArray[np.int32],
        lo: int,
        n: int,
        j: int,
        rcode: int,
        q: float,
        scratch: NDArray[np.float64],
    ) -> np.float64:
        # The CSR-offset analogue of _reduce_one_column: ``idx`` carries LOCAL
        # positions in [0, n) for one series and ``lo`` (= indptr[s]) maps each to a
        # global row of the flat panel data. Identical reducer semantics
        # (mean/var/std/quantile by rcode) so a num_series==1 panel matches the
        # single-series reduce, and the two-pass variance stays byte-identical.
        if rcode == 3:
            for t in range(n):
                scratch[t] = data[lo + idx[t], j]
            scratch.sort()
            return _quantile_linear(scratch, n, q)
        acc = 0.0
        for t in range(n):
            acc += data[lo + idx[t], j]
        mean = acc / n
        if rcode == 0:
            return np.float64(mean)
        sq = 0.0
        for t in range(n):
            dev = data[lo + idx[t], j] - mean
            sq += dev * dev
        var = sq / n  # ddof=0 population variance, matching numpy's default
        if rcode == 1:
            return np.float64(var)
        return np.sqrt(var)

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _iid_panel_reduce_kernel(  # pragma: no cover - njit-compiled to machine code
        flat_data: NDArray[np.float64],
        indptr: NDArray[np.int64],
        num_series: int,
        root_a: np.uint64,
        root_b: np.uint64,
        rcode: int,
        q: float,
        out: NDArray[np.float64],
    ) -> None:
        # Fused ragged-panel IID build -> CSR gather -> column reduce. prange over
        # B * num_series; each item reads only its own series slice via indptr and
        # writes out[b, s, :]. Distinct (b, s) rows are disjoint, so the result is
        # order- and thread-count-invariant. _replicate_key is recomputed per (b, s) rather
        # than hoisted per b: it is a few scalar ops, dwarfed by the per-item O(n) draw and
        # reduce below, and the flat prange keeps long and short series interleaved across
        # threads (a per-b hoist would need a separate pre-pass and lose that balance).
        B = out.shape[0]
        d = flat_data.shape[1]
        total_items = B * num_series
        for t in numba.prange(total_items):
            b = t // num_series
            s = t % num_series
            lo = indptr[s]
            n = indptr[s + 1] - lo
            rk_hi, rk_lo = _replicate_key(root_a, root_b, b)
            kh, kl = _fold_in_key(rk_hi, rk_lo, s)
            idx = np.empty(n, np.int32)
            scratch = np.empty(n, np.float64)
            _fill_iid_row(n, kh, kl, idx)
            for j in range(d):
                out[b, s, j] = _reduce_one_column_csr(flat_data, idx, lo, n, j, rcode, q, scratch)

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _stationary_panel_reduce_kernel(  # pragma: no cover - njit-compiled to machine code
        flat_data: NDArray[np.float64],
        indptr: NDArray[np.int64],
        num_series: int,
        p: float,
        root_a: np.uint64,
        root_b: np.uint64,
        rcode: int,
        q: float,
        out: NDArray[np.float64],
    ) -> None:
        # Ragged-panel stationary analogue of _iid_panel_reduce_kernel. ``p`` is the
        # geometric restart probability (a single panel-wide value); the per-series
        # length n drives every local index draw.
        B = out.shape[0]
        d = flat_data.shape[1]
        total_items = B * num_series
        for t in numba.prange(total_items):
            b = t // num_series
            s = t % num_series
            lo = indptr[s]
            n = indptr[s + 1] - lo
            rk_hi, rk_lo = _replicate_key(root_a, root_b, b)
            kh, kl = _fold_in_key(rk_hi, rk_lo, s)
            idx = np.empty(n, np.int32)
            scratch = np.empty(n, np.float64)
            _fill_stationary_row(n, p, kh, kl, idx)
            for j in range(d):
                out[b, s, j] = _reduce_one_column_csr(flat_data, idx, lo, n, j, rcode, q, scratch)

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _block_panel_reduce_kernel(  # pragma: no cover - njit-compiled to machine code
        flat_data: NDArray[np.float64],
        indptr: NDArray[np.int64],
        num_series: int,
        length: int,
        span_mode: np.int64,
        wrap: np.int64,
        root_a: np.uint64,
        root_b: np.uint64,
        rcode: int,
        q: float,
        out: NDArray[np.float64],
    ) -> None:
        # Ragged-panel fixed-length-block analogue (moving / circular /
        # non-overlapping). ``span`` is per-series (it depends on n), so it is
        # recomputed inside the loop from ``span_mode``:
        #   span_mode == 0 -> moving      (span = n - length + 1, wrap = 0)
        #   span_mode == 1 -> circular    (span = n,             wrap = 1)
        #   span_mode == 2 -> non-overlapping (span = -1 sentinel, wrap = 0)
        # The block length is clamped per series to [1, n] so a series shorter than
        # the panel block length still draws a valid (full-coverage) block.
        B = out.shape[0]
        d = flat_data.shape[1]
        total_items = B * num_series
        for t in numba.prange(total_items):
            b = t // num_series
            s = t % num_series
            lo = indptr[s]
            n = indptr[s + 1] - lo
            length_s = length if length < n else n
            if span_mode == 0:
                span = n - length_s + 1
            elif span_mode == 1:
                span = n
            else:
                span = np.int64(-1)
            rk_hi, rk_lo = _replicate_key(root_a, root_b, b)
            kh, kl = _fold_in_key(rk_hi, rk_lo, s)
            idx = np.empty(n, np.int32)
            scratch = np.empty(n, np.float64)
            _fill_block_row(n, length_s, span, wrap, kh, kl, idx)
            for j in range(d):
                out[b, s, j] = _reduce_one_column_csr(flat_data, idx, lo, n, j, rcode, q, scratch)

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _iid_panel_indices_kernel(  # pragma: no cover - njit-compiled
        indptr: NDArray[np.int64],
        num_series: int,
        root_a: np.uint64,
        root_b: np.uint64,
        out_flat: NDArray[np.int32],
    ) -> None:
        # Build each item's LOCAL resample positions (in [0, n_s)) into a flat
        # (B, total_N) array with the same CSR layout on axis 1 (series s occupies
        # columns indptr[s]:indptr[s+1]). For raggedness / standalone-equality
        # validation; uses the same fold_in and fill device functions the reduce
        # kernel uses, so the resample is identical.
        B = out_flat.shape[0]
        total_items = B * num_series
        for t in numba.prange(total_items):
            b = t // num_series
            s = t % num_series
            lo = indptr[s]
            n = indptr[s + 1] - lo
            rk_hi, rk_lo = _replicate_key(root_a, root_b, b)
            kh, kl = _fold_in_key(rk_hi, rk_lo, s)
            _fill_iid_row(n, kh, kl, out_flat[b, lo : lo + n])

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _stationary_panel_indices_kernel(  # pragma: no cover - njit-compiled
        indptr: NDArray[np.int64],
        num_series: int,
        p: float,
        root_a: np.uint64,
        root_b: np.uint64,
        out_flat: NDArray[np.int32],
    ) -> None:
        B = out_flat.shape[0]
        total_items = B * num_series
        for t in numba.prange(total_items):
            b = t // num_series
            s = t % num_series
            lo = indptr[s]
            n = indptr[s + 1] - lo
            rk_hi, rk_lo = _replicate_key(root_a, root_b, b)
            kh, kl = _fold_in_key(rk_hi, rk_lo, s)
            _fill_stationary_row(n, p, kh, kl, out_flat[b, lo : lo + n])

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _block_panel_indices_kernel(  # pragma: no cover - njit-compiled
        indptr: NDArray[np.int64],
        num_series: int,
        length: int,
        span_mode: np.int64,
        wrap: np.int64,
        root_a: np.uint64,
        root_b: np.uint64,
        out_flat: NDArray[np.int32],
    ) -> None:
        B = out_flat.shape[0]
        total_items = B * num_series
        for t in numba.prange(total_items):
            b = t // num_series
            s = t % num_series
            lo = indptr[s]
            n = indptr[s + 1] - lo
            length_s = length if length < n else n
            if span_mode == 0:
                span = n - length_s + 1
            elif span_mode == 1:
                span = n
            else:
                span = np.int64(-1)
            rk_hi, rk_lo = _replicate_key(root_a, root_b, b)
            kh, kl = _fold_in_key(rk_hi, rk_lo, s)
            _fill_block_row(n, length_s, span, wrap, kh, kl, out_flat[b, lo : lo + n])

    # --- recursive residual AR fused reduce ------------------------------------
    # ResidualBootstrap with an AR model (and the order-selected SieveAR share the
    # same fitted context, but only ResidualBootstrap(AR) is wired). Each replicate
    # resamples the centered residuals IID via its own Philox stream, runs the AR
    # recurrence in-kernel from the fitted coefficients + intercept + initial
    # values, and reduces the generated path with the shared reducer, never
    # materialising the (B, n) resampled-residual array or the (B, n) path. The
    # numpy path (recursive._ar_batched) draws n + burn_in - p IID innovations,
    # seeds the recurrence from the initial block, simulates, then trims the first
    # burn_in steps; this kernel reproduces that construction in distribution.

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _ar_residual_reduce_kernel(  # pragma: no cover - njit-compiled to machine code
        eps: NDArray[np.float64],
        ar_coefs: NDArray[np.float64],
        intercept: float,
        series: NDArray[np.float64],
        p: int,
        burn_in: int,
        n: int,
        fixed: bool,
        root_a: np.uint64,
        root_b: np.uint64,
        rcode: int,
        q: float,
        out: NDArray[np.float64],
    ) -> None:
        # eps: (n_resid,) centered residuals; ar_coefs: (p,); series: (n_series,)
        # observed series (for the initial block); out: (B, 1).
        #
        # The recurrence keeps the last p generated values in a small rolling
        # window ``hist`` (hist[0] is the most recent). The full path has length
        # p + n_gen where n_gen = n + burn_in - p generated steps; the kept window
        # is path[burn_in : burn_in + n]. To honour burn_in without a length-n+burn
        # buffer, a length-n scratch holds exactly the kept window for the var/std
        # two-pass reduce; the mean is streamed (no scratch read-back needed but the
        # scratch is filled uniformly so all reducers share one path-generation loop).
        B = out.shape[0]
        n_resid = eps.shape[0]
        n_series = series.shape[0]
        n_gen = n + burn_in - p
        for b in numba.prange(B):
            kh, kl = _replicate_key(root_a, root_b, b)
            buf, st = _new_draw_state()
            hist = np.empty(p, np.float64)
            scratch = np.empty(n, np.float64)
            # initial block: fixed -> series[:p]; random_block -> a uniform start in
            # [0, n_series - p]. The numpy path draws this start from the same
            # per-replicate generator before the innovations; here it is the first
            # buffered Philox draw of the replicate (distinct stream, equal in
            # distribution).
            # The ternary short-circuits, so a Philox draw is consumed only on the
            # random-block path; the fixed path consumes none, matching the numpy stream.
            start = 0 if fixed else np.int32(_next_u01(kh, kl, buf, st) * (n_series - p + 1))
            for j in range(p):
                hist[p - 1 - j] = series[start + j]  # hist[0] = most recent init value
            # The first p path values are the initial block; they land in the kept
            # window only when burn_in < p. Fill the kept-window scratch as we go.
            kept = 0
            # emit the p initial values at global path positions 0..p-1
            for j in range(p):
                gpos = j  # global path position of this initial value
                if burn_in <= gpos < burn_in + n:
                    scratch[kept] = series[start + j]
                    kept += 1
            # generate n_gen steps at global positions p .. p + n_gen - 1
            for s in range(n_gen):
                u = _next_u01(kh, kl, buf, st)
                ridx = np.int32(u * n_resid)
                val = intercept + eps[ridx]
                for j in range(p):
                    val += ar_coefs[j] * hist[j]
                for j in range(p - 1, 0, -1):
                    hist[j] = hist[j - 1]
                hist[0] = val
                gpos = p + s
                if burn_in <= gpos < burn_in + n:
                    scratch[kept] = val
                    kept += 1
            # reduce the kept window (length n) through the shared 1-D window
            # reducer (the quantile branch sorts the per-replicate scratch in place).
            out[b, 0] = _reduce_window_1d(scratch, n, rcode, q)

    # --- recursive residual VAR fused reduce -----------------------------------
    # ResidualBootstrap with a VAR model. This is the multivariate analogue of the
    # AR fused reduce: each replicate resamples the centered vector residual rows
    # IID via its own Philox stream, runs the fitted VAR recurrence in-kernel from
    # the coefficient matrices + intercept vector + initial block, and reduces each
    # of the d generated columns with the shared 1-D window reducer, never
    # materialising the (B, n, d) path. The numpy path (recursive._var_batched ->
    # engines.var.simulate_var_batched) draws n + burn_in - p IID residual ROWS,
    # seeds the recurrence from the initial block, simulates, then trims the first
    # burn_in steps; this kernel reproduces that construction in distribution.

    @numba.njit(parallel=True, fastmath=False, cache=True)
    def _var_residual_reduce_kernel(  # pragma: no cover - njit-compiled to machine code
        coefs: NDArray[np.float64],
        intercept: NDArray[np.float64],
        eps: NDArray[np.float64],
        series: NDArray[np.float64],
        p: int,
        burn_in: int,
        n: int,
        fixed: bool,
        root_a: np.uint64,
        root_b: np.uint64,
        rcode: int,
        q: float,
        out: NDArray[np.float64],
    ) -> None:
        # coefs: (p, d, d) lag matrices; intercept: (d,); eps: (n_resid, d) centered
        # vector residuals; series: (n_series, d) observed series (for the initial
        # block); out: (B, d).
        #
        # The recurrence keeps the last p generated vector observations in a rolling
        # window ``hist`` (p, d) with hist[0] the most recent. The kept window is the
        # global path positions burn_in .. burn_in + n - 1; a per-replicate (n, d)
        # ``scratch`` holds exactly that window so every column can be reduced.
        #
        # The matvec uses the EXPLICIT triple loop (acc[i] += coefs[j,i,k]*hist[j,k]),
        # NOT ``@``: at small d the @ form allocates a temporary d-vector per lag per
        # step and that allocation dominates the arithmetic (measured ~3.3x slower in
        # the design spike). This mirrors engines.var._var_recurrence_numba's form and
        # uses coefs[j] directly (fit_var already baked the transpose into coefs, so
        # the engine computes coefs[j] @ x, not coefs[j].T @ x).
        B = out.shape[0]
        n_resid = eps.shape[0]
        n_series = series.shape[0]
        d = coefs.shape[1]
        n_gen = n + burn_in - p
        for b in numba.prange(B):
            kh, kl = _replicate_key(root_a, root_b, b)
            buf, st = _new_draw_state()
            hist = np.empty((p, d), np.float64)
            scratch = np.empty((n, d), np.float64)
            col = np.empty(n, np.float64)  # contiguous per-column buffer for the reduce
            val = np.empty(d, np.float64)  # per-step generated vector
            # initial block: fixed -> series[:p]; random_block -> a uniform start in
            # [0, n_series - p]. The numpy path draws this start before the
            # innovations; here it is the first buffered Philox draw of the replicate
            # (distinct stream, equal in distribution). The ternary short-circuits, so
            # a draw is consumed only on the random-block path, matching the AR kernel.
            start = 0 if fixed else np.int32(_next_u01(kh, kl, buf, st) * (n_series - p + 1))
            for j in range(p):
                for i in range(d):
                    hist[p - 1 - j, i] = series[start + j, i]  # hist[0] = most recent init
            # The first p path values are the initial block; fill the kept-window
            # scratch for any that fall inside burn_in .. burn_in + n - 1.
            kept = 0
            for j in range(p):
                gpos = j  # global path position of this initial value
                if burn_in <= gpos < burn_in + n:
                    for i in range(d):
                        scratch[kept, i] = series[start + j, i]
                    kept += 1
            # generate n_gen steps at global positions p .. p + n_gen - 1
            for s in range(n_gen):
                u = _next_u01(kh, kl, buf, st)
                ridx = np.int32(u * n_resid)
                for i in range(d):
                    acc = intercept[i] + eps[ridx, i]
                    for j in range(p):
                        for k in range(d):
                            acc += coefs[j, i, k] * hist[j, k]
                    val[i] = acc
                # roll the window: shift older lags back, drop the new value in front
                for j in range(p - 1, 0, -1):
                    for i in range(d):
                        hist[j, i] = hist[j - 1, i]
                for i in range(d):
                    hist[0, i] = val[i]
                gpos = p + s
                if burn_in <= gpos < burn_in + n:
                    for i in range(d):
                        scratch[kept, i] = val[i]
                    kept += 1
            # reduce each generated column through the shared 1-D window reducer.
            for c in range(d):
                for t in range(n):
                    col[t] = scratch[t, c]
                out[b, c] = _reduce_window_1d(col, n, rcode, q)

    _HAVE_NUMBA = True

    def _warm_compiled_kernels() -> None:
        """Trigger one-time JIT compilation off the hot path (registered warm-up)."""
        data = np.zeros((2, 1), dtype=np.float64)
        root_a = np.uint64(0)
        root_b = np.uint64(0)
        out = np.empty((1, 1), dtype=np.float64)
        out_idx = np.empty((1, 2), dtype=np.int32)
        out_val = np.empty((1, 2, 1), dtype=np.float64)
        # Warm every reducer code so var/std/quantile are compiled off the hot path.
        for rc in (_RCODE_MEAN, _RCODE_VAR, _RCODE_STD, _RCODE_QUANTILE):
            _stationary_reduce_kernel(data, 0.5, root_a, root_b, rc, 0.5, out)
            _iid_reduce_kernel(data, root_a, root_b, rc, 0.5, out)
            # Moving (wrap=0), circular (wrap=1), and non-overlapping (span<0).
            _block_reduce_kernel(data, 1, np.int64(2), np.int64(0), root_a, root_b, rc, 0.5, out)
            _block_reduce_kernel(data, 1, np.int64(2), np.int64(1), root_a, root_b, rc, 0.5, out)
            _block_reduce_kernel(data, 1, np.int64(-1), np.int64(0), root_a, root_b, rc, 0.5, out)
        _stationary_indices_kernel(2, 0.5, root_a, root_b, out_idx)
        _stationary_values_kernel(data, 0.5, root_a, root_b, out_val, out_idx)
        _iid_indices_kernel(2, root_a, root_b, out_idx)
        _iid_values_kernel(data, root_a, root_b, out_val, out_idx)
        _block_indices_kernel(2, 1, np.int64(2), np.int64(0), root_a, root_b, out_idx)
        _block_values_kernel(data, 1, np.int64(2), np.int64(0), root_a, root_b, out_val, out_idx)
        _block_indices_kernel(2, 1, np.int64(2), np.int64(1), root_a, root_b, out_idx)
        _block_values_kernel(data, 1, np.int64(2), np.int64(1), root_a, root_b, out_val, out_idx)
        _block_indices_kernel(2, 1, np.int64(-1), np.int64(0), root_a, root_b, out_idx)
        _block_values_kernel(data, 1, np.int64(-1), np.int64(0), root_a, root_b, out_val, out_idx)
        # Ragged-panel fused reduce: warm every reducer code and every family off
        # the hot path with a two-series toy panel (indptr [0, 2, 4], num_series 2).
        panel_flat = np.zeros((4, 1), dtype=np.float64)
        panel_indptr = np.array([0, 2, 4], dtype=np.int64)
        panel_out = np.empty((1, 2, 1), dtype=np.float64)
        panel_idx = np.empty((1, 4), dtype=np.int32)
        for rc in (_RCODE_MEAN, _RCODE_VAR, _RCODE_STD, _RCODE_QUANTILE):
            _iid_panel_reduce_kernel(
                panel_flat, panel_indptr, 2, root_a, root_b, rc, 0.5, panel_out
            )
            _stationary_panel_reduce_kernel(
                panel_flat, panel_indptr, 2, 0.5, root_a, root_b, rc, 0.5, panel_out
            )
            for sm in (np.int64(0), np.int64(1), np.int64(2)):
                _block_panel_reduce_kernel(
                    panel_flat,
                    panel_indptr,
                    2,
                    1,
                    sm,
                    np.int64(1 if sm == 1 else 0),
                    root_a,
                    root_b,
                    rc,
                    0.5,
                    panel_out,
                )
        _iid_panel_indices_kernel(panel_indptr, 2, root_a, root_b, panel_idx)
        _stationary_panel_indices_kernel(panel_indptr, 2, 0.5, root_a, root_b, panel_idx)
        for sm in (np.int64(0), np.int64(1), np.int64(2)):
            _block_panel_indices_kernel(
                panel_indptr, 2, 1, sm, np.int64(1 if sm == 1 else 0), root_a, root_b, panel_idx
            )
        # Recursive residual AR fused reduce: warm every reducer code and both the
        # fixed and random-block initial-state branches off the hot path.
        eps_w = np.zeros(2, dtype=np.float64)
        ar_w = np.zeros(1, dtype=np.float64)
        series_w = np.zeros(2, dtype=np.float64)
        out_ar = np.empty((1, 1), dtype=np.float64)
        for rc in (_RCODE_MEAN, _RCODE_VAR, _RCODE_STD, _RCODE_QUANTILE):
            _ar_residual_reduce_kernel(
                eps_w, ar_w, 0.0, series_w, 1, 0, 2, True, root_a, root_b, rc, 0.5, out_ar
            )
            _ar_residual_reduce_kernel(
                eps_w, ar_w, 0.0, series_w, 1, 0, 2, False, root_a, root_b, rc, 0.5, out_ar
            )
        # Recursive residual VAR fused reduce: warm every reducer code and both the
        # fixed and random-block initial-state branches off the hot path with a small
        # d>=2 toy (p=1, d=2, n=2): one lag matrix, an intercept vector, two residual
        # rows, and a two-row series for the initial block.
        coefs_w = np.zeros((1, 2, 2), dtype=np.float64)
        intercept_w = np.zeros(2, dtype=np.float64)
        eps_v = np.zeros((2, 2), dtype=np.float64)
        series_v = np.zeros((2, 2), dtype=np.float64)
        out_var = np.empty((1, 2), dtype=np.float64)
        for rc in (_RCODE_MEAN, _RCODE_VAR, _RCODE_STD, _RCODE_QUANTILE):
            _var_residual_reduce_kernel(
                coefs_w,
                intercept_w,
                eps_v,
                series_v,
                1,
                0,
                2,
                True,
                root_a,
                root_b,
                rc,
                0.5,
                out_var,
            )
            _var_residual_reduce_kernel(
                coefs_w,
                intercept_w,
                eps_v,
                series_v,
                1,
                0,
                2,
                False,
                root_a,
                root_b,
                rc,
                0.5,
                out_var,
            )

    register_warmup(_warm_compiled_kernels)
    # Retained name for callers that warmed the stationary kernels by this name.
    _warm_stationary_kernel = _warm_compiled_kernels
except ImportError:  # pragma: no cover - exercised only without the accel extra
    _HAVE_NUMBA = False


def _root_words(root_key: tuple[int, int]) -> tuple[np.uint64, np.uint64]:
    """Cast the packed 128-bit run root to the two ``uint64`` scalars the kernels take.

    See :func:`tsbootstrap.api._root_key_from` for how the root is packed and
    :func:`_replicate_key` for how each kernel derives its per-replicate Philox key.
    """
    return np.uint64(root_key[0]), np.uint64(root_key[1])


def _require_numba() -> None:
    """Raise a typed, actionable error when the optional numba accelerator is absent.

    The compiled fast paths are an explicit opt-in, so a missing numba fails loudly here
    rather than silently falling back to the default stream.
    """
    if not _HAVE_NUMBA:
        raise MethodConfigError(
            "The compiled fast path requires the optional 'numba' dependency, "
            "which is not installed.",
            code=Codes.BACKEND_NOT_INSTALLED,
            hint="Install the accelerator extra: pip install 'tsbootstrap[accel]'.",
        )


def _validate_common(
    data: NDArray[np.floating],
    sim_dtype: np.dtype[np.floating],
    reducer: str,
) -> tuple[NDArray[np.float64], int]:
    """Validate data, dtype, and reducer at the Python boundary.

    Returns the contiguous float64 data (1-D promoted to a single column) and the
    observation count. Raises clear, typed errors so the numba kernels never see
    malformed input and cannot emit a cryptic typing error. Block-length
    validation is method-specific and handled by the callers.
    """
    _require_numba()
    if reducer not in _SUPPORTED_REDUCERS:
        raise MethodConfigError(
            f"Unsupported reducer {reducer!r} for the compiled fast path.",
            code=Codes.INVALID_PARAMETER,
            context={"reducer": reducer, "supported": sorted(_SUPPORTED_REDUCERS)},
            hint=f"Use one of: {sorted(_SUPPORTED_REDUCERS)}.",
        )

    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise MethodConfigError(
            f"data must be 1-D or 2-D, got {arr.ndim} dimensions.",
            code=Codes.INVALID_SHAPE,
            context={"ndim": arr.ndim, "shape": tuple(arr.shape)},
        )
    n_obs = arr.shape[0]
    if n_obs < 1:
        raise MethodConfigError(
            "data must contain at least one observation.",
            code=Codes.TOO_FEW_OBSERVATIONS,
            context={"n_obs": n_obs},
        )

    # Kernel arithmetic runs in float64; the caller's sim_dtype is honored on the
    # returned statistics, mirroring the pure-numpy path's sim_dtype handling.
    np.dtype(sim_dtype)  # validates the dtype, raises TypeError on a bad value
    data_f64 = np.ascontiguousarray(arr, dtype=np.float64)
    return data_f64, n_obs


def _reducer_code(reducer: str) -> int:
    """Translate a validated reducer string to its integer kernel code.

    ``_validate_common`` has already rejected any unsupported reducer with a typed
    error, so this lookup cannot miss; it exists only to keep the string-to-code
    mapping in one place.
    """
    return _REDUCER_CODES[reducer]


def _resolve_q(reducer: str, q: float | None) -> float:
    """Validate ``q`` for the quantile reducer and return the float passed to kernels.

    For the quantile reducer ``q`` must be a single float in ``[0, 1]``; for every
    other reducer ``q`` is ignored and a harmless ``0.0`` is threaded into the
    kernel (which never reads it). This keeps the mean/var/std paths byte-identical
    while letting the quantile reducer carry its parameter through the same
    signature.
    """
    if reducer != REDUCER_QUANTILE:
        return 0.0
    if q is None:
        raise MethodConfigError(
            "reducer='quantile' requires a quantile level q (a float in [0, 1]).",
            code=Codes.INVALID_PARAMETER,
            context={"reducer": reducer, "q": q},
            hint="Pass q=0.5 for the median, q=0.95 for the upper tail, etc.",
        )
    q_float = float(q)
    if not (0.0 <= q_float <= 1.0):
        raise MethodConfigError(
            f"quantile level q must lie in [0, 1]; got {q_float}.",
            code=Codes.INVALID_PARAMETER,
            context={"q": q_float},
        )
    return q_float


def _clamp_block_length(length: int, n_obs: int, *, field: str) -> int:
    """Validate a positive integer block length and clamp it to ``[1, n_obs]``."""
    length = int(length)
    if length < 1:
        raise MethodConfigError(
            f"{field} must be a positive integer.",
            code=Codes.INVALID_PARAMETER,
            context={field: length},
        )
    return max(1, min(length, n_obs))


def _validate_and_prepare(
    data: NDArray[np.floating],
    n_bootstraps: int,
    avg_block_length: int,
    sim_dtype: np.dtype[np.floating],
    reducer: str,
) -> tuple[NDArray[np.float64], int, int, float]:
    """Validate stationary inputs at the Python boundary and return kernel inputs."""
    data_f64, n_obs = _validate_common(data, sim_dtype, reducer)
    avg_length = _clamp_block_length(avg_block_length, n_obs, field="avg_block_length")
    p = 1.0 / avg_length
    return data_f64, n_obs, n_bootstraps, p


def stationary_reduce(
    data: NDArray[np.floating],
    root_key: tuple[int, int],
    avg_block_length: int,
    sim_dtype: np.dtype[np.floating] = _FLOAT64,
    reducer: str = REDUCER_MEAN,
    q: float | None = None,
    *,
    n_bootstraps: int,
) -> NDArray[np.floating]:
    """Run the fused stationary fast path and return the reduced statistics.

    Builds each replicate's stationary index sequence, gathers the resampled
    rows, and applies ``reducer`` to each replicate, all inside one compiled
    parallel kernel that never materialises the full ``(B, n, d)`` sample.

    Parameters
    ----------
    data : array of shape ``(n,)`` or ``(n, d)``
        The observed series. 1-D input is treated as a single column.
    root_key : tuple of two int
        The run's packed 128-bit RNG root; replicate ``b``'s Philox key is derived
        in-kernel from ``(root_key, b)``.
    n_bootstraps : int
        Number of replicates ``B`` (keyword-only, so it can never be transposed with
        an adjacent positional int).
    avg_block_length : int
        Mean geometric block length; the restart probability is ``1 / avg_block_length``.
        Clamped to ``[1, n]`` exactly as the pure-numpy path clamps it.
    sim_dtype : numpy dtype, optional
        Dtype of the returned statistics. Defaults to float64.
    reducer : str, optional
        Per-replicate reducer, one of ``"mean"`` (column means), ``"var"``
        (population variance, ddof=0), ``"std"`` (population std, ddof=0), or
        ``"quantile"`` (the ``q``-quantile with numpy's default linear interpolation).
    q : float, optional
        Quantile level in ``[0, 1]``, required when ``reducer="quantile"`` and
        ignored otherwise.

    Returns
    -------
    array of shape ``(B, d)`` and dtype ``sim_dtype``
        The reduced statistic for each replicate.

    Raises
    ------
    MethodConfigError
        If numba is not installed, the reducer is unsupported, ``q`` is missing or
        out of ``[0, 1]`` for the quantile reducer, or the inputs are malformed
        (bad shape, no observations, non-positive block length).
    """
    data_f64, n_obs, B, p = _validate_and_prepare(
        data, n_bootstraps, avg_block_length, sim_dtype, reducer
    )
    q_val = _resolve_q(reducer, q)
    root_a, root_b = _root_words(root_key)
    d = data_f64.shape[1]
    out = np.empty((B, d), dtype=np.float64)
    if B > 0 and n_obs > 0:
        _stationary_reduce_kernel(data_f64, p, root_a, root_b, _reducer_code(reducer), q_val, out)
    return out.astype(sim_dtype, copy=False)


def stationary_indices(
    data_or_n: NDArray[np.floating] | int,
    root_key: tuple[int, int],
    avg_block_length: int,
    *,
    n_bootstraps: int,
) -> NDArray[np.int32]:
    """Build the ``(B, n)`` stationary index matrix from the Philox stream.

    Exposed for distributional validation and for callers that need the raw
    indices rather than a reduced statistic. Uses the same RNG and carry as
    :func:`stationary_reduce`.

    Parameters
    ----------
    data_or_n : array or int
        Either the observed series (its first axis gives ``n``) or ``n`` directly.
    root_key : tuple of two int
        The run's packed 128-bit RNG root; replicate ``b``'s key is derived from it.
    avg_block_length : int
        Mean geometric block length (restart probability ``1 / avg_block_length``).
    n_bootstraps : int
        Number of replicates ``B`` (keyword-only).

    Returns
    -------
    array of shape ``(B, n)`` and dtype int32
        The resampled observation index for each replicate and time step.
    """
    # A length-n_obs zero column reuses the shared validation for both the int and
    # array forms without allocating the caller's full series.
    probe: NDArray[np.floating]
    if isinstance(data_or_n, (int, np.integer)):
        probe = np.zeros((int(data_or_n), 1), dtype=np.float64)
    else:
        probe = data_or_n
    _, n_obs, B, p = _validate_and_prepare(
        probe, n_bootstraps, avg_block_length, np.dtype(np.float64), REDUCER_MEAN
    )
    root_a, root_b = _root_words(root_key)
    out_idx = np.empty((B, n_obs), dtype=np.int32)
    if B > 0:
        _stationary_indices_kernel(n_obs, p, root_a, root_b, out_idx)
    return out_idx


def iid_reduce(
    data: NDArray[np.floating],
    root_key: tuple[int, int],
    sim_dtype: np.dtype[np.floating] = _FLOAT64,
    reducer: str = REDUCER_MEAN,
    q: float | None = None,
    *,
    n_bootstraps: int,
) -> NDArray[np.floating]:
    """Run the fused IID fast path and return the per-replicate reduced statistics.

    Each replicate draws ``n`` independent uniform positions in ``[0, n)`` and the
    selected ``reducer`` is applied per column in one compiled parallel kernel that
    never materialises the full ``(B, n, d)`` sample. Equal in distribution to the
    numpy IID executor; uses the distinct Philox stream keyed in-kernel from
    ``(root_key, b)``.

    The ``reducer`` is one of ``"mean"``, ``"var"`` (population variance, ddof=0),
    ``"std"`` (population std, ddof=0), or ``"quantile"`` (the ``q``-quantile with
    numpy's default linear interpolation; ``q`` in ``[0, 1]`` is required for it and
    ignored otherwise).
    """
    data_f64, n_obs = _validate_common(data, sim_dtype, reducer)
    q_val = _resolve_q(reducer, q)
    B = n_bootstraps
    root_a, root_b = _root_words(root_key)
    d = data_f64.shape[1]
    out = np.empty((B, d), dtype=np.float64)
    if B > 0 and n_obs > 0:
        _iid_reduce_kernel(data_f64, root_a, root_b, _reducer_code(reducer), q_val, out)
    return out.astype(sim_dtype, copy=False)


def iid_indices(
    data_or_n: NDArray[np.floating] | int,
    root_key: tuple[int, int],
    *,
    n_bootstraps: int,
) -> NDArray[np.int32]:
    """Build the ``(B, n)`` IID index matrix from the Philox stream (for validation)."""
    probe: NDArray[np.floating]
    if isinstance(data_or_n, (int, np.integer)):
        probe = np.zeros((int(data_or_n), 1), dtype=np.float64)
    else:
        probe = data_or_n
    _, n_obs = _validate_common(probe, _FLOAT64, REDUCER_MEAN)
    B = n_bootstraps
    root_a, root_b = _root_words(root_key)
    out_idx = np.empty((B, n_obs), dtype=np.int32)
    if B > 0:
        _iid_indices_kernel(n_obs, root_a, root_b, out_idx)
    return out_idx


# Block-family parameters for the fused fixed-length-block kernel. ``span`` and
# ``wrap`` together select the start-draw rule and the gather wraparound:
#   moving:          span = n - length + 1, wrap = 0
#   circular:        span = n,              wrap = 1
#   non_overlapping: span = -1 (sentinel),  wrap = 0  (starts are multiples of length)
_MOVING: Final = "moving"
_CIRCULAR: Final = "circular"
_NON_OVERLAPPING: Final = "non_overlapping"


def _block_span(family: str, n_obs: int, length: int) -> int:
    if family == _MOVING:
        return n_obs - length + 1
    if family == _CIRCULAR:
        return n_obs
    return -1  # non-overlapping sentinel


def block_reduce(
    family: str,
    data: NDArray[np.floating],
    root_key: tuple[int, int],
    block_length: int,
    sim_dtype: np.dtype[np.floating] = _FLOAT64,
    reducer: str = REDUCER_MEAN,
    q: float | None = None,
    *,
    n_bootstraps: int,
) -> NDArray[np.floating]:
    """Run a fused fixed-length-block fast path (moving/circular/non-overlapping).

    Equal in distribution to the matching numpy executor in
    :mod:`tsbootstrap.block.indices`; uses the distinct Philox stream keyed in-kernel
    from ``(root_key, b)``. Fuses block build, gather, and the selected column reduce
    in one compiled parallel kernel that never materialises the full ``(B, n, d)``
    sample. The ``reducer`` is one of ``"mean"``, ``"var"`` (population variance,
    ddof=0), ``"std"`` (population std, ddof=0), or ``"quantile"`` (the ``q``-quantile
    with numpy's default linear interpolation; ``q`` in ``[0, 1]`` is required for it
    and ignored otherwise).
    """
    data_f64, n_obs = _validate_common(data, sim_dtype, reducer)
    q_val = _resolve_q(reducer, q)
    length = _clamp_block_length(block_length, n_obs, field="block_length")
    span = _block_span(family, n_obs, length)
    wrap = np.int64(1 if family == _CIRCULAR else 0)
    B = n_bootstraps
    root_a, root_b = _root_words(root_key)
    d = data_f64.shape[1]
    out = np.empty((B, d), dtype=np.float64)
    if B > 0 and n_obs > 0:
        _block_reduce_kernel(
            data_f64,
            length,
            np.int64(span),
            wrap,
            root_a,
            root_b,
            _reducer_code(reducer),
            q_val,
            out,
        )
    return out.astype(sim_dtype, copy=False)


def block_indices(
    family: str,
    data_or_n: NDArray[np.floating] | int,
    root_key: tuple[int, int],
    block_length: int,
    *,
    n_bootstraps: int,
) -> NDArray[np.int32]:
    """Build the ``(B, n)`` fixed-length-block index matrix (for validation)."""
    probe: NDArray[np.floating]
    if isinstance(data_or_n, (int, np.integer)):
        probe = np.zeros((int(data_or_n), 1), dtype=np.float64)
    else:
        probe = data_or_n
    _, n_obs = _validate_common(probe, _FLOAT64, REDUCER_MEAN)
    length = _clamp_block_length(block_length, n_obs, field="block_length")
    span = _block_span(family, n_obs, length)
    wrap = np.int64(1 if family == _CIRCULAR else 0)
    B = n_bootstraps
    root_a, root_b = _root_words(root_key)
    out_idx = np.empty((B, n_obs), dtype=np.int32)
    if B > 0:
        _block_indices_kernel(n_obs, length, np.int64(span), wrap, root_a, root_b, out_idx)
    return out_idx


def stationary_values(
    data: NDArray[np.floating],
    root_key: tuple[int, int],
    avg_block_length: int,
    sim_dtype: np.dtype[np.floating] = _FLOAT64,
    *,
    n_bootstraps: int,
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    """Materialise the stationary fast path: gathered ``(B, n, d)`` values plus indices.

    Builds each replicate's stationary index sequence and gathers the resampled
    rows in one compiled parallel kernel that writes both outputs in a single
    pass. The indices are built by the same device function the reduce path uses,
    so for the same root the resample is identical to :func:`stationary_reduce`.

    Returns
    -------
    (values, indices)
        ``values`` of shape ``(B, n, d)`` and dtype ``sim_dtype``; ``indices`` of
        shape ``(B, n)`` and dtype int32, with ``values[b, t] == data[indices[b, t]]``.
    """
    data_f64, n_obs, B, p = _validate_and_prepare(
        data, n_bootstraps, avg_block_length, sim_dtype, REDUCER_MEAN
    )
    root_a, root_b = _root_words(root_key)
    d = data_f64.shape[1]
    out_val = np.empty((B, n_obs, d), dtype=np.float64)
    out_idx = np.empty((B, n_obs), dtype=np.int32)
    if B > 0 and n_obs > 0:
        _stationary_values_kernel(data_f64, p, root_a, root_b, out_val, out_idx)
    return out_val.astype(sim_dtype, copy=False), out_idx


def iid_values(
    data: NDArray[np.floating],
    root_key: tuple[int, int],
    sim_dtype: np.dtype[np.floating] = _FLOAT64,
    *,
    n_bootstraps: int,
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    """Materialise the IID fast path: gathered ``(B, n, d)`` values plus ``(B, n)`` indices.

    Same Philox stream and per-replicate keying as :func:`iid_reduce`; the indices
    are built by the shared IID device function, so the resample matches the
    reduce path for the same root and ``values[b, t] == data[indices[b, t]]``.
    """
    data_f64, n_obs = _validate_common(data, sim_dtype, REDUCER_MEAN)
    B = n_bootstraps
    root_a, root_b = _root_words(root_key)
    d = data_f64.shape[1]
    out_val = np.empty((B, n_obs, d), dtype=np.float64)
    out_idx = np.empty((B, n_obs), dtype=np.int32)
    if B > 0 and n_obs > 0:
        _iid_values_kernel(data_f64, root_a, root_b, out_val, out_idx)
    return out_val.astype(sim_dtype, copy=False), out_idx


def block_values(
    family: str,
    data: NDArray[np.floating],
    root_key: tuple[int, int],
    block_length: int,
    sim_dtype: np.dtype[np.floating] = _FLOAT64,
    *,
    n_bootstraps: int,
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    """Materialise a fixed-length-block fast path (moving/circular/non-overlapping).

    Builds the block index row with the same device function the reduce path uses
    and gathers ``(B, n, d)`` values in one compiled parallel pass, so the
    resample matches :func:`block_reduce` for the same root and
    ``values[b, t] == data[indices[b, t]]``.
    """
    data_f64, n_obs = _validate_common(data, sim_dtype, REDUCER_MEAN)
    length = _clamp_block_length(block_length, n_obs, field="block_length")
    span = _block_span(family, n_obs, length)
    wrap = np.int64(1 if family == _CIRCULAR else 0)
    B = n_bootstraps
    root_a, root_b = _root_words(root_key)
    d = data_f64.shape[1]
    out_val = np.empty((B, n_obs, d), dtype=np.float64)
    out_idx = np.empty((B, n_obs), dtype=np.int32)
    if B > 0 and n_obs > 0:
        _block_values_kernel(
            data_f64, length, np.int64(span), wrap, root_a, root_b, out_val, out_idx
        )
    return out_val.astype(sim_dtype, copy=False), out_idx


def _validate_ar_reducer(reducer: str) -> None:
    """Reject an unsupported reducer for the AR fused path with a typed error."""
    if reducer not in _SUPPORTED_REDUCERS:
        raise MethodConfigError(
            f"Unsupported reducer {reducer!r} for the compiled fast path.",
            code=Codes.INVALID_PARAMETER,
            context={"reducer": reducer, "supported": sorted(_SUPPORTED_REDUCERS)},
            hint=f"Use one of: {sorted(_SUPPORTED_REDUCERS)}.",
        )


def ar_residual_reduce(
    ctx: _ARContext,
    root_key: tuple[int, int],
    n_obs: int,
    sim_dtype: np.dtype[np.floating] = _FLOAT64,
    reducer: str = REDUCER_MEAN,
    q: float | None = None,
    *,
    n_bootstraps: int,
) -> NDArray[np.floating]:
    """Run the fused recursive residual AR fast path and return reduced statistics.

    For each replicate this resamples the centered residuals IID through that
    replicate's Philox stream, runs the fitted AR recurrence in the kernel, and
    applies ``reducer`` to the generated length-``n_obs`` path, never materialising
    the ``(B, n)`` resampled-residual array or the ``(B, n)`` path. Equal in
    distribution to the numpy :func:`tsbootstrap.model.recursive._ar_batched`
    reduce; uses the distinct Philox stream (keyed in-kernel from ``(root_key, b)``)
    with its own goldens.

    Parameters
    ----------
    ctx : tsbootstrap.model.recursive._ARContext
        The fitted AR context (coefficients, intercept, centered residuals, the
        observed series for the initial block, burn-in, and initial-state mode).
        Held-fixed exogenous regressors are not supported on this fast path.
    root_key : tuple of two int
        The run's packed 128-bit RNG root; replicate ``b``'s key is derived from it.
    n_obs : int
        Length of each generated path (matches the observed series length).
    n_bootstraps : int
        Number of replicates ``B`` (keyword-only, so it can never be transposed with
        the adjacent ``n_obs``).
    sim_dtype : numpy dtype, optional
        Dtype of the returned statistics. Defaults to float64.
    reducer : str, optional
        Per-replicate reducer, one of ``"mean"``, ``"var"`` (population variance,
        ddof=0), ``"std"`` (population std, ddof=0), or ``"quantile"`` (the
        ``q``-quantile with numpy's default linear interpolation).
    q : float, optional
        Quantile level in ``[0, 1]``, required when ``reducer="quantile"`` and
        ignored otherwise.

    Returns
    -------
    array of shape ``(B, 1)`` and dtype ``sim_dtype``
        The reduced statistic for each replicate (AR is univariate).

    Raises
    ------
    MethodConfigError
        If numba is not installed, the reducer is unsupported, ``q`` is missing or
        out of ``[0, 1]`` for the quantile reducer, or the context carries exogenous
        regressors (not supported on this fast path).
    """
    _require_numba()
    _validate_ar_reducer(reducer)
    q_val = _resolve_q(reducer, q)
    if getattr(ctx, "exog_state", None) is not None:
        raise MethodConfigError(
            "backend='compiled' does not support exogenous regressors for the "
            "recursive residual AR fast path.",
            code=Codes.INVALID_PARAMETER,
            hint="Use the default backend='numpy' for an ARX model.",
        )
    np.dtype(sim_dtype)  # validates the dtype, raises TypeError on a bad value
    fit = ctx.fit
    p = int(fit.order)
    eps = np.ascontiguousarray(ctx.resampling_innovations, dtype=np.float64)
    ar_coefs = np.ascontiguousarray(fit.ar_coefs, dtype=np.float64)
    series = np.ascontiguousarray(ctx.series, dtype=np.float64)
    intercept = float(fit.intercept)
    burn_in = int(ctx.burn_in)
    fixed = ctx.initial == "fixed"
    B = n_bootstraps
    out = np.empty((B, 1), dtype=np.float64)
    if B > 0 and n_obs > 0:
        root_a, root_b = _root_words(root_key)
        _ar_residual_reduce_kernel(
            eps,
            ar_coefs,
            intercept,
            series,
            p,
            burn_in,
            n_obs,
            fixed,
            root_a,
            root_b,
            _reducer_code(reducer),
            q_val,
            out,
        )
    return out.astype(sim_dtype, copy=False)


def var_residual_reduce(
    ctx: _VARContext,
    root_key: tuple[int, int],
    n_obs: int,
    sim_dtype: np.dtype[np.floating] = _FLOAT64,
    reducer: str = REDUCER_MEAN,
    q: float | None = None,
    *,
    n_bootstraps: int,
) -> NDArray[np.floating]:
    """Run the fused recursive residual VAR fast path and return reduced statistics.

    The multivariate analogue of :func:`ar_residual_reduce`. For each replicate this
    resamples the centered vector residual rows IID through that replicate's Philox
    stream, runs the fitted VAR recurrence in the kernel, and applies ``reducer`` to
    each of the ``d`` generated columns of the length-``n_obs`` path, never
    materialising the ``(B, n, d)`` resampled-residual array or the ``(B, n, d)``
    path. Equal in distribution to the numpy
    :func:`tsbootstrap.model.recursive._var_batched` reduce; uses the distinct
    Philox stream (keyed in-kernel from ``(root_key, b)``) with its own goldens.

    Parameters
    ----------
    ctx : tsbootstrap.model.recursive._VARContext
        The fitted VAR context (coefficient matrices, intercept vector, centered
        vector residuals, the observed series for the initial block, burn-in, and
        initial-state mode). Held-fixed exogenous regressors are not supported on
        this fast path.
    root_key : tuple of two int
        The run's packed 128-bit RNG root; replicate ``b``'s key is derived from it.
    n_obs : int
        Length of each generated path (matches the observed series row count).
    n_bootstraps : int
        Number of replicates ``B`` (keyword-only, so it can never be transposed with
        the adjacent ``n_obs``).
    sim_dtype : numpy dtype, optional
        Dtype of the returned statistics. Defaults to float64.
    reducer : str, optional
        Per-replicate reducer, one of ``"mean"``, ``"var"`` (population variance,
        ddof=0), ``"std"`` (population std, ddof=0), or ``"quantile"`` (the
        ``q``-quantile with numpy's default linear interpolation).
    q : float, optional
        Quantile level in ``[0, 1]``, required when ``reducer="quantile"`` and
        ignored otherwise.

    Returns
    -------
    array of shape ``(B, d)`` and dtype ``sim_dtype``
        The reduced statistic for each replicate and each of the ``d`` series.

    Raises
    ------
    MethodConfigError
        If numba is not installed, the reducer is unsupported, ``q`` is missing or
        out of ``[0, 1]`` for the quantile reducer, or the context carries exogenous
        regressors (not supported on this fast path).
    """
    _require_numba()
    _validate_ar_reducer(reducer)
    q_val = _resolve_q(reducer, q)
    if getattr(ctx, "exog_state", None) is not None:
        raise MethodConfigError(
            "backend='compiled' does not support exogenous regressors for the "
            "recursive residual VAR fast path.",
            code=Codes.INVALID_PARAMETER,
            hint="Use the default backend='numpy' for a VARX model.",
        )
    np.dtype(sim_dtype)  # validates the dtype, raises TypeError on a bad value
    fit = ctx.fit
    p = int(fit.order)
    eps = np.ascontiguousarray(ctx.resampling_innovations, dtype=np.float64)  # (n_resid, d)
    coefs = np.ascontiguousarray(fit.coefs, dtype=np.float64)  # (p, d, d)
    intercept = np.ascontiguousarray(fit.intercept, dtype=np.float64)  # (d,)
    series = np.ascontiguousarray(ctx.series, dtype=np.float64)  # (n_series, d)
    burn_in = int(ctx.burn_in)
    fixed = ctx.initial == "fixed"
    d = series.shape[1]
    B = n_bootstraps
    out = np.empty((B, d), dtype=np.float64)
    if B > 0 and n_obs > 0:
        root_a, root_b = _root_words(root_key)
        _var_residual_reduce_kernel(
            coefs,
            intercept,
            eps,
            series,
            p,
            burn_in,
            n_obs,
            fixed,
            root_a,
            root_b,
            _reducer_code(reducer),
            q_val,
            out,
        )
    return out.astype(sim_dtype, copy=False)


# --- ragged-panel boundary + entry points ----------------------------------
# A ragged panel is a flat (total_N, d) data array plus a CSR ``indptr`` of length
# num_series+1, exactly the layout the kernels above gather through. The Python
# boundary validates indptr (the njit kernels would otherwise read out of bounds on
# a malformed one) and resolves the family parameters, then runs one fused parallel
# pass returning the dense (B, num_series, d) statistics. ``.values()`` on a ragged
# panel is mathematically incoherent across unequal lengths, so the reduce IS the
# panel API; there is no panel values kernel.

_PANEL_SPAN_MODE: Final = {_MOVING: 0, _CIRCULAR: 1, _NON_OVERLAPPING: 2}


def _validate_indptr_shape(indptr: NDArray[np.integer]) -> NDArray[np.int64]:
    """Validate a panel CSR ``indptr``'s shape and monotonicity, returning it as int64.

    Checks 1-D, length num_series + 1 >= 2, starts at 0, non-decreasing, and no empty
    series. It does NOT cross-check the final offset against a flat-data row count: the
    panel local-index builders have no flat data, so they self-derive total_N as
    ``indptr[-1]`` and call this directly; :func:`_validate_indptr` layers the total_N
    check on top for the reduce path. Validating the shape here also turns a zero-length
    ``indptr`` into a typed error instead of a raw ``IndexError``.
    """
    arr = np.ascontiguousarray(indptr)
    if arr.ndim != 1:
        raise MethodConfigError(
            f"indptr must be 1-D, got {arr.ndim} dimensions.",
            code=Codes.INVALID_SHAPE,
            context={"ndim": arr.ndim, "shape": tuple(arr.shape)},
        )
    if arr.shape[0] < 2:
        raise MethodConfigError(
            "indptr must have length num_series + 1 >= 2 (at least one series).",
            code=Codes.INVALID_SHAPE,
            context={"len_indptr": int(arr.shape[0])},
        )
    arr64 = arr.astype(np.int64, copy=False)
    if int(arr64[0]) != 0:
        raise MethodConfigError(
            f"indptr must start at 0; got indptr[0]={int(arr64[0])}.",
            code=Codes.INVALID_PARAMETER,
            context={"indptr_0": int(arr64[0])},
        )
    diffs = np.diff(arr64)
    if np.any(diffs < 0):
        bad = int(np.argmin(diffs))
        raise MethodConfigError(
            f"indptr must be non-decreasing; it decreases at position {bad} "
            f"(indptr[{bad}]={int(arr64[bad])} > indptr[{bad + 1}]={int(arr64[bad + 1])}).",
            code=Codes.INVALID_PARAMETER,
            context={"position": bad},
        )
    if np.any(diffs < 1):
        bad = int(np.argmin(diffs))
        raise MethodConfigError(
            f"every series must have at least one observation; series {bad} is empty "
            f"(indptr[{bad}]==indptr[{bad + 1}]={int(arr64[bad])}).",
            code=Codes.TOO_FEW_OBSERVATIONS,
            context={"series": bad},
        )
    return arr64


def _validate_indptr(indptr: NDArray[np.integer], total_n: int) -> NDArray[np.int64]:
    """Validate a CSR ``indptr`` at the Python boundary and return it as int64.

    Layers the final-offset check (``indptr[-1] == total_n``, the number of flat rows)
    on top of :func:`_validate_indptr_shape`, so a malformed ``indptr`` cannot make the
    njit kernel read out of bounds.
    """
    arr64 = _validate_indptr_shape(indptr)
    if int(arr64[-1]) != total_n:
        raise MethodConfigError(
            f"indptr must end at total_N={total_n} (the number of flat rows); "
            f"got indptr[-1]={int(arr64[-1])}.",
            code=Codes.INVALID_PARAMETER,
            context={"indptr_last": int(arr64[-1]), "total_n": total_n},
        )
    return arr64


def _validate_panel_common(
    flat_data: NDArray[np.floating],
    indptr: NDArray[np.integer],
    sim_dtype: np.dtype[np.floating],
    reducer: str,
) -> tuple[NDArray[np.float64], NDArray[np.int64], int, int]:
    """Validate a ragged panel at the Python boundary; return kernel-ready inputs.

    Returns ``(flat_f64, indptr64, num_series, d)``. The flat data is promoted to a
    contiguous 2-D float64 array (1-D is one column); the reducer and dtype are
    validated exactly as the rectangular path validates them.
    """
    _require_numba()
    if reducer not in _SUPPORTED_REDUCERS:
        raise MethodConfigError(
            f"Unsupported reducer {reducer!r} for the compiled fast path.",
            code=Codes.INVALID_PARAMETER,
            context={"reducer": reducer, "supported": sorted(_SUPPORTED_REDUCERS)},
            hint=f"Use one of: {sorted(_SUPPORTED_REDUCERS)}.",
        )
    arr = np.asarray(flat_data)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise MethodConfigError(
            f"panel flat_data must be 1-D or 2-D (total_N[, d]); got {arr.ndim} dimensions.",
            code=Codes.INVALID_SHAPE,
            context={"ndim": arr.ndim, "shape": tuple(arr.shape)},
        )
    total_n = arr.shape[0]
    if total_n < 1:
        raise MethodConfigError(
            "panel flat_data must contain at least one observation.",
            code=Codes.TOO_FEW_OBSERVATIONS,
            context={"total_n": total_n},
        )
    indptr64 = _validate_indptr(indptr, total_n)
    num_series = int(indptr64.shape[0]) - 1
    np.dtype(sim_dtype)  # validates the dtype, raises TypeError on a bad value
    flat_f64 = np.ascontiguousarray(arr, dtype=np.float64)
    return flat_f64, indptr64, num_series, flat_f64.shape[1]


def panel_iid_reduce(
    flat_data: NDArray[np.floating],
    indptr: NDArray[np.integer],
    root_key: tuple[int, int],
    sim_dtype: np.dtype[np.floating] = _FLOAT64,
    reducer: str = REDUCER_MEAN,
    q: float | None = None,
    *,
    n_bootstraps: int,
) -> NDArray[np.floating]:
    """Fused ragged-panel IID reduce, returning ``(B, num_series, d)`` statistics."""
    flat_f64, indptr64, num_series, d = _validate_panel_common(
        flat_data, indptr, sim_dtype, reducer
    )
    q_val = _resolve_q(reducer, q)
    B = n_bootstraps
    out = np.empty((B, num_series, d), dtype=np.float64)
    if B > 0:
        root_a, root_b = _root_words(root_key)
        _iid_panel_reduce_kernel(
            flat_f64, indptr64, num_series, root_a, root_b, _reducer_code(reducer), q_val, out
        )
    return out.astype(sim_dtype, copy=False)


def panel_stationary_reduce(
    flat_data: NDArray[np.floating],
    indptr: NDArray[np.integer],
    root_key: tuple[int, int],
    avg_block_length: int,
    sim_dtype: np.dtype[np.floating] = _FLOAT64,
    reducer: str = REDUCER_MEAN,
    q: float | None = None,
    *,
    n_bootstraps: int,
) -> NDArray[np.floating]:
    """Fused ragged-panel stationary reduce, returning ``(B, num_series, d)`` statistics.

    ``avg_block_length`` is a single panel-wide mean geometric block length (the
    restart probability is ``1 / avg_block_length``); it is not clamped per series
    here, the geometric carry wraps within each series's own length.
    """
    flat_f64, indptr64, num_series, d = _validate_panel_common(
        flat_data, indptr, sim_dtype, reducer
    )
    q_val = _resolve_q(reducer, q)
    avg_length = int(avg_block_length)
    if avg_length < 1:
        raise MethodConfigError(
            "avg_block_length must be a positive integer.",
            code=Codes.INVALID_PARAMETER,
            context={"avg_block_length": avg_length},
        )
    p = 1.0 / avg_length
    B = n_bootstraps
    out = np.empty((B, num_series, d), dtype=np.float64)
    if B > 0:
        root_a, root_b = _root_words(root_key)
        _stationary_panel_reduce_kernel(
            flat_f64, indptr64, num_series, p, root_a, root_b, _reducer_code(reducer), q_val, out
        )
    return out.astype(sim_dtype, copy=False)


def panel_block_reduce(
    family: str,
    flat_data: NDArray[np.floating],
    indptr: NDArray[np.integer],
    root_key: tuple[int, int],
    block_length: int,
    sim_dtype: np.dtype[np.floating] = _FLOAT64,
    reducer: str = REDUCER_MEAN,
    q: float | None = None,
    *,
    n_bootstraps: int,
) -> NDArray[np.floating]:
    """Fused ragged-panel fixed-length-block reduce (moving/circular/non-overlapping).

    ``block_length`` is a single panel-wide block length; the kernel clamps it to a
    series's own length when a series is shorter than the requested block.
    """
    flat_f64, indptr64, num_series, d = _validate_panel_common(
        flat_data, indptr, sim_dtype, reducer
    )
    q_val = _resolve_q(reducer, q)
    length = int(block_length)
    if length < 1:
        raise MethodConfigError(
            "block_length must be a positive integer.",
            code=Codes.INVALID_PARAMETER,
            context={"block_length": length},
        )
    span_mode = _PANEL_SPAN_MODE[family]
    wrap = np.int64(1 if family == _CIRCULAR else 0)
    B = n_bootstraps
    out = np.empty((B, num_series, d), dtype=np.float64)
    if B > 0:
        root_a, root_b = _root_words(root_key)
        _block_panel_reduce_kernel(
            flat_f64,
            indptr64,
            num_series,
            length,
            np.int64(span_mode),
            wrap,
            root_a,
            root_b,
            _reducer_code(reducer),
            q_val,
            out,
        )
    return out.astype(sim_dtype, copy=False)


def panel_iid_local_indices(
    indptr: NDArray[np.integer],
    root_key: tuple[int, int],
    *,
    n_bootstraps: int,
) -> NDArray[np.int32]:
    """Build the flat ``(B, total_N)`` local IID index matrix for a panel (for validation).

    Series ``s`` of replicate ``b`` occupies columns ``indptr[s]:indptr[s+1]``; the
    values are LOCAL positions in ``[0, n_s)`` (the caller maps them through
    ``indptr[s]`` to global rows). Uses the same fold_in and fill device functions as
    :func:`panel_iid_reduce`, so the resample is identical.
    """
    indptr64 = _validate_indptr_shape(indptr)
    total_n = int(indptr64[-1])
    num_series = int(indptr64.shape[0]) - 1
    _require_numba()
    B = n_bootstraps
    out_flat = np.empty((B, total_n), dtype=np.int32)
    if B > 0:
        root_a, root_b = _root_words(root_key)
        _iid_panel_indices_kernel(indptr64, num_series, root_a, root_b, out_flat)
    return out_flat


def panel_stationary_local_indices(
    indptr: NDArray[np.integer],
    root_key: tuple[int, int],
    avg_block_length: int,
    *,
    n_bootstraps: int,
) -> NDArray[np.int32]:
    """Build the flat ``(B, total_N)`` local stationary index matrix for a panel (validation)."""
    indptr64 = _validate_indptr_shape(indptr)
    total_n = int(indptr64[-1])
    num_series = int(indptr64.shape[0]) - 1
    _require_numba()
    p = 1.0 / int(avg_block_length)
    B = n_bootstraps
    out_flat = np.empty((B, total_n), dtype=np.int32)
    if B > 0:
        root_a, root_b = _root_words(root_key)
        _stationary_panel_indices_kernel(indptr64, num_series, p, root_a, root_b, out_flat)
    return out_flat


def panel_block_local_indices(
    family: str,
    indptr: NDArray[np.integer],
    root_key: tuple[int, int],
    block_length: int,
    *,
    n_bootstraps: int,
) -> NDArray[np.int32]:
    """Build the flat ``(B, total_N)`` local fixed-length-block index matrix (validation)."""
    indptr64 = _validate_indptr_shape(indptr)
    total_n = int(indptr64[-1])
    num_series = int(indptr64.shape[0]) - 1
    _require_numba()
    span_mode = _PANEL_SPAN_MODE[family]
    wrap = np.int64(1 if family == _CIRCULAR else 0)
    B = n_bootstraps
    out_flat = np.empty((B, total_n), dtype=np.int32)
    if B > 0:
        root_a, root_b = _root_words(root_key)
        _block_panel_indices_kernel(
            indptr64,
            num_series,
            int(block_length),
            np.int64(span_mode),
            wrap,
            root_a,
            root_b,
            out_flat,
        )
    return out_flat


_SUPPORTED_METHOD_NAMES: Final = (
    "IID",
    "MovingBlock",
    "CircularBlock",
    "StationaryBlock",
    "NonOverlappingBlock",
    "ResidualBootstrap(AR)",
    "ResidualBootstrap(VAR)",
)


def compiled_supports(method: object) -> bool:
    """True if the compiled backend has a kernel for this method's family.

    Cheap, fit-free check on the method type alone, so a caller can reject an
    unsupported method before doing any setup work (e.g. a model fit).
    """
    from tsbootstrap.methods import (
        AR,
        IID,
        VAR,
        CircularBlock,
        MovingBlock,
        NonOverlappingBlock,
        ResidualBootstrap,
        StationaryBlock,
    )

    if isinstance(method, ResidualBootstrap):
        # The recursive residual AR and VAR fast paths are built; ARIMA models stay
        # on the numpy backend (the unified entry raises the typed error for them).
        # The residual kernels resample IID innovations inside the compiled loop, so
        # any other innovation (the wild multiplier family, block innovations) must be
        # rejected here: running the IID kernel for them would silently return a
        # different bootstrap distribution than the one the spec asks for.
        if not isinstance(method.innovation, IID):
            return False
        return isinstance(method.model, (AR, VAR))
    return isinstance(
        method, (IID, MovingBlock, CircularBlock, StationaryBlock, NonOverlappingBlock)
    )


def unsupported_method_error(method: object) -> MethodConfigError:
    """The typed error raised when the compiled backend has no kernel for a method."""
    from tsbootstrap.methods import AR, IID, VAR, ResidualBootstrap

    # A residual bootstrap whose MODEL is supported but whose INNOVATION is not:
    # name the innovation, not the method, so the message does not falsely imply
    # that ResidualBootstrap itself lacks a compiled kernel.
    if (
        isinstance(method, ResidualBootstrap)
        and isinstance(method.model, (AR, VAR))
        and not isinstance(method.innovation, IID)
    ):
        innovation_name = type(method.innovation).__name__
        return MethodConfigError(
            f"backend='compiled' does not support {innovation_name} innovations; "
            "the compiled residual kernels resample IID innovations only",
            code=Codes.INVALID_PARAMETER,
            context={"method": type(method).__name__, "innovation": innovation_name},
            hint="Use the default backend='numpy' for wild-type or block innovations.",
        )
    return MethodConfigError(
        f"backend='compiled' does not support {type(method).__name__}; "
        f"supported methods: {list(_SUPPORTED_METHOD_NAMES)}.",
        code=Codes.INVALID_PARAMETER,
        context={"method": type(method).__name__, "supported": list(_SUPPORTED_METHOD_NAMES)},
        hint="Use the default backend='numpy' for this method.",
    )


_PANEL_SUPPORTED_METHOD_NAMES: Final = (
    "IID",
    "MovingBlock",
    "CircularBlock",
    "StationaryBlock",
    "NonOverlappingBlock",
)


def compiled_panel_supports(method: object) -> bool:
    """True if the ragged-panel compiled backend has a kernel for this method family.

    The v1 panel scope is the observation methods only (pure index work). Recursive
    (model-based) panels would need per-series fitted-context arrays threaded through
    the kernel and are out of scope, so they are rejected here.
    """
    from tsbootstrap.methods import (
        IID,
        CircularBlock,
        MovingBlock,
        NonOverlappingBlock,
        StationaryBlock,
    )

    return isinstance(
        method, (IID, MovingBlock, CircularBlock, StationaryBlock, NonOverlappingBlock)
    )


def unsupported_panel_method_error(method: object) -> MethodConfigError:
    """The typed error raised when the panel backend has no kernel for a method."""
    return MethodConfigError(
        f"the ragged-panel backend does not support {type(method).__name__}; supported "
        f"panel methods: {list(_PANEL_SUPPORTED_METHOD_NAMES)}. Recursive (model-based) "
        "panels are out of the v1 scope.",
        code=Codes.INVALID_PARAMETER,
        context={
            "method": type(method).__name__,
            "supported": list(_PANEL_SUPPORTED_METHOD_NAMES),
        },
        hint="Use an observation method (IID, MovingBlock, CircularBlock, "
        "StationaryBlock, NonOverlappingBlock) for a ragged panel.",
    )


def _resolve_panel_block_length(
    value: BlockLength,
    flat_data: NDArray[np.float64],
    indptr: NDArray[np.int64],
    kind: str,
) -> int:
    """Resolve a panel's ``"auto"`` (or explicit) block length to a single int.

    A panel uses one block length across every series. An explicit int is passed
    through; ``"auto"`` is resolved against the LONGEST series (its block-length
    estimate is the most data-supported), matching how the rectangular path resolves
    ``"auto"`` from the one series it sees.
    """
    from tsbootstrap.block.pwsd import resolve_block_length

    if value != "auto":
        return int(value)
    lengths = np.diff(indptr)
    s_longest = int(np.argmax(lengths))
    lo = int(indptr[s_longest])
    hi = int(indptr[s_longest + 1])
    return resolve_block_length("auto", flat_data[lo:hi], kind=kind)


def compiled_panel_reduce(
    method: object,
    flat_data: NDArray[np.floating],
    indptr: NDArray[np.integer],
    root_key: tuple[int, int],
    sim_dtype: np.dtype[np.floating] = _FLOAT64,
    reducer: str = REDUCER_MEAN,
    q: float | None = None,
    *,
    n_bootstraps: int,
) -> NDArray[np.floating]:
    """Unified entry: dispatch an observation-method spec to its ragged-panel fast path.

    Resolves any ``"auto"`` block / average length against the panel's longest series,
    validates the flat data and CSR ``indptr`` at the Python boundary, and returns the
    dense per-replicate, per-series reduced statistic of shape ``(B, num_series, d)``.

    Parameters
    ----------
    method : method spec
        One of ``IID``, ``MovingBlock``, ``CircularBlock``, ``StationaryBlock``, or
        ``NonOverlappingBlock``. Recursive (model-based) methods are out of v1 scope.
    flat_data : array of shape ``(total_N,)`` or ``(total_N, d)``
        The concatenated per-series observations (1-D is one column).
    indptr : array of shape ``(num_series + 1,)``
        CSR offsets; series ``s`` is ``flat_data[indptr[s]:indptr[s+1]]``.
    root_key : tuple of two int
        The run's packed 128-bit RNG root; replicate ``b``'s Philox key is derived
        in-kernel from ``(root_key, b)`` (then folded per series slot).
    n_bootstraps : int
        Number of replicates ``B`` (keyword-only).
    sim_dtype, reducer, q
        As for the rectangular :func:`compiled_reduce`.

    Returns
    -------
    array of shape ``(B, num_series, d)`` and dtype ``sim_dtype``
        The reduced statistic for each replicate and each series.

    Raises
    ------
    MethodConfigError
        If numba is not installed, the method is unsupported on a panel, the reducer
        is unsupported, ``q`` is missing or out of ``[0, 1]`` for the quantile reducer,
        or the flat data / ``indptr`` are malformed.
    """
    from tsbootstrap.methods import (
        IID,
        CircularBlock,
        MovingBlock,
        NonOverlappingBlock,
        StationaryBlock,
    )

    if not compiled_panel_supports(method):
        raise unsupported_panel_method_error(method)

    # Validate once here so "auto" resolution sees a clean int64 indptr / float64 data;
    # the per-family entry re-validates cheaply (idempotent) before its kernel call.
    flat_f64, indptr64, _num_series, _d = _validate_panel_common(
        flat_data, indptr, sim_dtype, reducer
    )

    if isinstance(method, IID):
        return panel_iid_reduce(
            flat_f64, indptr64, root_key, sim_dtype, reducer, q, n_bootstraps=n_bootstraps
        )
    if isinstance(method, StationaryBlock):
        avg_length = _resolve_panel_block_length(
            method.avg_block_length, flat_f64, indptr64, "stationary"
        )
        return panel_stationary_reduce(
            flat_f64,
            indptr64,
            root_key,
            avg_length,
            sim_dtype,
            reducer,
            q,
            n_bootstraps=n_bootstraps,
        )
    if isinstance(method, MovingBlock):
        length = _resolve_panel_block_length(method.block_length, flat_f64, indptr64, "circular")
        return panel_block_reduce(
            _MOVING,
            flat_f64,
            indptr64,
            root_key,
            length,
            sim_dtype,
            reducer,
            q,
            n_bootstraps=n_bootstraps,
        )
    if isinstance(method, CircularBlock):
        length = _resolve_panel_block_length(method.block_length, flat_f64, indptr64, "circular")
        return panel_block_reduce(
            _CIRCULAR,
            flat_f64,
            indptr64,
            root_key,
            length,
            sim_dtype,
            reducer,
            q,
            n_bootstraps=n_bootstraps,
        )
    if isinstance(method, NonOverlappingBlock):
        length = _resolve_panel_block_length(method.block_length, flat_f64, indptr64, "circular")
        return panel_block_reduce(
            _NON_OVERLAPPING,
            flat_f64,
            indptr64,
            root_key,
            length,
            sim_dtype,
            reducer,
            q,
            n_bootstraps=n_bootstraps,
        )

    raise unsupported_panel_method_error(method)


def compiled_reduce(
    method: object,
    data: object,
    root_key: tuple[int, int],
    sim_dtype: np.dtype[np.floating] = _FLOAT64,
    reducer: str = REDUCER_MEAN,
    q: float | None = None,
    *,
    n_bootstraps: int,
) -> NDArray[np.floating]:
    """Unified entry point: dispatch a method spec to its compiled fast path.

    Resolves any ``"auto"`` block / average length internally via
    :func:`tsbootstrap.block.pwsd.resolve_block_length` (mirroring how the numpy
    executors resolve it, with the same ``kind=`` per method), validates inputs
    with typed :class:`~tsbootstrap.errors.MethodConfigError`, and returns the
    per-replicate reduced statistic of shape ``(B, d)``.

    Parameters
    ----------
    method : method spec
        One of ``IID``, ``MovingBlock``, ``CircularBlock``, ``StationaryBlock``,
        or ``NonOverlappingBlock``.
    data : array of shape ``(n,)`` or ``(n, d)``
        The observed series. 1-D input is treated as a single column.
    root_key : tuple of two int
        The run's packed 128-bit RNG root; replicate ``b``'s Philox key is derived
        in-kernel from ``(root_key, b)``.
    n_bootstraps : int
        Number of replicates ``B`` (keyword-only).
    sim_dtype : numpy dtype, optional
        Dtype of the returned statistics. Defaults to float64.
    reducer : str, optional
        Per-replicate reducer, one of ``"mean"`` (column means), ``"var"``
        (population variance, ddof=0), ``"std"`` (population std, ddof=0), or
        ``"quantile"`` (the ``q``-quantile with numpy's default linear interpolation).
    q : float, optional
        Quantile level in ``[0, 1]``, required when ``reducer="quantile"`` and
        ignored otherwise.

    Returns
    -------
    array of shape ``(B, d)`` and dtype ``sim_dtype``
        The reduced statistic for each replicate.

    Raises
    ------
    MethodConfigError
        If numba is not installed, the method is unsupported, the reducer is
        unsupported, ``q`` is missing or out of ``[0, 1]`` for the quantile reducer,
        or the inputs are malformed.
    """
    from tsbootstrap.block.pwsd import resolve_block_length
    from tsbootstrap.methods import (
        AR,
        IID,
        VAR,
        CircularBlock,
        MovingBlock,
        NonOverlappingBlock,
        ResidualBootstrap,
        StationaryBlock,
    )

    if isinstance(method, ResidualBootstrap):
        # For recursive methods the api layer passes the fitted context (not raw
        # observations) as ``data``: the AR and VAR models have compiled kernels, and
        # only the matching context reaches each (the ARIMA context hits the typed
        # error below). ``n_obs`` is the observed series length carried on the context.
        if isinstance(method.model, AR):
            ar_ctx = cast("_ARContext", data)
            n_obs = int(np.asarray(ar_ctx.series).shape[0])
            return ar_residual_reduce(
                ar_ctx, root_key, n_obs, sim_dtype, reducer, q, n_bootstraps=n_bootstraps
            )
        if isinstance(method.model, VAR):
            var_ctx = cast("_VARContext", data)
            n_obs = int(np.asarray(var_ctx.series).shape[0])
            return var_residual_reduce(
                var_ctx, root_key, n_obs, sim_dtype, reducer, q, n_bootstraps=n_bootstraps
            )
        raise unsupported_method_error(method)

    # Every remaining method resamples observations: the api layer passes the
    # coerced float64 series as ``data`` here (never a context).
    obs = cast("NDArray[np.float64]", data)
    if isinstance(method, IID):
        return iid_reduce(obs, root_key, sim_dtype, reducer, q, n_bootstraps=n_bootstraps)
    if isinstance(method, StationaryBlock):
        avg_length = resolve_block_length(method.avg_block_length, obs, kind="stationary")
        return stationary_reduce(
            obs, root_key, avg_length, sim_dtype, reducer, q, n_bootstraps=n_bootstraps
        )
    if isinstance(method, MovingBlock):
        length = resolve_block_length(method.block_length, obs, kind="circular")
        return block_reduce(
            _MOVING, obs, root_key, length, sim_dtype, reducer, q, n_bootstraps=n_bootstraps
        )
    if isinstance(method, CircularBlock):
        length = resolve_block_length(method.block_length, obs, kind="circular")
        return block_reduce(
            _CIRCULAR, obs, root_key, length, sim_dtype, reducer, q, n_bootstraps=n_bootstraps
        )
    if isinstance(method, NonOverlappingBlock):
        length = resolve_block_length(method.block_length, obs, kind="circular")
        return block_reduce(
            _NON_OVERLAPPING,
            obs,
            root_key,
            length,
            sim_dtype,
            reducer,
            q,
            n_bootstraps=n_bootstraps,
        )

    raise unsupported_method_error(method)


def compiled_values(
    method: object,
    data: object,
    root_key: tuple[int, int],
    sim_dtype: np.dtype[np.floating] = _FLOAT64,
    *,
    n_bootstraps: int,
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    """Unified entry point for the materialised (``.values()``) compiled fast path.

    Dispatches a method spec to its fused values kernel and returns the gathered
    ``(B, n, d)`` values plus the ``(B, n)`` int32 indices. The index logic is the
    same device function the reduce path uses, so for the same ``root_key`` and method
    ``compiled_values`` and :func:`compiled_reduce` see an identical resample, and
    ``values[b, t] == data[indices[b, t]]``.

    Resolves any ``"auto"`` block / average length internally exactly as
    :func:`compiled_reduce` does (same ``kind=`` per method) and raises the same
    typed :class:`~tsbootstrap.errors.MethodConfigError` for an unsupported method
    or when numba is not installed.

    Parameters
    ----------
    method : method spec
        One of ``IID``, ``MovingBlock``, ``CircularBlock``, ``StationaryBlock``,
        or ``NonOverlappingBlock``.
    data : array of shape ``(n,)`` or ``(n, d)``
        The observed series. 1-D input is treated as a single column and returned
        as ``(B, n, 1)``; the api layer squeezes it back to ``(B, n)`` via ``was_1d``.
    root_key : tuple of two int
        The run's packed 128-bit RNG root; replicate ``b``'s Philox key is derived
        in-kernel from ``(root_key, b)``.
    n_bootstraps : int
        Number of replicates ``B`` (keyword-only).
    sim_dtype : numpy dtype, optional
        Dtype of the returned values. Defaults to float64; kernel math is float64
        and the values are cast at the boundary.

    Returns
    -------
    (values, indices)
        ``values`` of shape ``(B, n, d)`` and dtype ``sim_dtype``; ``indices`` of
        shape ``(B, n)`` and dtype int32.

    Raises
    ------
    MethodConfigError
        If numba is not installed, the method is unsupported, or the inputs are
        malformed.
    """
    from tsbootstrap.block.pwsd import resolve_block_length
    from tsbootstrap.methods import (
        IID,
        CircularBlock,
        MovingBlock,
        NonOverlappingBlock,
        ResidualBootstrap,
        StationaryBlock,
    )

    if isinstance(method, ResidualBootstrap):
        # The compiled recursive AR fast path is a fused reduce only; it never
        # materialises the (B, n) path, so there is no compiled values kernel for
        # it. bootstrap_reduce(..., backend='compiled') is the supported entry.
        raise MethodConfigError(
            "backend='compiled' supports the recursive ResidualBootstrap(AR) and "
            "ResidualBootstrap(VAR) fast paths only through bootstrap_reduce (a fused "
            "reduce that never materialises the path); the full .values() "
            "materialisation is not available on the compiled backend for recursive "
            "methods.",
            code=Codes.INVALID_PARAMETER,
            context={"method": type(method).__name__},
            hint="Use bootstrap_reduce(..., backend='compiled') or the default "
            "backend='numpy' for bootstrap().",
        )

    # Every materialised method resamples observations: the api layer passes the
    # coerced float64 series as ``data`` here (recursive methods raised above).
    obs = cast("NDArray[np.float64]", data)
    if isinstance(method, IID):
        return iid_values(obs, root_key, sim_dtype, n_bootstraps=n_bootstraps)
    if isinstance(method, StationaryBlock):
        avg_length = resolve_block_length(method.avg_block_length, obs, kind="stationary")
        return stationary_values(obs, root_key, avg_length, sim_dtype, n_bootstraps=n_bootstraps)
    if isinstance(method, MovingBlock):
        length = resolve_block_length(method.block_length, obs, kind="circular")
        return block_values(_MOVING, obs, root_key, length, sim_dtype, n_bootstraps=n_bootstraps)
    if isinstance(method, CircularBlock):
        length = resolve_block_length(method.block_length, obs, kind="circular")
        return block_values(_CIRCULAR, obs, root_key, length, sim_dtype, n_bootstraps=n_bootstraps)
    if isinstance(method, NonOverlappingBlock):
        length = resolve_block_length(method.block_length, obs, kind="circular")
        return block_values(
            _NON_OVERLAPPING, obs, root_key, length, sim_dtype, n_bootstraps=n_bootstraps
        )

    raise unsupported_method_error(method)


__all__ = [
    "REDUCER_MEAN",
    "REDUCER_VAR",
    "REDUCER_STD",
    "REDUCER_QUANTILE",
    "stationary_reduce",
    "stationary_indices",
    "stationary_values",
    "iid_reduce",
    "iid_indices",
    "iid_values",
    "block_reduce",
    "block_indices",
    "block_values",
    "ar_residual_reduce",
    "var_residual_reduce",
    "compiled_reduce",
    "compiled_values",
    "panel_iid_reduce",
    "panel_stationary_reduce",
    "panel_block_reduce",
    "panel_iid_local_indices",
    "panel_stationary_local_indices",
    "panel_block_local_indices",
    "compiled_panel_reduce",
    "compiled_panel_supports",
]
