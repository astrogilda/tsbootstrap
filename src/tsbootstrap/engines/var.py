"""Recursive VAR (vector autoregression) simulation.

A direct vector recursion: ``X_t = c + sum_j A_j X_{t-j} + e_t``. The time axis is a
genuine sequential dependency, so it cannot be vectorised away; at large ``n`` the
per-step dispatch over small ``(B, d)`` arrays dominates. When the optional ``[accel]``
extra (numba) is installed, a compiled kernel runs the whole ``(B, n, d)`` recursion in
fused machine code and is selected automatically; otherwise a pure-numpy fallback is
used. The two backends are each deterministic but differ by a few ULPs (compiled
scalar accumulation vs BLAS), the same class of variation as a different BLAS already
introduces, results are reproducible for a fixed backend.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.rng import register_warmup

try:  # optional [accel] extra: a compiled, replicate-parallel kernel for the time loop
    import numba

    @numba.njit(parallel=True, cache=True)
    def _var_recurrence_numba(
        coefs: NDArray[np.float64],
        intercept: NDArray[np.float64],
        path: NDArray[np.float64],
        innovations: NDArray[np.float64],
        p: int,
        m: int,
    ) -> None:  # pragma: no cover - numba njit-compiles this body to machine code, so coverage.py cannot trace the Python lines even when the kernel runs; correctness is pinned by test_var_numba_and_numpy_backends_agree
        # Fill path[:, p:] in place with X_t[i] = c[i] + e_t[i] + sum_j sum_k A_j[i,k] X_{t-1-j}[k].
        # The B replicates are independent and write disjoint path[b] slices, so prange over
        # b is data-race-free and deterministic; the time loop stays sequential per path.
        n_paths = path.shape[0]
        d = path.shape[2]
        for b in numba.prange(n_paths):
            for t in range(p, p + m):
                for i in range(d):
                    acc = intercept[i] + innovations[b, t - p, i]
                    for j in range(p):
                        for k in range(d):
                            acc += coefs[j, i, k] * path[b, t - 1 - j, k]
                    path[b, t, i] = acc

    _HAVE_NUMBA = True

    def _warm_var_kernel() -> None:
        """Trigger one-time JIT compilation off the hot path (registered warm-up)."""
        coefs = np.zeros((1, 1, 1), dtype=np.float64)
        intercept = np.zeros(1, dtype=np.float64)
        path = np.zeros((1, 2, 1), dtype=np.float64)
        innovations = np.zeros((1, 1, 1), dtype=np.float64)
        _var_recurrence_numba(coefs, intercept, path, innovations, 1, 1)

    register_warmup(_warm_var_kernel)
except ImportError:  # pragma: no cover - exercised only without the accel extra
    _HAVE_NUMBA = False


def simulate_var(
    coefs: NDArray[np.float64],
    intercept: NDArray[np.float64],
    init: NDArray[np.float64],
    innovations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Simulate one VAR path.

    Parameters
    ----------
    coefs : (p, d, d) array
        Coefficient matrices A_1 .. A_p.
    intercept : (d,) array
        Constant vector c.
    init : (p, d) array
        Initial vector observations X_0 .. X_{p-1}.
    innovations : (m, d) array
        Resampled innovation rows e* for the m generated steps.

    Returns
    -------
    (p + m, d) array
        The initial rows followed by the generated rows.
    """
    p, d, _ = coefs.shape
    m = innovations.shape[0]
    path = np.empty((p + m, d), dtype=np.float64)
    path[:p] = init
    for t in range(p, p + m):
        acc = intercept.copy()
        for j in range(p):
            acc += coefs[j] @ path[t - 1 - j]
        path[t] = acc + innovations[t - p]
    return path


def _var_recurrence_numpy(
    coefs: NDArray[np.float64],
    intercept: NDArray[np.float64],
    path: NDArray[np.float64],
    innovations: NDArray[np.float64],
    p: int,
    m: int,
) -> None:
    """Pure-numpy fallback: carry the B dimension with batched ``(B, d) @ (d, d)`` matmuls."""
    coefs_t = [np.ascontiguousarray(coefs[j].T) for j in range(p)]
    for t in range(p, p + m):
        acc = intercept + innovations[:, t - p]
        for j in range(p):
            acc = acc + path[:, t - 1 - j] @ coefs_t[j]
        path[:, t] = acc


def simulate_var_batched(
    coefs: NDArray[np.float64],
    intercept: NDArray[np.float64],
    inits: NDArray[np.float64],
    innovations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Simulate ``B`` VAR paths at once. ``inits`` is ``(B, p, d)``, ``innovations`` ``(B, m, d)``.

    Uses the compiled kernel when numba (the ``[accel]`` extra) is installed, else a
    pure-numpy batched recursion. Both are deterministic; they agree to within a few ULPs.
    """
    inits = np.ascontiguousarray(inits, dtype=np.float64)
    innovations = np.ascontiguousarray(innovations, dtype=np.float64)
    coefs = np.ascontiguousarray(coefs, dtype=np.float64)
    intercept = np.ascontiguousarray(intercept, dtype=np.float64)
    n_paths, p, d = inits.shape
    m = innovations.shape[1]
    path = np.empty((n_paths, p + m, d), dtype=np.float64)
    path[:, :p] = inits
    if _HAVE_NUMBA:
        _var_recurrence_numba(coefs, intercept, path, innovations, p, m)
    else:
        _var_recurrence_numpy(coefs, intercept, path, innovations, p, m)
    return path


__all__ = ["simulate_var", "simulate_var_batched"]
