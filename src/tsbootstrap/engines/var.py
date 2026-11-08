"""Recursive VAR (vector autoregression) simulation.

A direct vector recursion: ``X_t = c + sum_j A_j X_{t-j} + e_t``. The time axis is a
genuine sequential dependency, so it cannot be vectorised away; at large ``n`` the
per-step dispatch over small ``(B, d)`` arrays dominates. When the optional ``[accel]``
extra (numba) is installed, a compiled kernel runs the whole ``(B, n, d)`` recursion in
fused machine code and is selected automatically; otherwise a pure-numpy fallback is
used. Both engines accumulate each replicate in the same scalar order (intercept plus
innovation, then every lag ``j`` and component ``k`` in turn, one rounding per multiply and
per add), so their bytes are identical: installing numba changes speed, never results, and
neither engine depends on the chunk size or on ``B``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
    """Pure-numpy fallback in the compiled kernel's exact accumulation order.

    Vectorised over replicates only: for every step it forms ``c + e`` and then adds
    ``coefs[j, i, k] * x[k]`` for each lag ``j`` and component ``k`` in the kernel's order,
    with one rounding per multiply and per add. A BLAS matmul would sum in a shape-dependent
    order, which made the bytes depend on the engine, the chunk size and ``B``.
    """
    d = path.shape[2]
    acc = np.empty_like(innovations[:, 0])
    term = np.empty_like(acc)
    for t in range(p, p + m):
        np.add(intercept, innovations[:, t - p], out=acc)
        for j in range(p):
            prev = path[:, t - 1 - j]
            for k in range(d):
                np.multiply(coefs[j, :, k], prev[:, k : k + 1], out=term)
                acc += term
        path[:, t] = acc


def simulate_var_batched(
    coefs: NDArray[np.float64],
    intercept: NDArray[np.float64],
    inits: NDArray[np.float64],
    innovations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Simulate ``B`` VAR paths at once. ``inits`` is ``(B, p, d)``, ``innovations`` ``(B, m, d)``.

    Uses the compiled kernel when numba (the ``[accel]`` extra) is installed, else a
    pure-numpy batched recursion. The two are byte-identical (same accumulation order).
    """
    # The sole caller (recursive._var_batched via _draw_innovations_and_inits) always builds
    # inits/innovations fresh and C-contiguous, so np.asarray suffices; keep float64 coercion.
    inits = np.asarray(inits, dtype=np.float64)
    innovations = np.asarray(innovations, dtype=np.float64)
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
