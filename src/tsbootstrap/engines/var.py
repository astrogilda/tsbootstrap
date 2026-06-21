"""Recursive VAR (vector autoregression) simulation.

A direct vector recursion: ``X_t = c + sum_j A_j X_{t-j} + e_t``. It is correct
and clear; a compiled kernel is a later performance concern, not a correctness
one.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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


def simulate_var_batched(
    coefs: NDArray[np.float64],
    intercept: NDArray[np.float64],
    inits: NDArray[np.float64],
    innovations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Simulate ``B`` VAR paths at once. ``inits`` is ``(B, p, d)``, ``innovations`` ``(B, m, d)``.

    The B dimension is carried by batched ``(B, d) @ (d, d)`` matmuls, so the time loop
    runs ``m`` times rather than ``B * m``. Because BLAS chooses its accumulation order
    by matrix shape, this matches the per-path recursion to within a few ULPs, not
    bit-for-bit; a golden-master regression test pins the batched output.
    """
    inits = np.ascontiguousarray(inits, dtype=np.float64)
    innovations = np.ascontiguousarray(innovations, dtype=np.float64)
    n_paths, p, d = inits.shape
    m = innovations.shape[1]
    path = np.empty((n_paths, p + m, d), dtype=np.float64)
    path[:, :p] = inits
    coefs_t = [np.ascontiguousarray(coefs[j].T) for j in range(p)]
    for t in range(p, p + m):
        acc = intercept + innovations[:, t - p]
        for j in range(p):
            acc = acc + path[:, t - 1 - j] @ coefs_t[j]
        path[:, t] = acc
    return path


__all__ = ["simulate_var", "simulate_var_batched"]
