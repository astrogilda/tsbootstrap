"""Recursive AR/ARMA simulation via ``scipy.signal.lfilter`` (compiled C).

The AR recursion ``X_t = c + sum_j phi_j X_{t-j} + e_t`` is the all-pole filter
with denominator ``a = [1, -phi_1, ..., -phi_p]`` driven by the forcing
``c + e_t``. Initial conditions are injected through the filter state (``lfiltic``)
so the generated path continues from the supplied initial values.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter, lfiltic


def simulate_ar(
    ar_coefs: NDArray[np.float64],
    intercept: float,
    init: NDArray[np.float64],
    innovations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Simulate one AR path.

    Parameters
    ----------
    ar_coefs : (p,) array
        AR coefficients phi_1 .. phi_p.
    intercept : float
        Constant term c.
    init : (p,) array
        Initial values X_0 .. X_{p-1}.
    innovations : (m,) array
        Resampled innovations e* for the m generated steps.

    Returns
    -------
    (p + m,) array
        The full path: the initial values followed by the generated values.
    """
    p = len(ar_coefs)
    innovations = np.ascontiguousarray(innovations, dtype=np.float64)
    if p == 0:
        return intercept + innovations
    a = np.empty(p + 1, dtype=np.float64)
    a[0] = 1.0
    a[1:] = -np.asarray(ar_coefs, dtype=np.float64)
    b = np.array([1.0])
    forcing = intercept + innovations
    zi = lfiltic(b, a, np.asarray(init, dtype=np.float64)[::-1])
    generated, _ = lfilter(b, a, forcing, zi=zi)
    return np.concatenate([np.asarray(init, dtype=np.float64), generated])


def simulate_arma(
    ar_coefs: NDArray[np.float64],
    ma_coefs: NDArray[np.float64],
    innovations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Simulate a zero-mean ARMA(p, q) driven by ``innovations`` from a zero state.

    ``phi(L) w_t = theta(L) e_t`` is the rational filter with denominator
    ``a = [1, -phi_1, ..., -phi_p]`` and numerator ``b = [1, theta_1, ..., theta_q]``.
    The initial state is zero, so the caller should prepend burn-in innovations and
    discard the corresponding outputs to remove the transient.
    """
    e = np.ascontiguousarray(innovations, dtype=np.float64)
    a = np.concatenate([[1.0], -np.asarray(ar_coefs, dtype=np.float64)])
    b = np.concatenate([[1.0], np.asarray(ma_coefs, dtype=np.float64)])
    out, _ = lfilter(b, a, e, zi=np.zeros(max(len(a), len(b)) - 1))
    return out


def simulate_ar_batched(
    ar_coefs: NDArray[np.float64],
    intercept: float,
    inits: NDArray[np.float64],
    innovations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Simulate ``B`` AR paths at once. ``inits`` is ``(B, p)``, ``innovations`` ``(B, m)``.

    Filtering along ``axis=1`` is a per-row sequential recurrence, so the result is
    bit-identical to ``B`` separate :func:`simulate_ar` calls.
    """
    inits = np.ascontiguousarray(inits, dtype=np.float64)
    innovations = np.ascontiguousarray(innovations, dtype=np.float64)
    n_paths, p = inits.shape
    if p == 0:
        return intercept + innovations
    a = np.empty(p + 1, dtype=np.float64)
    a[0] = 1.0
    a[1:] = -np.asarray(ar_coefs, dtype=np.float64)
    b = np.array([1.0])
    forcing = intercept + innovations
    zi = np.stack([lfiltic(b, a, inits[i, ::-1]) for i in range(n_paths)])
    generated, _ = lfilter(b, a, forcing, axis=1, zi=zi)
    return np.concatenate([inits, generated], axis=1)


def simulate_arma_batched(
    ar_coefs: NDArray[np.float64],
    ma_coefs: NDArray[np.float64],
    innovations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Simulate ``B`` zero-mean ARMA paths from a zero state. ``innovations`` is ``(B, m)``."""
    e = np.ascontiguousarray(innovations, dtype=np.float64)
    a = np.concatenate([[1.0], -np.asarray(ar_coefs, dtype=np.float64)])
    b = np.concatenate([[1.0], np.asarray(ma_coefs, dtype=np.float64)])
    zi = np.zeros((e.shape[0], max(len(a), len(b)) - 1), dtype=np.float64)
    out, _ = lfilter(b, a, e, axis=1, zi=zi)
    return out


__all__ = ["simulate_ar", "simulate_arma", "simulate_ar_batched", "simulate_arma_batched"]
