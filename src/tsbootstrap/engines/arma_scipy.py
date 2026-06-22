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
    a = np.empty(p + 1, dtype=np.float64)
    a[0] = 1.0
    a[1:] = -np.asarray(ar_coefs, dtype=np.float64)
    b = np.array([1.0])
    forcing = intercept + innovations
    zi = lfiltic(b, a, np.asarray(init, dtype=np.float64)[::-1])
    generated, _ = lfilter(b, a, forcing, zi=zi)
    return np.concatenate([np.asarray(init, dtype=np.float64), generated])


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
    a = np.empty(p + 1, dtype=np.float64)
    a[0] = 1.0
    a[1:] = -np.asarray(ar_coefs, dtype=np.float64)
    b = np.array([1.0])
    forcing = intercept + innovations
    # For an all-pole filter, lfiltic reduces to zi[:, m] = -sum_k a[m+1+k] * rev[:, k]
    # (rev = init values, most-recent-first). Computing it as a numpy axis-1 reduction
    # matches lfiltic's own pairwise sum bit-for-bit (verified) while replacing the
    # per-path Python loop with p vectorized reductions.
    rev = inits[:, ::-1]
    zi = np.empty((n_paths, p), dtype=np.float64)
    for m in range(p):
        zi[:, m] = -np.sum(a[m + 1 : p + 1] * rev[:, : p - m], axis=1)
    generated, _ = lfilter(b, a, forcing, axis=1, zi=zi)
    return np.concatenate([inits, generated], axis=1)


def simulate_arma_batched(
    ar_coefs: NDArray[np.float64],
    ma_coefs: NDArray[np.float64],
    innovations: NDArray[np.float64],
    *,
    init_state: NDArray[np.float64] | None = None,
    init_values: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Simulate ``B`` zero-mean ARMA paths. ``innovations`` is ``(B, m)``.

    Zero-state by default. Pass ``init_state`` (the lfilter ``zi`` from
    :func:`tsbootstrap.model.arima.arma_initial_state`) and ``init_values`` (the observed
    initial differenced values) to condition the simulation on the observed initial state -
    the observed values are prepended to the output, so the result is ``(B, len(init_values) + m)``.
    """
    if (init_state is None) != (init_values is None):
        # Paired precondition: one without the other would either raise an opaque numpy error
        # (init_state set, init_values None) or silently fall back to the zero-state path and
        # ignore the supplied init_values (init_values set, init_state None).
        raise ValueError("init_state and init_values must be provided together (both or neither)")
    e = np.ascontiguousarray(innovations, dtype=np.float64)
    a = np.concatenate([[1.0], -np.asarray(ar_coefs, dtype=np.float64)])
    b = np.concatenate([[1.0], np.asarray(ma_coefs, dtype=np.float64)])
    k = max(len(a), len(b)) - 1
    if init_state is None:
        zi = np.zeros((e.shape[0], k), dtype=np.float64)
        out, _ = lfilter(b, a, e, axis=1, zi=zi)
        return out
    zi = np.broadcast_to(np.asarray(init_state, dtype=np.float64), (e.shape[0], k)).copy()
    tail, _ = lfilter(b, a, e, axis=1, zi=zi)
    init = np.asarray(init_values, dtype=np.float64)
    init_b = np.broadcast_to(init, (e.shape[0], init.shape[0]))
    return np.concatenate([init_b, tail], axis=1)


# Internal engine module: the public surface is tsbootstrap.bootstrap. These simulators are
# imported explicitly by the executors, the forecast/UQ layer, and the property tests.
__all__: list[str] = []
