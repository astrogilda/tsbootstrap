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


__all__ = ["simulate_ar"]
