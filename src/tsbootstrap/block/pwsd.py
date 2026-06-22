"""Automatic block-length selection.

Implements the Politis & White (2004) spectral-density plug-in selector with the
Patton, Politis & White (2009) correction. The optimal (long-run-variance MSE
minimising) block length is

    b_opt = (2 * g^2 / d * n) ** (1/3)

with d = 2 * sigma^4 for the stationary bootstrap and (4/3) * sigma^4 for the
circular bootstrap, where g and sigma are flat-top-lag-window estimates of
sum |k| R(k) and the long-run variance. The tuning lag m is chosen as the first
lag past which k_n consecutive autocorrelations all fall inside the conservative
band +/- 2 * sqrt(log10(n) / n).

References
----------
Politis, D. N. & White, H. (2004). Automatic Block-Length Selection for the
Dependent Bootstrap. Econometric Reviews 23(1), 53-70.
Patton, A., Politis, D. N. & White, H. (2009). Correction. Econometric Reviews 28(4), 372-375.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.errors import Codes, MethodConfigError
from tsbootstrap.methods import BlockLength


def _flat_top_kernel(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Politis-Romano flat-top lag window: 1 for |x|<1/2, 2(1-|x|) for 1/2<=|x|<=1, else 0."""
    ax = np.abs(x)
    return np.where(ax <= 0.5, 1.0, np.where(ax <= 1.0, 2.0 * (1.0 - ax), 0.0))


def _autocovariances(x: NDArray[np.float64], max_lag: int) -> NDArray[np.float64]:
    """Biased sample autocovariances R(0..max_lag) (divided by n)."""
    n = x.shape[0]
    xc = x - x.mean()
    acv = np.empty(max_lag + 1, dtype=np.float64)
    for k in range(max_lag + 1):
        acv[k] = float(np.dot(xc[: n - k], xc[k:])) / n
    return acv


def _select_m(x: NDArray[np.float64], n: int) -> int:
    """Choose the tuning lag m via Politis' adaptive rule (c=2)."""
    kn = int(max(5, np.sqrt(np.log10(n))))
    m_max = int(np.ceil(np.sqrt(n))) + kn
    m_max = min(m_max, n - 1)
    if m_max < 1:
        return 1
    acv = _autocovariances(x, m_max)
    if acv[0] <= 0.0:
        return 1
    rho = acv[1:] / acv[0]
    crit = 2.0 * np.sqrt(np.log10(n) / n)
    insignificant = np.abs(rho) < crit

    m_hat: int | None = None
    for g in range(rho.shape[0] - kn + 1):
        if insignificant[g : g + kn].all():
            m_hat = g + 1  # lag of rho[g]
            break
    if m_hat is None:
        significant = np.flatnonzero(~insignificant)
        m_hat = int(significant[-1] + 1) if significant.size else 1

    return max(1, min(2 * m_hat, m_max))


def _pwsd_1d(x: NDArray[np.float64]) -> tuple[float, float]:
    """Optimal (stationary, circular) block lengths for a single series."""
    n = x.shape[0]
    if n < 4:
        return 1.0, 1.0
    m = _select_m(x, n)
    acv = _autocovariances(x, m)
    ks = np.arange(-m, m + 1)
    acv_sym = acv[np.abs(ks)]
    w = _flat_top_kernel(ks / m)
    g_hat = float(np.sum(w * np.abs(ks) * acv_sym))
    sigma_sq = float(np.sum(w * acv_sym))  # flat-top long-run variance estimate
    b_max = float(np.ceil(min(3.0 * np.sqrt(n), n / 3.0)))

    def b_star(d: float) -> float:
        if d <= 0.0 or abs(g_hat) <= 1e-12:
            return 1.0
        b = (2.0 * g_hat**2 / d) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
        return float(min(max(b, 1.0), b_max))

    return b_star(2.0 * sigma_sq**2), b_star((4.0 / 3.0) * sigma_sq**2)


def optimal_block_length(
    arr: NDArray[np.float64],
    *,
    kind: str = "circular",
) -> int:
    """Automatic optimal block length for a ``(n, d)`` array.

    ``kind`` selects the target estimator: ``"stationary"`` uses the stationary
    bootstrap constant, anything else uses the circular/moving constant. For
    multivariate input the per-series optima are aggregated by the maximum
    (the most conservative, preserving the longest dependence), rounded up.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    n = a.shape[0]
    use_sb = kind == "stationary"
    best = 1.0
    for j in range(a.shape[1]):
        b_sb, b_cb = _pwsd_1d(np.ascontiguousarray(a[:, j]))
        best = max(best, b_sb if use_sb else b_cb)
    return int(max(1, min(int(np.ceil(best)), n)))


def resolve_block_length(value: BlockLength, arr: NDArray[np.float64], *, kind: str) -> int:
    """Resolve a spec's block length (an int or ``"auto"``) to a concrete int.

    Raises :class:`MethodConfigError` if an explicit length exceeds the series length.
    """
    n = int(arr.shape[0])
    if value == "auto":
        return optimal_block_length(arr, kind=kind)
    b = int(value)
    if b > n:
        raise MethodConfigError(
            f"block_length {b} exceeds series length {n}",
            code=Codes.BLOCK_LENGTH_GT_N,
            context={"block_length": b, "n": n},
        )
    return b


__all__ = ["optimal_block_length", "resolve_block_length"]
