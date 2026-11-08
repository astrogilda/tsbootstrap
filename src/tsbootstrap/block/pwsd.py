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

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.errors import Codes, DegenerateBlockBootstrapWarning, MethodConfigError
from tsbootstrap.methods import (
    BlockLength,
    CircularBlock,
    MovingBlock,
    NonOverlappingBlock,
    StationaryBlock,
)


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


def _select_m(x: NDArray[np.float64], n: int) -> tuple[int, NDArray[np.float64]]:
    """Choose the tuning lag m via Politis' adaptive rule (c=2).

    Also returns the autocovariances ``R(0..m_max)`` it computes so the caller can reuse
    the prefix ``R(0..m)`` (``m <= m_max``) instead of recomputing the autocovariance pass.
    """
    kn = int(max(5, np.sqrt(np.log10(n))))
    m_max = int(np.ceil(np.sqrt(n))) + kn
    m_max = min(m_max, n - 1)
    if m_max < 1:
        return 1, _autocovariances(x, 1)
    acv = _autocovariances(x, m_max)
    if acv[0] <= 0.0:
        return 1, acv
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

    return max(1, min(2 * m_hat, m_max)), acv


def _pwsd_1d(x: NDArray[np.float64]) -> tuple[float, float]:
    """Optimal (stationary, circular) block lengths for a single series."""
    n = x.shape[0]
    if n < 4:
        return 1.0, 1.0
    m, acv_full = _select_m(x, n)
    acv = acv_full[: m + 1]  # m <= m_max, so the prefix is exactly R(0..m); avoids a second pass
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
) -> float:
    """Automatic optimal block length for a ``(n, d)`` array.

    ``kind`` selects the target estimator: ``"stationary"`` uses the stationary
    bootstrap constant, anything else uses the circular/moving constant. For
    multivariate input the per-series optima are aggregated by the maximum
    (the most conservative, preserving the longest dependence). The stationary mean
    length is returned as a real number; the fixed block length is rounded up to an int.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    # Fortran order once makes every column a contiguous view (_pwsd_1d only reads it),
    # replacing the per-column np.ascontiguousarray copy.
    a = np.asfortranarray(a)
    n = a.shape[0]
    use_sb = kind == "stationary"
    best = 1.0
    for j in range(a.shape[1]):
        b_sb, b_cb = _pwsd_1d(a[:, j])
        best = max(best, b_sb if use_sb else b_cb)
    if use_sb:  # the stationary mean block length is real-valued: no rounding
        return float(max(1.0, min(best, float(n))))
    return max(1, min(int(np.ceil(best)), n))


# Set only while the numpy panel path runs one series of a panel whose lengths were already
# resolved by resolve_panel_block_lengths: a series shortened to its own length there is
# degenerate by design (the panel rule kept it because other series are not), so the
# one-series resolution must not refuse it again.
_PANEL_SERIES: ContextVar[bool] = ContextVar("_PANEL_SERIES", default=False)


@contextmanager
def panel_series_resolution() -> Iterator[None]:
    """Run one already-resolved panel series without the all-degenerate refusal."""
    token = _PANEL_SERIES.set(True)
    try:
        yield
    finally:
        _PANEL_SERIES.reset(token)


def resolve_block_length(
    value: BlockLength, arr: NDArray[np.float64], *, kind: str, stacklevel: int = 3
) -> float:
    """Resolve a spec's block length (a number or ``"auto"``) for one series.

    The one-series case of :func:`resolve_panel_block_lengths`, with the same rule: an
    explicit length that makes the series degenerate raises ``BLOCK_LENGTH_GT_N``. The
    stationary bootstrap's mean length is real-valued; the fixed-length families return
    an integral value.
    """
    n = int(arr.shape[0])
    lengths = resolve_panel_block_lengths(
        value, arr, np.array([0, n], dtype=np.int64), kind=kind, stacklevel=stacklevel + 1
    )
    return float(lengths[0])


def resolve_panel_block_lengths(
    value: BlockLength,
    flat_data: NDArray[np.float64],
    indptr: NDArray[np.integer],
    *,
    kind: str,
    stacklevel: int = 2,
) -> NDArray[np.float64]:
    """Resolve one block length per series: the single rule every backend uses.

    ``"auto"`` is resolved on each series's own observations. An explicit length applies
    to every series. A series is degenerate when the length reaches its own length for the
    fixed-length block families (``kind="circular"``: the series is one block, so every
    replicate repeats it) or exceeds it for the stationary bootstrap. An explicit length
    that makes EVERY series degenerate raises ``BLOCK_LENGTH_GT_N``: the bootstrap would
    have no variability and every interval would have zero width. One that leaves at least
    one series non-degenerate shortens the others to their own length with ONE
    :class:`DegenerateBlockBootstrapWarning` naming them. The stationary mean length is
    real-valued; the fixed-length families get integers (as float64).

    Raises :class:`MethodConfigError` for an explicit length below 1, a non-integral
    fixed block length, or a length that leaves every series degenerate.
    """
    ptr = np.asarray(indptr, dtype=np.int64)
    num_series = int(ptr.size) - 1
    sizes = np.diff(ptr).astype(np.float64)
    stationary = kind == "stationary"
    if value == "auto":
        requested = np.array(
            [
                optimal_block_length(flat_data[int(ptr[s]) : int(ptr[s + 1])], kind=kind)
                for s in range(num_series)
            ],
            dtype=np.float64,
        )
    else:
        b = float(value)
        if not b >= 1.0 or (not stationary and b != int(b)):
            raise MethodConfigError(
                "block_length must be "
                + ("a real number of at least 1" if stationary else "a positive integer")
                + f"; got {value!r}",
                code=Codes.INVALID_PARAMETER,
                context={"block_length": b},
            )
        requested = np.full(num_series, b, dtype=np.float64)
    affected = requested > sizes if stationary else requested >= sizes
    if value != "auto" and num_series and bool(np.all(affected)) and not _PANEL_SERIES.get():
        raise MethodConfigError(
            f"block length {value} leaves no series with more than one possible block "
            f"(longest series {int(sizes.max())}); use a shorter block length",
            code=Codes.BLOCK_LENGTH_GT_N,
            context={"block_length": float(value), "n_max": int(sizes.max())},
        )
    if np.any(affected):
        slots = [int(s) for s in np.flatnonzero(affected)]
        warnings.warn(
            DegenerateBlockBootstrapWarning(
                f"the block length reaches the length of {len(slots)} series "
                f"(slots {slots}); each is resampled with at most its own length",
                context={
                    "series": slots,
                    "requested": [float(requested[s]) for s in slots],
                    "n": [int(sizes[s]) for s in slots],
                },
            ),
            stacklevel=stacklevel,
        )
    return np.minimum(requested, sizes)


def panel_method_block_lengths(
    method: object,
    flat_data: NDArray[np.float64],
    indptr: NDArray[np.integer],
    *,
    stacklevel: int = 3,
) -> tuple[str, NDArray[np.float64] | NDArray[np.int64]] | None:
    """Return ``(field, lengths)`` for a block method on a panel, or ``None`` for IID.

    ``field`` is the spec field the lengths fill (``"block_length"`` or
    ``"avg_block_length"``); ``lengths`` comes from :func:`resolve_panel_block_lengths`.
    """
    if isinstance(method, StationaryBlock):
        return "avg_block_length", resolve_panel_block_lengths(
            method.avg_block_length, flat_data, indptr, kind="stationary", stacklevel=stacklevel
        )
    if isinstance(method, (MovingBlock, CircularBlock, NonOverlappingBlock)):
        return "block_length", resolve_panel_block_lengths(
            method.block_length, flat_data, indptr, kind="circular", stacklevel=stacklevel
        ).astype(np.int64)
    return None


__all__ = [
    "optimal_block_length",
    "panel_method_block_lengths",
    "panel_series_resolution",
    "resolve_block_length",
    "resolve_panel_block_lengths",
]
