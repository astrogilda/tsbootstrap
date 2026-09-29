"""Calibrators over a time-ordered out-of-bag residual buffer.

An EnbPI ensemble turns the calibration problem into a single question: given the
time-ordered out-of-bag absolute residuals, what half-width should each prediction
carry? Each calibrator below answers it differently:

- :func:`static_halfwidths`, one global ``1 - alpha`` quantile, the same width for
  every row. This is the original EnbPI behaviour and the simple default.
- :func:`sliding_window_halfwidths`, a rolling ``1 - alpha`` quantile over a trailing
  window of the residuals, so the width tracks local volatility (true time-local EnbPI,
  Xu & Xie 2021). This is the headline adaptive capability.

The drift-adaptive calibrators (Adaptive Conformal Inference and nonexchangeable /
recency-weighted quantiles) live in :mod:`tsbootstrap.uq.adaptive` as
:func:`~tsbootstrap.uq.adaptive.aci_halfwidths` and
:func:`~tsbootstrap.uq.adaptive.nexcp_quantile`; the ensemble delegates to them.

All calibrators are pure functions of their inputs (no hidden state, no RNG), so the
same residual buffer and parameters always yield the same widths. Coverage is
approximate / asymptotic under temporal dependence, not finite-sample
distribution-free, consistent with the rest of the UQ layer.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def static_halfwidths(
    residuals: NDArray[np.float64], n_rows: int, *, alpha: float = 0.1
) -> NDArray[np.float64]:
    """Constant half-width: the global ``1 - alpha`` quantile, broadcast to ``n_rows``.

    Parameters
    ----------
    residuals : ndarray, shape (m,)
        Time-ordered out-of-bag absolute residuals (the calibration scores).
    n_rows : int
        Number of prediction rows to emit a width for.
    alpha : float
        Target miscoverage; the interval target coverage is ``1 - alpha``.

    Returns
    -------
    ndarray, shape (n_rows,)
        The same scalar ``1 - alpha`` quantile repeated for every row.
    """
    res = np.asarray(residuals, dtype=np.float64).ravel()  # ravel already yields a contiguous 1-D
    if res.size == 0:
        raise ValueError("residuals must be non-empty")
    width = float(np.quantile(res, 1.0 - alpha))
    return np.full(n_rows, width, dtype=np.float64)


def sliding_window_halfwidths(
    residuals: NDArray[np.float64],
    n_rows: int,
    *,
    alpha: float = 0.1,
    window: int | None = None,
    start_index: int = 0,
) -> NDArray[np.float64]:
    """Prospective time-local half-widths from preceding calibration scores.

    For prediction row ``t``, use the most recent ``window`` scores strictly before
    ``start_index + t``. The default aligns predictions with the in-sample rows;
    ``start_index=len(residuals)`` predicts after the calibration period. A row with
    no preceding finite score has an undefined interval width (``nan``). This
    prevents its own realized error from affecting a prospective interval.

    Parameters
    ----------
    residuals : ndarray, shape (m,)
        Time-ordered out-of-bag absolute residuals. Missing out-of-bag scores may
        be ``nan``; they keep their time positions and are ignored within a window.
    n_rows : int
        Number of prediction rows to emit a width for.
    alpha : float
        Target miscoverage; the interval target coverage is ``1 - alpha``.
    window : int, optional
        Trailing window length. Defaults to ``min(len(residuals), 50)``.
    start_index : int, default 0
        Index of the first prediction relative to the calibration buffer. Values
        greater than ``m`` reuse the last observed window until new scores arrive.

    Returns
    -------
    ndarray, shape (n_rows,)
        Per-row half-width; ``nan`` where no preceding finite score exists.
    """
    res = np.asarray(residuals, dtype=np.float64).ravel()  # ravel already yields a contiguous 1-D
    m = res.size
    if m == 0:
        raise ValueError("residuals must be non-empty")
    if isinstance(n_rows, bool) or not isinstance(n_rows, (int, np.integer)) or n_rows < 0:
        raise ValueError("n_rows must be a non-negative integer")
    win = min(m, 50) if window is None else int(window)
    if win < 1:
        raise ValueError("window must be >= 1")
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, (int, np.integer))
        or start_index < 0
    ):
        raise ValueError("start_index must be a non-negative integer")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    if n_rows == 0:
        return np.empty(0, dtype=np.float64)

    q = 1.0 - alpha
    # Ends are exclusive and nondecreasing. Quantile the final training window
    # once even if a large out-of-sample horizon reuses it thousands of times.
    ends = np.minimum(start_index + np.arange(n_rows), m)
    unique_ends, inverse = np.unique(ends, return_inverse=True)
    unique_widths = np.empty(unique_ends.size, dtype=np.float64)
    ramp = unique_ends < win
    for i in np.flatnonzero(ramp):
        end = int(unique_ends[i])
        finite = res[:end][np.isfinite(res[:end])]
        unique_widths[i] = float(np.quantile(finite, q)) if finite.size else np.nan

    # A strided view avoids materializing all trailing windows. Bound each batch
    # to roughly 2 MiB of float64 window values before the quantile reduction.
    full = np.flatnonzero(~ramp)
    if full.size:
        view = np.lib.stride_tricks.sliding_window_view(res, win)
        batch_size = max(1, 262_144 // win)
        for offset in range(0, full.size, batch_size):
            positions = full[offset : offset + batch_size]
            windows = view[unique_ends[positions] - win]
            clean = np.all(np.isfinite(windows), axis=1)
            if np.any(clean):
                unique_widths[positions[clean]] = np.quantile(windows[clean], q, axis=1)
            for i, window_values in zip(positions[~clean], windows[~clean], strict=True):
                finite = window_values[np.isfinite(window_values)]
                unique_widths[i] = float(np.quantile(finite, q)) if finite.size else np.nan
    return unique_widths[inverse]


__all__ = ["static_halfwidths", "sliding_window_halfwidths"]
