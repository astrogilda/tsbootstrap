"""Calibrators over a time-ordered out-of-bag residual buffer.

An EnbPI ensemble turns the calibration problem into a single question: given the
time-ordered out-of-bag absolute residuals, what half-width should each prediction
carry? Each calibrator below answers it differently:

- :func:`static_halfwidths` — one global ``1 - alpha`` quantile, the same width for
  every row. This is the original EnbPI behaviour and the simple default.
- :func:`sliding_window_halfwidths` — a rolling ``1 - alpha`` quantile over a trailing
  window of the residuals, so the width tracks local volatility (true time-local EnbPI,
  Xu & Xie 2021). This is the headline adaptive capability.

The drift-adaptive calibrators (Adaptive Conformal Inference and nonexchangeable /
recency-weighted quantiles) live in :mod:`tsbootstrap.uq.adaptive` as
:func:`~tsbootstrap.uq.adaptive.aci_halfwidths` and
:func:`~tsbootstrap.uq.adaptive.nexcp_quantile`; the ensemble delegates to them.

All calibrators are pure functions of their inputs (no hidden state, no RNG), so the
same residual buffer and parameters always yield the same widths. Coverage is
approximate / asymptotic under temporal dependence, not finite-sample
distribution-free — consistent with the rest of the UQ layer.
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
    res = np.ascontiguousarray(np.asarray(residuals, dtype=np.float64).ravel())
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
) -> NDArray[np.float64]:
    """Time-local half-widths: a rolling ``1 - alpha`` quantile of the residuals.

    For row ``t`` the width is the ``1 - alpha`` quantile of the most recent ``window``
    residuals ending at ``t`` (the trailing window shrinks at the start of the series,
    where fewer residuals are available). The width therefore widens in high-volatility
    stretches and tightens in calm ones, which is the defining time-local mechanism of
    EnbPI (Xu & Xie 2021) and the static calibrator's missing piece.

    Parameters
    ----------
    residuals : ndarray, shape (m,)
        Time-ordered out-of-bag absolute residuals (the calibration scores).
    n_rows : int
        Number of prediction rows to emit a width for. Each row ``t`` uses the window
        of residuals ending at ``min(t, m - 1)``, so out-of-sample rows beyond the
        calibration set reuse the final trailing window.
    alpha : float
        Target miscoverage; the interval target coverage is ``1 - alpha``.
    window : int, optional
        Trailing window length. Defaults to ``min(len(residuals), 50)``.

    Returns
    -------
    ndarray, shape (n_rows,)
        Per-row half-width; non-constant whenever local volatility varies.
    """
    res = np.ascontiguousarray(np.asarray(residuals, dtype=np.float64).ravel())
    m = res.size
    if m == 0:
        raise ValueError("residuals must be non-empty")
    win = min(m, 50) if window is None else int(window)
    if win < 1:
        raise ValueError("window must be >= 1")

    widths = np.empty(n_rows, dtype=np.float64)
    for t in range(n_rows):
        end = min(t, m - 1)  # rows past the calibration set reuse the last window
        start = max(0, end - win + 1)
        widths[t] = float(np.quantile(res[start : end + 1], 1.0 - alpha))
    return widths


__all__ = ["static_halfwidths", "sliding_window_halfwidths"]
