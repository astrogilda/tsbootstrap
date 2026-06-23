"""Adaptive and nonexchangeable conformal calibration for distribution shift.

Base EnbPI (and any fixed-calibration conformal method) assumes the calibration
residuals are representative of the test residuals. Under distribution shift or
volatility clustering that fails, and intervals silently under- or over-cover. These
two methods adapt the calibration to recent behaviour and compose on the per-replicate
nonconformity scores produced by the bootstrap (e.g. via :func:`bootstrap_reduce`):

- :func:`aci_halfwidths`, Adaptive Conformal Inference (Gibbs & Candès 2021): adapt the
  quantile *level* online from realized coverage errors, so long-run coverage tracks the
  target even when the score distribution drifts.
- :func:`nexcp_quantile`, Nonexchangeable Conformal Prediction (Barber et al. 2023): a
  recency-weighted quantile of the scores, so recent residuals dominate the interval.

Coverage is approximate / long-run under temporal dependence, not finite-sample
distribution-free, consistent with the rest of the UQ layer.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def aci_halfwidths(
    calibration_scores: object,
    test_scores: object,
    *,
    alpha: float = 0.1,
    gamma: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Adaptive Conformal Inference: online-adapted interval half-widths.

    Parameters
    ----------
    calibration_scores : array-like, shape (m,)
        Nonconformity scores (e.g. ``|residual|``) from the bootstrap calibration set.
    test_scores : array-like, shape (T,)
        Realized scores ``|y_t - prediction_t|`` over the test sequence, in time order.
    alpha : float
        Target miscoverage (interval target coverage is ``1 - alpha``).
    gamma : float
        Adaptation step size. ``gamma = 0`` recovers static conformal.

    Returns
    -------
    halfwidths : ndarray, shape (T,)
        Interval half-width ``q_t`` to use at each step (``prediction_t ± q_t``).
    alphas : ndarray, shape (T,)
        The adapted miscoverage level used at each step.

    The update is ``alpha_{t+1} = alpha_t + gamma * (alpha - err_t)`` with
    ``err_t = 1`` when step ``t`` is miscovered: a miss shrinks the level and widens the
    next interval. Coverage converges to ``1 - alpha`` regardless of how the scores drift.
    """
    cal = np.asarray(calibration_scores, dtype=np.float64).ravel()  # ravel yields contiguous 1-D
    test = np.asarray(test_scores, dtype=np.float64).ravel()
    if cal.size == 0:
        raise ValueError("calibration_scores must be non-empty")

    a_t = float(alpha)
    halfwidths = np.empty(test.shape[0], dtype=np.float64)
    alphas = np.empty(test.shape[0], dtype=np.float64)
    # Sort the calibration scores once; each per-step quantile is then a constant-time
    # linear interpolation over the sorted array. This reproduces numpy's
    # ``np.quantile(cal, p, method="linear")`` bit-for-bit, including its two-branch
    # lerp (``a + (b - a) * t`` for ``t < 0.5``; ``b - (b - a) * (1 - t)`` otherwise).
    cs = np.sort(cal)
    last = cs.shape[0] - 1
    cs_last = float(cs[last])
    for t in range(test.shape[0]):
        a_clip = min(max(a_t, 0.0), 1.0)
        if a_clip <= 0.0:
            q = np.inf  # cover everything
        elif a_clip >= 1.0:
            q = 0.0
        else:
            h = (1.0 - a_clip) * last
            lo = int(np.floor(h))
            if lo >= last:
                q = cs_last
            else:
                frac = h - lo
                a = float(cs[lo])
                b = float(cs[lo + 1])
                diff = b - a
                q = b - diff * (1.0 - frac) if frac >= 0.5 else a + diff * frac
        halfwidths[t] = q
        alphas[t] = a_clip
        err = 1.0 if test[t] > q else 0.0
        a_t = a_t + gamma * (alpha - err)
    return halfwidths, alphas


def nexcp_quantile(scores: object, *, alpha: float = 0.1, decay: float = 0.99) -> float:
    """Recency-weighted (nonexchangeable) conformal quantile of the scores.

    Weights score ``i`` (0 = oldest, last = most recent) by ``decay ** (n - 1 - i)`` and
    returns the smallest score whose normalized weighted CDF reaches ``1 - alpha``. With
    ``decay = 1`` this is the ordinary empirical quantile; smaller ``decay`` puts more
    weight on recent residuals, widening the interval when recent volatility rises.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()  # ravel yields a contiguous 1-D
    n = s.shape[0]
    if n == 0:
        raise ValueError("scores must be non-empty")
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    weights = decay ** np.arange(n - 1, -1, -1, dtype=np.float64)
    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    cdf = np.cumsum(weights[order])
    # Compare the unnormalized weighted CDF against (1 - alpha) * total rather than
    # normalizing the weights first: the searchsorted boundary is the same and one pass
    # over the array is saved.
    target = (1.0 - alpha) * cdf[n - 1]
    idx = int(np.searchsorted(cdf, target, side="left"))
    return float(s_sorted[min(idx, n - 1)])


__all__ = ["aci_halfwidths", "nexcp_quantile"]
