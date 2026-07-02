"""Classical bootstrap confidence intervals over replicate statistics.

These are pure functions of a ``(B[, k])`` array of bootstrap replicate
statistics (e.g. the ``statistics`` field of a :func:`~tsbootstrap.bootstrap_reduce`
run). Each transforms the replicate distribution into a two-sided interval; none
holds hidden state or draws random numbers, so the same replicates and level
always yield the same bounds.

The interval families differ in what they assume and how fast their coverage
error shrinks:

- :func:`percentile_interval` and :func:`basic_interval` are first-order correct:
  the coverage error is ``O(1 / sqrt(n))``. They only need the replicate
  distribution and make no smoothness or symmetry assumption.
- :func:`studentized_interval` is second-order correct (``O(1 / n)``) for smooth
  statistics, but only when its per-replicate standard errors are themselves
  dependence-aware; with a block-jackknife SE (:func:`block_jackknife_se`) it
  stays valid under temporal dependence.
- :func:`bca_interval` corrects for bias and skewness through an acceleration
  constant. The delete-one jackknife acceleration (:func:`jackknife_acceleration`)
  is defined under independent sampling (Efron 1987); applying BCa to a
  dependent-data method is therefore gated to the IID specification at the
  orchestrator level, not in this module.

Coverage is approximate / asymptotic; these are not finite-sample
distribution-free guarantees, consistent with the rest of the UQ layer.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.errors import Codes, MethodConfigError


def _check_alpha(alpha: float) -> None:
    """Reject a miscoverage level outside the open interval ``(0, 1)``."""
    if not 0.0 < alpha < 1.0:
        raise MethodConfigError(
            f"alpha must be in the open interval (0, 1); got {alpha}",
            code=Codes.INVALID_PARAMETER,
            context={"alpha": alpha},
        )


def percentile_interval(
    statistics: NDArray[np.float64], *, alpha: float = 0.05
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Percentile interval: the ``alpha/2`` and ``1 - alpha/2`` replicate quantiles.

    Parameters
    ----------
    statistics : ndarray, shape (B,) or (B, k)
        Bootstrap replicate statistics, one row per replicate.
    alpha : float
        Target miscoverage; the interval target coverage is ``1 - alpha``.

    Returns
    -------
    lower, upper : ndarray, shape ``statistics.shape[1:]``
        The two-sided percentile bounds (0-d arrays for a scalar statistic). The
        quantiles use numpy's linear interpolation.
    """
    stats = np.asarray(statistics, dtype=np.float64)
    if stats.size == 0:
        raise ValueError("statistics must be non-empty")
    _check_alpha(alpha)
    lower = np.asarray(np.quantile(stats, alpha / 2.0, axis=0), dtype=np.float64)
    upper = np.asarray(np.quantile(stats, 1.0 - alpha / 2.0, axis=0), dtype=np.float64)
    return lower, upper


def basic_interval(
    statistics: NDArray[np.float64], theta_hat: NDArray[np.float64], *, alpha: float = 0.05
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Basic (reflected-percentile) interval around the point estimate.

    The percentile interval is reflected through ``theta_hat``: ``lower =
    2 * theta_hat - q_hi`` and ``upper = 2 * theta_hat - q_lo``, where ``q_lo`` and
    ``q_hi`` are the ``alpha/2`` and ``1 - alpha/2`` replicate quantiles. This
    corrects the percentile interval's bias when the replicate distribution is
    shifted relative to ``theta_hat``.

    Parameters
    ----------
    statistics : ndarray, shape (B,) or (B, k)
        Bootstrap replicate statistics, one row per replicate.
    theta_hat : ndarray, shape ``statistics.shape[1:]``
        The statistic evaluated on the original series.
    alpha : float
        Target miscoverage; the interval target coverage is ``1 - alpha``.

    Returns
    -------
    lower, upper : ndarray, shape ``statistics.shape[1:]``
        The two-sided basic-interval bounds.
    """
    stats = np.asarray(statistics, dtype=np.float64)
    if stats.size == 0:
        raise ValueError("statistics must be non-empty")
    _check_alpha(alpha)
    theta = np.asarray(theta_hat, dtype=np.float64)
    q_lo = np.asarray(np.quantile(stats, alpha / 2.0, axis=0), dtype=np.float64)
    q_hi = np.asarray(np.quantile(stats, 1.0 - alpha / 2.0, axis=0), dtype=np.float64)
    lower = np.asarray(2.0 * theta - q_hi, dtype=np.float64)
    upper = np.asarray(2.0 * theta - q_lo, dtype=np.float64)
    return lower, upper
