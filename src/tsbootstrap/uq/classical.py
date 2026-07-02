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

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr, ndtri

from tsbootstrap.errors import Codes, MethodConfigError

# A replicate reducer: ``(values, indices) -> scalar | array``. ``indices`` is the
# original-observation index vector for observation-resampling methods, or ``None``
# when the jackknife helpers operate on a raw array (mirrors the bootstrap_reduce
# statistic contract in tsbootstrap.api).
_Statistic = Callable[[NDArray[np.floating], NDArray[np.int32] | None], object]


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


def jackknife_statistics(x: NDArray[np.float64], statistic: _Statistic) -> NDArray[np.float64]:
    """Delete-one jackknife: the statistic recomputed on each leave-one-row-out sample.

    Parameters
    ----------
    x : ndarray, shape (n,) or (n, d)
        The original observations, one row per observation.
    statistic : callable ``(values, indices) -> scalar | array``
        The reducer to recompute on each leave-one-out sample. It is called with
        ``indices=None`` (the helper operates on a raw array with no resampling
        provenance), matching the :func:`~tsbootstrap.bootstrap_reduce` contract.

    Returns
    -------
    ndarray, shape ``(n,)`` or ``(n, k)``
        The ``n`` leave-one-out statistics stacked along axis 0.
    """
    arr = np.asarray(x, dtype=np.float64)
    n = arr.shape[0]
    thetas = [
        np.asarray(statistic(np.delete(arr, i, axis=0), None), dtype=np.float64) for i in range(n)
    ]
    return np.stack(thetas, axis=0)


def block_jackknife_se(
    values: NDArray[np.float64],
    statistic: _Statistic,
    *,
    block_length: int,
    indices: NDArray[np.int32] | None = None,
) -> NDArray[np.float64]:
    """Delete-a-group (block) jackknife standard error, Kunsch 1989.

    The rows are split into ``g = n // block_length`` non-overlapping blocks; the
    statistic is recomputed with each block deleted, and the standard error is
    ``sqrt((g - 1) / g * sum_j (theta_(j) - mean_j)^2)``. Deleting whole blocks rather
    than single rows keeps the estimate consistent under temporal dependence. With
    ``block_length=1`` this reduces exactly to the classic delete-one jackknife
    variance.

    Parameters
    ----------
    values : ndarray, shape (n,) or (n, d)
        The observations, one row per observation.
    statistic : callable ``(values, indices) -> scalar | array``
        The reducer to recompute on each block-deleted sample.
    block_length : int
        Number of consecutive rows per deleted block.
    indices : ndarray of int32, shape (n,), optional
        Original-observation indices to slice in lockstep with ``values``: when
        supplied, the same block of rows is removed from ``indices`` and passed to
        ``statistic`` alongside the deleted-block values. ``None`` (the default) passes
        ``None`` to the reducer.

    Returns
    -------
    ndarray, shape ``()`` or ``(k,)``
        The block-jackknife standard error per statistic component.
    """
    vals = np.asarray(values, dtype=np.float64)
    n = vals.shape[0]
    g = n // block_length
    if g < 2:
        raise MethodConfigError(
            "block jackknife needs at least 2 groups; "
            f"got n={n} and block_length={block_length} (g={g})",
            code=Codes.INVALID_PARAMETER,
            context={"n": n, "block_length": block_length, "groups": g},
        )
    idx = None if indices is None else np.asarray(indices)
    thetas = []
    for j in range(g):
        rows = np.arange(j * block_length, (j + 1) * block_length)
        vj = np.delete(vals, rows, axis=0)
        ij = None if idx is None else np.delete(idx, rows, axis=0)
        thetas.append(np.asarray(statistic(vj, ij), dtype=np.float64))
    theta = np.stack(thetas, axis=0)
    mean = theta.mean(axis=0)
    se_sq = (g - 1) / g * np.sum((theta - mean) ** 2, axis=0)
    return np.asarray(np.sqrt(se_sq), dtype=np.float64)


def studentized_interval(
    statistics: NDArray[np.float64],
    se_statistics: NDArray[np.float64],
    theta_hat: NDArray[np.float64],
    se_hat: NDArray[np.float64],
    *,
    alpha: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Studentized (bootstrap-t) interval from per-replicate standard errors.

    Each replicate is pivoted to ``t_b = (theta_b - theta_hat) / se_b``; the interval
    inverts the pivot, ``lower = theta_hat - t_{1 - alpha/2} * se_hat`` and
    ``upper = theta_hat - t_{alpha/2} * se_hat``. The pivot's upper quantile therefore
    sets the lower bound (the minus sign flips the orientation), which is what makes the
    interval second-order correct for smooth statistics when ``se_b`` and ``se_hat`` are
    dependence-aware (e.g. :func:`block_jackknife_se`).

    Parameters
    ----------
    statistics : ndarray, shape (B,) or (B, k)
        Bootstrap replicate statistics.
    se_statistics : ndarray, same shape as ``statistics``
        The per-replicate standard error of each statistic.
    theta_hat : ndarray, shape ``statistics.shape[1:]``
        The statistic on the original series.
    se_hat : ndarray, shape ``statistics.shape[1:]``
        The standard error of ``theta_hat`` on the original series.
    alpha : float
        Target miscoverage; the interval target coverage is ``1 - alpha``.

    Returns
    -------
    lower, upper : ndarray, shape ``statistics.shape[1:]``
        The two-sided studentized bounds.

    Raises
    ------
    ValueError
        If any per-replicate or point standard error is zero (the pivot is undefined).
    """
    stats = np.asarray(statistics, dtype=np.float64)
    se_b = np.asarray(se_statistics, dtype=np.float64)
    if stats.size == 0:
        raise ValueError("statistics must be non-empty")
    _check_alpha(alpha)
    theta = np.asarray(theta_hat, dtype=np.float64)
    se_h = np.asarray(se_hat, dtype=np.float64)
    if np.any(se_b == 0.0) or np.any(se_h == 0.0):
        raise ValueError(
            "studentized interval requires non-zero standard errors; got a zero se "
            "(supply a different se_statistic or a larger sample)"
        )
    t = (stats - theta) / se_b
    t_hi = np.asarray(np.quantile(t, 1.0 - alpha / 2.0, axis=0), dtype=np.float64)
    t_lo = np.asarray(np.quantile(t, alpha / 2.0, axis=0), dtype=np.float64)
    lower = np.asarray(theta - t_hi * se_h, dtype=np.float64)
    upper = np.asarray(theta - t_lo * se_h, dtype=np.float64)
    return lower, upper


def jackknife_acceleration(x: NDArray[np.float64], statistic: _Statistic) -> NDArray[np.float64]:
    """Efron's BCa acceleration constant from the delete-one jackknife.

    The acceleration is ``a = sum(d_i^3) / (6 * (sum(d_i^2))^{3/2})`` with
    ``d_i = mean(theta_jack) - theta_(i)`` the centred leave-one-out statistics
    (Efron 1987). It measures the skewness of the statistic's sampling distribution.
    Where the denominator is zero (a constant jackknife, e.g. a degenerate sample)
    the acceleration is defined as zero.

    Parameters
    ----------
    x : ndarray, shape (n,) or (n, d)
        The original observations, one row per observation.
    statistic : callable ``(values, indices) -> scalar | array``
        The reducer whose acceleration is estimated.

    Returns
    -------
    ndarray, shape ``()`` or ``(k,)``
        The acceleration per statistic component.
    """
    jack = jackknife_statistics(x, statistic)
    d = jack.mean(axis=0) - jack
    num = np.asarray(np.sum(d**3, axis=0), dtype=np.float64)
    den = np.asarray(6.0 * np.sum(d**2, axis=0) ** 1.5, dtype=np.float64)
    accel = np.divide(num, den, out=np.zeros_like(den), where=den != 0.0)
    return np.asarray(accel, dtype=np.float64)


def bca_interval(
    statistics: NDArray[np.float64],
    theta_hat: NDArray[np.float64],
    acceleration: NDArray[np.float64],
    *,
    alpha: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bias-corrected and accelerated (BCa) interval, Efron 1987.

    The two endpoint probability levels are adjusted for median bias (the
    bias-correction ``z0``) and skewness (the ``acceleration``) before reading the
    replicate quantiles. ``z0`` comes from the tie-adjusted fraction of replicates
    below ``theta_hat``, ``p0 = (#{theta_b < theta_hat} + 0.5 * #{theta_b == theta_hat})
    / B``. With ``z0 = 0`` and ``acceleration = 0`` the interval reduces exactly to
    :func:`percentile_interval`.

    This function is method-agnostic pure math: it takes a precomputed
    ``acceleration``. The jackknife acceleration (:func:`jackknife_acceleration`) is
    defined under independent sampling, so restricting BCa to the IID method spec is
    an orchestrator-level concern, not enforced here.

    Parameters
    ----------
    statistics : ndarray, shape (B,) or (B, k)
        Bootstrap replicate statistics.
    theta_hat : ndarray, shape ``statistics.shape[1:]``
        The statistic on the original series.
    acceleration : ndarray, shape ``statistics.shape[1:]``
        The precomputed acceleration constant per component.
    alpha : float
        Target miscoverage; the interval target coverage is ``1 - alpha``.

    Returns
    -------
    lower, upper : ndarray, shape ``statistics.shape[1:]``
        The two-sided BCa bounds.

    Raises
    ------
    ValueError
        If the bias-correction fraction ``p0`` is 0 or 1 for any component (``z0`` is
        infinite, so BCa is degenerate).
    """
    stats = np.asarray(statistics, dtype=np.float64)
    if stats.size == 0:
        raise ValueError("statistics must be non-empty")
    _check_alpha(alpha)
    theta = np.asarray(theta_hat, dtype=np.float64)
    accel = np.asarray(acceleration, dtype=np.float64)
    n_rep = stats.shape[0]
    p0 = (np.sum(stats < theta, axis=0) + 0.5 * np.sum(stats == theta, axis=0)) / n_rep
    if np.any(p0 <= 0.0) or np.any(p0 >= 1.0):
        raise ValueError(
            "BCa is degenerate: no replicate variation on one side of theta_hat "
            "(p0 is 0 or 1); use kind='percentile' or a larger sample"
        )
    z0 = ndtri(p0)
    z_lo = ndtri(alpha / 2.0)
    z_hi = ndtri(1.0 - alpha / 2.0)
    a1 = ndtr(z0 + (z0 + z_lo) / (1.0 - accel * (z0 + z_lo)))
    a2 = ndtr(z0 + (z0 + z_hi) / (1.0 - accel * (z0 + z_hi)))
    stats2d = stats.reshape(n_rep, -1)
    a1_flat = np.atleast_1d(np.asarray(a1, dtype=np.float64)).ravel()
    a2_flat = np.atleast_1d(np.asarray(a2, dtype=np.float64)).ravel()
    lower_flat = np.array(
        [np.quantile(stats2d[:, c], a1_flat[c]) for c in range(stats2d.shape[1])],
        dtype=np.float64,
    )
    upper_flat = np.array(
        [np.quantile(stats2d[:, c], a2_flat[c]) for c in range(stats2d.shape[1])],
        dtype=np.float64,
    )
    out_shape = stats.shape[1:]
    lower = np.asarray(lower_flat.reshape(out_shape), dtype=np.float64)
    upper = np.asarray(upper_flat.reshape(out_shape), dtype=np.float64)
    return lower, upper
