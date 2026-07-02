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
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr, ndtri

from tsbootstrap.errors import Codes, MethodConfigError
from tsbootstrap.methods import (
    IID,
    BaseMethodSpec,
    StationaryBlock,
)
from tsbootstrap.rng import RandomStateLike

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
    values: NDArray[np.floating],
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
    # Standard errors are nonnegative by construction (sqrt of a sum of squares),
    # so a nonpositive value is exactly the degenerate zero case.
    if np.any(se_b <= 0.0) or np.any(se_h <= 0.0):
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
    # den is nonnegative by construction, so strict positivity is the exact
    # complement of the degenerate constant-jackknife case.
    accel = np.divide(num, den, out=np.zeros_like(den), where=den > 0.0)
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


# --------------------------------------------------------------------------- #
# One-call orchestrators.
# --------------------------------------------------------------------------- #
IntervalKind = Literal["percentile", "basic", "studentized", "bca"]

_INTERVAL_KINDS: tuple[str, ...] = ("percentile", "basic", "studentized", "bca")


def _check_kind(kind: str) -> None:
    if kind not in _INTERVAL_KINDS:
        raise MethodConfigError(
            f"unknown interval kind {kind!r}; available: {list(_INTERVAL_KINDS)}",
            code=Codes.INVALID_PARAMETER,
            context={"kind": kind},
        )


def _resolve_statistic(
    statistic: str | tuple[str, float] | _Statistic,
) -> _Statistic:
    """Resolve the statistic union to one callable for point/jackknife evaluation.

    Mirrors the resolution :func:`~tsbootstrap.bootstrap_reduce` performs, so the
    point estimate is computed by exactly the function each replicate sees.
    """
    from tsbootstrap.api import _BUILTIN_REDUCERS

    if isinstance(statistic, tuple):
        name, q = statistic
        if name != "quantile" or not 0.0 < float(q) < 1.0:
            raise MethodConfigError(
                "a tuple statistic must be ('quantile', q) with 0 < q < 1",
                code=Codes.INVALID_PARAMETER,
                context={"statistic": statistic},
            )
        _q = float(q)

        def _quantile(values: NDArray[np.floating], indices: NDArray[np.int32] | None) -> object:
            return np.quantile(values, _q, axis=0)

        return _quantile
    if isinstance(statistic, str):
        if statistic not in _BUILTIN_REDUCERS:
            raise MethodConfigError(
                f"unknown built-in reducer {statistic!r}; available: "
                f"{sorted(_BUILTIN_REDUCERS)} (the quantile reducer is the tuple ('quantile', q))",
                code=Codes.INVALID_PARAMETER,
                context={"statistic": statistic},
            )
        return _BUILTIN_REDUCERS[statistic]
    return statistic


def _acceleration_for(
    method: BaseMethodSpec, x0: NDArray[np.float64], stat_fn: _Statistic
) -> NDArray[np.float64]:
    """The BCa gate: acceleration for IID resampling, a typed refusal otherwise.

    The delete-one jackknife acceleration is defined under independent sampling.
    For dependent data the literature's second-order-correct route is the
    studentized block bootstrap, not BCa: the dependent-data analogue of BCa in
    Gotze and Kunsch (1996, Annals of Statistics 24(5), Section 3) is a different
    construction with block-cumulant acceleration and extra correction terms, and
    no block-jackknife BCa is established practice. This helper is the single
    place that policy lives, so revisiting it never touches the interval math.
    """
    if isinstance(method, IID):
        return jackknife_acceleration(x0, stat_fn)
    raise MethodConfigError(
        f"BCa intervals are only supported with the IID method spec; got "
        f"{type(method).__name__}. The acceleration constant is a delete-one jackknife "
        "estimate defined under independent observations (Efron 1987); for dependent "
        "data the second-order-correct route is the studentized interval "
        "(Gotze and Kunsch 1996).",
        code=Codes.UNSUPPORTED_MODEL_FEATURE,
        context={"method": type(method).__name__, "kind": "bca"},
        hint="Use kind='studentized' or kind='percentile' with this method, or "
        "method=IID() when the observations are exchangeable.",
    )


def _resolve_se_block_length(
    method: BaseMethodSpec,
    arr: NDArray[np.float64],
    se_block_length: int | None,
) -> int:
    """Pick the block length for the studentized path's block-jackknife SE.

    Priority: an explicit ``se_block_length``; 1 for IID (the classic delete-one
    jackknife); the method spec's own explicit integer block length; otherwise the
    Politis-White rule on the original series. Residual/sieve methods use the
    Politis-White length of the series too: their replicate paths carry the same
    dependence scale as the data they were fitted on.
    """
    from tsbootstrap.block.pwsd import optimal_block_length

    if se_block_length is not None:
        if se_block_length < 1:
            raise MethodConfigError(
                f"se_block_length must be >= 1; got {se_block_length}",
                code=Codes.INVALID_PARAMETER,
                context={"se_block_length": se_block_length},
            )
        return int(se_block_length)
    if isinstance(method, IID):
        return 1
    spec_length = getattr(method, "block_length", None)
    if spec_length is None:
        spec_length = getattr(method, "avg_block_length", None)
    if isinstance(spec_length, int):
        return spec_length
    kind = "stationary" if isinstance(method, StationaryBlock) else "circular"
    return optimal_block_length(arr, kind=kind)


def _point_indices(method: BaseMethodSpec, n: int) -> NDArray[np.int32] | None:
    """The ``indices`` the statistic sees when evaluated on the original series.

    Observation-resampling methods hand replicates an index vector, so the point
    evaluation gets the identity indices; recursive methods hand ``None``.
    """
    from tsbootstrap.methods import OBSERVATION_RESAMPLING

    if isinstance(method, OBSERVATION_RESAMPLING):
        return np.arange(n, dtype=np.int32)
    return None


def _reject_panel_input(X: object, function_name: str) -> None:
    # A ragged list of series or a 3-D array is a panel; guessing an axis here
    # would be a silent statistical error, so the contract is explicit.
    if isinstance(X, (list, tuple)):
        raise MethodConfigError(
            f"{function_name} takes one series; for a collection of series use conf_int_panel",
            code=Codes.INVALID_PARAMETER,
            hint="conf_int_panel accepts a list of per-series arrays or a flat array + indptr.",
        )
    arr = np.asarray(X)
    if arr.ndim >= 3:
        raise MethodConfigError(
            f"{function_name} takes a (n,) or (n, d) series; got {arr.ndim} dimensions. "
            "For a panel of series use conf_int_panel",
            code=Codes.INVALID_PARAMETER,
            context={"ndim": arr.ndim},
        )


def conf_int(
    X: object,
    statistic: str | tuple[str, float] | _Statistic,
    *,
    method: BaseMethodSpec,
    kind: IntervalKind = "percentile",
    alpha: float = 0.05,
    n_bootstraps: int = 999,
    random_state: RandomStateLike = None,
    backend: Literal["numpy", "compiled"] = "numpy",
    se_statistic: _Statistic | None = None,
    se_block_length: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Bootstrap confidence interval for a statistic of one series, in one call.

    Runs a single :func:`~tsbootstrap.bootstrap_reduce` pass with ``method`` and
    reads the requested interval from the replicate statistics. To reuse an
    existing run instead, call the interval functions directly on its
    ``statistics`` array (e.g. ``percentile_interval(result.statistics)``).

    Parameters
    ----------
    X : array-like, shape (n,) or (n, d)
        The observed series (any input :func:`~tsbootstrap.bootstrap` accepts).
    statistic : str, ("quantile", q) tuple, or callable ``(values, indices) -> theta``
        The statistic to bootstrap. Built-in names (``"mean"``, ``"var"``, ``"std"``)
        are required for ``backend="compiled"``.
    method : BaseMethodSpec
        Any method spec. BCa additionally requires ``IID`` (see below).
    kind : {"percentile", "basic", "studentized", "bca"}
        The interval family. ``studentized`` computes a dependence-aware
        per-replicate standard error via :func:`block_jackknife_se` (or
        ``se_statistic``); ``bca`` is available for the ``IID`` spec only, because
        its jackknife acceleration is defined under independent sampling
        (Efron 1987; for dependent data use ``studentized``, the
        second-order-correct route of Gotze and Kunsch 1996).
    alpha : float
        Target miscoverage; the interval targets ``1 - alpha`` coverage.
    n_bootstraps : int
        Number of bootstrap replicates.
    random_state : int, Generator, SeedSequence, or None
        Seeding, with the library's per-replicate determinism contract.
    backend : {"numpy", "compiled"}
        ``"compiled"`` accelerates ``percentile``/``basic`` with a built-in string
        statistic; ``studentized``/``bca`` need Python callables per replicate and
        raise a typed error under the compiled backend.
    se_statistic : callable, optional
        Override for the per-replicate standard-error estimator (studentized only).
    se_block_length : int, optional
        Override for the block-jackknife block length (studentized only).

    Returns
    -------
    lower, upper, point : ndarray
        Interval bounds and the statistic on the original series, each shaped like
        one replicate's statistic (0-d for a scalar statistic).
    """
    from tsbootstrap.api import bootstrap_reduce
    from tsbootstrap.validation import coerce_observations

    _check_kind(kind)
    _check_alpha(alpha)
    if backend == "compiled" and kind in ("studentized", "bca"):
        raise MethodConfigError(
            f"backend='compiled' supports kind='percentile' and kind='basic'; "
            f"kind={kind!r} evaluates a Python callable per replicate, which the "
            "compiled backend cannot run",
            code=Codes.INVALID_PARAMETER,
            context={"kind": kind, "backend": backend},
            hint="Use backend='numpy' for studentized or BCa intervals.",
        )
    _reject_panel_input(X, "conf_int")
    stat_fn = _resolve_statistic(statistic)

    arr, was_1d = coerce_observations(X)
    x0 = arr[:, 0] if was_1d else arr
    n = int(arr.shape[0])
    ident = _point_indices(method, n)
    theta_hat = np.asarray(stat_fn(x0, ident), dtype=np.float64)

    acceleration: NDArray[np.float64] | None = None
    if kind == "bca":
        # Gate before the bootstrap run: an unsupported method should fail fast.
        acceleration = _acceleration_for(method, x0, stat_fn)

    if kind == "studentized":
        ell = _resolve_se_block_length(method, arr, se_block_length)
        se_fn: _Statistic = se_statistic or (
            lambda values, indices: block_jackknife_se(
                values, stat_fn, block_length=ell, indices=indices
            )
        )
        se_hat = np.asarray(se_fn(x0, ident), dtype=np.float64)
        k = int(np.atleast_1d(theta_hat).size)

        def _theta_and_se(
            values: NDArray[np.floating], indices: NDArray[np.int32] | None
        ) -> object:
            th = np.atleast_1d(np.asarray(stat_fn(values, indices), dtype=np.float64))
            se = np.atleast_1d(np.asarray(se_fn(values, indices), dtype=np.float64))
            return np.concatenate([th.ravel(), se.ravel()])

        result = bootstrap_reduce(
            X,
            method=method,
            statistic=_theta_and_se,
            n_bootstraps=n_bootstraps,
            random_state=random_state,
        )
        stats = _require_statistics(result)
        theta_b = stats[:, :k].reshape((stats.shape[0], *np.shape(theta_hat)))
        se_b = stats[:, k:].reshape((stats.shape[0], *np.shape(theta_hat)))
        lower, upper = studentized_interval(theta_b, se_b, theta_hat, se_hat, alpha=alpha)
        return lower, upper, theta_hat

    result = bootstrap_reduce(
        X,
        method=method,
        statistic=statistic,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
        backend=backend,
    )
    stats = _require_statistics(result)
    if kind == "percentile":
        lower, upper = percentile_interval(stats, alpha=alpha)
    elif kind == "basic":
        lower, upper = basic_interval(stats, theta_hat, alpha=alpha)
    else:  # bca
        assert acceleration is not None  # noqa: S101  (kind == "bca" bound it above)
        lower, upper = bca_interval(stats, theta_hat, acceleration, alpha=alpha)
    return lower, upper, theta_hat


def _require_statistics(result: object) -> NDArray[np.float64]:
    stats = getattr(result, "statistics", None)
    if stats is None:
        reason = getattr(getattr(result, "metadata", None), "failure_reason", None)
        raise ValueError(f"bootstrap run produced no statistics: {reason}")
    return np.asarray(stats, dtype=np.float64)


def conf_int_panel(
    panel: object,
    statistic: str | tuple[str, float] | _Statistic,
    *,
    method: BaseMethodSpec,
    indptr: object = None,
    kind: Literal["percentile", "basic", "studentized"] = "percentile",
    alpha: float = 0.05,
    n_bootstraps: int = 999,
    random_state: RandomStateLike = None,
    backend: Literal["numpy", "compiled"] = "numpy",
    se_statistic: _Statistic | None = None,
    se_block_length: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per-series bootstrap confidence intervals over a ragged panel, in one pass.

    The panel counterpart of :func:`conf_int`, built on
    :func:`~tsbootstrap.bootstrap_reduce_panel` (observation-resampling methods
    only, matching that function's contract). Returns arrays with a leading
    ``num_series`` axis.

    BCa is not offered for panels: it is gated to IID data at the single-series
    level and a per-series jackknife acceleration sweep is deliberately out of
    scope. The studentized kind requires an explicit block length (either
    ``se_block_length`` or an integer block length on the method spec): replicate
    reducers see one series at a time without its identity, so a per-series
    automatic block length cannot be resolved honestly, and one Politis-White fit
    on a mixed panel would be statistically arbitrary.
    """
    from tsbootstrap.api import bootstrap_reduce_panel

    if kind not in ("percentile", "basic", "studentized"):
        raise MethodConfigError(
            f"conf_int_panel supports kind='percentile', 'basic', and 'studentized'; got {kind!r} "
            "(BCa is gated to IID single-series data; a per-series jackknife acceleration sweep "
            "is out of scope)",
            code=Codes.INVALID_PARAMETER,
            context={"kind": kind},
        )
    _check_alpha(alpha)
    if backend == "compiled" and kind == "studentized":
        raise MethodConfigError(
            "backend='compiled' supports kind='percentile' and kind='basic' for panels; "
            "kind='studentized' evaluates a Python callable per replicate",
            code=Codes.INVALID_PARAMETER,
            context={"kind": kind, "backend": backend},
            hint="Use backend='numpy' for studentized panel intervals.",
        )
    stat_fn = _resolve_statistic(statistic)

    series_list = _panel_series_list(panel, indptr)
    ident_per_series = [_point_indices(method, s.shape[0]) for s in series_list]
    theta_hat = np.stack(
        [
            np.atleast_1d(np.asarray(stat_fn(s, i), dtype=np.float64))
            for s, i in zip(series_list, ident_per_series, strict=True)
        ],
        axis=0,
    )  # (num_series, k)

    if kind == "studentized":
        ell = se_block_length
        if ell is None:
            spec_length = getattr(method, "block_length", None)
            if spec_length is None:
                spec_length = getattr(method, "avg_block_length", None)
            ell = spec_length if isinstance(spec_length, int) else None
        if ell is None and isinstance(method, IID):
            ell = 1
        if ell is None:
            raise MethodConfigError(
                "studentized panel intervals need an explicit block length: pass "
                "se_block_length or use a method spec with an integer block length "
                "(replicate reducers see one series at a time without its identity, so "
                "an automatic per-series block length cannot be resolved)",
                code=Codes.INVALID_PARAMETER,
                hint="For example se_block_length=10, or MovingBlock(block_length=10).",
            )
        se_fn: _Statistic = se_statistic or (
            lambda values, indices: block_jackknife_se(
                values, stat_fn, block_length=ell, indices=indices
            )
        )
        se_hat = np.stack(
            [
                np.atleast_1d(np.asarray(se_fn(s, i), dtype=np.float64))
                for s, i in zip(series_list, ident_per_series, strict=True)
            ],
            axis=0,
        )
        k = theta_hat.shape[1]

        def _theta_and_se(
            values: NDArray[np.floating], indices: NDArray[np.int32] | None
        ) -> object:
            th = np.atleast_1d(np.asarray(stat_fn(values, indices), dtype=np.float64))
            se = np.atleast_1d(np.asarray(se_fn(values, indices), dtype=np.float64))
            return np.concatenate([th.ravel(), se.ravel()])

        result = bootstrap_reduce_panel(
            panel,
            indptr=indptr,
            method=method,
            statistic=_theta_and_se,
            n_bootstraps=n_bootstraps,
            random_state=random_state,
        )
        stats = _require_statistics(result)  # (B, num_series, 2k)
        theta_b = stats[:, :, :k]
        se_b = stats[:, :, k:]
        lower, upper = studentized_interval(theta_b, se_b, theta_hat, se_hat, alpha=alpha)
        return _squeeze_panel(lower), _squeeze_panel(upper), _squeeze_panel(theta_hat)

    result = bootstrap_reduce_panel(
        panel,
        indptr=indptr,
        method=method,
        statistic=statistic,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
        backend=backend,
    )
    stats = _require_statistics(result)  # (B, num_series, k) or (B, num_series)
    if stats.ndim == 2:
        stats = stats[:, :, None]
    if kind == "percentile":
        lower, upper = percentile_interval(stats, alpha=alpha)
    else:
        lower, upper = basic_interval(stats, theta_hat, alpha=alpha)
    return _squeeze_panel(lower), _squeeze_panel(upper), _squeeze_panel(theta_hat)


def _panel_series_list(panel: object, indptr: object) -> list[NDArray[np.float64]]:
    """Materialise the per-series views of a panel (list form or flat + indptr)."""
    if indptr is None:
        if not isinstance(panel, (list, tuple)) or len(panel) == 0:
            raise MethodConfigError(
                "a panel without indptr must be a non-empty list of per-series arrays",
                code=Codes.INVALID_PARAMETER,
            )
        return [np.asarray(s, dtype=np.float64) for s in panel]
    flat = np.asarray(panel, dtype=np.float64)
    ptr = np.asarray(indptr)
    return [flat[int(ptr[i]) : int(ptr[i + 1])] for i in range(ptr.shape[0] - 1)]


def _squeeze_panel(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Drop a trailing singleton statistic axis: (S, 1) -> (S,)."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr
