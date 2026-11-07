"""Fit autoregressive models and select the sieve order.

AR and VAR are fit by direct OLS (numpy only): this is the exact conditional
least-squares estimator the residual/sieve bootstrap theory assumes, so it needs no
optional dependency and is far faster than going through statsmodels. statsmodels is
required only for the ARIMA (MA / MLE) path and is imported lazily there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.errors import BackendError, Codes, InputDataError, MethodConfigError


@dataclass(frozen=True, slots=True)
class ARFit:
    """An estimated AR(p) model: ``x_t = c + sum_j phi_j x_{t-j} + e_t``."""

    order: int
    intercept: float
    ar_coefs: NDArray[np.float64]  # (p,)
    residuals: NDArray[np.float64]  # (n - p,) raw innovations (caller centers them)
    exog_coefs: NDArray[np.float64] | None = None  # (k,) coefficients on exogenous regressors


def _require_statsmodels() -> None:
    try:
        import statsmodels  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without statsmodels
        raise BackendError(
            "statsmodels is required for model-based (residual/sieve) bootstraps",
            code=Codes.BACKEND_NOT_INSTALLED,
            hint="Install the model extra: pip install 'tsbootstrap[models]'.",
        ) from exc


def _ols(design: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    """Least-squares solve (SVD) with a rank-deficiency guard.

    SVD (``lstsq``) is chosen over the normal equations because it does not square the
    condition number, important near a unit root. A rank-deficient design (a constant
    series or perfectly collinear regressors) makes the minimum-norm solution arbitrary,
    so we raise instead of silently fitting a hallucinated model.
    """
    beta, _residuals, rank, _singular = np.linalg.lstsq(design, target, rcond=None)
    if rank < design.shape[1]:
        raise InputDataError(
            f"design matrix is rank-deficient (rank {rank} < {design.shape[1]}); the series "
            "or exogenous regressors are perfectly collinear or constant",
            code=Codes.PERFECT_COLLINEARITY,
            context={"rank": int(rank), "n_params": int(design.shape[1])},
        )
    # lstsq's stub types the solution as floating[Any]; with float64 inputs it is
    # float64 at runtime, so narrow without a copy.
    return cast(NDArray[np.float64], beta)


def fit_ar(x: NDArray[np.float64], order: int, exog: NDArray[np.float64] | None = None) -> ARFit:
    """Fit an AR(``order``) model with an intercept and optional exogenous regressors by OLS."""
    series = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    n = series.shape[0]
    if order >= n:
        raise MethodConfigError(
            f"AR order {order} is too large for a series of length {n}",
            code=Codes.ORDER_TOO_LARGE,
            context={"order": order, "n": n},
        )
    p = order
    target = series[p:]
    # AR Hankel lag block via a zero-copy sliding window: row i, col c is series[i+p-1-c]
    # (lag c+1 of target row i), i.e. each length-p window reversed to most-recent-first.
    lags = np.lib.stride_tricks.sliding_window_view(series, p)[: n - p, ::-1]
    blocks = [np.ones((n - p, 1)), lags]
    if exog is not None:
        exog_arr = np.asarray(exog, dtype=np.float64)
        if exog_arr.ndim == 1:
            exog_arr = exog_arr.reshape(-1, 1)
        blocks.append(exog_arr[p:])
    # sliding_window_view is non-writeable and non-contiguous; restore a C-contiguous
    # design at the LAPACK boundary.
    design = np.ascontiguousarray(np.hstack(blocks))
    beta = _ols(design, target)
    intercept = float(beta[0])
    ar_coefs = np.ascontiguousarray(beta[1 : 1 + p])
    exog_coefs = None if exog is None else np.ascontiguousarray(beta[1 + p :])
    residuals = np.ascontiguousarray(target - design @ beta)
    return ARFit(
        order=order,
        intercept=intercept,
        ar_coefs=ar_coefs,
        residuals=residuals,
        exog_coefs=exog_coefs,
    )


# A candidate whose QR criterion lies within this relative distance of the QR minimum is
# re-scored by the per-order SVD solve, so a near tie is decided by the same arithmetic as
# a solve per order. QR and SVD residual sums agree to a few ulps times the condition
# number, far inside this band for any design the conditioning guard below admits.
_IC_TIE_RTOL = 1e-8
# The QR path is used only when the smallest singular value of the full lag design clears
# the rank threshold of numpy's lstsq (eps * rows * s_max) by this factor; nearer the
# threshold the per-order SVD path decides rank exactly as before.
_RANK_MARGIN = 1e3


def _ic_penalty(criterion: str, n_eff: int) -> float:
    if criterion == "aic":
        return 2.0
    if criterion == "hqic":
        return 2.0 * float(np.log(np.log(n_eff)))
    return float(np.log(n_eff))  # bic (default)


def _ic_by_ols(
    design: NDArray[np.float64], target: NDArray[np.float64], k: int, penalty: float
) -> float:
    """Information criterion of the leading ``k + 1`` columns by one SVD least-squares solve."""
    sub = design[:, : k + 1]
    beta = _ols(sub, target)
    resid = target - sub @ beta
    n_eff = target.shape[0]
    sigma2 = float(resid @ resid) / n_eff
    return n_eff * float(np.log(sigma2)) + penalty * (k + 1)


def _select_by_ols(
    design: NDArray[np.float64],
    target: NDArray[np.float64],
    orders: range | list[int],
    penalty: float,
) -> int:
    """First order (in ascending order) with the strictly smallest criterion, one solve each."""
    best_ic = np.inf
    best_order = orders[0]
    for k in orders:
        ic = _ic_by_ols(design, target, k, penalty)
        if ic < best_ic:
            best_ic = ic
            best_order = k
    return best_order


def select_ar_order(
    x: NDArray[np.float64],
    *,
    min_lag: int = 1,
    max_lag: int | None = None,
    criterion: str = "bic",
) -> int:
    """Select the AR order by an OLS information criterion (for the sieve bootstrap).

    Every candidate order is evaluated on the SAME sample (truncated to ``upper`` lags)
    so the criteria are comparable across orders. One Householder QR of the full lag
    design gives every nested residual sum of squares at once: with ``z = Q.T @ y``, the
    model on the leading ``k + 1`` columns leaves ``RSS_full + sum(z[k+1:] ** 2)``. Two
    guards keep the decision identical to one SVD solve per order: a design near rank
    deficiency takes the per-order path (which raises the same rank error), and candidates
    whose criterion ties the minimum within ``_IC_TIE_RTOL`` are re-scored by SVD.
    """
    series = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    n = series.shape[0]
    upper = max_lag if max_lag is not None else int(np.ceil(10 * np.log10(n)))
    upper = max(min_lag, min(upper, n // 2 - 1))
    target = series[upper:]
    n_eff = target.shape[0]
    penalty = _ic_penalty(criterion, n_eff)
    # Full design: intercept + lags 1..upper. Column j (j >= 1) is series[upper-j : n-j].
    lags = np.lib.stride_tricks.sliding_window_view(series, upper)[: n - upper, ::-1]
    full_design = np.empty((n_eff, upper + 1), dtype=np.float64)
    full_design[:, 0] = 1.0
    full_design[:, 1:] = lags
    orders = range(min_lag, upper + 1)

    q, r = np.linalg.qr(full_design, mode="reduced")
    sing = np.linalg.svd(r, compute_uv=False)
    rank_tol = float(np.finfo(np.float64).eps) * max(full_design.shape) * float(sing[0])
    if not float(sing[-1]) > _RANK_MARGIN * rank_tol:
        return _select_by_ols(full_design, target, orders, penalty)

    z = q.T @ target
    resid_full = target - q @ z
    rss_full = float(resid_full @ resid_full)
    # tail[k] = sum(z[k+1:] ** 2): the extra residual sum of the model on columns 0..k.
    z2 = z * z
    tail = np.concatenate((np.cumsum(z2[::-1])[::-1][1:], [0.0]))
    ks = np.arange(min_lag, upper + 1)
    rss = rss_full + tail[ks]
    with np.errstate(divide="ignore"):
        ic = n_eff * np.log(rss / n_eff) + penalty * (ks + 1)
    best = float(ic.min())
    if not np.isfinite(best):  # an exact fit (zero residual) has no finite criterion
        return _select_by_ols(full_design, target, orders, penalty)
    band = _IC_TIE_RTOL * max(1.0, abs(best))
    contenders = [int(k) for k, v in zip(ks, ic, strict=True) if v - best <= band]
    if len(contenders) == 1:
        return contenders[0]
    return _select_by_ols(full_design, target, contenders, penalty)


@dataclass(frozen=True, slots=True)
class VARFit:
    """An estimated VAR(p): ``X_t = c + sum_j A_j X_{t-j} + e_t`` (vector form)."""

    order: int
    intercept: NDArray[np.float64]  # (d,)
    coefs: NDArray[np.float64]  # (p, d, d)
    residuals: NDArray[np.float64]  # (n - p, d) vector innovations (caller centers them)
    exog_coefs: NDArray[np.float64] | None = None  # (k, d) coefficients on exogenous regressors


def fit_var(
    data: NDArray[np.float64], order: int, exog: NDArray[np.float64] | None = None
) -> VARFit:
    """Fit a VAR(``order``) with an intercept and optional exogenous regressors by multivariate OLS.

    With ``exog`` this is a VARX (``X_t = c + sum_j A_j X_{t-j} + B z_t + e_t``), still a
    linear model, so plain multivariate OLS, not VARMAX (no moving-average term).
    """
    arr = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise MethodConfigError(
            "VAR requires a multivariate series of shape (n, d) with d >= 2",
            code=Codes.VAR_REQUIRES_MULTIVARIATE,
        )
    n, d = arr.shape
    if order * d >= n:
        raise MethodConfigError(
            f"VAR order {order} is too large for shape ({n}, {d})",
            code=Codes.ORDER_TOO_LARGE,
            context={"order": order, "n": n, "d": d},
        )
    p = order
    target = arr[p:]  # (n - p, d)
    columns = [np.ones((n - p, 1)), *(arr[p - j : n - j, :] for j in range(1, p + 1))]
    if exog is not None:
        exog_arr = np.ascontiguousarray(np.asarray(exog, dtype=np.float64))
        if exog_arr.ndim == 1:
            exog_arr = exog_arr.reshape(-1, 1)
        columns.append(exog_arr[p:])
    design = np.column_stack(columns)  # (n - p, 1 + p*d [+ k])
    beta = _ols(design, target)
    intercept = np.ascontiguousarray(beta[0])  # (d,)
    # coefs[j] maps lag (j+1) to the response; transpose to match simulate_var_batched,
    # which forms path[:, t-1-j] @ coefs[j].T.
    coefs = np.ascontiguousarray(
        np.stack([beta[1 + j * d : 1 + (j + 1) * d, :].T for j in range(p)])
    )
    exog_coefs = None if exog is None else np.ascontiguousarray(beta[1 + p * d :])  # (k, d)
    residuals = np.ascontiguousarray(target - design @ beta)  # (n - p, d)
    return VARFit(
        order=order, intercept=intercept, coefs=coefs, residuals=residuals, exog_coefs=exog_coefs
    )


__all__ = ["ARFit", "VARFit", "fit_ar", "fit_var", "select_ar_order"]
