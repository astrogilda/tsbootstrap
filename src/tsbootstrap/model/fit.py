"""Fit autoregressive models and select the sieve order.

AR and VAR are fit by direct OLS (numpy only): this is the exact conditional
least-squares estimator the residual/sieve bootstrap theory assumes, so it needs no
optional dependency and is far faster than going through statsmodels. statsmodels is
required only for the ARIMA (MA / MLE) path and is imported lazily there.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    condition number — important near a unit root. A rank-deficient design (a constant
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
    return beta


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
    columns = [np.ones(n - p), *(series[p - j : n - j] for j in range(1, p + 1))]
    if exog is not None:
        exog_arr = np.ascontiguousarray(np.asarray(exog, dtype=np.float64))
        if exog_arr.ndim == 1:
            exog_arr = exog_arr.reshape(-1, 1)
        columns.extend(exog_arr[p:, k] for k in range(exog_arr.shape[1]))
    design = np.column_stack(columns)
    beta = _ols(design, target)
    intercept = float(beta[0])
    ar_coefs = np.ascontiguousarray(beta[1 : 1 + p])
    exog_coefs = None if exog is None else np.ascontiguousarray(beta[1 + p :])
    residuals = np.ascontiguousarray(target - design @ beta)
    return ARFit(
        order=order, intercept=intercept, ar_coefs=ar_coefs, residuals=residuals, exog_coefs=exog_coefs
    )


def select_ar_order(
    x: NDArray[np.float64],
    *,
    min_lag: int = 1,
    max_lag: int | None = None,
    criterion: str = "bic",
) -> int:
    """Select the AR order by an OLS information criterion (for the sieve bootstrap).

    Every candidate order is evaluated on the SAME sample (truncated to ``upper`` lags)
    so the criteria are comparable across orders.
    """
    series = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    n = series.shape[0]
    upper = max_lag if max_lag is not None else int(np.ceil(10 * np.log10(n)))
    upper = max(min_lag, min(upper, n // 2 - 1))
    target = series[upper:]
    n_eff = target.shape[0]
    if criterion == "aic":
        penalty = 2.0
    elif criterion == "hqic":
        penalty = 2.0 * float(np.log(np.log(n_eff)))
    else:  # bic (default)
        penalty = float(np.log(n_eff))
    best_ic = np.inf
    best_order = min_lag
    for k in range(min_lag, upper + 1):
        columns = [np.ones(n_eff), *(series[upper - j : n - j] for j in range(1, k + 1))]
        design = np.column_stack(columns)
        beta = _ols(design, target)
        resid = target - design @ beta
        sigma2 = float(resid @ resid) / n_eff
        ic = n_eff * float(np.log(sigma2)) + penalty * (k + 1)
        if ic < best_ic:
            best_ic = ic
            best_order = k
    return best_order


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

    With ``exog`` this is a VARX (``X_t = c + sum_j A_j X_{t-j} + B z_t + e_t``) — still a
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
    coefs = np.ascontiguousarray(np.stack([beta[1 + j * d : 1 + (j + 1) * d, :].T for j in range(p)]))
    exog_coefs = None if exog is None else np.ascontiguousarray(beta[1 + p * d :])  # (k, d)
    residuals = np.ascontiguousarray(target - design @ beta)  # (n - p, d)
    return VARFit(
        order=order, intercept=intercept, coefs=coefs, residuals=residuals, exog_coefs=exog_coefs
    )


__all__ = ["ARFit", "VARFit", "fit_ar", "fit_var", "select_ar_order"]
