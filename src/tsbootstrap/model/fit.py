"""Fit autoregressive models and select the sieve order (statsmodels reference).

statsmodels is imported lazily so the core package does not hard-require it; a
clear error is raised if a model-based bootstrap is requested without it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.errors import BackendError, Codes, MethodConfigError


@dataclass(frozen=True, slots=True)
class ARFit:
    """An estimated AR(p) model: ``x_t = c + sum_j phi_j x_{t-j} + e_t``."""

    order: int
    intercept: float
    ar_coefs: NDArray[np.float64]  # (p,)
    residuals: NDArray[np.float64]  # (n - p,) raw innovations (caller centers them)


def _require_statsmodels() -> None:
    try:
        import statsmodels  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without statsmodels
        raise BackendError(
            "statsmodels is required for model-based (residual/sieve) bootstraps",
            code=Codes.BACKEND_NOT_INSTALLED,
            hint="Install it, e.g. pip install 'tsbootstrap[stats]' or pip install statsmodels.",
        ) from exc


def fit_ar(x: NDArray[np.float64], order: int) -> ARFit:
    """Fit an AR(``order``) model with an intercept by conditional MLE/OLS."""
    _require_statsmodels()
    from statsmodels.tsa.ar_model import AutoReg

    series = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    n = series.shape[0]
    if order >= n:
        raise MethodConfigError(
            f"AR order {order} is too large for a series of length {n}",
            code=Codes.ORDER_TOO_LARGE,
            context={"order": order, "n": n},
        )
    res = AutoReg(series, lags=order, trend="c", old_names=False).fit()
    params = np.asarray(res.params, dtype=np.float64)
    intercept = float(params[0])
    ar_coefs = np.ascontiguousarray(params[1:])
    residuals = np.ascontiguousarray(np.asarray(res.resid, dtype=np.float64))
    return ARFit(order=order, intercept=intercept, ar_coefs=ar_coefs, residuals=residuals)


def select_ar_order(
    x: NDArray[np.float64],
    *,
    min_lag: int = 1,
    max_lag: int | None = None,
    criterion: str = "bic",
) -> int:
    """Select the AR order once on the original series (for the sieve bootstrap)."""
    _require_statsmodels()
    from statsmodels.tsa.ar_model import ar_select_order

    series = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    n = series.shape[0]
    upper = max_lag if max_lag is not None else int(np.ceil(10 * np.log10(n)))
    upper = max(min_lag, min(upper, n // 2 - 1))
    selected = ar_select_order(series, maxlag=upper, ic=criterion, trend="c", old_names=False)
    lags = selected.ar_lags
    order = int(max(lags)) if lags is not None and len(lags) > 0 else min_lag
    return max(min_lag, order)


__all__ = ["ARFit", "fit_ar", "select_ar_order"]
