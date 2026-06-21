"""Fit and difference/integrate for ARIMA recursive bootstraps.

ARIMA(p, d, q) is handled by differencing the series d times to a stationary
ARMA(p, q) scale, bootstrapping there, and inverse-differencing each replicate
back using the original initial levels. statsmodels is imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter, lfiltic

from tsbootstrap.errors import Codes, MethodConfigError
from tsbootstrap.model.fit import _require_statsmodels


@dataclass(frozen=True, slots=True)
class ARMAFit:
    """An estimated zero-mean ARMA(p, q) on the differenced scale.

    The series is fit demeaned, so ``mean`` is added back after simulation and
    the AR/MA recursion itself is zero-mean.
    """

    ar_coefs: NDArray[np.float64]  # (p,)
    ma_coefs: NDArray[np.float64]  # (q,)
    mean: float
    residuals: NDArray[np.float64]  # (m,) innovations (caller centers them)
    init_w: NDArray[np.float64]  # first max(p, q) demeaned values: the conditional initial state


def difference(x: NDArray[np.float64], d: int) -> tuple[NDArray[np.float64], list[float]]:
    """Difference ``x`` ``d`` times; return the result and the initial level of each order."""
    levels: list[float] = []
    cur = np.asarray(x, dtype=np.float64)
    for _ in range(d):
        levels.append(float(cur[0]))
        cur = np.diff(cur)
    return cur, levels


def integrate(w: NDArray[np.float64], levels: list[float]) -> NDArray[np.float64]:
    """Invert :func:`difference`: reconstruct the original-scale series from ``w``."""
    cur = np.asarray(w, dtype=np.float64)
    for level in reversed(levels):
        cur = np.concatenate([[level], level + np.cumsum(cur)])
    return cur


def fit_arma(w: NDArray[np.float64], p: int, q: int) -> ARMAFit:
    """Fit a demeaned ARMA(p, q) to the (already differenced) series ``w``."""
    _require_statsmodels()
    from statsmodels.tsa.arima.model import ARIMA as _SMARIMA

    series = np.ascontiguousarray(np.asarray(w, dtype=np.float64).ravel())
    n = series.shape[0]
    if p + q >= n:
        raise MethodConfigError(
            f"ARMA order p+q={p + q} is too large for a differenced series of length {n}",
            code=Codes.ORDER_TOO_LARGE,
            context={"p": p, "q": q, "n": n},
        )
    mean = float(series.mean())
    res = _SMARIMA(series - mean, order=(p, 0, q), trend="n").fit()
    ar_coefs = np.ascontiguousarray(np.asarray(res.arparams, dtype=np.float64))
    ma_coefs = np.ascontiguousarray(np.asarray(res.maparams, dtype=np.float64))
    # Derive the innovations in OUR engine's convention (scipy lfilter), not statsmodels'
    # Kalman one-step residuals: apply the inverse ARMA filter to the demeaned series. This
    # makes the resampled innovations consistent with the forward lfilter simulation (so the
    # engine can exactly reconstruct the fitted series) instead of mixing two innovation
    # definitions. statsmodels is used only for the parameter MLE (the part lfilter cannot do).
    b = np.concatenate([[1.0], ma_coefs])
    a = np.concatenate([[1.0], -ar_coefs])
    demeaned = series - mean
    residuals = np.ascontiguousarray(lfilter(a, b, demeaned))
    k = max(p, q)  # length of the conditional initial state
    init_w = np.ascontiguousarray(demeaned[:k])
    return ARMAFit(
        ar_coefs=ar_coefs, ma_coefs=ma_coefs, mean=mean, residuals=residuals, init_w=init_w
    )


def arma_initial_state(
    ar_coefs: NDArray[np.float64],
    ma_coefs: NDArray[np.float64],
    init_w: NDArray[np.float64],
    init_residuals: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The lfilter delay state (``zi``) conditioning the simulation on the observed initials.

    Built from the observed initial differenced values and the estimated initial innovations
    — the ARMA analogue of AR/VAR's ``initial="fixed"``.
    """
    a = np.concatenate([[1.0], -ar_coefs])
    b = np.concatenate([[1.0], ma_coefs])
    return lfiltic(b, a, init_w[::-1], init_residuals[::-1])


def fit_regression_arima_beta(
    y: NDArray[np.float64], order: tuple[int, int, int], exog: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Estimate exogenous coefficients for regression with ARIMA errors (statsmodels).

    The ARIMAX model is ``y_t = beta . z_t + eta_t`` with ``eta_t ~ ARIMA(order)``. Returns
    ``beta`` (shape ``(k,)``) estimated jointly so it accounts for the ARIMA error
    structure; the caller then bootstraps ``eta = y - exog @ beta`` and adds ``beta . z``
    back to each replicate.
    """
    _require_statsmodels()
    from statsmodels.tsa.arima.model import ARIMA as _SMARIMA

    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
    exog = np.ascontiguousarray(np.asarray(exog, dtype=np.float64))
    if exog.ndim == 1:
        exog = exog.reshape(-1, 1)
    p, d, q = order
    res = _SMARIMA(y, order=(p, d, q), exog=exog, trend="n").fit()
    # With trend="n", statsmodels orders the exogenous coefficients first.
    k = exog.shape[1]
    return np.ascontiguousarray(np.asarray(res.params[:k], dtype=np.float64))


__all__ = [
    "ARMAFit",
    "difference",
    "integrate",
    "fit_arma",
    "arma_initial_state",
    "fit_regression_arima_beta",
]
