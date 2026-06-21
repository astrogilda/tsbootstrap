"""Recursive residual and sieve bootstrap (autoregressive core).

The model is fit once (in the preparer); each replicate then regenerates a full
path recursively from the fitted coefficients and resampled, **centered**
innovations: ``X*_t = c + sum_j phi_j X*_{t-j} + e*_t``. The resampled shocks
propagate recursively through the fitted dynamics, as the model-based bootstrap
requires.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.dispatch import register_executor, register_preparer
from tsbootstrap.engines.arma_scipy import simulate_ar, simulate_arma
from tsbootstrap.engines.var import simulate_var
from tsbootstrap.errors import Codes, MethodConfigError
from tsbootstrap.methods import AR, ARIMA, IID, VAR, ResidualBootstrap, SieveAR
from tsbootstrap.model.arima import ARMAFit, difference, fit_arma, integrate
from tsbootstrap.model.fit import ARFit, VARFit, fit_ar, fit_var, select_ar_order
from tsbootstrap.model.stability import check_ar_stability, check_var_stability

# Burn-in for the zero-state ARMA simulation, so the transient clears before the
# kept window (the AR path seeds initial conditions instead and needs none).
_ARMA_BURN_FLOOR = 100


@dataclass(frozen=True, slots=True)
class _ARContext:
    series: NDArray[np.float64]
    fit: ARFit
    centered_resid: NDArray[np.float64]
    burn_in: int
    initial: str


@dataclass(frozen=True, slots=True)
class _ARIMAContext:
    arma: ARMAFit
    centered_resid: NDArray[np.float64]
    levels: list[float]
    d: int
    burn_in: int


@dataclass(frozen=True, slots=True)
class _VARContext:
    series: NDArray[np.float64]  # (n, d)
    fit: VARFit
    centered_resid: NDArray[np.float64]  # (m, d)
    burn_in: int
    initial: str


def _as_univariate(data: NDArray[np.float64], method_name: str) -> NDArray[np.float64]:
    if data.ndim == 2 and data.shape[1] > 1:
        raise MethodConfigError(
            f"{method_name} requires a univariate series; use a VAR model for multivariate data",
            code=Codes.VAR_REQUIRES_MULTIVARIATE,
        )
    return np.ascontiguousarray(data[:, 0] if data.ndim == 2 else data)


def _require_iid(innovation: object, method_name: str) -> None:
    if not isinstance(innovation, IID):
        raise MethodConfigError(
            f"{method_name} currently supports only IID innovations; "
            f"block innovation resampling is not yet implemented",
            code=Codes.UNSUPPORTED_MODEL_FEATURE,
        )


def _build_ar_context(
    series: NDArray[np.float64], fit: ARFit, burn_in: int, initial: str
) -> _ARContext:
    check_ar_stability(fit.ar_coefs)
    centered = fit.residuals - fit.residuals.mean()
    return _ARContext(series, fit, centered, burn_in, initial)


@register_preparer(ResidualBootstrap)
def _prepare_residual(
    data: NDArray[np.float64], spec: ResidualBootstrap
) -> _ARContext | _ARIMAContext:
    _require_iid(spec.innovation, "ResidualBootstrap")
    model = spec.model
    if isinstance(model, AR):
        series = _as_univariate(data, "ResidualBootstrap with an AR model")
        fit = fit_ar(series, model.order)
        return _build_ar_context(series, fit, model.burn_in, model.initial)
    if isinstance(model, ARIMA):
        series = _as_univariate(data, "ResidualBootstrap with an ARIMA model")
        return _prepare_arima(series, model)
    if isinstance(model, VAR):
        return _prepare_var(data, model)
    raise MethodConfigError(
        f"ResidualBootstrap with a {type(model).__name__} model is not yet implemented",
        code=Codes.UNSUPPORTED_MODEL_FEATURE,
    )


def _prepare_arima(series: NDArray[np.float64], model: ARIMA) -> _ARIMAContext:
    p, d, q = model.order
    w, levels = difference(series, d)
    arma = fit_arma(w, p, q)
    check_ar_stability(arma.ar_coefs)  # AR part must be stationary on the differenced scale
    centered = arma.residuals - arma.residuals.mean()
    return _ARIMAContext(arma=arma, centered_resid=centered, levels=levels, d=d, burn_in=model.burn_in)


def _prepare_var(data: NDArray[np.float64], model: VAR) -> _VARContext:
    arr = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    fit = fit_var(arr, model.order)  # raises if not multivariate
    check_var_stability(fit.coefs)
    centered = fit.residuals - fit.residuals.mean(axis=0)
    return _VARContext(series=arr, fit=fit, centered_resid=centered, burn_in=model.burn_in, initial=model.initial)


@register_preparer(SieveAR)
def _prepare_sieve(data: NDArray[np.float64], spec: SieveAR) -> _ARContext:
    _require_iid(spec.innovation, "SieveAR")
    series = _as_univariate(data, "SieveAR")
    order = select_ar_order(series, min_lag=spec.min_lag, max_lag=spec.max_lag, criterion=spec.criterion)
    fit = fit_ar(series, order)
    return _build_ar_context(series, fit, spec.burn_in, spec.initial)


def _ar_sample(ctx: _ARContext, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    p = ctx.fit.order
    eps = ctx.centered_resid
    n_steps = n + ctx.burn_in - p
    e_star = eps[rng.integers(0, eps.shape[0], size=n_steps)]
    if ctx.initial == "fixed":
        init = ctx.series[:p].copy()
    else:  # random_block: a random consecutive run of observed values
        start = int(rng.integers(0, ctx.series.shape[0] - p + 1))
        init = ctx.series[start : start + p].copy()
    path = simulate_ar(ctx.fit.ar_coefs, ctx.fit.intercept, init, e_star)
    sample = path[ctx.burn_in : ctx.burn_in + n] if ctx.burn_in else path[:n]
    return sample.reshape(-1, 1)


def _arima_sample(ctx: _ARIMAContext, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    m = n - ctx.d  # length of the differenced (ARMA-scale) series
    burn = max(ctx.burn_in, _ARMA_BURN_FLOOR)
    eps = ctx.centered_resid
    e_star = eps[rng.integers(0, eps.shape[0], size=m + burn)]
    w_centered = simulate_arma(ctx.arma.ar_coefs, ctx.arma.ma_coefs, e_star)
    w_star = w_centered[burn:] + ctx.arma.mean
    x_star = integrate(w_star, ctx.levels)
    return x_star.reshape(-1, 1)


def _var_sample(ctx: _VARContext, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    p = ctx.fit.order
    eps = ctx.centered_resid
    n_steps = n + ctx.burn_in - p
    # resample whole innovation rows to preserve the contemporaneous cross-covariance
    e_star = eps[rng.integers(0, eps.shape[0], size=n_steps)]
    if ctx.initial == "fixed":
        init = ctx.series[:p].copy()
    else:
        start = int(rng.integers(0, ctx.series.shape[0] - p + 1))
        init = ctx.series[start : start + p].copy()
    path = simulate_var(ctx.fit.coefs, ctx.fit.intercept, init, e_star)
    return path[ctx.burn_in : ctx.burn_in + n] if ctx.burn_in else path[:n]


def _one_sample(
    prepared: _ARContext | _ARIMAContext | _VARContext, n_obs: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    if isinstance(prepared, _VARContext):
        return _var_sample(prepared, n_obs, rng)
    if isinstance(prepared, _ARIMAContext):
        return _arima_sample(prepared, n_obs, rng)
    return _ar_sample(prepared, n_obs, rng)


@register_executor(ResidualBootstrap)
def _residual(
    prepared: _ARContext | _ARIMAContext | _VARContext,
    spec: ResidualBootstrap,
    generators: list[np.random.Generator],
    n_obs: int,
):
    return np.stack([_one_sample(prepared, n_obs, g) for g in generators]), None


@register_executor(SieveAR)
def _sieve(
    prepared: _ARContext, spec: SieveAR, generators: list[np.random.Generator], n_obs: int
):
    return np.stack([_ar_sample(prepared, n_obs, g) for g in generators]), None


__all__ = []
