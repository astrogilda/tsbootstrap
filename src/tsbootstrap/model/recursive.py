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
from tsbootstrap.engines.arma_scipy import simulate_ar
from tsbootstrap.errors import Codes, MethodConfigError
from tsbootstrap.methods import AR, IID, ResidualBootstrap, SieveAR
from tsbootstrap.model.fit import ARFit, fit_ar, select_ar_order
from tsbootstrap.model.stability import check_ar_stability


@dataclass(frozen=True, slots=True)
class _ARContext:
    series: NDArray[np.float64]
    fit: ARFit
    centered_resid: NDArray[np.float64]
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
def _prepare_residual(data: NDArray[np.float64], spec: ResidualBootstrap) -> _ARContext:
    if not isinstance(spec.model, AR):
        raise MethodConfigError(
            f"ResidualBootstrap with a {type(spec.model).__name__} model is not yet implemented",
            code=Codes.UNSUPPORTED_MODEL_FEATURE,
        )
    _require_iid(spec.innovation, "ResidualBootstrap")
    series = _as_univariate(data, "ResidualBootstrap with an AR model")
    fit = fit_ar(series, spec.model.order)
    return _build_ar_context(series, fit, spec.model.burn_in, spec.model.initial)


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


@register_executor(ResidualBootstrap)
def _residual(prepared: _ARContext, spec: ResidualBootstrap, rng: np.random.Generator, n_obs: int):
    return _ar_sample(prepared, n_obs, rng), None


@register_executor(SieveAR)
def _sieve(prepared: _ARContext, spec: SieveAR, rng: np.random.Generator, n_obs: int):
    return _ar_sample(prepared, n_obs, rng), None


__all__ = []
