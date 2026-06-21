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

from tsbootstrap.dispatch import PreparationFailed, register_executor, register_preparer
from tsbootstrap.engines.arma_scipy import simulate_ar_batched, simulate_arma_batched
from tsbootstrap.engines.var import simulate_var_batched
from tsbootstrap.errors import Codes, MethodConfigError, ModelStabilityError
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
    exog: NDArray[np.float64] | None = None  # (n, k) exogenous regressors, held fixed


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
    exog: NDArray[np.float64] | None = None  # (n, k) exogenous regressors, held fixed


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


def _stability_guard(coefs, check_fn, policy: str) -> PreparationFailed | None:
    """Apply the stability policy: PreparationFailed on "skip", re-raise on "raise", None if stable."""
    try:
        check_fn(coefs)
    except ModelStabilityError as exc:
        if policy == "skip":
            return PreparationFailed(str(exc))
        raise
    return None


def _build_ar_context(
    series: NDArray[np.float64],
    fit: ARFit,
    burn_in: int,
    initial: str,
    policy: str,
    exog: NDArray[np.float64] | None = None,
) -> _ARContext | PreparationFailed:
    failed = _stability_guard(fit.ar_coefs, check_ar_stability, policy)
    if failed is not None:
        return failed
    centered = fit.residuals - fit.residuals.mean()
    return _ARContext(series, fit, centered, burn_in, initial, exog)


def _check_exog_compatible(exog: object, burn_in: int, initial: str) -> None:
    if exog is None:
        return
    if initial != "fixed":
        raise MethodConfigError(
            "exogenous regressors require initial='fixed' (a random initial block would "
            "break the exog time alignment)",
            code=Codes.UNSUPPORTED_EXOG,
        )
    if burn_in != 0:
        raise MethodConfigError(
            "exogenous regressors require burn_in=0 (there is no exog for burn-in steps)",
            code=Codes.UNSUPPORTED_EXOG,
        )


@register_preparer(ResidualBootstrap)
def _prepare_residual(
    data: NDArray[np.float64], spec: ResidualBootstrap, exog: NDArray[np.float64] | None
) -> _ARContext | _ARIMAContext | _VARContext | PreparationFailed:
    _require_iid(spec.innovation, "ResidualBootstrap")
    model = spec.model
    if isinstance(model, AR):
        series = _as_univariate(data, "ResidualBootstrap with an AR model")
        _check_exog_compatible(exog, model.burn_in, model.initial)
        fit = fit_ar(series, model.order, exog)
        return _build_ar_context(series, fit, model.burn_in, model.initial, model.stability_policy, exog)
    if isinstance(model, VAR):
        _check_exog_compatible(exog, model.burn_in, model.initial)
        return _prepare_var(data, model, exog)
    if exog is not None:
        raise MethodConfigError(
            f"exogenous regressors are not yet supported for a {type(model).__name__} model",
            code=Codes.UNSUPPORTED_EXOG,
        )
    if isinstance(model, ARIMA):
        series = _as_univariate(data, "ResidualBootstrap with an ARIMA model")
        return _prepare_arima(series, model)
    raise MethodConfigError(
        f"ResidualBootstrap with a {type(model).__name__} model is not yet implemented",
        code=Codes.UNSUPPORTED_MODEL_FEATURE,
    )


def _prepare_arima(series: NDArray[np.float64], model: ARIMA) -> _ARIMAContext | PreparationFailed:
    p, d, q = model.order
    w, levels = difference(series, d)
    arma = fit_arma(w, p, q)
    failed = _stability_guard(arma.ar_coefs, check_ar_stability, model.stability_policy)
    if failed is not None:
        return failed
    centered = arma.residuals - arma.residuals.mean()
    return _ARIMAContext(arma=arma, centered_resid=centered, levels=levels, d=d, burn_in=model.burn_in)


def _prepare_var(
    data: NDArray[np.float64], model: VAR, exog: NDArray[np.float64] | None
) -> _VARContext | PreparationFailed:
    arr = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    fit = fit_var(arr, model.order, exog)  # raises if not multivariate
    failed = _stability_guard(fit.coefs, check_var_stability, model.stability_policy)
    if failed is not None:
        return failed
    centered = fit.residuals - fit.residuals.mean(axis=0)
    return _VARContext(
        series=arr, fit=fit, centered_resid=centered, burn_in=model.burn_in,
        initial=model.initial, exog=exog,
    )


@register_preparer(SieveAR)
def _prepare_sieve(
    data: NDArray[np.float64], spec: SieveAR, exog: NDArray[np.float64] | None
) -> _ARContext | PreparationFailed:
    if exog is not None:
        raise MethodConfigError(
            "exogenous regressors are not yet supported for SieveAR",
            code=Codes.UNSUPPORTED_EXOG,
        )
    _require_iid(spec.innovation, "SieveAR")
    series = _as_univariate(data, "SieveAR")
    order = select_ar_order(series, min_lag=spec.min_lag, max_lag=spec.max_lag, criterion=spec.criterion)
    fit = fit_ar(series, order)
    return _build_ar_context(series, fit, spec.burn_in, spec.initial, spec.stability_policy)


def _draw_innovations_and_inits(
    ctx: _ARContext | _VARContext, generators: list[np.random.Generator], n_steps: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-generator innovation resample + initial conditions, stacked over B.

    Drawing happens per generator (so determinism is independent of batching); the
    numeric simulation that follows is vectorised over the stacked tensors.
    """
    p = ctx.fit.order
    eps = ctx.centered_resid
    n_resid = eps.shape[0]
    series = ctx.series
    n_series = series.shape[0]
    e_star = np.empty((len(generators), n_steps, *eps.shape[1:]), dtype=np.float64)
    inits = np.empty((len(generators), p, *eps.shape[1:]), dtype=np.float64)
    fixed = ctx.initial == "fixed"
    for i, gen in enumerate(generators):
        e_star[i] = eps[gen.integers(0, n_resid, size=n_steps)]
        if fixed:
            inits[i] = series[:p]
        else:
            start = int(gen.integers(0, n_series - p + 1))
            inits[i] = series[start : start + p]
    return e_star, inits


def _ar_batched(ctx: _ARContext, n: int, generators: list[np.random.Generator]) -> NDArray[np.float64]:
    p = ctx.fit.order
    e_star, inits = _draw_innovations_and_inits(ctx, generators, n + ctx.burn_in - p)
    if ctx.exog is not None:
        # Exog is held fixed; the generated steps are times p..n-1 (burn_in is 0 with exog),
        # so add the deterministic exog contribution to each step's forcing.
        exog_contrib = ctx.exog[p : p + e_star.shape[1]] @ ctx.fit.exog_coefs
        e_star = e_star + exog_contrib[None, :]
    paths = simulate_ar_batched(ctx.fit.ar_coefs, ctx.fit.intercept, inits, e_star)
    samples = paths[:, ctx.burn_in : ctx.burn_in + n] if ctx.burn_in else paths[:, :n]
    return samples[:, :, None]


def _arima_batched(ctx: _ARIMAContext, n: int, generators: list[np.random.Generator]) -> NDArray[np.float64]:
    eps = ctx.centered_resid
    n_resid = eps.shape[0]
    n_kept = n - ctx.d
    burn = max(ctx.burn_in, _ARMA_BURN_FLOOR)
    e_star = np.empty((len(generators), n_kept + burn), dtype=np.float64)
    for i, gen in enumerate(generators):
        e_star[i] = eps[gen.integers(0, n_resid, size=n_kept + burn)]
    w_centered = simulate_arma_batched(ctx.arma.ar_coefs, ctx.arma.ma_coefs, e_star)
    w_star = w_centered[:, burn:] + ctx.arma.mean
    samples = np.stack([integrate(w_star[i], ctx.levels) for i in range(len(generators))])
    return samples[:, :, None]


def _var_batched(ctx: _VARContext, n: int, generators: list[np.random.Generator]) -> NDArray[np.float64]:
    p = ctx.fit.order
    e_star, inits = _draw_innovations_and_inits(ctx, generators, n + ctx.burn_in - p)
    if ctx.exog is not None:
        # Exog held fixed; generated steps are times p..n-1 (burn_in is 0 with exog). Fold the
        # deterministic B z_t contribution into each step's forcing (mirrors the ARX path).
        exog_contrib = ctx.exog[p : p + e_star.shape[1]] @ ctx.fit.exog_coefs  # (m, d)
        e_star = e_star + exog_contrib[None, :, :]
    paths = simulate_var_batched(ctx.fit.coefs, ctx.fit.intercept, inits, e_star)
    return paths[:, ctx.burn_in : ctx.burn_in + n] if ctx.burn_in else paths[:, :n]


@register_executor(ResidualBootstrap)
def _residual(
    prepared: _ARContext | _ARIMAContext | _VARContext,
    spec: ResidualBootstrap,
    generators: list[np.random.Generator],
    n_obs: int,
):
    if isinstance(prepared, _VARContext):
        return _var_batched(prepared, n_obs, generators), None
    if isinstance(prepared, _ARIMAContext):
        return _arima_batched(prepared, n_obs, generators), None
    return _ar_batched(prepared, n_obs, generators), None


@register_executor(SieveAR)
def _sieve(
    prepared: _ARContext, spec: SieveAR, generators: list[np.random.Generator], n_obs: int
):
    return _ar_batched(prepared, n_obs, generators), None


__all__ = []
