"""Recursive residual and sieve bootstrap (autoregressive core).

The model is fit once (in the preparer); each replicate then regenerates a full
path recursively from the fitted coefficients and resampled, **centered**
innovations: ``X*_t = c + sum_j phi_j X*_{t-j} + e*_t``. The resampled shocks
propagate recursively through the fitted dynamics, as the model-based bootstrap
requires.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.block.pwsd import optimal_block_length
from tsbootstrap.dispatch import PreparationFailed, register_chunk_executor, register_preparer
from tsbootstrap.engines.arma_scipy import simulate_ar_batched, simulate_arma_batched
from tsbootstrap.engines.var import simulate_var_batched
from tsbootstrap.errors import (
    Codes,
    DegenerateBlockBootstrapWarning,
    MethodConfigError,
    ModelStabilityError,
)
from tsbootstrap.methods import AR, ARIMA, IID, VAR, BlockWild, ResidualBootstrap, SieveAR, Wild
from tsbootstrap.model.arima import (
    ARMAFit,
    arma_initial_state,
    difference,
    fit_arma,
    fit_regression_arima_beta,
    integrate_batched,
)
from tsbootstrap.model.fit import ARFit, VARFit, fit_ar, fit_var, select_ar_order
from tsbootstrap.model.stability import check_ar_stability, check_var_stability
from tsbootstrap.rng import generators_from_seeds


@dataclass(frozen=True, slots=True)
class _ExogState:
    """Held-fixed exogenous regressors paired with their fitted coefficients.

    The two arrays are always present together (a fit either has exog or it does
    not), so bundling them lets a single ``None`` check narrow both at once.
    """

    exog: NDArray[np.float64]  # (n, k) exogenous regressors, held fixed
    coefs: NDArray[np.float64]  # (k,) for AR/ARIMA, (k, d) for VAR


@dataclass(frozen=True, slots=True)
class _WildPlan:
    """Resolved wild-innovation execution plan.

    The spec declares the intent (:class:`~tsbootstrap.methods.Wild` /
    :class:`~tsbootstrap.methods.BlockWild`); this carries the realisation the
    engines need: the multiplier distribution and, for block-wild, the block
    length resolved against the centered residuals at fit time.
    """

    distribution: str  # "rademacher" | "gaussian" | "mammen"
    block_length: int | None  # None: one multiplier per residual (classic wild)


@dataclass(frozen=True, slots=True)
class _ARContext:
    series: NDArray[np.float64]
    fit: ARFit
    resampling_innovations: NDArray[np.float64]
    burn_in: int
    initial: str
    exog_state: _ExogState | None = None
    wild: _WildPlan | None = None


@dataclass(frozen=True, slots=True)
class _ARIMAContext:
    arma: ARMAFit
    resampling_innovations: NDArray[np.float64]
    levels: list[float]
    d: int
    exog_state: _ExogState | None = None
    wild: _WildPlan | None = None


@dataclass(frozen=True, slots=True)
class _VARContext:
    series: NDArray[np.float64]  # (n, d)
    fit: VARFit
    resampling_innovations: NDArray[np.float64]  # (m, d)
    burn_in: int
    initial: str
    exog_state: _ExogState | None = None
    wild: _WildPlan | None = None


def _as_univariate(data: NDArray[np.float64], method_name: str) -> NDArray[np.float64]:
    if data.ndim == 2 and data.shape[1] > 1:
        raise MethodConfigError(
            f"{method_name} requires a univariate series; use a VAR model for multivariate data",
            code=Codes.VAR_REQUIRES_MULTIVARIATE,
        )
    return np.ascontiguousarray(data[:, 0] if data.ndim == 2 else data)


def _check_innovation(innovation: object, method_name: str) -> None:
    # Positive allowlist: IID index resampling and the wild multiplier family are
    # executable; block innovation resampling remains spec-constructible but not
    # yet implemented (same typed error as before, so the contract is unchanged).
    if not isinstance(innovation, (IID, Wild, BlockWild)):
        raise MethodConfigError(
            f"{method_name} currently supports IID and wild-type (Wild/BlockWild) innovations; "
            f"block innovation resampling is not yet implemented",
            code=Codes.UNSUPPORTED_MODEL_FEATURE,
        )


def _check_wild_compatible(innovation: object, burn_in: int, initial: str) -> None:
    if not isinstance(innovation, (Wild, BlockWild)):
        return
    if initial != "fixed":
        raise MethodConfigError(
            "wild innovations require initial='fixed' (the multiplier stream is aligned "
            "one-to-one with the residuals, conditional on the observed initial values)",
            code=Codes.UNSUPPORTED_MODEL_FEATURE,
        )
    if burn_in != 0:
        raise MethodConfigError(
            "wild innovations require burn_in=0 (there is one multiplier per residual; "
            "burn-in steps would have no residual to multiply)",
            code=Codes.UNSUPPORTED_MODEL_FEATURE,
        )


def _wild_plan(innovation: object, centered: NDArray[np.float64]) -> _WildPlan | None:
    """Resolve a wild-type innovation spec against the centered residuals.

    ``BlockWild(block_length="auto")`` resolves via the Politis-White rule on the
    residuals themselves: the multiplier blocks must match the dependence
    length-scale left IN the residuals (near 1 for a well-specified model), not
    the scale of the original series.
    """
    if isinstance(innovation, Wild):
        return _WildPlan(distribution=innovation.distribution, block_length=None)
    if not isinstance(innovation, BlockWild):
        return None
    m = int(centered.shape[0])
    if innovation.block_length == "auto":
        arr2d = centered if centered.ndim == 2 else centered.reshape(-1, 1)
        length = optimal_block_length(arr2d, kind="circular")
    else:
        length = int(innovation.block_length)
        if length > m:
            raise MethodConfigError(
                f"block_length {length} exceeds the number of residuals {m}",
                code=Codes.BLOCK_LENGTH_GT_RESIDUALS,
                context={"block_length": length, "n_residuals": m},
            )
    if length >= m:
        warnings.warn(
            DegenerateBlockBootstrapWarning(
                f"block-wild block length {length} >= residual count {m}; "
                "one multiplier covers the whole path",
                context={"block_length": length, "n_residuals": m},
            ),
            stacklevel=4,
        )
        length = m
    return _WildPlan(distribution=innovation.distribution, block_length=length)


# Mammen (1993) two-point multiplier: mean 0, variance 1, third moment 1.
_MAMMEN_P = (np.sqrt(5.0) + 1.0) / (2.0 * np.sqrt(5.0))
_MAMMEN_LO = -(np.sqrt(5.0) - 1.0) / 2.0
_MAMMEN_HI = (np.sqrt(5.0) + 1.0) / 2.0


def _draw_multipliers(
    gen: np.random.Generator, distribution: str, size: int
) -> NDArray[np.float64]:
    """One fixed-shape multiplier draw (mean 0, variance 1) from a bound generator.

    The draw shape depends only on replicate-invariant quantities, so byte
    stability across chunking and worker counts is inherited from the
    per-replicate generator binding.
    """
    if distribution == "rademacher":
        return gen.integers(0, 2, size=size).astype(np.float64) * 2.0 - 1.0
    if distribution == "gaussian":
        return gen.standard_normal(size)
    u = gen.random(size)  # mammen
    return np.where(u < _MAMMEN_P, _MAMMEN_LO, _MAMMEN_HI)


def _stability_guard(
    coefs: NDArray[np.float64],
    check_fn: Callable[[NDArray[np.float64]], float],
    policy: str,
) -> PreparationFailed | None:
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
    innovation: object = None,
) -> _ARContext | PreparationFailed:
    failed = _stability_guard(fit.ar_coefs, check_ar_stability, policy)
    if failed is not None:
        return failed
    centered = fit.residuals - fit.residuals.mean()
    exog_state = None
    if exog is not None and fit.exog_coefs is not None:
        exog_state = _ExogState(exog=exog, coefs=fit.exog_coefs)
    return _ARContext(
        series, fit, centered, burn_in, initial, exog_state, _wild_plan(innovation, centered)
    )


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
    _check_innovation(spec.innovation, "ResidualBootstrap")
    model = spec.model
    if isinstance(model, AR):
        series = _as_univariate(data, "ResidualBootstrap with an AR model")
        _check_exog_compatible(exog, model.burn_in, model.initial)
        _check_wild_compatible(spec.innovation, model.burn_in, model.initial)
        fit = fit_ar(series, model.order, exog)
        return _build_ar_context(
            series,
            fit,
            model.burn_in,
            model.initial,
            model.stability_policy,
            exog,
            spec.innovation,
        )
    if isinstance(model, VAR):
        _check_exog_compatible(exog, model.burn_in, model.initial)
        _check_wild_compatible(spec.innovation, model.burn_in, model.initial)
        return _prepare_var(data, model, exog, spec.innovation)
    if isinstance(model, ARIMA):
        series = _as_univariate(data, "ResidualBootstrap with an ARIMA model")
        # Exog is added at the level after inverse-differencing, so no alignment constraint.
        # ARIMA has no burn_in/initial fields: the multiplier stream aligns with the
        # residual tail automatically, so no wild-compatibility constraint either.
        return _prepare_arima(series, model, exog, spec.innovation)
    if exog is not None:
        raise MethodConfigError(
            f"exogenous regressors are not yet supported for a {type(model).__name__} model",
            code=Codes.UNSUPPORTED_EXOG,
        )
    raise MethodConfigError(
        f"ResidualBootstrap with a {type(model).__name__} model is not yet implemented",
        code=Codes.UNSUPPORTED_MODEL_FEATURE,
    )


def _prepare_arima(
    series: NDArray[np.float64],
    model: ARIMA,
    exog: NDArray[np.float64] | None,
    innovation: object = None,
) -> _ARIMAContext | PreparationFailed:
    p, d, q = model.order
    exog_coefs = None
    eta = series  # the ARIMA-distributed part; with exog, eta = y - z @ beta
    if exog is not None:
        exog_coefs = fit_regression_arima_beta(series, model.order, exog)
        eta = series - exog @ exog_coefs
    w, levels = difference(eta, d)
    arma = fit_arma(w, p, q)
    failed = _stability_guard(arma.ar_coefs, check_ar_stability, model.stability_policy)
    if failed is not None:
        return failed
    centered = arma.residuals - arma.residuals.mean()
    exog_state = None
    if exog is not None and exog_coefs is not None:
        exog_state = _ExogState(exog=exog, coefs=exog_coefs)
    return _ARIMAContext(
        arma=arma,
        resampling_innovations=centered,
        levels=levels,
        d=d,
        exog_state=exog_state,
        wild=_wild_plan(innovation, centered),
    )


def _prepare_var(
    data: NDArray[np.float64],
    model: VAR,
    exog: NDArray[np.float64] | None,
    innovation: object = None,
) -> _VARContext | PreparationFailed:
    arr = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    fit = fit_var(arr, model.order, exog)  # raises if not multivariate
    failed = _stability_guard(fit.coefs, check_var_stability, model.stability_policy)
    if failed is not None:
        return failed
    centered = fit.residuals - fit.residuals.mean(axis=0)
    exog_state = None
    if exog is not None and fit.exog_coefs is not None:
        exog_state = _ExogState(exog=exog, coefs=fit.exog_coefs)
    return _VARContext(
        series=arr,
        fit=fit,
        resampling_innovations=centered,
        burn_in=model.burn_in,
        initial=model.initial,
        exog_state=exog_state,
        wild=_wild_plan(innovation, centered),
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
    _check_innovation(spec.innovation, "SieveAR")
    _check_wild_compatible(spec.innovation, spec.burn_in, spec.initial)
    series = _as_univariate(data, "SieveAR")
    order = select_ar_order(
        series, min_lag=spec.min_lag, max_lag=spec.max_lag, criterion=spec.criterion
    )
    fit = fit_ar(series, order)
    return _build_ar_context(
        series, fit, spec.burn_in, spec.initial, spec.stability_policy, None, spec.innovation
    )


def _draw_innovations_and_inits(
    ctx: _ARContext | _VARContext, generators: list[np.random.Generator], n_steps: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-generator innovation resample + initial conditions, stacked over B.

    Drawing happens per generator (so determinism is independent of batching); the
    numeric simulation that follows is vectorised over the stacked tensors.
    """
    p = ctx.fit.order
    eps = ctx.resampling_innovations
    n_resid = eps.shape[0]
    series = ctx.series
    n_series = series.shape[0]
    e_star = np.empty((len(generators), n_steps, *eps.shape[1:]), dtype=np.float64)
    inits = np.empty((len(generators), p, *eps.shape[1:]), dtype=np.float64)
    fixed = ctx.initial == "fixed"
    plan = ctx.wild
    # Wild multipliers pair one-to-one with the residuals in time order, so the
    # spec-level gate guarantees n_steps == n_resid here (burn_in=0, initial="fixed").
    # Block-wild draws one multiplier per contiguous block and repeats it; the draw
    # shape depends only on replicate-invariant quantities (byte-stable).
    n_draw = 0
    if plan is not None:
        n_draw = n_steps if plan.block_length is None else -(-n_steps // plan.block_length)
    for i, gen in enumerate(generators):
        if plan is None:
            e_star[i] = eps[gen.integers(0, n_resid, size=n_steps)]
        else:
            v = _draw_multipliers(gen, plan.distribution, n_draw)
            if plan.block_length is not None:
                v = np.repeat(v, plan.block_length)[:n_steps]
            # VAR residuals are (m, d): one scalar multiplier per time step, broadcast
            # across the d components, preserves the cross-sectional residual covariance.
            e_star[i] = eps * (v[:, None] if eps.ndim == 2 else v)
        if fixed:
            inits[i] = series[:p]
        else:
            start = int(gen.integers(0, n_series - p + 1))
            inits[i] = series[start : start + p]
    return e_star, inits


def _ar_batched(
    ctx: _ARContext,
    n: int,
    generators: list[np.random.Generator],
    sim_dtype: np.dtype[np.floating],
) -> NDArray[np.floating]:
    p = ctx.fit.order
    e_star, inits = _draw_innovations_and_inits(ctx, generators, n + ctx.burn_in - p)
    if ctx.exog_state is not None:
        # Exog is held fixed; the generated steps are times p..n-1 (burn_in is 0 with exog),
        # so add the deterministic exog contribution to each step's forcing.
        exog_contrib = ctx.exog_state.exog[p : p + e_star.shape[1]] @ ctx.exog_state.coefs
        e_star += exog_contrib[None]
    paths = simulate_ar_batched(ctx.fit.ar_coefs, ctx.fit.intercept, inits, e_star)
    samples = paths[:, ctx.burn_in : ctx.burn_in + n] if ctx.burn_in else paths[:, :n]
    # The simulation runs in float64; the burn-in / length trim leaves a non-contiguous view, so
    # cast to sim_dtype at this final contiguity boundary and return a contiguous (B, n, 1).
    return np.ascontiguousarray(samples, dtype=sim_dtype)[:, :, None]


def _arima_batched(
    ctx: _ARIMAContext,
    n: int,
    generators: list[np.random.Generator],
    sim_dtype: np.dtype[np.floating],
) -> NDArray[np.floating]:
    arma = ctx.arma
    eps = ctx.resampling_innovations
    n_resid = eps.shape[0]
    n_kept = n - ctx.d  # length of the differenced series w
    k = arma.init_w.shape[0]  # conditional initial-state length = max(p, q)
    # Condition on the observed initial differenced state (the ARMA analogue of AR/VAR's
    # initial="fixed"): seed the filter from the observed initial values (arma.init_w) and the
    # RAW initial residuals arma.residuals[:k] -- deliberately NOT the centered
    # resampling_innovations -- so the first k simulated values reproduce the observed series
    # exactly. The continuation below resamples the mean-zero resampling_innovations; this
    # raw-seed / centered-continuation seam is required for exact conditional reconstruction
    # (centering the seed instead would inject a fictional initial state and drift the level).
    init_state = arma_initial_state(arma.ar_coefs, arma.ma_coefs, arma.init_w, arma.residuals[:k])
    # > 0 always: fit_arma's ORDER_TOO_LARGE guard guarantees max(p, q) < n_w == n_kept.
    m_tail = n_kept - k
    e_star = np.empty((len(generators), m_tail), dtype=np.float64)
    plan = ctx.wild
    # Wild multiplies the CENTERED continuation tail eps[k:] in place: residuals[i]
    # corresponds to w[i], so the continuation times k..n_kept-1 pair with eps[k:]
    # exactly, and the raw-seed / centered-continuation seam above is untouched.
    n_draw = 0
    if plan is not None:
        n_draw = m_tail if plan.block_length is None else -(-m_tail // plan.block_length)
    for i, gen in enumerate(generators):
        if plan is None:
            e_star[i] = eps[gen.integers(0, n_resid, size=m_tail)]
        else:
            v = _draw_multipliers(gen, plan.distribution, n_draw)
            if plan.block_length is not None:
                v = np.repeat(v, plan.block_length)[:m_tail]
            e_star[i] = eps[k:] * v
    w_centered = simulate_arma_batched(
        arma.ar_coefs, arma.ma_coefs, e_star, init_state=init_state, init_values=arma.init_w
    )
    w_star = w_centered + arma.mean
    samples = integrate_batched(w_star, ctx.levels)
    if ctx.exog_state is not None:
        # samples are the ARIMA part eta*; add the held-fixed regression level beta . z back.
        samples = samples + (ctx.exog_state.exog @ ctx.exog_state.coefs)[None, :]
    # The simulation runs in float64; cast to sim_dtype at this final contiguity boundary.
    return np.ascontiguousarray(samples, dtype=sim_dtype)[:, :, None]


def _var_batched(
    ctx: _VARContext,
    n: int,
    generators: list[np.random.Generator],
    sim_dtype: np.dtype[np.floating],
) -> NDArray[np.floating]:
    p = ctx.fit.order
    e_star, inits = _draw_innovations_and_inits(ctx, generators, n + ctx.burn_in - p)
    if ctx.exog_state is not None:
        # Exog held fixed; generated steps are times p..n-1 (burn_in is 0 with exog). Fold the
        # deterministic B z_t contribution into each step's forcing (mirrors the ARX path).
        exog_contrib = ctx.exog_state.exog[p : p + e_star.shape[1]] @ ctx.exog_state.coefs  # (m, d)
        e_star += exog_contrib[None]
    paths = simulate_var_batched(ctx.fit.coefs, ctx.fit.intercept, inits, e_star)
    samples = paths[:, ctx.burn_in : ctx.burn_in + n] if ctx.burn_in else paths[:, :n]
    # The simulation runs in float64; the burn-in / length trim leaves a non-contiguous view, so
    # cast to sim_dtype at this final contiguity boundary and return a contiguous (B, n, d).
    return np.ascontiguousarray(samples, dtype=sim_dtype)


@register_chunk_executor(ResidualBootstrap)
def _residual(
    prepared: _ARContext | _ARIMAContext | _VARContext,
    spec: ResidualBootstrap,
    seeds: list[np.random.SeedSequence],
    n_obs: int,
    sim_dtype: np.dtype[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.int32] | None]:
    generators = generators_from_seeds(seeds)
    if isinstance(prepared, _VARContext):
        return _var_batched(prepared, n_obs, generators, sim_dtype), None
    if isinstance(prepared, _ARIMAContext):
        return _arima_batched(prepared, n_obs, generators, sim_dtype), None
    return _ar_batched(prepared, n_obs, generators, sim_dtype), None


@register_chunk_executor(SieveAR)
def _sieve(
    prepared: _ARContext,
    spec: SieveAR,
    seeds: list[np.random.SeedSequence],
    n_obs: int,
    sim_dtype: np.dtype[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.int32] | None]:
    return _ar_batched(prepared, n_obs, generators_from_seeds(seeds), sim_dtype), None


__all__ = []
