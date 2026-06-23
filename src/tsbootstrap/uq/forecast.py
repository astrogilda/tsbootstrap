"""Forecast prediction intervals from forward bootstrap simulation.

Fit the model once, then simulate it ``horizon`` steps beyond the data many times
with resampled, centered innovations, and read off empirical quantiles per step.
Coverage is approximate / asymptotic under temporal dependence, not finite-sample
distribution-free.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.engines.arma_scipy import simulate_ar_batched
from tsbootstrap.errors import Codes, MethodConfigError
from tsbootstrap.methods import AR
from tsbootstrap.model.fit import fit_ar
from tsbootstrap.model.stability import check_ar_stability
from tsbootstrap.rng import RandomStateLike, resolve_seed_sequence, spawn_generators


def forecast_intervals(
    X: object,
    *,
    model: object,
    horizon: int,
    alpha: float = 0.1,
    n_bootstraps: int = 999,
    random_state: RandomStateLike = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Bootstrap forecast intervals: ``(lower, upper, median)``, each ``(horizon,)``.

    The fitted model is simulated ``horizon`` steps past the data ``n_bootstraps``
    times with resampled, centered innovations; the per-step quantiles form the
    interval.
    """
    if not isinstance(model, AR):
        raise MethodConfigError(
            f"forecast_intervals currently supports an AR model; got {type(model).__name__}",
            code=Codes.UNSUPPORTED_MODEL_FEATURE,
        )
    if horizon < 1:
        raise MethodConfigError(
            "horizon must be >= 1", code=Codes.INVALID_PARAMETER, context={"horizon": horizon}
        )

    x = np.ascontiguousarray(np.asarray(X, dtype=np.float64).ravel())
    fit = fit_ar(x, model.order)
    check_ar_stability(fit.ar_coefs)
    eps = fit.residuals - fit.residuals.mean()
    p = fit.order

    generators = spawn_generators(resolve_seed_sequence(random_state), n_bootstraps)
    inits = np.tile(x[-p:], (n_bootstraps, 1))  # continue from the last p observations
    # One integers() draw per generator (same size/order as before, so each replicate's RNG
    # stream is byte-identical) into a (B, horizon) index matrix, then a single fancy-index gather.
    idx = np.empty((n_bootstraps, horizon), dtype=np.intp)
    n_resid = eps.shape[0]
    for b, g in enumerate(generators):
        idx[b] = g.integers(0, n_resid, size=horizon)
    e_star = eps[idx]
    paths = simulate_ar_batched(fit.ar_coefs, fit.intercept, inits, e_star)
    forecasts = paths[:, p:]  # the generated horizon, beyond the initial values

    # One sort-and-interpolate pass for all three levels instead of three independent
    # passes over ``forecasts``.
    q = np.quantile(forecasts, [alpha / 2.0, 1.0 - alpha / 2.0, 0.5], axis=0)
    lower, upper, median = q
    return lower, upper, median


__all__ = ["forecast_intervals"]
