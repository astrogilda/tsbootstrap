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
    e_star = np.stack([eps[g.integers(0, eps.shape[0], size=horizon)] for g in generators])
    paths = simulate_ar_batched(fit.ar_coefs, fit.intercept, inits, e_star)
    forecasts = paths[:, p:]  # the generated horizon, beyond the initial values

    lower = np.quantile(forecasts, alpha / 2.0, axis=0)
    upper = np.quantile(forecasts, 1.0 - alpha / 2.0, axis=0)
    median = np.quantile(forecasts, 0.5, axis=0)
    return lower, upper, median


__all__ = ["forecast_intervals"]
