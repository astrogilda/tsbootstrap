"""``diagnose(X)``: inspect a series and recommend bootstrap methods.

A lightweight, honest advisor — it measures serial dependence and stationarity
and maps them to suitable method specs. It does not choose for you; it explains
what it sees and what fits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.validation import coerce_observations

_DEPENDENCE_THRESHOLD = 0.2


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """What ``diagnose`` found and what it recommends."""

    n_obs: int
    n_series: int
    lag1_autocorr: float
    dependent: bool
    nonstationary: bool
    recommended_methods: tuple[str, ...]
    notes: tuple[str, ...]


def _max_lag1_autocorr(arr: NDArray[np.float64]) -> float:
    best = 0.0
    for j in range(arr.shape[1]):
        col = arr[:, j]
        if col.std() > 0:
            best = max(best, abs(float(np.corrcoef(col[:-1], col[1:])[0, 1])))
    return best


def _looks_nonstationary(arr: NDArray[np.float64], lag1: float) -> bool:
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:  # pragma: no cover - fall back without statsmodels
        return lag1 > 0.95
    for j in range(arr.shape[1]):
        col = arr[:, j]
        if col.std() == 0:
            continue
        try:
            pvalue = float(adfuller(col, autolag="AIC")[1])
        except ValueError:  # series too short for the test
            return lag1 > 0.95
        if pvalue > 0.05:  # fail to reject the unit-root null
            return True
    return False


def diagnose(X: object) -> Diagnosis:
    """Inspect ``X`` and recommend bootstrap methods for it."""
    arr, _ = coerce_observations(X)
    n_obs, n_series = arr.shape
    lag1 = _max_lag1_autocorr(arr)
    dependent = lag1 > _DEPENDENCE_THRESHOLD
    nonstationary = _looks_nonstationary(arr, lag1)

    recommended: list[str] = []
    notes: list[str] = []

    if nonstationary:
        recommended += ["ResidualBootstrap(model=ARIMA(...))", "SieveAR"]
        notes.append("Series looks non-stationary (unit root): difference it via ARIMA, or use the sieve.")
    elif dependent:
        recommended += ["StationaryBlock", "MovingBlock", "SieveAR"]
        notes.append(f"Serial dependence present (lag-1 autocorrelation {lag1:.2f}): use a block method or the sieve.")
    else:
        recommended += ["IID", "MovingBlock"]
        notes.append("Serial dependence is weak: i.i.d. resampling is acceptable; a block method is a safe default.")

    if n_series > 1:
        recommended.insert(0, "ResidualBootstrap(model=VAR(...))")
        notes.append("Multivariate input: VAR captures cross-series dependence; block methods preserve it by resampling whole rows.")

    if dependent and not nonstationary:
        from tsbootstrap.block.pwsd import optimal_block_length

        suggested = optimal_block_length(arr, kind="stationary")
        notes.append(f"Suggested automatic block length (Politis-White): {suggested}.")

    return Diagnosis(
        n_obs=n_obs,
        n_series=n_series,
        lag1_autocorr=lag1,
        dependent=dependent,
        nonstationary=nonstationary,
        recommended_methods=tuple(dict.fromkeys(recommended)),
        notes=tuple(notes),
    )


__all__ = ["Diagnosis", "diagnose"]
