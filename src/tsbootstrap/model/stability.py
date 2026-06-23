"""Stationarity / stability checks for fitted autoregressive coefficients.

A recursive bootstrap of a non-stationary fitted model produces explosive paths.
We refuse to simulate from an unstable model (spectral radius >= 1) and warn near
a unit root, rather than silently clipping coefficients or rejecting paths, which
would bias the bootstrap distribution.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.errors import Codes, ModelStabilityError, NearUnitRootWarning


def ar_spectral_radius(ar_coefs: NDArray[np.float64]) -> float:
    """Largest absolute eigenvalue of the AR companion matrix."""
    p = len(ar_coefs)
    if p == 0:
        return 0.0
    companion = np.zeros((p, p), dtype=np.float64)
    companion[0, :] = ar_coefs
    if p > 1:
        companion[1:, :-1] = np.eye(p - 1)
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def check_ar_stability(
    ar_coefs: NDArray[np.float64], *, near_unit_threshold: float = 0.98
) -> float:
    """Raise if the fitted AR model is non-stationary; warn if near a unit root."""
    radius = ar_spectral_radius(ar_coefs)
    if radius >= 1.0:
        raise ModelStabilityError(
            f"fitted AR model is non-stationary (companion spectral radius {radius:.4f} >= 1); "
            f"a recursive bootstrap would produce explosive paths",
            code=Codes.UNSTABLE_MODEL,
            context={"spectral_radius": radius},
        )
    if radius >= near_unit_threshold:
        warnings.warn(
            NearUnitRootWarning(
                f"fitted AR model is near a unit root (companion spectral radius {radius:.4f}); "
                f"bootstrap paths may be unreliable",
                context={"spectral_radius": radius},
            ),
            stacklevel=3,
        )
    return radius


def var_spectral_radius(coefs: NDArray[np.float64]) -> float:
    """Largest absolute eigenvalue of the VAR companion matrix. ``coefs`` is ``(p, d, d)``."""
    p, d, _ = coefs.shape
    if p == 0:
        return 0.0
    companion = np.zeros((p * d, p * d), dtype=np.float64)
    companion[:d, :] = coefs.transpose(1, 0, 2).reshape(d, p * d)
    if p > 1:
        companion[d:, : (p - 1) * d] = np.eye((p - 1) * d)
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def check_var_stability(coefs: NDArray[np.float64], *, near_unit_threshold: float = 0.98) -> float:
    """Raise if the fitted VAR is non-stationary; warn if near a unit root."""
    radius = var_spectral_radius(coefs)
    if radius >= 1.0:
        raise ModelStabilityError(
            f"fitted VAR model is non-stationary (companion spectral radius {radius:.4f} >= 1); "
            f"a recursive bootstrap would produce explosive paths",
            code=Codes.UNSTABLE_MODEL,
            context={"spectral_radius": radius},
        )
    if radius >= near_unit_threshold:
        warnings.warn(
            NearUnitRootWarning(
                f"fitted VAR model is near a unit root (companion spectral radius {radius:.4f}); "
                f"bootstrap paths may be unreliable",
                context={"spectral_radius": radius},
            ),
            stacklevel=3,
        )
    return radius


__all__ = [
    "ar_spectral_radius",
    "check_ar_stability",
    "var_spectral_radius",
    "check_var_stability",
]
