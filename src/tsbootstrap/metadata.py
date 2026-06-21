"""Machine-readable metadata for every method, keyed by spec type.

This is the introspection surface that ``diagnose()`` and tooling read to reason
about which method fits a series and what each guarantees. It is data, not a
string factory: dispatch keys off the concrete spec type, not a name.
"""

from __future__ import annotations

from dataclasses import dataclass

from tsbootstrap.methods import (
    IID,
    CircularBlock,
    MovingBlock,
    NonOverlappingBlock,
    ResidualBootstrap,
    SieveAR,
    StationaryBlock,
    TaperedBlock,
)


@dataclass(frozen=True, slots=True)
class MethodMetadata:
    """Declarative facts about a bootstrap method."""

    name: str
    assumptions: tuple[str, ...]
    supports_multivariate: bool
    supports_exog: bool
    supports_indices: bool  # produces original-observation indices (OOB-capable)
    supports_oob: bool
    preserves_temporal_dependence: bool
    references: tuple[str, ...]
    complexity: str
    failure_modes: tuple[str, ...]


METHODS: dict[type, MethodMetadata] = {
    IID: MethodMetadata(
        name="iid",
        assumptions=("exchangeability (no serial dependence)",),
        supports_multivariate=True,
        supports_exog=False,
        supports_indices=True,
        supports_oob=True,
        preserves_temporal_dependence=False,
        references=("Efron 1979",),
        complexity="O(B*n)",
        failure_modes=("invalid under serial dependence; baseline only",),
    ),
    MovingBlock: MethodMetadata(
        name="moving_block",
        assumptions=("stationarity", "weak dependence"),
        supports_multivariate=True,
        supports_exog=False,
        supports_indices=True,
        supports_oob=True,
        preserves_temporal_dependence=True,
        references=("Kunsch 1989", "Liu-Singh 1992"),
        complexity="O(B*n)",
        failure_modes=("sensitive to block length", "edge effects"),
    ),
    CircularBlock: MethodMetadata(
        name="circular_block",
        assumptions=("stationarity", "weak dependence"),
        supports_multivariate=True,
        supports_exog=False,
        supports_indices=True,
        supports_oob=True,
        preserves_temporal_dependence=True,
        references=("Politis-Romano 1992",),
        complexity="O(B*n)",
        failure_modes=("sensitive to block length",),
    ),
    StationaryBlock: MethodMetadata(
        name="stationary_block",
        assumptions=("stationarity", "weak dependence"),
        supports_multivariate=True,
        supports_exog=False,
        supports_indices=True,
        supports_oob=True,
        preserves_temporal_dependence=True,
        references=("Politis-Romano 1994",),
        complexity="O(B*n)",
        failure_modes=("sensitive to expected block length",),
    ),
    NonOverlappingBlock: MethodMetadata(
        name="non_overlapping_block",
        assumptions=("stationarity", "weak dependence"),
        supports_multivariate=True,
        supports_exog=False,
        supports_indices=True,
        supports_oob=True,
        preserves_temporal_dependence=True,
        references=("Carlstein 1986",),
        complexity="O(B*n)",
        failure_modes=("higher variance than overlapping blocks",),
    ),
    TaperedBlock: MethodMetadata(
        name="tapered_block",
        assumptions=("stationarity", "weak dependence"),
        supports_multivariate=True,
        supports_exog=False,
        supports_indices=True,
        supports_oob=True,
        preserves_temporal_dependence=True,
        references=("Paparoditis-Politis 2001",),
        complexity="O(B*n)",
        failure_modes=("requires correct window energy normalization",),
    ),
    ResidualBootstrap: MethodMetadata(
        name="residual",
        assumptions=("correctly specified AR/ARIMA/VAR model", "stable (non-explosive) dynamics"),
        supports_multivariate=True,
        supports_exog=False,
        supports_indices=False,
        supports_oob=False,
        preserves_temporal_dependence=True,
        references=("Efron-Tibshirani 1993", "Kreiss-Lahiri 2012"),
        complexity="O(B*n*p) AR; O(B*n*p*d^2) VAR",
        failure_modes=("model misspecification", "near-unit-root instability"),
    ),
    SieveAR: MethodMetadata(
        name="sieve_ar",
        assumptions=("linear process admitting an AR(inf) representation",),
        supports_multivariate=False,
        supports_exog=False,
        supports_indices=False,
        supports_oob=False,
        preserves_temporal_dependence=True,
        references=("Buhlmann 1997", "Kreiss 1992"),
        complexity="O(B*n*p_hat)",
        failure_modes=("poor order selection", "near-unit-root instability"),
    ),
}


def metadata_for(spec: object) -> MethodMetadata:
    """Return the :class:`MethodMetadata` for a method spec instance."""
    return METHODS[type(spec)]


__all__ = ["MethodMetadata", "METHODS", "metadata_for"]
