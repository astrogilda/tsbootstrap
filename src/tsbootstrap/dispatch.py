"""Spec-type -> executor registry.

Decouples the typed configuration (:mod:`tsbootstrap.methods`) from execution.
Each method registers a pure ``Executor`` keyed by its spec type. The registry
lives in its own module so engine modules can register without importing the
entry point (no import cycles).

An ``Executor`` produces ALL ``B`` bootstrap replicates at once:

    executor(prepared, spec, seeds, n_obs) -> (values, indices)

- ``prepared``   : the prepared state (the data array by default, or a fitted
  model context from a preparer).
- ``spec``       : the validated method spec.
- ``seeds``      : the list of ``B`` per-replicate ``numpy.random.SeedSequence`` objects
  (seed ``i`` is bound to replicate ``i``). A NumPy executor materializes its generators
  via :func:`tsbootstrap.rng.generators_from_seeds`; a compiled/GPU backend can instead
  derive counter-based keys from the seed entropy. Then it vectorises the numeric work.
- ``n_obs``      : number of observations in each replicate.
- returns ``(values (B, n[, d]) float64, indices (B, n) intp or None)``.
  ``indices`` is the original-observation indices for observation-resampling
  methods, or ``None`` for recursive methods.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.errors import Codes, MethodConfigError


@dataclass(frozen=True, slots=True)
class PreparationFailed:
    """Sentinel a preparer returns instead of a context when setup fails recoverably.

    Used by ``stability_policy="skip"``: a non-stationary fit produces this instead
    of raising, and the entry point returns an empty result flagged as failed.
    """

    reason: str


Executor = Callable[
    [object, object, "list[np.random.SeedSequence]", int],
    "tuple[NDArray[np.float64], NDArray[np.intp] | None]",
]
# Preparer: (data, spec, exog) -> prepared. Runs ONCE per bootstrap() call (e.g.
# fit a model). The prepared value is passed to the executor for every replicate.
# ``exog`` is the optional (n, k) exogenous array (None for most methods). The
# default preparer returns the data array unchanged.
Preparer = Callable[[NDArray[np.float64], object, object], object]

_EXECUTORS: dict[type, Executor] = {}
_PREPARERS: dict[type, Preparer] = {}

# Each concrete executor/preparer is registered with its precise parameter types
# (e.g. its concrete spec type) and returned UNCHANGED, so the registration site
# keeps full type checking. The registry stores them type-erased via a single
# localized ``cast`` -- the contravariant spec parameter is the only mismatch and
# it is resolved at lookup time by ``get_executor`` keying on the spec's type.
_E = TypeVar(
    "_E",
    bound=Callable[..., "tuple[NDArray[np.float64], NDArray[np.intp] | None]"],
)
_P = TypeVar("_P", bound=Callable[..., object])


def _identity_preparer(data: NDArray[np.float64], spec: object, exog: object) -> object:
    return data


def register_executor(spec_type: type) -> Callable[[_E], _E]:
    """Decorator: register ``fn`` as the executor for ``spec_type``."""

    def decorator(fn: _E) -> _E:
        _EXECUTORS[spec_type] = cast(Executor, fn)
        return fn

    return decorator


def get_executor(spec: object) -> Executor:
    """Look up the executor for a spec instance, or raise a structured error."""
    try:
        return _EXECUTORS[type(spec)]
    except KeyError:
        raise MethodConfigError(
            f"method {type(spec).__name__!r} is not implemented in this build",
            code=Codes.UNSUPPORTED_MODEL_FEATURE,
            hint="Supported methods are the MethodSpec union in tsbootstrap.methods.",
        ) from None


def is_registered(spec_type: type) -> bool:
    """Whether an executor is registered for ``spec_type``."""
    return spec_type in _EXECUTORS


def register_preparer(spec_type: type) -> Callable[[_P], _P]:
    """Decorator: register a one-time setup function for ``spec_type``."""

    def decorator(fn: _P) -> _P:
        _PREPARERS[spec_type] = cast(Preparer, fn)
        return fn

    return decorator


def get_preparer(spec: object) -> Preparer:
    """Return the preparer for a spec instance (the identity preparer by default)."""
    return _PREPARERS.get(type(spec), _identity_preparer)


__all__ = [
    "Executor",
    "Preparer",
    "PreparationFailed",
    "register_executor",
    "register_preparer",
    "get_executor",
    "get_preparer",
    "is_registered",
]
