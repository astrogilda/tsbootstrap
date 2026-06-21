"""Spec-type -> executor registry.

Decouples the typed configuration (:mod:`tsbootstrap.methods`) from execution.
Each method registers a pure ``Executor`` keyed by its spec type. The registry
lives in its own module so engine modules can register without importing the
entry point (no import cycles).

An ``Executor`` produces ALL ``B`` bootstrap replicates at once:

    executor(prepared, spec, generators, n_obs) -> (values, indices)

- ``prepared``   : the prepared state (the data array by default, or a fitted
  model context from a preparer).
- ``spec``       : the validated method spec.
- ``generators`` : the list of ``B`` per-replicate Generators (generator ``i`` is
  bound to replicate ``i``). The executor draws each replicate's randoms from its
  own generator, then vectorises the numeric work.
- ``n_obs``      : number of observations in each replicate.
- returns ``(values (B, n[, d]) float64, indices (B, n) intp or None)``.
  ``indices`` is the original-observation indices for observation-resampling
  methods, or ``None`` for recursive methods.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
    [object, object, "list[np.random.Generator]", int],
    "tuple[NDArray[np.float64], NDArray[np.intp] | None]",
]
# Preparer: (data, spec) -> prepared. Runs ONCE per bootstrap() call (e.g. fit a
# model). The prepared value is passed to the executor for every replicate. The
# default preparer returns the data array unchanged.
Preparer = Callable[[NDArray[np.float64], object], object]

_EXECUTORS: dict[type, Executor] = {}
_PREPARERS: dict[type, Preparer] = {}


def _identity_preparer(data: NDArray[np.float64], spec: object) -> object:
    return data


def register_executor(spec_type: type) -> Callable[[Executor], Executor]:
    """Decorator: register ``fn`` as the executor for ``spec_type``."""

    def decorator(fn: Executor) -> Executor:
        _EXECUTORS[spec_type] = fn
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


def register_preparer(spec_type: type) -> Callable[[Preparer], Preparer]:
    """Decorator: register a one-time setup function for ``spec_type``."""

    def decorator(fn: Preparer) -> Preparer:
        _PREPARERS[spec_type] = fn
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
