"""Spec-type -> executor registry.

Decouples the typed configuration (:mod:`tsbootstrap.methods`) from execution.
Each method registers a pure ``Executor`` keyed by its spec type. The registry
lives in its own module so engine modules can register without importing the
entry point (no import cycles).

An ``Executor`` produces ONE bootstrap replicate:

    executor(data, spec, rng, n_obs) -> (values, indices)

- ``data``  : canonical ``(n, d)`` float64 array.
- ``spec``  : the validated method spec.
- ``rng``   : the per-replicate Generator (already bound to the sample index).
- ``n_obs`` : number of observations.
- returns ``(values (n, d) float64, indices (n,) intp or None)``. ``indices`` is
  the original-observation indices for observation-resampling methods, or
  ``None`` for recursive methods.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from tsbootstrap.errors import Codes, MethodConfigError

Executor = Callable[
    [NDArray[np.float64], object, np.random.Generator, int],
    "tuple[NDArray[np.float64], NDArray[np.intp] | None]",
]

_EXECUTORS: dict[type, Executor] = {}


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
            hint="See tsbootstrap.registry.METHODS for available methods.",
        ) from None


def is_registered(spec_type: type) -> bool:
    """Whether an executor is registered for ``spec_type``."""
    return spec_type in _EXECUTORS


__all__ = ["Executor", "register_executor", "get_executor", "is_registered"]
