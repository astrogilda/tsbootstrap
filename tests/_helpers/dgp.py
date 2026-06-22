"""Synthetic data-generating processes shared across the test suite.

Plain functions, not pytest fixtures: tests call them inline with literal arguments
(often several times per test with different parameters), so a callable is the right
shape, a fixture injects a single value per test and cannot be parameterised at the
call site. Two AR(1) variants are intentionally distinct and must not be merged:

- :func:`ar1` starts from a transient (``x[0] = e[0]``); the unit tests use it.
- :func:`ar1_stationary` starts from the stationary marginal
  (``x[0] = e[0]/sqrt(1-phi**2)``); the coverage/reference statistical gates need a
  well-defined mean and variance from ``t = 0`` so finite-sample coverage is unbiased.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ar1(phi: float, n: int, seed: int, c: float = 0.0) -> NDArray[np.float64]:
    """AR(1) from a transient start: ``x[0] = e[0]``, ``x[t] = c + phi*x[t-1] + e[t]``."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = c + phi * x[t - 1] + e[t]
    return x


def ar1_stationary(n: int, phi: float, seed: int) -> NDArray[np.float64]:
    """AR(1) from the stationary marginal: ``x[0] = e[0]/sqrt(1-phi**2)``, mean 0, unit variance."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1.0 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def acf1(x: NDArray[np.float64]) -> float:
    """Lag-1 autocorrelation of a 1-D series."""
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])
