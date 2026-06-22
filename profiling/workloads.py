"""Canonical profiling workloads, one per method, exercising the batched paths.

Single source of truth shared by every profiler runner (cProfile, line_profiler,
memory_profiler, scalene, py-spy) so they all attribute time and memory to the
SAME representative calls of the public API. Edit sizes here once; every tool
follows.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tsbootstrap import (
    AR,
    ARIMA,
    IID,
    VAR,
    CircularBlock,
    MovingBlock,
    NonOverlappingBlock,
    ResidualBootstrap,
    SieveAR,
    StationaryBlock,
    TaperedBlock,
    bootstrap,
)

N = 2000
B = 999
SEED = 0


def _ar1(n: int = N) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + e[t]
    return x


def _var1(n: int = N, d: int = 3) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    a = 0.4 * np.eye(d) + 0.1
    x = np.zeros((n, d))
    for t in range(1, n):
        x[t] = a @ x[t - 1] + rng.standard_normal(d)
    return x


_X = _ar1()
_XV = _var1()


# (name, thunk), each thunk runs one full bootstrap() call.
WORKLOADS: list[tuple[str, Callable[[], object]]] = [
    ("iid", lambda: bootstrap(_X, method=IID(), n_bootstraps=B, random_state=SEED)),
    (
        "moving_block",
        lambda: bootstrap(
            _X, method=MovingBlock(block_length=20), n_bootstraps=B, random_state=SEED
        ),
    ),
    (
        "circular_block",
        lambda: bootstrap(
            _X, method=CircularBlock(block_length=20), n_bootstraps=B, random_state=SEED
        ),
    ),
    (
        "stationary_block",
        lambda: bootstrap(
            _X, method=StationaryBlock(avg_block_length=20), n_bootstraps=B, random_state=SEED
        ),
    ),
    (
        "nonoverlap_block",
        lambda: bootstrap(
            _X, method=NonOverlappingBlock(block_length=20), n_bootstraps=B, random_state=SEED
        ),
    ),
    (
        "tapered_block",
        lambda: bootstrap(
            _X, method=TaperedBlock(block_length=20), n_bootstraps=B, random_state=SEED
        ),
    ),
    (
        "moving_block_auto",
        lambda: bootstrap(
            _X, method=MovingBlock(block_length="auto"), n_bootstraps=B, random_state=SEED
        ),
    ),
    (
        "residual_ar",
        lambda: bootstrap(
            _X, method=ResidualBootstrap(model=AR(order=2)), n_bootstraps=B, random_state=SEED
        ),
    ),
    (
        "residual_arima",
        lambda: bootstrap(
            _X,
            method=ResidualBootstrap(model=ARIMA(order=(1, 1, 1))),
            n_bootstraps=B,
            random_state=SEED,
        ),
    ),
    (
        "residual_var",
        lambda: bootstrap(
            _XV, method=ResidualBootstrap(model=VAR(order=1)), n_bootstraps=B, random_state=SEED
        ),
    ),
    ("sieve_ar", lambda: bootstrap(_X, method=SieveAR(), n_bootstraps=B, random_state=SEED)),
]


def run_all_workloads() -> None:
    """Run every workload once (used as the scalene / py-spy entry point)."""
    for _name, thunk in WORKLOADS:
        thunk()
