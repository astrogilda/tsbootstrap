"""airspeed-velocity benchmarks for the bootstrap engines.

Run with ``asv run`` (tracks speed and peak memory per version, so regressions in
the vectorized engines are caught). Sizes are chosen to exercise the batched paths,
not to be exhaustive.
"""

from __future__ import annotations

import numpy as np

from tsbootstrap import (
    AR,
    VAR,
    MovingBlock,
    ResidualBootstrap,
    StationaryBlock,
    bootstrap,
)


def _ar1(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + e[t]
    return x


def _var1(n: int, d: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    a = 0.4 * np.eye(d) + 0.1
    x = np.zeros((n, d))
    for t in range(1, n):
        x[t] = a @ x[t - 1] + rng.standard_normal(d)
    return x


class BlockBootstrap:
    params = ([500, 2000], [999])
    param_names = ["n", "n_bootstraps"]

    def setup(self, n: int, n_bootstraps: int) -> None:
        self.x = _ar1(n)

    def time_moving_block(self, n: int, n_bootstraps: int) -> None:
        bootstrap(
            self.x, method=MovingBlock(block_length=20), n_bootstraps=n_bootstraps, random_state=0
        )

    def time_stationary_block(self, n: int, n_bootstraps: int) -> None:
        bootstrap(
            self.x,
            method=StationaryBlock(avg_block_length=20),
            n_bootstraps=n_bootstraps,
            random_state=0,
        )

    def peakmem_moving_block(self, n: int, n_bootstraps: int) -> None:
        bootstrap(
            self.x, method=MovingBlock(block_length=20), n_bootstraps=n_bootstraps, random_state=0
        )


class RecursiveBootstrap:
    params = ([500, 2000], [999])
    param_names = ["n", "n_bootstraps"]

    def setup(self, n: int, n_bootstraps: int) -> None:
        self.x = _ar1(n)
        self.xv = _var1(n, 3)

    def time_ar_residual(self, n: int, n_bootstraps: int) -> None:
        bootstrap(
            self.x,
            method=ResidualBootstrap(model=AR(order=2)),
            n_bootstraps=n_bootstraps,
            random_state=0,
        )

    def time_var_residual(self, n: int, n_bootstraps: int) -> None:
        bootstrap(
            self.xv,
            method=ResidualBootstrap(model=VAR(order=1)),
            n_bootstraps=n_bootstraps,
            random_state=0,
        )

    def peakmem_ar_residual(self, n: int, n_bootstraps: int) -> None:
        bootstrap(
            self.x,
            method=ResidualBootstrap(model=AR(order=2)),
            n_bootstraps=n_bootstraps,
            random_state=0,
        )
