"""Head-to-head speed benchmark: tsbootstrap vs the ``arch`` library.

This is the competitor baseline whose absence let a 4x performance regression ship.
It measures the PUBLIC tsbootstrap entry points (``bootstrap(...).values()`` and
``bootstrap_reduce(..., statistic=mean)``), NOT the raw engine, and reports the ratio
``tsbootstrap_time / arch_time`` so a regression against the reference is visible.

The three block methods line up one-to-one:

    arch.MovingBlockBootstrap     <-> tsbootstrap.MovingBlock
    arch.CircularBlockBootstrap   <-> tsbootstrap.CircularBlock
    arch.StationaryBootstrap      <-> tsbootstrap.StationaryBlock

Sizes (``params = ([200, 2000], [999, 10000])`` i.e. ``(n, B)``) deliberately span
BOTH regimes the old asv suite missed:

  * engine-dominated: large ``n`` where the resampling math dominates.
  * wrapping-dominated: small ``n`` with large ``B`` where the public-API per-replicate
    Python wrapping (sample objects, metadata, coercion) dominates over the engine.

Run as an asv suite (``asv run``) or as a standalone table (``python benchmarks/bench_vs_arch.py``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial

import numpy as np

# arch is a BENCHMARK-ONLY dependency (it lives in the dev extra, never in the runtime
# core). Importing at module top is fine: asv installs the dev extra into its env, and
# the standalone runner prints a clear hint if it is missing.
from arch.bootstrap import (
    CircularBlockBootstrap,
    MovingBlockBootstrap,
    StationaryBootstrap,
)

from tsbootstrap import (
    CircularBlock,
    MovingBlock,
    StationaryBlock,
    bootstrap,
    bootstrap_reduce,
)

# --- fail-gate threshold (intentionally disabled for now) ------------------------------
# We are currently ~4x slower than arch, so a hard fail-gate would block CI. Leave it
# OFF until the optimization effort lands, then flip the constant to a real ratio.
#
# TODO(perf): once Phase 1 optimization lands and the ratio is at parity, set this to a
# real number (e.g. 1.1) and wire it into a CI assertion so future regressions fail loud.
RATIO_FAIL_THRESHOLD = None  # set to 1.1 once Phase 1 optimization lands

# Fixed seed: deterministic series + reproducible timings across runs.
SEED = 0
BLOCK_LENGTH = 20


def _ar1(n: int) -> np.ndarray:
    """A length-``n`` AR(1) series, the same shape the asv block suite uses."""
    rng = np.random.default_rng(SEED)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + e[t]
    return x


def _mean(values: np.ndarray, indices: np.ndarray | None) -> float:
    """Reducer for ``bootstrap_reduce`` (mean of a single replicate)."""
    return float(np.mean(values))


# Each entry maps a method name to its (arch class, tsbootstrap spec factory) pair.
# The spec factory takes no args; block length is the shared constant above.
_METHODS: dict[str, tuple[type, Callable[[], object]]] = {
    "MovingBlock": (
        MovingBlockBootstrap,
        lambda: MovingBlock(block_length=BLOCK_LENGTH),
    ),
    "CircularBlock": (
        CircularBlockBootstrap,
        lambda: CircularBlock(block_length=BLOCK_LENGTH),
    ),
    "StationaryBlock": (
        StationaryBootstrap,
        lambda: StationaryBlock(avg_block_length=BLOCK_LENGTH),
    ),
}


def _run_arch(arch_cls: type, x: np.ndarray, b: int) -> np.ndarray:
    """Run the arch reference path: ``.apply(np.mean, reps=B)``."""
    bs = arch_cls(BLOCK_LENGTH, x, seed=SEED)
    return bs.apply(np.mean, reps=b)


def _run_ts_values(spec: object, x: np.ndarray, b: int) -> np.ndarray:
    """Run the public path that materialises every replicate, then ``.values()``."""
    return bootstrap(x, method=spec, n_bootstraps=b, random_state=SEED).values()


def _run_ts_reduce(spec: object, x: np.ndarray, b: int) -> object:
    """Run the public streaming path: reduce each replicate to a statistic (no full array)."""
    return bootstrap_reduce(
        x, method=spec, statistic=_mean, n_bootstraps=b, random_state=SEED
    ).statistics


class VsArch:
    """asv-style head-to-head against arch on the three shared block methods.

    Tracks are kept separate so the three things are independently attributable:

      * ``time_arch_*``       -- the reference engine (arch's ``.apply``).
      * ``time_ts_values_*``  -- the public ``bootstrap(...).values()`` path.
      * ``time_ts_reduce_*``  -- the public streaming ``bootstrap_reduce`` path.

    The ``track_ratio_*`` methods report ``tsbootstrap_time / arch_time`` directly so the
    asv history surfaces the competitive standing, not just absolute wall-clock.
    """

    params = ([200, 2000], [999, 10000])
    param_names = ["n", "B"]

    # asv default repeat-count timing can be noisy for the ratio tracks; keep the engines
    # warm and the series fixed so the ratio is dominated by real work, not setup.
    def setup(self, n: int, b: int) -> None:
        self.x = _ar1(n)
        # Warm both engines once so JIT / import / kernel warmup is out of the timing.
        for arch_cls, spec_factory in _METHODS.values():
            _run_arch(arch_cls, self.x, 1)
            _run_ts_values(spec_factory(), self.x, 1)
            _run_ts_reduce(spec_factory(), self.x, 1)

    # --- absolute timing tracks (one per method x engine) ----------------------------- #

    def time_arch_moving(self, n: int, b: int) -> None:
        _run_arch(MovingBlockBootstrap, self.x, b)

    def time_arch_circular(self, n: int, b: int) -> None:
        _run_arch(CircularBlockBootstrap, self.x, b)

    def time_arch_stationary(self, n: int, b: int) -> None:
        _run_arch(StationaryBootstrap, self.x, b)

    def time_ts_values_moving(self, n: int, b: int) -> None:
        _run_ts_values(MovingBlock(block_length=BLOCK_LENGTH), self.x, b)

    def time_ts_values_circular(self, n: int, b: int) -> None:
        _run_ts_values(CircularBlock(block_length=BLOCK_LENGTH), self.x, b)

    def time_ts_values_stationary(self, n: int, b: int) -> None:
        _run_ts_values(StationaryBlock(avg_block_length=BLOCK_LENGTH), self.x, b)

    def time_ts_reduce_moving(self, n: int, b: int) -> None:
        _run_ts_reduce(MovingBlock(block_length=BLOCK_LENGTH), self.x, b)

    def time_ts_reduce_circular(self, n: int, b: int) -> None:
        _run_ts_reduce(CircularBlock(block_length=BLOCK_LENGTH), self.x, b)

    def time_ts_reduce_stationary(self, n: int, b: int) -> None:
        _run_ts_reduce(StationaryBlock(avg_block_length=BLOCK_LENGTH), self.x, b)

    # --- ratio tracks (tsbootstrap / arch; > 1 means tsbootstrap is slower) ----------- #

    def track_ratio_values_moving(self, n: int, b: int) -> float:
        return _measure_ratio(
            MovingBlockBootstrap, MovingBlock(block_length=BLOCK_LENGTH), self.x, b, _run_ts_values
        )

    def track_ratio_values_circular(self, n: int, b: int) -> float:
        return _measure_ratio(
            CircularBlockBootstrap,
            CircularBlock(block_length=BLOCK_LENGTH),
            self.x,
            b,
            _run_ts_values,
        )

    def track_ratio_values_stationary(self, n: int, b: int) -> float:
        return _measure_ratio(
            StationaryBootstrap,
            StationaryBlock(avg_block_length=BLOCK_LENGTH),
            self.x,
            b,
            _run_ts_values,
        )

    def track_ratio_reduce_moving(self, n: int, b: int) -> float:
        return _measure_ratio(
            MovingBlockBootstrap, MovingBlock(block_length=BLOCK_LENGTH), self.x, b, _run_ts_reduce
        )

    def track_ratio_reduce_circular(self, n: int, b: int) -> float:
        return _measure_ratio(
            CircularBlockBootstrap,
            CircularBlock(block_length=BLOCK_LENGTH),
            self.x,
            b,
            _run_ts_reduce,
        )

    def track_ratio_reduce_stationary(self, n: int, b: int) -> float:
        return _measure_ratio(
            StationaryBootstrap,
            StationaryBlock(avg_block_length=BLOCK_LENGTH),
            self.x,
            b,
            _run_ts_reduce,
        )


def _time_once(fn: Callable[[], object]) -> float:
    """Best-of-3 wall-clock seconds for ``fn`` (median-ish, robust to one-off jitter)."""
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def _measure_ratio(
    arch_cls: type,
    spec: object,
    x: np.ndarray,
    b: int,
    ts_runner: Callable[[object, np.ndarray, int], object],
) -> float:
    """Return ``tsbootstrap_time / arch_time`` for one method/size cell."""
    arch_t = _time_once(partial(_run_arch, arch_cls, x, b))
    ts_t = _time_once(partial(ts_runner, spec, x, b))
    return ts_t / arch_t if arch_t > 0 else float("inf")


def _print_table() -> None:
    """Print a plain pre-optimization standings table for human reading."""
    header = (
        f"{'method':<16}{'n':>7}{'B':>8}"
        f"{'arch ms':>12}{'ts.values ms':>16}{'ratio':>9}"
        f"{'ts.reduce ms':>16}{'ratio':>9}"
    )
    print(header)
    print("-" * len(header))

    ns, bs = VsArch.params
    for name, (arch_cls, spec_factory) in _METHODS.items():
        for n in ns:
            x = _ar1(n)
            for b in bs:
                # Warm both engines for this cell so timing is real work, not warmup.
                _run_arch(arch_cls, x, 1)
                _run_ts_values(spec_factory(), x, 1)
                _run_ts_reduce(spec_factory(), x, 1)

                arch_t = _time_once(partial(_run_arch, arch_cls, x, b))
                vals_t = _time_once(partial(_run_ts_values, spec_factory(), x, b))
                red_t = _time_once(partial(_run_ts_reduce, spec_factory(), x, b))

                vals_ratio = vals_t / arch_t if arch_t > 0 else float("inf")
                red_ratio = red_t / arch_t if arch_t > 0 else float("inf")
                print(
                    f"{name:<16}{n:>7}{b:>8}"
                    f"{arch_t * 1e3:>12.2f}{vals_t * 1e3:>16.2f}{vals_ratio:>9.2f}"
                    f"{red_t * 1e3:>16.2f}{red_ratio:>9.2f}"
                )

    print()
    if RATIO_FAIL_THRESHOLD is None:
        print("RATIO_FAIL_THRESHOLD is None: fail-gate disabled (we are still slower than arch).")
        print("Flip it to a real ratio (e.g. 1.1) once the optimization waves land.")
    else:
        print(
            f"RATIO_FAIL_THRESHOLD = {RATIO_FAIL_THRESHOLD} (lower ratio is better; > 1 means slower than arch)."
        )


if __name__ == "__main__":
    _print_table()
