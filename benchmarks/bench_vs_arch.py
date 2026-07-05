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

import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np

# arch is a BENCHMARK-ONLY dependency (it lives in the dev extra, never in the runtime
# core). Importing at module top is fine: asv installs the dev extra into its env, and
# the standalone runner prints a clear hint if it is missing.
from arch.bootstrap import (
    CircularBlockBootstrap,
    IIDBootstrap,
    MovingBlockBootstrap,
    StationaryBootstrap,
)

from tsbootstrap import (
    IID,
    CircularBlock,
    MovingBlock,
    StationaryBlock,
    bootstrap,
    bootstrap_reduce,
)

# --- fail-gate threshold ---------------------------------------------------------------
# The compiled backend beats arch on every overlapping method (compiled-reduce ratio
# 0.37-0.80 on a clean 8-core box). This gate guards against a future regression that makes
# the compiled reduce stop beating arch. It is intentionally generous (1.2, not 1.0) because
# shared-runner timing is noisy; the gate catches a gross regression, not a tight margin, and
# the authoritative check is a run on a dedicated/clean box. It gates ONLY the compiled-reduce
# path (cc.red); the numpy path materializes every replicate and is expected to trail arch.
RATIO_FAIL_THRESHOLD: float | None = 1.2

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
    # All four resampling methods tsbootstrap and arch both implement (the full overlap):
    # the i.i.d. bootstrap plus the three block families.
    "IID": (
        IIDBootstrap,
        IID,
    ),
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
    """Run the arch reference path: ``.apply(np.mean, reps=B)``.

    The i.i.d. bootstrap takes no block length; the block families take it positionally.
    """
    bs = (
        arch_cls(x, seed=SEED) if arch_cls is IIDBootstrap else arch_cls(BLOCK_LENGTH, x, seed=SEED)
    )
    return bs.apply(np.mean, reps=b)


def _run_ts_values(spec: object, x: np.ndarray, b: int) -> np.ndarray:
    """Run the public path that materialises every replicate, then ``.values()``."""
    return bootstrap(x, method=spec, n_bootstraps=b, random_state=SEED).values()


def _run_ts_reduce(spec: object, x: np.ndarray, b: int) -> object:
    """Run the public streaming path: reduce each replicate to a statistic (no full array)."""
    return bootstrap_reduce(
        x, method=spec, statistic=_mean, n_bootstraps=b, random_state=SEED
    ).statistics


def _run_ts_values_compiled(spec: object, x: np.ndarray, b: int) -> np.ndarray:
    """Run the opt-in compiled materialising path (fused index build plus gather)."""
    return bootstrap(x, method=spec, n_bootstraps=b, random_state=SEED, backend="compiled").values()


def _run_ts_reduce_compiled(spec: object, x: np.ndarray, b: int) -> object:
    """Run the opt-in compiled streaming path (fused index build, gather, and reduce)."""
    return bootstrap_reduce(
        x, method=spec, statistic="mean", n_bootstraps=b, random_state=SEED, backend="compiled"
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
        # Warm every engine once so JIT / import / kernel warmup is out of the timing
        # (the compiled backend compiles its kernels on first call).
        for arch_cls, spec_factory in _METHODS.values():
            _run_arch(arch_cls, self.x, 1)
            _run_ts_values(spec_factory(), self.x, 1)
            _run_ts_reduce(spec_factory(), self.x, 1)
            _run_ts_values_compiled(spec_factory(), self.x, 1)
            _run_ts_reduce_compiled(spec_factory(), self.x, 1)

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

    # --- compiled-backend ratio tracks (the opt-in fast path vs arch) ----------------- #

    def track_ratio_compiled_values_moving(self, n: int, b: int) -> float:
        return _measure_ratio(
            MovingBlockBootstrap,
            MovingBlock(block_length=BLOCK_LENGTH),
            self.x,
            b,
            _run_ts_values_compiled,
        )

    def track_ratio_compiled_values_circular(self, n: int, b: int) -> float:
        return _measure_ratio(
            CircularBlockBootstrap,
            CircularBlock(block_length=BLOCK_LENGTH),
            self.x,
            b,
            _run_ts_values_compiled,
        )

    def track_ratio_compiled_values_stationary(self, n: int, b: int) -> float:
        return _measure_ratio(
            StationaryBootstrap,
            StationaryBlock(avg_block_length=BLOCK_LENGTH),
            self.x,
            b,
            _run_ts_values_compiled,
        )

    def track_ratio_compiled_reduce_moving(self, n: int, b: int) -> float:
        return _measure_ratio(
            MovingBlockBootstrap,
            MovingBlock(block_length=BLOCK_LENGTH),
            self.x,
            b,
            _run_ts_reduce_compiled,
        )

    def track_ratio_compiled_reduce_circular(self, n: int, b: int) -> float:
        return _measure_ratio(
            CircularBlockBootstrap,
            CircularBlock(block_length=BLOCK_LENGTH),
            self.x,
            b,
            _run_ts_reduce_compiled,
        )

    def track_ratio_compiled_reduce_stationary(self, n: int, b: int) -> float:
        return _measure_ratio(
            StationaryBootstrap,
            StationaryBlock(avg_block_length=BLOCK_LENGTH),
            self.x,
            b,
            _run_ts_reduce_compiled,
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


def _print_table(cells: list[dict] | None = None) -> float:
    """Print a standings table for human reading: numpy and compiled paths vs arch.

    Ratios are ``tsbootstrap_time / arch_time``; below 1.0 means tsbootstrap is faster than
    arch. The compiled columns are the opt-in fast path; they need the ``[accel]`` extra.
    Returns the worst (largest) compiled-reduce ratio across all cells, for the fail-gate.

    If ``cells`` is passed, each cell's timings and ratios are appended to it as a dict, so
    the caller can serialize the full grid (see the ``--json`` mode) without re-timing.
    """
    worst_cc_red = 0.0
    header = (
        f"{'method':<16}{'n':>7}{'B':>8}{'arch ms':>11}"
        f"{'np.val ms':>12}{'r':>7}{'np.red ms':>12}{'r':>7}"
        f"{'cc.val ms':>12}{'r':>7}{'cc.red ms':>12}{'r':>7}"
    )
    print(header)
    print("-" * len(header))

    ns, bs = VsArch.params
    for name, (arch_cls, spec_factory) in _METHODS.items():
        for n in ns:
            x = _ar1(n)
            for b in bs:
                # Warm every engine for this cell so timing is real work, not warmup.
                _run_arch(arch_cls, x, 1)
                _run_ts_values(spec_factory(), x, 1)
                _run_ts_reduce(spec_factory(), x, 1)
                _run_ts_values_compiled(spec_factory(), x, 1)
                _run_ts_reduce_compiled(spec_factory(), x, 1)

                arch_t = _time_once(partial(_run_arch, arch_cls, x, b))
                vals_t = _time_once(partial(_run_ts_values, spec_factory(), x, b))
                red_t = _time_once(partial(_run_ts_reduce, spec_factory(), x, b))
                cc_vals_t = _time_once(partial(_run_ts_values_compiled, spec_factory(), x, b))
                cc_red_t = _time_once(partial(_run_ts_reduce_compiled, spec_factory(), x, b))

                def _r(t: float, base: float = arch_t) -> float:
                    return t / base if base > 0 else float("inf")

                worst_cc_red = max(worst_cc_red, _r(cc_red_t))
                if cells is not None:
                    cells.append(
                        {
                            "method": name,
                            "n": n,
                            "B": b,
                            "arch_ms": round(arch_t * 1e3, 2),
                            "np_val_ms": round(vals_t * 1e3, 2),
                            "np_val_r": round(_r(vals_t), 2),
                            "np_red_ms": round(red_t * 1e3, 2),
                            "np_red_r": round(_r(red_t), 2),
                            "cc_val_ms": round(cc_vals_t * 1e3, 2),
                            "cc_val_r": round(_r(cc_vals_t), 2),
                            "cc_red_ms": round(cc_red_t * 1e3, 2),
                            "cc_red_r": round(_r(cc_red_t), 2),
                        }
                    )
                print(
                    f"{name:<16}{n:>7}{b:>8}{arch_t * 1e3:>11.2f}"
                    f"{vals_t * 1e3:>12.2f}{_r(vals_t):>7.2f}{red_t * 1e3:>12.2f}{_r(red_t):>7.2f}"
                    f"{cc_vals_t * 1e3:>12.2f}{_r(cc_vals_t):>7.2f}"
                    f"{cc_red_t * 1e3:>12.2f}{_r(cc_red_t):>7.2f}"
                )

    print()
    print("Columns: np.* = default numpy backend, cc.* = opt-in compiled backend; r = ts/arch.")
    print("Ratio below 1.0 means tsbootstrap is faster than arch for that cell.")
    if RATIO_FAIL_THRESHOLD is None:
        print("RATIO_FAIL_THRESHOLD is None: fail-gate disabled.")
    else:
        print(
            f"RATIO_FAIL_THRESHOLD = {RATIO_FAIL_THRESHOLD} on the compiled-reduce path "
            f"(lower is better; worst compiled-reduce ratio this run = {worst_cc_red:.2f})."
        )
    return worst_cc_red


def _version(module: str) -> str | None:
    """Best-effort installed version of ``module`` (None if it is not importable)."""
    try:
        import importlib

        return getattr(importlib.import_module(module), "__version__", None)
    except Exception:
        return None


def _git_sha() -> str | None:
    """Short git SHA of the working tree, or None if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _provenance(worst_cc_red: float) -> dict:
    """Machine, OS, and library versions for the current run, for the JSON header."""
    return {
        "source": "bench_vs_arch.py --json",
        "machine": platform.node(),
        "cpu_model": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "arch": _version("arch"),
        "numba": _version("numba"),
        "tsbootstrap": _version("tsbootstrap"),
        "git_sha": _git_sha(),
        "date": datetime.now(timezone.utc).date().isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ratio_fail_threshold": RATIO_FAIL_THRESHOLD,
        "worst_cc_red_ratio": round(worst_cc_red, 2),
        "metric": (
            "ms = median-of-3 wall time; *_r = ts/arch time ratio "
            "(below 1.0 favors tsbootstrap); speedup = 1 / cc_red_r"
        ),
    }


if __name__ == "__main__":
    # Optional: ``--json PATH`` writes the full grid plus a provenance block alongside the
    # usual stdout table and PASS/FAIL exit, so a run can be committed under benchmarks/results/.
    json_path: str | None = None
    if "--json" in sys.argv:
        i = sys.argv.index("--json")
        if i + 1 >= len(sys.argv):
            print("usage: bench_vs_arch.py [--json PATH]")
            sys.exit(2)
        json_path = sys.argv[i + 1]

    cells: list[dict] = []
    worst = _print_table(cells)

    if json_path is not None:
        payload = {"provenance": _provenance(worst), "cells": cells}
        Path(json_path).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {json_path} ({len(cells)} cells)")

    if RATIO_FAIL_THRESHOLD is not None and worst > RATIO_FAIL_THRESHOLD:
        print(
            f"\nFAIL: worst compiled-reduce ratio {worst:.2f} exceeds the gate "
            f"{RATIO_FAIL_THRESHOLD} -- the compiled reduce regressed against arch."
        )
        sys.exit(1)
    print("\nPASS: compiled reduce stays within the gate vs arch.")
