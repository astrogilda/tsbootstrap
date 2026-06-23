# tsbootstrap benchmarks

This directory contains the benchmark suite for tsbootstrap. It includes two
complementary tools: a head-to-head comparison against the `arch` library
(`bench_vs_arch.py`) and an internal regression suite for the block and
residual bootstrap engines (`bench_bootstrap.py`), both runnable as
[airspeed velocity](https://asv.readthedocs.io/en/stable/) (asv) benchmark
suites or as plain Python scripts.

---

## Contents

| File | Purpose |
|------|---------|
| `bench_vs_arch.py` | Head-to-head timing comparison against `arch` for the four overlapping resampling methods |
| `bench_bootstrap.py` | Internal speed and peak-memory regression suite for all block and residual engines |
| `__init__.py` | Package marker required by asv |

---

## Background: what gets compared

Both tsbootstrap and `arch` implement four overlapping resampling methods:

| tsbootstrap | arch |
|-------------|------|
| `IID` | `IIDBootstrap` |
| `MovingBlock` | `MovingBlockBootstrap` |
| `CircularBlock` | `CircularBlockBootstrap` |
| `StationaryBlock` | `StationaryBootstrap` |

The fairest comparison for speed is the **reduce path on both sides**:

- tsbootstrap: `bootstrap_reduce(x, method=..., statistic=mean, backend="compiled")`
- arch: `bs.apply(np.mean, reps=B)`

Both functions stream one replicate at a time, compute a statistic, and
accumulate only the scalar results. Neither materializes the full
`(B, n)` array of resampled paths. Comparing the materializing
`bootstrap(...).values()` path against `arch.apply` is also valid, because
materialization is an identical floor that both sides pay.

---

## The compiled backend

tsbootstrap 0.3.0 ships an optional compiled backend (powered by `numba`)
that is enabled by passing `backend="compiled"` to `bootstrap` or
`bootstrap_reduce`. It requires the `[accel]` extra:

```sh
pip install "tsbootstrap[accel]"
# or
uv add "tsbootstrap[accel]"
```

The default backend (pure NumPy + PCG64 random generator) is unchanged and
remains reproducible with no extra dependencies. The compiled backend is
faster but produces samples that are statistically equivalent to the default
rather than bit-identical (each backend uses its own deterministic stream).

---

## Head-to-head results (compiled reduce vs arch.apply)

Numbers below are from an 8-core dedicated CPU. The metric is the
**speedup: arch time / tsbootstrap time** (higher is better).

The benchmark spans four methods, two series lengths (n), and two replicate
counts (B). All 16 cells exceed 1.0x, faster than `arch` at every combination.

### 8 threads (default for an 8-core machine)

| Method | n=200, B=999 | n=200, B=10000 | n=2000, B=999 | n=2000, B=10000 |
|----------------|-------------|---------------|--------------|----------------|
| IID | 1.3x | 1.3x | 1.8x | 1.8x |
| MovingBlock | 1.6x | 1.6x | 1.9x | 1.8x |
| CircularBlock | 1.7x | 1.7x | 2.4x | 2.4x |
| StationaryBlock | 1.6x | 1.6x | 2.5x | 2.7x |

### 1 thread (single-threaded, removes parallelism advantage)

All four methods stay above 1.0x, ranging from 1.25x to 2.1x.

---

## Features not in arch

### Fused multivariate (VAR) reduce

For multivariate (VAR) residual bootstraps, `bootstrap_reduce` with the
compiled backend generates, gathers, and reduces one replicate at a time
without ever constructing the `(B, n, d)` tensor. Memory use is flat in the
number of replicates.

Example: B=20000 replicates, n=1000 observations, d=3 variables.

| Path | Peak memory |
|------|------------|
| tsbootstrap `bootstrap_reduce` (compiled) | ~1.4 MB |
| Materializing all paths first | ~480 MB |

### Panel-scale reduce

`bootstrap_reduce` accepts a list of ragged series and bootstraps the entire
panel in one pass. Compared to looping over each series individually:

Example: B=1000 replicates, 1000 series.

| Path | Wall time | Peak memory |
|------|----------|------------|
| `bootstrap_reduce` on the full panel | ~1x (baseline) | ~8 MB |
| Per-series loop | ~14x slower | ~1.6 GB |

---

## How to reproduce

### Prerequisites

```sh
# Install tsbootstrap with the compiled backend and arch
pip install "tsbootstrap[accel]"
pip install arch

# Or with uv
uv add "tsbootstrap[accel]" arch
```

### Standalone table (no asv required)

```sh
python benchmarks/bench_vs_arch.py
```

This prints a table with columns for each engine path (default NumPy
backend, compiled backend) and the speed ratio against `arch` for every
method-and-size combination.

### Full asv regression suite

The asv suite tracks speed and peak memory across commits, so regressions
in any engine are caught automatically.

```sh
# One-time setup
pip install asv virtualenv

# Run all benchmarks on the current commit
asv run HEAD^!

# Run on a range of commits (e.g. to compare main vs a feature branch)
asv run main..HEAD

# View results in a browser
asv publish
asv preview
```

asv installs `numba` and `arch` into its managed environment automatically
(they are listed in the `matrix.req` block in `asv.conf.json`).

### Internal regression suite only (no arch dependency)

```sh
asv run --bench BlockBootstrap HEAD^!
asv run --bench RecursiveBootstrap HEAD^!
```

---

## Reading the output

The standalone script (`python benchmarks/bench_vs_arch.py`) prints one row
per method-and-size cell with these columns:

- `arch ms`: wall-clock time for arch's `.apply(np.mean, reps=B)`
- `np.val ms` / `r`: tsbootstrap default backend, materializing path; ratio to arch
- `np.red ms` / `r`: tsbootstrap default backend, streaming reduce path; ratio to arch
- `cc.val ms` / `r`: tsbootstrap compiled backend, materializing path; ratio to arch
- `cc.red ms` / `r`: tsbootstrap compiled backend, streaming reduce path; ratio to arch

All timing values are the median of 3 back-to-back runs. Engines are warmed
before timing begins so that JIT compilation and import costs are excluded.
The ratio column `r` is tsbootstrap time over arch time, so below 1.0 favors
tsbootstrap. The speedup figures in the results tables above are the reciprocal
(`speedup = 1 / r`).

---

## When to use each path

| Goal | Recommended call |
|------|-----------------|
| Fast, reproducible samples (default) | `bootstrap(x, method=..., n_bootstraps=B)` |
| Fast, reproducible samples with opt-in speed | `bootstrap(x, ..., backend="compiled")` |
| Compute a single statistic per replicate (memory-efficient) | `bootstrap_reduce(x, ..., statistic=fn)` |
| Compute a statistic, maximum throughput | `bootstrap_reduce(x, ..., statistic="mean", backend="compiled")` |
| Multivariate (VAR) residual bootstrap at scale | `bootstrap_reduce(x, method=ResidualBootstrap(model=VAR(...)), ...)` |
| Panel of many series | `bootstrap_reduce(panel, ...)` |
