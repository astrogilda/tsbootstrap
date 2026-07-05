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
| `bench_vs_arch.py` | Head-to-head timing comparison against `arch` for the four overlapping resampling methods (add `--json PATH` to emit the full grid plus provenance) |
| `bench_bootstrap.py` | Internal speed and peak-memory regression suite for all block and residual engines |
| `plot_launch.py` | Renders the README performance figure from the committed results (no hardcoded numbers) |
| `plot_mem.py` | Renders the peak-memory-vs-B figure from the committed results |
| `results/` | Committed benchmark data, the single source of truth for every published number |
| `__init__.py` | Package marker required by asv |

### Committed results (`results/`)

Every published benchmark number (the README figure, the tables below, and the
papers) is read from committed JSON so nothing is hand-transcribed and the figure
cannot drift from the data:

| File | Source run | Contents |
|------|-----------|----------|
| `results/vs_arch_ccx33_2026-07-05.json` | Hetzner ccx33, 2026-07-05 | the 16-cell speed grid (`arch_ms`, per-path ratios) plus a provenance block (machine, CPU count, OS/kernel, python/numpy/arch/numba/tsbootstrap versions, git SHA, date, fail threshold, worst ratio) |
| `results/membench_2026-07-04.json` | Hetzner ccx33, 2026-07-04 | the streaming-reduce vs materialize-all peak-memory sweep over B (n=2000), plus the same style of provenance block |

Regenerate the figures from these files with `python benchmarks/plot_launch.py`
and `python benchmarks/plot_mem.py`. Produce a fresh speed grid on a clean box with
`python benchmarks/bench_vs_arch.py --json results/vs_arch_<box>_<date>.json`.

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

tsbootstrap ships an optional compiled backend (powered by `numba`)
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

Numbers below are read from `results/vs_arch_ccx33_2026-07-05.json` (Hetzner ccx33,
AMD EPYC-Milan, 8 vCPU; python 3.12.3, numpy 2.4.6, arch 8.0.0, numba 0.65.1). The
metric is the **speedup: arch time / tsbootstrap time** (higher is better), i.e. the
reciprocal of the committed `cc_red_r` ratio for each cell.

The benchmark spans four methods, two series lengths (n), and two replicate
counts (B). All 16 cells exceed 1.0x, faster than `arch` at every combination.

### 8 threads (default for an 8-core machine)

| Method | n=200, B=999 | n=200, B=10000 | n=2000, B=999 | n=2000, B=10000 |
|----------------|-------------|---------------|--------------|----------------|
| IID | 25x | 50x | 3.8x | 8.3x |
| MovingBlock | 100x | 100x | 12.5x | 25x |
| CircularBlock | 100x | 100x | 14x | 33x |
| StationaryBlock | 25x | 20x | 4.8x | 12.5x |

Read these as sustained gains of roughly 3.8x to 33x on the larger n=2000
workloads. The very large small-n multiples reflect `arch`'s per-replicate Python
callback overhead in `bs.apply`, which dominates its runtime when each resample is
cheap, so those cells measure that overhead as much as the compiled kernel.

### 1 thread (single-threaded, removes parallelism advantage)

All four methods stay above 1.0x single-threaded as well; re-run the harness with
one thread to reproduce the exact figures on your machine.

---

## Features not in arch

### Fused multivariate (VAR) reduce

For multivariate (VAR) residual bootstraps, `bootstrap_reduce` with the
compiled backend generates, gathers, and reduces one replicate at a time
without ever constructing the `(B, n, d)` tensor. Memory use is flat in the
number of replicates.

The reduce paths share one architecture, so a clean-box measurement of the
univariate case shows the shape (MovingBlock mean, n=2000, measured on an 8-core
machine). These figures are read from `results/membench_2026-07-04.json`:

| B | tsbootstrap `bootstrap_reduce` (compiled) | Materializing every replicate |
|------|------------|------------|
| 10000 | ~3.8 MB | ~389 MB (103x) |
| 50000 | ~20 MB | ~1.94 GB (96x) |

### Panel-scale reduce

`bootstrap_reduce` accepts a list of ragged series and bootstraps the entire
panel in one pass, fusing the work over all series without materializing the
full panel tensor, so peak memory tracks the statistic output rather than the
replicate count. Re-run the panel benchmark on your machine for exact figures.

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
