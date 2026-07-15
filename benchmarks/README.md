# tsbootstrap benchmarks

This directory contains the benchmark suite for tsbootstrap. It includes two
complementary tools: a head-to-head comparison against the `arch` library
(`bench_vs_arch.py`) and an internal regression suite for the block and
residual bootstrap engines (`bench_bootstrap.py`), both runnable as
[airspeed velocity](https://asv.readthedocs.io/en/stable/) (asv) benchmark
suites or as plain Python scripts.

Every multiplier in this file names, in the same sentence, the baseline it was
measured against and the axis (time or memory) it lives on.

---

## Contents

| File | Purpose |
|------|---------|
| `bench_vs_arch.py` | Head-to-head timing comparison against `arch` for the four overlapping resampling methods (add `--json PATH` to emit the full grid plus provenance) |
| `compare_vs_arch.py` | The only valid way to compare two committed grids (see [Comparing two grids](#comparing-two-grids)) |
| `_timing.py` | The settled timing loop shared by the harness and its unit tests (stdlib-only) |
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
| `results/vs_arch_ccx33_2026-07-11_settled.json` | Hetzner dedicated 8-vCPU EPYC-Milan, 2026-07-11 | **the canonical speed receipt**: the 16-cell grid from the settled min-of-15 harness (`*_ms` medians, `*_ms_min` settled mins, touchstone ratios, per-method single-thread sentinel) plus the extended provenance block (box state, repeat/discard parameters, versions, git SHA) |
| `results/vs_arch_ccx33_2026-07-05.json` | Hetzner ccx33, 2026-07-05 | historical grid from the old median-of-3 single-visit instrument. Its medians are single draws from a distribution with a 2-4x spread and are NOT citable; kept for history only |
| `results/vs_arch_ccx33_2026-07-11.json` | Hetzner ccx33, 2026-07-11 | historical grid from the same old instrument, taken on a degraded box (its arch column sits 22-60% above the 07-05 run, so its absolute ms are additionally not comparable to any other receipt); kept for history only |
| `results/cache_counters_ccx33_2026-07-11.json` | Hetzner ccx33, 2026-07-11 | hardware-counter comparison (perf stat) of the materializing vs fused-reduce paths |
| `results/membench_2026-07-04.json` | Hetzner ccx33, 2026-07-04 | the streaming-reduce vs materialize-all peak-memory sweep over B (n=2000), plus the same style of provenance block |

Regenerate the figures from these files with `python benchmarks/plot_launch.py`
and `python benchmarks/plot_mem.py`. Produce a fresh speed grid on a clean box with
`python benchmarks/bench_vs_arch.py --json results/vs_arch_<box>_<date>.json`.

Preflight for a receipt run: a dedicated, idle box (no other jobs), the performance
CPU governor where sysfs exposes one, `uv sync --extra dev --extra accel`, and fresh
caches (remove `__pycache__` directories and any numba cache). The provenance block
records the load average and governor so a violated preflight is visible in the
receipt itself.

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

Numbers below are read from `results/vs_arch_ccx33_2026-07-11_settled.json` (Hetzner
dedicated 8-vCPU AMD EPYC-Milan; python 3.12.3, numpy 2.4.6, arch 8.0.0, numba
0.65.1, tsbootstrap 0.6.1). The metric is the **speedup: arch time / tsbootstrap
time** (higher is better), i.e. the reciprocal of the committed `cc_red_r_min`
settled-min ratio for each cell (see [Reading the output](#reading-the-output)).

The benchmark spans four methods, two series lengths (n), and two replicate
counts (B). All 16 cells exceed 1.0x, faster than `arch` at every combination.

### 8 threads (default for an 8-core machine)

| Method | n=200, B=999 | n=200, B=10000 | n=2000, B=999 | n=2000, B=10000 |
|-----------------|--------------|----------------|---------------|-----------------|
| IID | 15x | 19x | 4.7x | 8.6x |
| MovingBlock | 38x | 61x | 9.8x | 26x |
| CircularBlock | 41x | 66x | 13x | 33x |
| StationaryBlock | 19x | 24x | 6.8x | 12x |

Read these as sustained gains of roughly 4.7x to 33x on the larger n=2000
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

`bootstrap_reduce_panel` bootstraps an entire panel of series in one pass,
fusing the work over all series without materializing the full panel tensor,
so peak memory tracks the statistic output rather than the replicate count.
It accepts two input forms: a list of unequal-length 1-D arrays, or a single
flat values array paired with CSR-style offsets (`indptr`).

The reduce returns the full per-series bootstrap distribution of the statistic
(`n_bootstraps x num_series`), so quantile and tail workflows on an estimator
are served directly with no replicate tensor. Use the materializing path only
when the workflow consumes the resampled paths themselves.

Three ways to compute the same per-series statistics on one panel, at three
panel sizes (AR(1) panel, n=200 observations per series, MovingBlock with
block_length=20, statistic mean, B=1000 replicates):

- **fused `bootstrap_reduce_panel`** with `backend="compiled"`: one fused pass
  over the whole panel
- **materialize-then-reduce**: build the full `(num_series, B, n)` tensor of
  resampled paths, then reduce it
- **per-series loop**: a Python loop calling `bootstrap_reduce` once per series

| Series | Fused time (s) | Fused mem (MiB) | Materialize time (s) | Materialize mem (MiB) | Loop time (s) | Loop mem (MiB) |
|-------:|---------------:|----------------:|----------------------:|----------------------:|--------------:|---------------:|
| 100    | 0.0125 | 0.9   | 1.846 | 157.1   | 1.794 | 3.9  |
| 1,000  | 0.0832 | 9.4   | 16.37 | 1,534.3 | 17.96 | 11.4 |
| 10,000 | 0.821  | 108.4 | 185.7 | 15,336.7 | 180.6 | 86.4 |

At 10,000 series the fused path is about 220x faster than the per-series loop
on the time axis (per-series-loop baseline), about 226x faster than
materialize-then-reduce on the time axis (materialize baseline), and about
141x lighter than materialize-then-reduce on the memory axis (materialize
baseline). Measured on a dedicated 8-vCPU Hetzner ccx33 box on 2026-07-14,
tsbootstrap 0.7.0 (Python 3.12.3, numpy 2.4.6, numba 0.66.0), with the settled
timing methodology in `_timing.py` (settling runs at the timed shape discarded,
then the median of the recorded repeats reported).

Memory values are peak-RSS deltas in binary megabytes (MiB, 2^20 bytes).

---

## Comparing two grids

Cross-day or cross-box comparison of absolute milliseconds is valid ONLY through the
comparator:

```sh
python benchmarks/compare_vs_arch.py results/vs_arch_OLD.json results/vs_arch_NEW.json
```

Exit code 0 is a pass, 1 flags a regression, 2 rules the grids incomparable. The rules
it encodes, learned from a phantom regression report (issue #247, where a grid from a
degraded box was read cell-by-cell against a healthy one as a 3.8x code regression):

- **The arch column is the box-state control.** arch is external pinned code, so only
  the box can move it. A median arch drift beyond 10% between two grids means the box
  states differ and no absolute-ms verdict is emitted.
- **The within-run ratios are the box-robust statistics.** `cc_red_r` (median) and
  `cc_red_r_min` (settled min) divide two timings from the same run on the same box,
  so they are compared across any pair of receipts. A worsening beyond 1.5x fails
  above a material floor (0.6 median, 0.3 settled min) and warns below it.
- **The single-thread sentinel** (`sentinel_1t_ms`, one compiled-reduce cell per method
  at one numba thread) tracks per-replicate cost without the parallel-scheduling
  noise. Its drift is a code regression only when the arch columns already prove the
  boxes comparable; on incomparable boxes it degrades to a re-run request, because
  single-threaded wall time moves with a degraded box too.
- **Milliseconds are compared like-for-like only**: settled min against settled min,
  or median against median when either receipt predates the min fields. Never min
  against median.

Known accepted blind spot (see DEC-017 in `docs/development/DECISION_LOG.md`): a real
regression that stays below the warning-to-failure floors and does not move the
sentinel is reported as a warning, not a failure. The structural dispatch tests in
`tests/unit/test_compiled.py` cover the known mechanism classes independently of any
timer.

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
- `cc.red ms` / `min` / `r`: tsbootstrap compiled backend, streaming reduce path; the
  settled minimum; ratio to arch

Every path in every cell is timed with the settled loop in `_timing.py`: three
untimed warm runs at the real `(n, B)` shape (JIT compilation, kernel-cache load,
thread-pool ramp, and the post-materialization memory-reclaim window all land
there), then fifteen recorded samples. The printed `ms` is the median of those
samples and tracks the box; the JSON additionally records the minimum per path
(`*_ms_min`), which tracks the code: a short multi-threaded numba kernel spreads
2-4x run to run on cloud vCPUs, so the median moves with scheduling load while the
settled minimum is stable. The headline `cc.red` ratio is taken against a short
arch touchstone re-timed immediately before that path, so drift across the grid
pass cannot leak into it. The ratio column `r` is tsbootstrap time over arch time,
so below 1.0 favors tsbootstrap. The speedup figures in the results tables above
are the reciprocal of the settled-min ratio (`speedup = 1 / cc_red_r_min`).

---

## When to use each path

| Goal | Recommended call |
|------|-----------------|
| Fast, reproducible samples (default) | `bootstrap(x, method=..., n_bootstraps=B)` |
| Fast, reproducible samples with opt-in speed | `bootstrap(x, ..., backend="compiled")` |
| Compute a single statistic per replicate (memory-efficient) | `bootstrap_reduce(x, ..., statistic=fn)` |
| Compute a statistic, maximum throughput | `bootstrap_reduce(x, ..., statistic="mean", backend="compiled")` |
| Multivariate (VAR) residual bootstrap at scale | `bootstrap_reduce(x, method=ResidualBootstrap(model=VAR(...)), ...)` |
| Panel of many series | `bootstrap_reduce_panel(panel, ...)` |
