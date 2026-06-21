# Profiling harness

Finds where `tsbootstrap` spends time and memory, so optimization targets the real
hot paths instead of guesses. Complements `benchmarks/` (asv tracks speed/memory
*across versions*; this harness tells you *where within a version* the cost is).

All runners share one workload set — `profiling/workloads.py` — one full
`bootstrap()` call per method at a representative size (`n=2000`, `B=999`). No
`@profile` decorators live in `src/`: the line profiler and cProfile attribute time
to the real source functions by wrapping them at runtime, so the shipped library
stays clean and importable without any profiling dependency.

## Install

```bash
pip install -e ".[profile]"   # scalene, line_profiler, memory_profiler, py-spy, snakeviz
```

## Tools and what each finds

| Tool | Command | What it answers |
|------|---------|-----------------|
| **cProfile** | `python -m profiling.cprofile_run [workload]` | Deterministic call graph — which functions own the self-time. Dumps `REPORTS/cprofile_<name>.prof` (open with `snakeviz`). |
| **line_profiler** | `python -m profiling.line_profile_run [workload]` | Per-line time inside the hot functions (`profiling/hotpaths.py`). Writes `REPORTS/line_profile.txt`. |
| **tracemalloc** | `python -m profiling.memory_run` | Peak Python-heap per workload — which methods allocate most. Writes `REPORTS/memory.txt`. |
| **scalene** | `scalene --html --outfile profiling/REPORTS/scalene.html -m profiling.scalene_target` | CPU split Python-vs-native + line memory. Best for "stuck in slow Python where native would do". |
| **py-spy** | `py-spy record -o profiling/REPORTS/flame.svg -- python -m profiling.scalene_target` | Sampling flamegraph, zero instrumentation overhead — sanity-checks the deterministic profilers. |

Everything deterministic in one shot:

```bash
python -m profiling.run_all
```

Reports land in `profiling/REPORTS/` (gitignored).

## Hot paths under watch

`profiling/hotpaths.py` lists the functions the line profiler instruments: the
block index kernels and executors, the PWSD auto-length path, and the recursive
AR/ARMA/VAR batched simulators. Add a function there when a new hot path appears.
