"""Run cProfile + line_profiler + tracemalloc across every workload.

One command to refresh all of REPORTS/. Sampling profilers (scalene, py-spy) wrap
the process and are run separately — see the README.

    python -m profiling.run_all
"""

from __future__ import annotations

from pathlib import Path

from profiling import cprofile_run, line_profile_run, memory_run

REPORTS = Path(__file__).resolve().parent / "REPORTS"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    print("\n##### cProfile (call graph, self-time) #####")
    cprofile_run.main()
    print("\n##### line_profiler (per-line, hot functions) #####")
    line_profile_run.main()
    print("\n##### tracemalloc (peak heap per workload) #####")
    memory_run.main()
    print(f"\nAll reports written under {REPORTS}")


if __name__ == "__main__":
    main()
