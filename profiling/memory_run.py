"""Peak Python-heap memory per workload (tracemalloc).

Surfaces which methods allocate the most, the (B, n[, d]) materialization and any
avoidable intermediate copies. For line-level memory of a single function use
scalene (``--memory``) or memory_profiler; tracemalloc gives the cheap per-call peak.

    python -m profiling.memory_run
    cat profiling/REPORTS/memory.txt
"""

from __future__ import annotations

import sys
import tracemalloc
from pathlib import Path

from profiling.workloads import WORKLOADS

REPORTS = Path(__file__).resolve().parent / "REPORTS"


def peak_mb(thunk) -> float:
    tracemalloc.start()
    try:
        thunk()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1e6


def main(argv: list[str] | None = None) -> None:
    wanted = set(argv) if argv else None
    rows: list[tuple[str, float]] = []
    for name, thunk in WORKLOADS:
        if wanted and name not in wanted:
            continue
        mb = peak_mb(thunk)
        rows.append((name, mb))
        print(f"{name:24s} peak={mb:8.2f} MB")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "memory.txt").write_text("".join(f"{n}\t{mb:.3f}\n" for n, mb in rows))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
