"""Deterministic call-graph profiling with cProfile.

Per workload: dump a ``.prof`` (open with snakeviz) and print the top functions
by total (self) time — the call-graph view of where wall-clock goes.

    python -m profiling.cprofile_run                # all workloads
    python -m profiling.cprofile_run residual_ar    # one workload
    snakeviz profiling/REPORTS/cprofile_residual_ar.prof
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
from pathlib import Path

from profiling.workloads import WORKLOADS

REPORTS = Path(__file__).resolve().parent / "REPORTS"


def profile_one(name: str, thunk, *, top: int = 15) -> str:
    pr = cProfile.Profile()
    pr.enable()
    thunk()
    pr.disable()
    REPORTS.mkdir(exist_ok=True)
    pr.dump_stats(str(REPORTS / f"cprofile_{name}.prof"))
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(top)
    return buf.getvalue()


def main(argv: list[str] | None = None) -> None:
    wanted = set(argv) if argv else None
    for name, thunk in WORKLOADS:
        if wanted and name not in wanted:
            continue
        print(f"\n{'=' * 72}\ncProfile: {name}\n{'=' * 72}")
        print(profile_one(name, thunk))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
