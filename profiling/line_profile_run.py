"""Line-level timing of the hot internal functions (programmatic line_profiler).

No ``@profile`` decorators in ``src``: ``LineProfiler`` matches on code objects,
so wrapping the real functions at runtime times them no matter how they are
called from inside ``bootstrap()``. The library stays clean and importable
without line_profiler installed.

    python -m profiling.line_profile_run             # all workloads
    python -m profiling.line_profile_run residual_ar
    cat profiling/REPORTS/line_profile.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

from line_profiler import LineProfiler

from profiling.hotpaths import HOT_FUNCTIONS
from profiling.workloads import WORKLOADS

REPORTS = Path(__file__).resolve().parent / "REPORTS"


def main(argv: list[str] | None = None) -> None:
    wanted = set(argv) if argv else None
    lp = LineProfiler()
    for fn in HOT_FUNCTIONS:
        lp.add_function(fn)
    lp.enable_by_count()
    try:
        for name, thunk in WORKLOADS:
            if wanted and name not in wanted:
                continue
            thunk()
    finally:
        lp.disable_by_count()
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "line_profile.txt"
    with out.open("w") as fh:
        lp.print_stats(stream=fh, output_unit=1e-3)
    lp.print_stats(output_unit=1e-3)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
