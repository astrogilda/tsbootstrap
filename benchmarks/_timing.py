"""Settled timing statistics for the benchmark harness.

Kept in its own stdlib-only module (no arch, no numpy, no tsbootstrap import) so the
unit tests can exercise the timing loop on any environment, including ones without
the benchmark-only dependencies installed.

Why settled min-of-many instead of median-of-3: a short multi-threaded numba kernel
(a 4-18 ms parallel region) has a 2-4x run-to-run wall-time spread on cloud vCPUs,
and the samples taken right after a large allocate-and-free window are systematically
1.5-3x slower than settled ones. A median-of-3 single visit per cell honestly records
one draw from that distribution, and comparing two such draws across days manufactured
the phantom 3.8x compiled-reduce regression reported in issue #247. The remedy is to
discard a few settling runs at the timed shape, record many samples, and report both
the median (tracks the box) and the min (tracks the code).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from statistics import median

# Recorded samples per (path, cell): the median tracks the box, the min tracks the code.
REPEATS = 15
# Settling runs at the timed shape, never recorded: JIT and kernel-cache load, thread-pool
# ramp, and the post-materialization memory-reclaim window all land here.
DISCARD = 3


def time_stats(
    fn: Callable[[], object],
    repeats: int = REPEATS,
    discard: int = DISCARD,
    timer: Callable[[], float] = time.perf_counter,
) -> tuple[float, float]:
    """Return ``(median, min)`` wall seconds over ``repeats`` samples after ``discard`` runs.

    The ``discard`` settling runs execute ``fn`` at the real timed shape but are never
    timed or recorded, so warmup and reclaim effects cannot leak into either statistic.
    ``timer`` is injectable for deterministic tests only; production callers use the
    default ``time.perf_counter``.
    """
    for _ in range(discard):
        fn()
    samples = []
    for _ in range(repeats):
        start = timer()
        fn()
        samples.append(timer() - start)
    return float(median(samples)), float(min(samples))
