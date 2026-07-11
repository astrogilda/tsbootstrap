"""Mechanical comparator for two committed vs-arch benchmark grids.

Cross-day or cross-box comparison of raw milliseconds is only valid when the two
boxes were in the same state, and issue #247 is what happens without that check: a
grid from a degraded box was compared cell-by-cell in absolute ms against a healthy
one and read as a 3.8x code regression that did not exist. This tool encodes the
comparison rules so that verdict can never be produced by hand again:

1. The arch column is the box-state control: arch is external pinned code, so if its
   median absolute drift between the grids exceeds ``ARCH_DRIFT_LIMIT`` the boxes are
   in different states and NO absolute-ms verdict is emitted (``INCOMPARABLE``).
2. The within-run cc_red/arch ratios are box-robust and always compared; a cell whose
   ratio worsened by more than ``RATIO_WORSEN_LIMIT`` fails only when the new ratio is
   also material (above the floor); below the floor it is reported as a warning flag
   so a method-selective drift is still visible.
3. The single-thread sentinel is compared only when both grids carry it, and drift is
   a failure only when the boxes are otherwise comparable: single-threaded wall time
   moves under box degradation too (the #247 grids' own arch columns moved 22-60%).
4. Absolute cc_red milliseconds are compared like-for-like only: settled-min against
   settled-min, or median against median when either grid predates the min fields
   (never min against median, which would hide a real regression behind the tighter
   statistic).

Stdlib-only on purpose: the unit tests import this module on every environment, and
a receipt comparison must never depend on the library under test.

Exit codes: 0 = PASS, 1 = FAIL (regression flagged), 2 = INCOMPARABLE (box drift;
the box-robust checks were still run and reported).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

# Median |arch delta| beyond this means the box states are incomparable: arch is pinned
# external code, so only the box can move it. (The #247 pair sits far beyond: 22-60%.)
ARCH_DRIFT_LIMIT = 0.10
# A within-run cc_red/arch ratio (or like-for-like ms) worsened by more than this factor
# is flagged; it FAILS only above the matching ratio floor.
RATIO_WORSEN_LIMIT = 1.5
# Median-ratio failure floor: half the harness gate (1.2), so a worsening that still
# beats arch by 1.7x or more warns instead of failing.
RATIO_FLOOR = 0.6
# Min-ratio failure floor: settled mins are much less thread-scheduling-amplified than
# medians (both sides come from the same run), so the floor is tighter and catches a
# large regression that still hides below the median floor.
RATIO_FLOOR_MIN = 0.3
# Relative single-thread sentinel drift beyond this, on otherwise-comparable boxes,
# is a code/toolchain regression.
SENTINEL_DRIFT_LIMIT = 0.10

# Flag kinds that make the verdict FAIL; everything else is a warning.
_FAIL_KINDS = frozenset(
    {"median_ratio_regression", "min_ratio_regression", "ms_regression", "sentinel_regression"}
)


def _cells_by_key(grid: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Index a grid's cells by ``(method, n, B)``."""
    return {(c["method"], int(c["n"]), int(c["B"])): c for c in grid.get("cells", [])}


def _sentinel_map(grid: dict[str, Any]) -> dict[str, float]:
    """The per-method single-thread sentinel from a grid's provenance (empty if absent)."""
    value = grid.get("provenance", {}).get("sentinel_1t_ms")
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items() if v is not None}
    # Defensive: a scalar sentinel is treated as the MovingBlock cell.
    return {"MovingBlock": float(value)}


def _check_ratio(
    flags: list[dict[str, Any]],
    key: tuple[str, int, int],
    old_r: float | None,
    new_r: float | None,
    statistic: str,
    floor: float,
) -> None:
    """Flag one cell's ratio worsening: FAIL above the floor, WARN below it."""
    if old_r is None or new_r is None or old_r <= 0:
        return
    worsened = new_r / old_r
    if worsened <= RATIO_WORSEN_LIMIT:
        return
    kind = f"{statistic}_regression" if new_r > floor else f"{statistic}_worsened"
    flags.append(
        {
            "kind": kind,
            "cell": list(key),
            "old": old_r,
            "new": new_r,
            "worsened": round(worsened, 2),
        }
    )


def compare_grids(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Compare two ``bench_vs_arch.py --json`` payloads under the receipt rules.

    Returns ``{"verdict": "PASS" | "FAIL" | "INCOMPARABLE", "comparable": bool,
    "arch_drift_median": float, "cells": int, "flags": [...]}``. ``flags`` carries
    every warning and failure with its cell and numbers; a flag whose kind is in the
    failure set makes the verdict FAIL even when the boxes are incomparable (the
    ratio checks are within-run and therefore box-robust).
    """
    old_cells = _cells_by_key(old)
    new_cells = _cells_by_key(new)
    keys = sorted(set(old_cells) & set(new_cells))
    if not keys:
        raise ValueError("the two grids share no (method, n, B) cells; nothing to compare")

    flags: list[dict[str, Any]] = []

    # 1. Box-state control: median absolute drift of the arch column.
    drifts = [
        abs(new_cells[k]["arch_ms"] / old_cells[k]["arch_ms"] - 1.0)
        for k in keys
        if old_cells[k].get("arch_ms")
    ]
    arch_drift_median = float(median(drifts)) if drifts else 0.0
    comparable = arch_drift_median <= ARCH_DRIFT_LIMIT
    if not comparable:
        flags.append(
            {
                "kind": "box_drift",
                "detail": (
                    f"median |arch drift| {arch_drift_median:.1%} exceeds "
                    f"{ARCH_DRIFT_LIMIT:.0%}: the boxes are in different states, "
                    "absolute-ms columns are not comparable"
                ),
            }
        )

    # 2. Within-run ratios: box-robust, always compared.
    for k in keys:
        oc, nc = old_cells[k], new_cells[k]
        _check_ratio(flags, k, oc.get("cc_red_r"), nc.get("cc_red_r"), "median_ratio", RATIO_FLOOR)
        _check_ratio(
            flags, k, oc.get("cc_red_r_min"), nc.get("cc_red_r_min"), "min_ratio", RATIO_FLOOR_MIN
        )

    # 3. Single-thread sentinel: a code regression only on otherwise-comparable boxes;
    #    under box drift it degrades to a re-run request (1T wall time moves with the
    #    box too, so treating it as box-independent would re-manufacture #247).
    old_sentinel = _sentinel_map(old)
    new_sentinel = _sentinel_map(new)
    for method in sorted(set(old_sentinel) & set(new_sentinel)):
        if old_sentinel[method] <= 0:
            continue
        drift = new_sentinel[method] / old_sentinel[method] - 1.0
        if abs(drift) > SENTINEL_DRIFT_LIMIT:
            flags.append(
                {
                    "kind": "sentinel_regression" if comparable else "sentinel_drift_needs_rerun",
                    "method": method,
                    "old": old_sentinel[method],
                    "new": new_sentinel[method],
                    "drift": round(drift, 3),
                }
            )

    # 4. Absolute cc_red milliseconds, only when the boxes are comparable, and only
    #    like-for-like: min vs min, or median vs median when either grid predates the
    #    min fields.
    if comparable:
        legacy = any(
            "cc_red_ms_min" not in old_cells[k] or "cc_red_ms_min" not in new_cells[k] for k in keys
        )
        field = "cc_red_ms" if legacy else "cc_red_ms_min"
        if legacy:
            flags.append(
                {
                    "kind": "legacy_median_comparison",
                    "detail": (
                        "at least one grid predates the settled-min fields; the ms check "
                        "compares medians on BOTH sides (never min against median)"
                    ),
                }
            )
        for k in keys:
            old_ms = old_cells[k].get(field)
            new_ms = new_cells[k].get(field)
            if not old_ms or new_ms is None:
                continue
            worsened = new_ms / old_ms
            if worsened > RATIO_WORSEN_LIMIT:
                flags.append(
                    {
                        "kind": "ms_regression",
                        "cell": list(k),
                        "field": field,
                        "old": old_ms,
                        "new": new_ms,
                        "worsened": round(worsened, 2),
                    }
                )

    if any(f["kind"] in _FAIL_KINDS for f in flags):
        verdict = "FAIL"
    elif not comparable:
        verdict = "INCOMPARABLE"
    else:
        verdict = "PASS"
    return {
        "verdict": verdict,
        "comparable": comparable,
        "arch_drift_median": round(arch_drift_median, 4),
        "cells": len(keys),
        "flags": flags,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: compare OLD.json NEW.json and exit 0 (PASS), 1 (FAIL), or 2 (INCOMPARABLE)."""
    parser = argparse.ArgumentParser(
        description="Mechanical comparator for two committed vs-arch benchmark grids."
    )
    parser.add_argument("old", type=Path, help="the earlier grid receipt (JSON)")
    parser.add_argument("new", type=Path, help="the later grid receipt (JSON)")
    args = parser.parse_args(argv)

    old = json.loads(args.old.read_text())
    new = json.loads(args.new.read_text())
    try:
        result = compare_grids(old, new)
    except ValueError as exc:
        print(f"INCOMPARABLE: {exc}")
        return 2

    print(json.dumps(result, indent=2))
    if result["verdict"] == "FAIL":
        return 1
    if result["verdict"] == "INCOMPARABLE":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
