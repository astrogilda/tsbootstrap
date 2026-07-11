"""Pins for the benchmark receipt comparator and the settled timing loop.

These are the structural pins of the phantom compiled-reduce regression (issue #247):
two honest single-draw grids from differently-conditioned boxes were compared in
absolute milliseconds and read as a 3.8x code regression that interleaved A/B and
hardware counters later refuted. The comparator rules that refuse that comparison,
and the timing loop that replaces the single-draw instrument, are pinned here with
synthetic fixtures plus the two real committed receipts. Everything imported is
stdlib-only, so no test in this module ever skips.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks._timing import DISCARD, REPEATS, time_stats
from benchmarks.compare_vs_arch import (
    ARCH_DRIFT_LIMIT,
    RATIO_WORSEN_LIMIT,
    compare_grids,
)

RESULTS = Path(__file__).resolve().parents[2] / "benchmarks" / "results"
RECEIPT_2026_07_05 = RESULTS / "vs_arch_ccx33_2026-07-05.json"
RECEIPT_2026_07_11 = RESULTS / "vs_arch_ccx33_2026-07-11.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _cell(method: str, n: int, b: int, arch_ms: float, cc_red_ms: float, **extra) -> dict:
    cell = {
        "method": method,
        "n": n,
        "B": b,
        "arch_ms": arch_ms,
        "cc_red_ms": cc_red_ms,
        "cc_red_r": round(cc_red_ms / arch_ms, 2),
    }
    cell.update(extra)
    return cell


def _grid(cells: list[dict], sentinel: dict[str, float] | None = None) -> dict:
    provenance: dict = {}
    if sentinel is not None:
        provenance["sentinel_1t_ms"] = sentinel
    return {"provenance": provenance, "cells": cells}


def _fail_flags(result: dict) -> list[dict]:
    fail_kinds = {
        "median_ratio_regression",
        "min_ratio_regression",
        "ms_regression",
        "sentinel_regression",
    }
    return [f for f in result["flags"] if f["kind"] in fail_kinds]


class TestComparatorVerdicts:
    def test_perfreg1_pattern_is_ruled_incomparable(self):
        # The exact issue-#247 shape, with the real committed numbers: arch (external
        # pinned code) drifted +22..+60%, so the boxes are in different states and the
        # 4.69 -> 17.90 ms cell must NOT be read as a code regression. The worsened
        # within-run ratio stays visible as a warning (it is below the failure floor).
        old = _grid(
            [
                _cell("IID", 2000, 10000, arch_ms=116.57, cc_red_ms=14.47),
                _cell("MovingBlock", 2000, 10000, arch_ms=121.75, cc_red_ms=4.69),
                _cell("CircularBlock", 2000, 10000, arch_ms=167.04, cc_red_ms=4.92),
                _cell("StationaryBlock", 2000, 10000, arch_ms=201.2, cc_red_ms=16.87),
            ]
        )
        new = _grid(
            [
                _cell("IID", 2000, 10000, arch_ms=186.97, cc_red_ms=22.13),
                _cell("MovingBlock", 2000, 10000, arch_ms=156.35, cc_red_ms=17.9),
                _cell("CircularBlock", 2000, 10000, arch_ms=204.2, cc_red_ms=16.01),
                _cell("StationaryBlock", 2000, 10000, arch_ms=284.96, cc_red_ms=27.53),
            ]
        )
        result = compare_grids(old, new)
        assert result["verdict"] == "INCOMPARABLE"
        assert not result["comparable"]
        assert result["arch_drift_median"] > ARCH_DRIFT_LIMIT
        assert _fail_flags(result) == []
        # The MovingBlock ratio worsening is reported, as a warning, not a failure.
        assert any(f["kind"] == "median_ratio_worsened" for f in result["flags"])

    def test_committed_receipts_are_incomparable(self):
        # The real receipt pair that spawned issue #247 now mechanically produces the
        # verdict it should have received on day one.
        result = compare_grids(_load(RECEIPT_2026_07_05), _load(RECEIPT_2026_07_11))
        assert result["verdict"] == "INCOMPARABLE"
        assert result["cells"] == 16
        assert _fail_flags(result) == []

    def test_true_regression_is_flagged(self):
        # Arch stable (boxes comparable), the within-run ratio collapses from beating
        # arch 6.7x to barely beating it: a genuine code regression, named by cell.
        old = _grid([_cell("MovingBlock", 2000, 10000, arch_ms=120.0, cc_red_ms=18.0)])
        new = _grid([_cell("MovingBlock", 2000, 10000, arch_ms=122.0, cc_red_ms=110.0)])
        result = compare_grids(old, new)
        assert result["verdict"] == "FAIL"
        flags = [f for f in result["flags"] if f["kind"] == "median_ratio_regression"]
        assert flags and flags[0]["cell"] == ["MovingBlock", 2000, 10000]

    def test_identical_committed_grid_passes(self):
        # Contract with the real receipt schema, including the pre-min-field fallback:
        # a grid compared against itself passes with zero drift.
        grid = _load(RECEIPT_2026_07_05)
        result = compare_grids(grid, grid)
        assert result["verdict"] == "PASS"
        assert result["arch_drift_median"] == 0.0
        assert _fail_flags(result) == []
        assert any(f["kind"] == "legacy_median_comparison" for f in result["flags"])

    def test_disjoint_grids_raise(self):
        old = _grid([_cell("IID", 200, 999, arch_ms=10.0, cc_red_ms=1.0)])
        new = _grid([_cell("MovingBlock", 2000, 10000, arch_ms=120.0, cc_red_ms=5.0)])
        with pytest.raises(ValueError, match="share no"):
            compare_grids(old, new)


class TestSentinelSemantics:
    def test_sentinel_drift_under_box_drift_warns_not_fails(self):
        # Single-threaded wall time moves with a degraded box too (the #247 grids'
        # own arch columns moved 22-60% single-threaded), so sentinel drift on
        # incomparable boxes is a re-run request, never a code-regression verdict.
        old = _grid(
            [_cell("MovingBlock", 2000, 10000, arch_ms=120.0, cc_red_ms=5.0)],
            sentinel={"MovingBlock": 24.36},
        )
        new = _grid(
            [_cell("MovingBlock", 2000, 10000, arch_ms=156.0, cc_red_ms=6.5)],
            sentinel={"MovingBlock": 30.0},
        )
        result = compare_grids(old, new)
        assert result["verdict"] == "INCOMPARABLE"
        assert any(f["kind"] == "sentinel_drift_needs_rerun" for f in result["flags"])
        assert not any(f["kind"] == "sentinel_regression" for f in result["flags"])

    def test_sentinel_drift_on_stable_box_fails(self):
        # Same sentinel drift with a stable arch column: the box did not move, so a
        # >10% single-thread kernel-time move IS a code/toolchain regression.
        old = _grid(
            [_cell("MovingBlock", 2000, 10000, arch_ms=120.0, cc_red_ms=5.0)],
            sentinel={"MovingBlock": 24.36},
        )
        new = _grid(
            [_cell("MovingBlock", 2000, 10000, arch_ms=121.0, cc_red_ms=5.1)],
            sentinel={"MovingBlock": 30.0},
        )
        result = compare_grids(old, new)
        assert result["verdict"] == "FAIL"
        assert any(f["kind"] == "sentinel_regression" for f in result["flags"])


class TestLikeForLikeStatistics:
    def test_legacy_receipt_compares_median_to_median_never_min(self):
        # The old grid predates the min fields, so the ms check must fall back to
        # median-vs-median on BOTH sides. Comparing the new grid's min (4 ms) against
        # the old median (10 ms) would hide the 2.5x median regression entirely.
        old = _grid([_cell("MovingBlock", 2000, 10000, arch_ms=120.0, cc_red_ms=10.0)])
        new_cell = _cell(
            "MovingBlock",
            2000,
            10000,
            arch_ms=121.0,
            cc_red_ms=25.0,
            cc_red_ms_min=4.0,
        )
        # Keep the within-run ratio below the worsen limit so the ms check is the
        # only signal under test.
        new_cell["cc_red_r"] = old["cells"][0]["cc_red_r"]
        result = compare_grids(old, _grid([new_cell]))
        assert result["verdict"] == "FAIL"
        assert any(f["kind"] == "legacy_median_comparison" for f in result["flags"])
        ms_flags = [f for f in result["flags"] if f["kind"] == "ms_regression"]
        assert ms_flags and ms_flags[0]["field"] == "cc_red_ms"
        assert ms_flags[0]["worsened"] == pytest.approx(2.5)

    def test_min_ratio_regression_is_caught_below_the_median_floor(self):
        # A settled-min ratio collapse (0.04 -> 0.35) that still hides below the 0.6
        # median floor must fail via the min-ratio check: mins from the same run are
        # far less scheduling-amplified, so the tighter floor applies.
        old_cell = _cell("MovingBlock", 2000, 10000, arch_ms=120.0, cc_red_ms=12.0)
        old_cell.update({"cc_red_ms_min": 4.8, "cc_red_r_min": 0.04})
        new_cell = _cell("MovingBlock", 2000, 10000, arch_ms=121.0, cc_red_ms=12.1)
        new_cell.update({"cc_red_ms_min": 42.0, "cc_red_r_min": 0.35})
        result = compare_grids(_grid([old_cell]), _grid([new_cell]))
        assert result["verdict"] == "FAIL"
        assert any(f["kind"] == "min_ratio_regression" for f in result["flags"])


class TestTimeStats:
    def test_settling_runs_are_executed_but_never_timed(self):
        # The substantive contract: exactly discard + repeats invocations, exactly
        # 2 * repeats timer reads (so the settling runs cannot leak into either
        # statistic), and the returned (median, min) computed over the recorded
        # durations only. The fake timer makes the durations exact: 5, 1, 2, 3.
        calls = {"fn": 0, "timer": 0}

        def fn() -> None:
            calls["fn"] += 1

        ticks = iter([0.0, 5.0, 10.0, 11.0, 20.0, 22.0, 30.0, 33.0])

        def timer() -> float:
            calls["timer"] += 1
            return next(ticks)

        med, mn = time_stats(fn, repeats=4, discard=2, timer=timer)
        assert calls["fn"] == 6
        assert calls["timer"] == 8
        assert med == pytest.approx(2.5)
        assert mn == pytest.approx(1.0)

    def test_default_counts(self):
        calls = {"fn": 0}

        def fn() -> None:
            calls["fn"] += 1

        med, mn = time_stats(fn)
        assert calls["fn"] == DISCARD + REPEATS
        assert mn <= med

    def test_worsen_limit_is_material(self):
        # The comparator's worsen limit must stay meaningfully above run-to-run noise
        # (the settled statistics vary far less than 50%) while catching a 2x move.
        assert 1.2 <= RATIO_WORSEN_LIMIT <= 2.0
