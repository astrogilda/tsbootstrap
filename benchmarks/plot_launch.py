"""Render the performance summary figure used in the project README.

Every number in this figure is read from the committed benchmark results under
``benchmarks/results/`` (no hand-transcribed literals), so the figure cannot drift
from the data:

  * speed: ``results/vs_arch_ccx33_2026-07-05.json`` (the 16-cell head-to-head grid
    emitted by ``bench_vs_arch.py --json``).
  * memory: ``results/membench_2026-07-04.json`` (the streaming-reduce vs
    materialize-all peak-memory sweep).

Two panels: left, the compiled-reduce speedup over the arch library on the four
overlapping methods (n=2000, B=10000); right, peak memory of the streaming reduce
versus materializing every replicate (n=2000), on a linear scale so the reduction is
shown at true magnitude. The subtitle is populated from the run provenance so the
figure self-documents the machine, versions, and dates it came from.

Run:  python benchmarks/plot_launch.py   (needs the examples extra for matplotlib)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"

BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
BLUE = "#58a6ff"
GREEN = "#3fb950"
GREY = "#8b949e"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": FG,
        "axes.labelcolor": FG,
        "xtick.color": FG,
        "ytick.color": FG,
        "axes.edgecolor": GRID,
        "font.size": 12,
    }
)


def _fmt_mb(v: float) -> str:
    """Format a megabyte figure the way the panel labels read it."""
    if v >= 1000:
        return f"{v / 1000:.2f} GB"
    if v >= 10:
        return f"{v:.0f} MB"
    return f"{v:.1f} MB"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _speedup_at(cells: list[dict], method: str, n: int, b: int) -> float:
    """Compiled-reduce speedup (arch / ts = 1 / ratio) for one grid cell, one decimal."""
    cell = next(c for c in cells if c["method"] == method and c["n"] == n and c["B"] == b)
    return round(1.0 / cell["cc_red_r"], 1)


def _mem_delta(records: list[dict], path: str, b: int) -> float:
    """Peak-memory delta (MB over floor) for one path at one replicate count."""
    return next(r["delta_mb"] for r in records if r["path"] == path and r["B"] == b)


def _caption(speed_prov: dict, mem_prov: dict) -> str:
    """One-line provenance banner for the figure, from the two result headers."""
    return (
        f"{speed_prov['cpu_model']}, {speed_prov['cpu_count']} vCPU, {speed_prov['os']}  |  "
        f"python {speed_prov['python']}, numpy {speed_prov['numpy']}, "
        f"arch {speed_prov['arch']}, numba {speed_prov['numba']}, "
        f"tsbootstrap {speed_prov['tsbootstrap']}  |  "
        f"speed {speed_prov['date']}, memory {mem_prov['date']}"
    )


def main() -> None:
    speed = _load("vs_arch_ccx33_2026-07-05.json")
    mem = _load("membench_2026-07-04.json")
    cells = speed["cells"]
    records = mem["records"]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 6.2))

    # Left: compiled-reduce speedup vs arch at the headline workload (n=2000, B=10000).
    methods = ["IID", "MovingBlock", "CircularBlock", "StationaryBlock"]
    speedup = [_speedup_at(cells, m, 2000, 10000) for m in methods]
    yp = range(len(methods))
    ax_l.barh(yp, speedup, color=BLUE, height=0.6, zorder=3)
    ax_l.axvline(1.0, color=GREY, ls="--", lw=1.2, zorder=2)
    ax_l.text(1.04, -0.72, "arch = 1.0x", color=GREY, fontsize=10)
    for i, s in enumerate(speedup):
        ax_l.text(
            s + 0.6, i, f"{s:g}x", va="center", ha="left", color=FG, fontsize=14, fontweight="bold"
        )
    ax_l.set_yticks(list(yp))
    ax_l.set_yticklabels(methods, fontsize=12)
    ax_l.set_xlim(0, 40)
    ax_l.invert_yaxis()
    ax_l.set_xlabel("speedup vs the arch library", fontsize=12)
    ax_l.set_title("Faster", fontsize=18, fontweight="bold", color=FG, loc="left")
    ax_l.grid(axis="x", color=GRID, ls=":", zorder=0)
    for s in ("top", "right", "left"):
        ax_l.spines[s].set_visible(False)

    # Right: peak memory before/after, linear scale (streaming reduce vs materialize-all,
    # MovingBlock mean at n=2000).
    b_grid = [10000, 50000]
    work = [f"Streaming reduce\n(n=2000, B={b})" for b in b_grid]
    before = [_mem_delta(records, "values", b) for b in b_grid]
    after = [_mem_delta(records, "reduce", b) for b in b_grid]
    red = [f"{round(bf / af)}x less" for bf, af in zip(before, after)]
    lab_before = [_fmt_mb(v) for v in before]
    lab_after = [_fmt_mb(v) for v in after]
    x = range(len(work))
    w = 0.36
    ax_r.bar(
        [i - w / 2 for i in x],
        before,
        width=w,
        color=GREY,
        label="materialize every path",
        zorder=3,
    )
    ax_r.bar(
        [i + w / 2 for i in x],
        after,
        width=w,
        color=GREEN,
        label="tsbootstrap streaming reduce",
        zorder=3,
    )
    ax_r.set_ylim(0, 1850)
    for i in x:
        ax_r.text(
            i - w / 2,
            before[i] + 35,
            lab_before[i],
            ha="center",
            color=FG,
            fontsize=13,
            fontweight="bold",
        )
        ax_r.text(
            i + w / 2,
            after[i] + 35,
            f"{lab_after[i]}\n({red[i]})",
            ha="center",
            color=GREEN,
            fontsize=12.5,
            fontweight="bold",
            linespacing=1.2,
        )
    ax_r.set_xticks(list(x))
    ax_r.set_xticklabels(work, fontsize=11)
    ax_r.set_ylabel("peak memory (MB)", fontsize=12)
    ax_r.set_title("Lighter", fontsize=18, fontweight="bold", color=FG, loc="left")
    ax_r.grid(axis="y", color=GRID, ls=":", zorder=0)
    ax_r.legend(loc="upper left", fontsize=10, facecolor=BG, edgecolor=GRID, labelcolor=FG)
    for s in ("top", "right"):
        ax_r.spines[s].set_visible(False)

    fig.suptitle(
        "tsbootstrap: faster than arch, a fraction of the memory",
        fontsize=18,
        fontweight="bold",
        color=FG,
        x=0.5,
        y=0.985,
    )
    fig.text(
        0.5,
        0.008,
        _caption(speed["provenance"], mem["provenance"]),
        ha="center",
        va="bottom",
        color=GREY,
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    out = Path(__file__).parent / "launch_speed_memory.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
