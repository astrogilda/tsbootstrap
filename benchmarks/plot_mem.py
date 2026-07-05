"""Render the peak-memory-vs-B figure: streaming reduce vs materialize-all.

Every point is read from the committed memory sweep under
``benchmarks/results/membench_2026-07-04.json`` (no hand-transcribed literals), so
the figure cannot drift from the data. It plots peak resident memory (MB over the
process floor) for the streaming ``bootstrap_reduce`` path against the materializing
``bootstrap(...).values()`` path as the replicate count B grows, at fixed n=2000, on
log-log axes. The subtitle is populated from the run provenance so the figure
self-documents the machine, versions, and date it came from.

Run:  python benchmarks/plot_mem.py   (needs the examples extra for matplotlib)
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


def _series(records: list[dict], path: str) -> tuple[list[int], list[float]]:
    rows = sorted((r for r in records if r["path"] == path), key=lambda r: r["B"])
    return [r["B"] for r in rows], [r["delta_mb"] for r in rows]


def _caption(prov: dict) -> str:
    return (
        f"{prov['cpu_model']}, {prov['cpu_count']} vCPU, {prov['os']}  |  "
        f"python {prov['python']}, numpy {prov['numpy']}, "
        f"numba {prov['numba']}, tsbootstrap {prov['tsbootstrap']}  |  {prov['date']}"
    )


def main() -> None:
    mem = json.loads((RESULTS / "membench_2026-07-04.json").read_text())
    records = mem["records"]
    b_v, values_mb = _series(records, "values")
    b_r, reduce_mb = _series(records, "reduce")

    # headline reduction at the largest common B, straight from the data
    b_max = max(set(b_v) & set(b_r))
    ratio = round(
        next(v for b, v in zip(b_v, values_mb) if b == b_max)
        / next(v for b, v in zip(b_r, reduce_mb) if b == b_max)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(
        b_v, values_mb, marker="s", ls="--", color=GREY, lw=2.2, ms=6.5,
        label="materialize every path, then reduce",
    )
    ax.loglog(
        b_r, reduce_mb, marker="o", ls="-", color=GREEN, lw=2.4, ms=7.5,
        label="tsbootstrap streaming reduce",
    )
    ax.annotate(
        "",
        xy=(b_max, next(v for b, v in zip(b_v, values_mb) if b == b_max)),
        xytext=(b_max, next(v for b, v in zip(b_r, reduce_mb) if b == b_max)),
        arrowprops=dict(arrowstyle="<|-|>", color=GREY, lw=1.4, mutation_scale=13),
    )
    ax.text(
        b_max * 0.9,
        next(v for b, v in zip(b_r, reduce_mb) if b == b_max) * 6,
        f"~{ratio}x lighter",
        color=GREEN,
        fontsize=13,
        fontweight="bold",
        ha="right",
    )

    ax.set_xlabel("number of bootstrap replicates B", fontsize=12)
    ax.set_ylabel("peak memory (MB)", fontsize=12)
    ax.grid(True, which="major", color=GRID, ls=":", zorder=0)
    ax.grid(True, which="minor", color=GRID, ls=":", lw=0.4, alpha=0.4, zorder=0)
    ax.legend(loc="upper left", fontsize=11, facecolor=BG, edgecolor=GRID, labelcolor=FG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle(
        "Streaming reduce holds peak memory flat as B grows (n=2000)",
        fontsize=14,
        fontweight="bold",
        color=FG,
    )
    fig.text(
        0.5, 0.008, _caption(mem["provenance"]), ha="center", va="bottom",
        color=GREY, fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = Path(__file__).parent / "memory_vs_B.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
