"""Render the performance summary figure used in the project README.

Two panels from the measured benchmark numbers (see bench_vs_arch.py and
bench_bootstrap.py): left, the compiled-reduce speedup over the arch library on
the four overlapping methods (8 cores, n=2000, B=10000); right, peak memory of
the streaming reduce versus materializing every replicate (n=2000), on a linear
scale so the reduction is shown at true magnitude.

Run:  python benchmarks/plot_launch.py   (needs the examples extra for matplotlib)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def main() -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 6.2))

    # Left: speedup vs arch (compiled reduce, 8 cores, n=2000, B=10000).
    methods = ["IID", "MovingBlock", "CircularBlock", "StationaryBlock"]
    speed = [8.3, 25.0, 33.3, 12.5]
    yp = range(len(methods))
    ax_l.barh(yp, speed, color=BLUE, height=0.6, zorder=3)
    ax_l.axvline(1.0, color=GREY, ls="--", lw=1.2, zorder=2)
    ax_l.text(1.04, -0.72, "arch = 1.0x", color=GREY, fontsize=10)
    for i, s in enumerate(speed):
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

    # Right: peak memory before/after, linear scale (clean-box streaming reduce vs
    # materialize-all, MovingBlock mean at n=2000; measured on an 8-core machine).
    work = ["Streaming reduce\n(n=2000, B=10000)", "Streaming reduce\n(n=2000, B=50000)"]
    before = [389.0, 1944.0]
    after = [3.8, 20.2]
    red = ["103x less", "96x less"]
    lab_before = ["389 MB", "1.94 GB"]
    lab_after = ["3.8 MB", "20 MB"]
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
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(__file__).parent / "launch_speed_memory.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
