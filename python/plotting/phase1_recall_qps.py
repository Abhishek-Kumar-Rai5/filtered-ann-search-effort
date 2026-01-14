"""Phase 1 figures: Recall@10 vs QPS and vs distance computations for the
ACORN SIFT1M reproduction (cf. ACORN paper Fig. 7a / Table 3).

Usage:
  python python/plotting/phase1_recall_qps.py RUN_DIR --out FIG_DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Default validated categorical palette, slots 1-2 (dataviz skill; checked
# with validate_palette.js on the light surface: all checks pass).
SERIES = {"acorn_gamma": ("#2a78d6", "ACORN-γ", "o"),
          "acorn_1": ("#eb6834", "ACORN-1", "s")}
SURFACE = "#fcfcfb"
INK = "#1a1a19"
MUTED = "#6b6a63"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e4e3dc", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    # One experiment directory: <run_dir>/<method>/{aggregate,timing_trials}.csv
    agg = pd.concat(pd.read_csv(args.run_dir / m / "aggregate.csv")
                    for m in SERIES)
    timing = pd.concat(pd.read_csv(args.run_dir / m / "timing_trials.csv")
                       for m in SERIES)
    qps = timing.groupby(["method", "efs"]).qps.mean().reset_index()
    trials = int(timing.groupby(["method", "efs"]).size().min())
    curve = qps.merge(agg, on=["method", "efs"])
    args.out.mkdir(parents=True, exist_ok=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    for method, (color, label, marker) in SERIES.items():
        c = curve[curve.method == method].sort_values("efs")
        a1.plot(c["qps"], c.mean_recall, color=color, linewidth=2,
                marker=marker, markersize=5, markeredgecolor=SURFACE,
                label=label)
        d = agg[agg.method == method].sort_values("efs")
        a2.plot(d.mean_ndis, d.mean_recall, color=color, linewidth=2,
                label=label)
    a1.set_xscale("log")
    a1.set_xlabel(f"QPS (mean of {trials} batch trials, log scale)",
                  color=INK, fontsize=9)
    a1.set_ylabel("Recall@10", color=INK, fontsize=9)
    a1.set_title("Recall vs QPS (efs 10…760)", color=INK, fontsize=10,
                 loc="left")
    a2.axhline(0.8, color=MUTED, linewidth=0.8, linestyle="--")
    a2.text(a2.get_xlim()[0], 0.802, " Recall@10 = 0.8 (Table 3)",
            color=MUTED, fontsize=8, va="bottom")
    a2.set_xlabel("Mean distance computations per query", color=INK,
                  fontsize=9)
    a2.set_title("Recall vs effort (efs 10…760)", color=INK, fontsize=10,
                 loc="left")
    for ax in (a1, a2):
        _style(ax)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("ACORN on SIFT1M, equals(y) over 12 uniform attribute "
                 "values (selectivity ≈ 0.083)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "phase1_recall_qps_ndis.png", dpi=160,
                facecolor=SURFACE)
    print(args.out / "phase1_recall_qps_ndis.png")


if __name__ == "__main__":
    main()
