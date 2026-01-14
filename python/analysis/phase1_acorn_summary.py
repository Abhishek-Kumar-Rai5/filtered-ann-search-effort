"""Phase 1: summarise the ACORN SIFT1M reproduction runs against the paper.

Implements the pre-registered procedures of docs/phase1_acorn_repro.md
(section 3.5) on the raw outputs of `fse_acorn_repro`:
  * Table-3 value: distance computations at mean Recall@10 = 0.8, by linear
    interpolation between the first bracketing pair of efSearch values;
  * QPS: mean / median / stdev over the 50 timed batch trials per efSearch;
and compares with the published values (section 4).

Usage:
  python python/analysis/phase1_acorn_summary.py RUN_DIR [RUN_DIR ...] \
      --out results/phase1/summary
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Published SIFT1M values (paper Tables 3-6; Fig. 7a read approximately).
PAPER = {
    "table3_ndis_at_recall_0.8": {"acorn_gamma": 611.0, "acorn_1": 999.6},
    "table6_out_degree_acorn_gamma": [87.5, 384.0, 363.0, 25.3, 0.0],
    "table5_index_size_gb": {"acorn_gamma": 0.98, "acorn_1": 0.93},
    "table4_tti_s_96vcpu": {"acorn_gamma": 148.9, "acorn_1": 8.6},
}
RECALL_TARGET = 0.8


def recall_crossing(efs, recall, ndis, target=RECALL_TARGET):
    """First upward crossing of `target` in mean recall over sorted efs.

    Returns a dict with the bracketing efs values, the linearly interpolated
    mean distance computations at `target`, and the (uninterpolated) value at
    the upper bracket. If recall is already >= target at the smallest efs,
    or never reaches it, the corresponding fields are None and `status`
    says why.
    """
    order = np.argsort(np.asarray(efs))
    e = np.asarray(efs, dtype=float)[order]
    r = np.asarray(recall, dtype=float)[order]
    n = np.asarray(ndis, dtype=float)[order]
    if len(e) == 0:
        raise ValueError("empty sweep")
    if r[0] >= target:
        return {"status": "above_target_at_min_efs", "efs_lo": None,
                "efs_hi": float(e[0]), "ndis_interp": None,
                "ndis_at_hi": float(n[0]), "recall_lo": None,
                "recall_hi": float(r[0])}
    for i in range(1, len(e)):
        if r[i - 1] < target <= r[i]:
            w = (target - r[i - 1]) / (r[i] - r[i - 1])
            return {"status": "ok", "efs_lo": float(e[i - 1]),
                    "efs_hi": float(e[i]),
                    "ndis_interp": float(n[i - 1] + w * (n[i] - n[i - 1])),
                    "ndis_at_hi": float(n[i]), "recall_lo": float(r[i - 1]),
                    "recall_hi": float(r[i])}
    return {"status": "never_reaches_target", "efs_lo": None, "efs_hi": None,
            "ndis_interp": None, "ndis_at_hi": None, "recall_lo": None,
            "recall_hi": None}


def qps_at_recall(recall, qps, target):
    """Descriptive only: QPS at `target` recall, interpolating log10(QPS)
    linearly in recall between the bracketing points (curve sorted by
    recall). None if the curve does not span `target`."""
    pts = sorted(zip(recall, qps))
    for (r0, q0), (r1, q1) in zip(pts, pts[1:]):
        if r0 <= target <= r1 and r1 > r0:
            w = (target - r0) / (r1 - r0)
            return float(10 ** (math.log10(q0) + w * (math.log10(q1) -
                                                      math.log10(q0))))
    return None


def rel_gap(ours, ref):
    return None if ours is None else (ours - ref) / ref


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def summarise_run(run_dir: Path) -> dict:
    """One experiment directory as written by `fse_acorn_repro` stages:
    <run_dir>/gt_checks.json (optional) and <run_dir>/<method>/{build,sweep,
    verify,timing}.json + aggregate.csv + timing_trials.csv."""
    gt = _read_json(run_dir / "gt_checks.json")
    out = {"run_dir": str(run_dir), "experiment": run_dir.name,
           "gt_checks": gt, "methods": {}}
    for mdir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        method = mdir.name
        if method not in PAPER["table3_ndis_at_recall_0.8"]:
            continue
        build = _read_json(mdir / "build.json")
        sweep = _read_json(mdir / "sweep.json")
        verify = _read_json(mdir / "verify.json")
        timing_meta = _read_json(mdir / "timing.json")
        out.setdefault("draw", (build or sweep or {}).get("draw_label"))
        out.setdefault("build", (build or sweep or {}).get("build_label"))
        g = pd.read_csv(mdir / "aggregate.csv").sort_values("efs")
        cross = recall_crossing(g.efs, g.mean_recall, g.mean_ndis)
        ref3 = PAPER["table3_ndis_at_recall_0.8"][method]
        entry = {
            "table3": cross,
            "table3_paper": ref3,
            "table3_rel_gap": rel_gap(cross["ndis_interp"], ref3),
            "max_mean_recall": float(g.mean_recall.max()),
            "recall_at_min_efs": float(g.mean_recall.iloc[0]),
            "efs_values_swept": int(len(g)),
            "filter_violations": int(g.filter_violations.sum()),
            "distance_mismatches": int(g.distance_mismatches.sum()),
            "build": build, "sweep": sweep, "verify": verify,
            "timing": timing_meta,
        }
        tpath = mdir / "timing_trials.csv"
        if tpath.exists():
            t = pd.read_csv(tpath).groupby("efs").qps.agg(
                ["mean", "median", "std", "count"]).reset_index()
            curve = t.merge(g[["efs", "mean_recall", "mean_ndis"]], on="efs")
            entry["qps_at_recall_0.9_descriptive"] = qps_at_recall(
                curve.mean_recall, curve["mean"], 0.9)
            entry["curve"] = curve.to_dict(orient="records")
        out["methods"][method] = entry
    return out


def comparison_markdown(runs: list[dict]) -> str:
    lines = ["| run | method | ndis @ R@10=0.8 (interp.) | paper | gap | "
             "bracket efs | ndis @ upper efs | max recall |",
             "|---|---|---|---|---|---|---|---|"]
    for run in runs:
        for m, s in run["methods"].items():
            c = s["table3"]
            interp = c["ndis_interp"]
            gap = s["table3_rel_gap"]
            lines.append(
                f"| build {run['build']} / draw {run['draw']} | {m} | "
                f"{'n/a' if interp is None else f'{interp:.1f}'} | "
                f"{s['table3_paper']:.1f} | "
                f"{'n/a' if gap is None else f'{100 * gap:+.1f} %'} | "
                f"{c['efs_lo']}–{c['efs_hi']} | "
                f"{'n/a' if c['ndis_at_hi'] is None else f'{c['ndis_at_hi']:.1f}'} | "
                f"{s['max_mean_recall']:.4f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    runs = [summarise_run(d) for d in args.run_dirs]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps({"paper": PAPER, "runs": runs}, indent=2))
    (args.out / "table3_comparison.md").write_text(
        comparison_markdown(runs) + "\n")
    print(comparison_markdown(runs))


if __name__ == "__main__":
    main()
