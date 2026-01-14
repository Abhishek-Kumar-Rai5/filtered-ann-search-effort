"""Work out the oracle effort for the Phase 4 matrix and run its validation checks.

Usage:
  python python/analysis/phase4_matrix.py --matrix results/phase4/matrix \
      --rerun results/phase4/rerun --audit results/phase4/matrix/audit_acorn.json \
      [--nscan results/phase4/nscan] --out results/phase4/analysis
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ("prefilter", "postfilter", "acorn")
GRAPH_METHODS = ("postfilter", "acorn")
RESULT_COLS = ["query_id", "budget", "effective_list", "recall", "dist_exact",
               "dist_native", "filter_checks", "rounds", "last_fetch",
               "n_valid", "filter_violations", "distance_mismatches",
               "gt_tie_at_k"]


def read_neighbor_table(path: Path):
    raw = path.read_bytes()
    nq, k = np.frombuffer(raw[:16], dtype=np.uint64)
    nq, k = int(nq), int(k)
    ids = np.frombuffer(raw[16:16 + 8 * nq * k], dtype=np.int64)
    dist = np.frombuffer(raw[16 + 8 * nq * k:], dtype=np.float32)
    return ids.reshape(nq, k), dist.reshape(nq, k)


def load_rows(matrix_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(matrix_dir.glob("*/*/per_query.csv"))]
    if not frames:
        raise FileNotFoundError(f"no per_query.csv under {matrix_dir}")
    return pd.concat(frames, ignore_index=True)


def load_sweeps(matrix_dir: Path) -> list[dict]:
    return [json.loads(p.read_text())
            for p in sorted(matrix_dir.glob("*/*/sweep.json"))]


def check_completeness(df: pd.DataFrame, conditions, budgets, n_queries):
    expected = len(conditions) * n_queries * (1 + 2 * len(budgets))
    key = ["condition", "method", "budget", "query_id"]
    dup = int(df.duplicated(key).sum())
    missing = []
    for cond in conditions:
        for method in METHODS:
            bs = [0] if method == "prefilter" else budgets
            for b in bs:
                sub = df[(df.condition == cond) & (df.method == method)
                         & (df.budget == b)]
                if sub.query_id.nunique() != n_queries or len(sub) != n_queries:
                    missing.append((cond, method, b, len(sub)))
    numeric = df.select_dtypes("number")
    nonfinite = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())
    return {"expected_rows": expected, "rows": int(len(df)),
            "duplicates": dup, "incomplete_cells": missing,
            "nonfinite_values": nonfinite,
            "pass": len(df) == expected and dup == 0 and not missing
            and nonfinite == 0}


def check_filter(df: pd.DataFrame):
    v = int(df.filter_violations.sum())
    m = int(df.distance_mismatches.sum())
    return {"filter_violations": v, "distance_mismatches": m,
            "pass": v == 0 and m == 0}


def check_accounting(df: pd.DataFrame, thresholds: dict):
    pre = df[df.method == "prefilter"]
    pre_bad = int((pre.dist_exact != pre.condition.map(thresholds)).sum())
    ac = df[df.method == "acorn"]
    ac_bad = int((ac.dist_exact != ac.dist_native + 1).sum())
    po = df[df.method == "postfilter"]
    po_bad = int((po.dist_exact != po.dist_native).sum())
    return {"prefilter_not_T": pre_bad, "acorn_not_native_plus_1": ac_bad,
            "postfilter_exact_ne_native": po_bad,
            "pass": pre_bad == 0 and ac_bad == 0 and po_bad == 0}


def check_monotone(df: pd.DataFrame, recall_tol: float = 0.002):
    failures = []
    g = (df[df.method.isin(GRAPH_METHODS)]
         .groupby(["method", "condition", "budget"])[["recall", "dist_exact"]]
         .mean().reset_index().sort_values(["method", "condition", "budget"]))
    for (method, cond), sub in g.groupby(["method", "condition"]):
        r = sub.recall.to_numpy()
        d = sub.dist_exact.to_numpy()
        if np.any(np.diff(r) < -recall_tol):
            failures.append((method, cond, "recall"))
        if np.any(np.diff(d) < -1e-9):
            failures.append((method, cond, "dist_exact"))
    pre = df[df.method == "prefilter"]
    pre_bad = int((pre.recall != 1.0).sum())
    return {"failures": failures, "prefilter_recall_not_1": pre_bad,
            "pass": not failures and pre_bad == 0}


def check_s1_identity(df: pd.DataFrame, s1_random: str, s1_clustered: str,
                      matrix_dir: Path | None = None):
    a = df[df.condition == s1_random].sort_values(["method", "budget", "query_id"])
    b = df[df.condition == s1_clustered].sort_values(["method", "budget", "query_id"])
    cols = ["method"] + RESULT_COLS
    rows_equal = (len(a) == len(b) and len(a) > 0
                  and a[cols].reset_index(drop=True)
                  .equals(b[cols].reset_index(drop=True)))
    raw_diffs = []
    if matrix_dir is not None:
        for method in METHODS:
            for pa in sorted((matrix_dir / method / s1_random).glob("raw_b*.bin")):
                pb = matrix_dir / method / s1_clustered / pa.name
                if not pb.exists() or pa.read_bytes() != pb.read_bytes():
                    raw_diffs.append(f"{method}/{pa.name}")
    return {"rows_equal": bool(rows_equal), "raw_differences": raw_diffs,
            "pass": bool(rows_equal) and not raw_diffs}


def check_inert(df: pd.DataFrame, k: int, matrix_dir: Path | None = None):
    failures = []
    po = df[df.method == "postfilter"]
    cols = ["recall", "dist_exact", "filter_checks", "rounds", "last_fetch",
            "n_valid"]
    for cond, sub in po.groupby("condition"):
        s = float(sub.s_achieved.iloc[0])
        bound = math.ceil(k / s - 1e-9)
        inert = sorted(b for b in sub.budget.unique() if b <= bound)
        if len(inert) < 2:
            continue
        ref = sub[sub.budget == inert[0]].sort_values("query_id")[cols]
        ref = ref.reset_index(drop=True)
        for b in inert[1:]:
            other = sub[sub.budget == b].sort_values("query_id")[cols]
            if not ref.equals(other.reset_index(drop=True)):
                failures.append((cond, int(b), "rows"))
            if matrix_dir is not None:
                pa = matrix_dir / "postfilter" / cond / f"raw_b{inert[0]}.bin"
                pb = matrix_dir / "postfilter" / cond / f"raw_b{b}.bin"
                if pa.read_bytes() != pb.read_bytes():
                    failures.append((cond, int(b), "raw"))
    return {"failures": failures, "pass": not failures}


def compute_oracle(df: pd.DataFrame, target: float = 0.9) -> pd.DataFrame:
    out = []
    for (method, cond), sub in df.groupby(["method", "condition"]):
        s = float(sub.s_achieved.iloc[0])
        corr = sub.correlation.iloc[0]
        ok = sub[sub.recall >= target - 1e-12].sort_values(["query_id", "budget"])
        first = ok.groupby("query_id").first()
        qids = np.sort(sub.query_id.unique())
        o = pd.DataFrame({"query_id": qids})
        o["method"] = method
        o["condition"] = cond
        o["correlation"] = corr
        o["s_achieved"] = s
        o["oracle_budget"] = o.query_id.map(first.budget).astype(float)
        o["oracle_dist"] = o.query_id.map(first.dist_exact).astype(float)
        o["censored"] = o.oracle_budget.isna()
        o.loc[o.censored, "oracle_dist"] = np.inf
        out.append(o)
    return pd.concat(out, ignore_index=True)


def check_v10(oracle: pd.DataFrame, s_low: float, s_high: float):
    results = []
    for method in GRAPH_METHODS:
        for corr in sorted(oracle.correlation.unique()):
            sub = oracle[(oracle.method == method) & (oracle.correlation == corr)]
            lo = sub[np.isclose(sub.s_achieved, s_low)].oracle_dist.to_numpy()
            hi = sub[np.isclose(sub.s_achieved, s_high)].oracle_dist.to_numpy()
            med_lo = float(np.median(lo))
            med_hi = float(np.median(hi))
            results.append({"method": method, "correlation": corr,
                            "median_oracle_low_s": med_lo,
                            "median_oracle_s1": med_hi,
                            "pass": med_lo > med_hi})
    return {"cells": results, "pass": all(r["pass"] for r in results)}


def join_scan(df: pd.DataFrame, scan: pd.DataFrame):
    key = ["condition", "budget", "query_id"]
    acorn = df[df.method == "acorn"]
    s = scan[key + ["n_scanned", "dist_exact", "dist_native", "recall"]]
    j = acorn[key + ["dist_exact", "dist_native", "recall"]].merge(
        s, on=key, how="left", suffixes=("", "_scan"))
    integrity = {
        "acorn_rows": len(acorn),
        "unmatched": int(j.n_scanned.isna().sum()),
        "scan_rows": len(scan),
        "dist_exact_mismatch": int((j.dist_exact != j.dist_exact_scan).sum()),
        "dist_native_mismatch": int((j.dist_native != j.dist_native_scan).sum()),
        "recall_mismatch": int((j.recall != j.recall_scan).sum())}
    integrity["pass"] = (integrity["unmatched"] == 0
                         and integrity["scan_rows"] == len(acorn)
                         and integrity["dist_exact_mismatch"] == 0
                         and integrity["dist_native_mismatch"] == 0
                         and integrity["recall_mismatch"] == 0)
    out = df.merge(s[key + ["n_scanned"]].assign(method="acorn"),
                   on=key + ["method"], how="left")
    return out, integrity


def oracle_scan(oracle: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    a = df[df.method == "acorn"][["condition", "budget", "query_id", "n_scanned"]]
    a = a.astype({"budget": float})
    o = oracle.merge(a.rename(columns={"budget": "oracle_budget",
                                       "n_scanned": "oracle_scan"})
                     .assign(method="acorn"),
                     on=["method", "condition", "query_id", "oracle_budget"],
                     how="left")
    acorn = o.method == "acorn"
    o.loc[acorn & o.censored, "oracle_scan"] = np.inf
    return o


def not_dominated(d_lo: float, d_hi: float, f_lo: float, f_hi: float) -> bool:
    return not (d_lo <= d_hi and f_lo <= f_hi)


def check_v10_revised(oracle: pd.DataFrame, s_low: float, s_high: float):
    results = []
    for method in GRAPH_METHODS:
        for corr in sorted(oracle.correlation.unique()):
            sub = oracle[(oracle.method == method) & (oracle.correlation == corr)]
            lo = sub[np.isclose(sub.s_achieved, s_low)]
            hi = sub[np.isclose(sub.s_achieved, s_high)]
            cell = {"method": method, "correlation": corr,
                    "median_D_low_s": float(np.median(lo.oracle_dist)),
                    "median_D_s1": float(np.median(hi.oracle_dist))}
            if method == "acorn":
                cell["median_F_low_s"] = float(np.median(lo.oracle_scan))
                cell["median_F_s1"] = float(np.median(hi.oracle_scan))
                cell["D_decreases"] = cell["median_D_low_s"] < cell["median_D_s1"]
                cell["pass"] = not_dominated(cell["median_D_low_s"],
                                             cell["median_D_s1"],
                                             cell["median_F_low_s"],
                                             cell["median_F_s1"])
            else:
                cell["pass"] = cell["median_D_low_s"] > cell["median_D_s1"]
            results.append(cell)
    return {"cells": results, "pass": all(r["pass"] for r in results)}


def check_determinism(main: pd.DataFrame, rerun: pd.DataFrame,
                      main_dir: Path | None = None,
                      rerun_dir: Path | None = None):
    key = ["condition", "method", "budget", "query_id"]
    cols = key + [c for c in RESULT_COLS if c not in key]
    m = main.merge(rerun[key], on=key)[cols].sort_values(key)
    r = rerun[cols].sort_values(key)
    equal = len(m) == len(r) and len(r) > 0 and \
        m.reset_index(drop=True).equals(r.reset_index(drop=True))
    raw_diffs = []
    if main_dir is not None and rerun_dir is not None:
        for p in sorted(rerun_dir.glob("*/*/raw_b*.bin")):
            rel = p.relative_to(rerun_dir)
            ids_r, d_r = read_neighbor_table(p)
            ids_m, d_m = read_neighbor_table(main_dir / rel)
            sub = rerun[(rerun.method == rel.parts[0])
                        & (rerun.condition == rel.parts[1])]
            qids = np.sort(sub.query_id.unique())
            allq = np.sort(main[(main.method == rel.parts[0])
                                & (main.condition == rel.parts[1])]
                           .query_id.unique())
            pos = np.searchsorted(allq, qids)
            if not (np.array_equal(ids_m[pos], ids_r)
                    and np.array_equal(d_m[pos], d_r)):
                raw_diffs.append(str(rel))
    return {"rows_compared": int(len(r)), "rows_equal": bool(equal),
            "raw_differences": raw_diffs,
            "pass": bool(equal) and not raw_diffs}


def aggregates(df: pd.DataFrame, target: float = 0.9) -> pd.DataFrame:
    g = df.groupby(["method", "condition", "correlation", "s_achieved",
                    "budget"])
    a = g.agg(mean_recall=("recall", "mean"),
              median_dist=("dist_exact", "median"),
              mean_dist=("dist_exact", "mean"),
              mean_latency_us=("latency_us", "mean"),
              queries=("query_id", "count")).reset_index()
    failed = g.recall.apply(lambda r: float((r < target - 1e-12).mean()))
    a["failed_fraction"] = failed.to_numpy()
    return a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--rerun", type=Path)
    ap.add_argument("--audit", type=Path)
    ap.add_argument("--nscan", type=Path,
                    help="ACORN sweeps with n_scanned (revised V10, section 16)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--budgets", type=str,
                    default="10,20,40,80,160,320,640,1280,2560")
    ap.add_argument("--queries", type=int, default=10000)
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]

    df = load_rows(args.matrix)
    sweeps = load_sweeps(args.matrix)
    conditions = sorted(df.condition.unique())
    thresholds = {s["condition"]: int(s["threshold"]) for s in sweeps}
    v = {}
    v["V1_completeness"] = check_completeness(df, conditions, budgets,
                                              args.queries)
    qh = {s["query_hash"] for s in sweeps}
    v["V2_same_query_set"] = {"query_hashes": sorted(qh),
                              "pass": len(qh) == 1}
    v["V3_ground_truth_identity"] = {
        "sweeps": len(sweeps),
        "conditions_with_identity": sorted({s["condition"] for s in sweeps}),
        "pass": len(sweeps) == len(conditions) * len(METHODS)}
    v["V4_filter"] = check_filter(df)
    acc = check_accounting(df, thresholds)
    replay = sum(int(s.get("replay_mismatches", 0)) for s in sweeps)
    replay_n = sum(int(s.get("replay_checked", 0)) for s in sweeps)
    acc["postfilter_replay_checked"] = replay_n
    acc["postfilter_replay_mismatches"] = replay
    audit_ok = False
    if args.audit and args.audit.exists():
        audit = json.loads(args.audit.read_text())
        acc["acorn_audit"] = {kk: audit[kk] for kk in
                              ("query_searches_checked", "offset_errors",
                               "behaviour_differences", "pass")}
        audit_ok = bool(audit["pass"])
    acc["pass"] = acc["pass"] and replay == 0 and replay_n > 0 and audit_ok
    v["V5_accounting"] = acc
    v["V6_monotone"] = check_monotone(df)
    s1 = [c for c in conditions if c.endswith("_s1.0000")]
    v["V7_s1_identity"] = check_s1_identity(
        df, "random_s1.0000", "clustered_s1.0000", args.matrix) \
        if len(s1) == 2 else {"pass": False, "reason": "s=1 conditions missing"}
    v["V8_inert_budgets"] = check_inert(df, args.k, args.matrix)
    if args.rerun:
        v["V9_determinism"] = check_determinism(df, load_rows(args.rerun),
                                                args.matrix, args.rerun)
    else:
        v["V9_determinism"] = {"pass": False, "reason": "no re-run given"}
    oracle = compute_oracle(df, args.target)
    s_vals = sorted(df.s_achieved.unique())
    v["V10_design_checkpoint"] = check_v10(oracle, s_vals[0], s_vals[-1])
    revised = None
    if args.nscan:
        df, scan_integrity = join_scan(df, load_rows(args.nscan))
        oracle = oracle_scan(oracle, df)
        revised = check_v10_revised(oracle, s_vals[0], s_vals[-1])
        revised["scan_join"] = scan_integrity
        revised["pass"] = revised["pass"] and scan_integrity["pass"]
    args.out.mkdir(parents=True, exist_ok=True)
    oracle.to_csv(args.out / "oracle.csv.gz", index=False)
    agg = aggregates(df, args.target)
    agg.to_csv(args.out / "aggregates.csv", index=False)
    osum = (oracle.assign(od=oracle.oracle_dist)
            .groupby(["method", "condition", "correlation", "s_achieved"])
            .agg(median_oracle_dist=("od", "median"),
                 censored_fraction=("censored", "mean"),
                 queries=("query_id", "count")).reset_index())
    if revised is not None:
        fs = (oracle[oracle.method == "acorn"]
              .groupby("condition").oracle_scan.median().rename("median_oracle_scan"))
        osum = osum.merge(fs, on="condition", how="left")
        osum.loc[osum.method != "acorn", "median_oracle_scan"] = np.nan
    osum.to_csv(args.out / "oracle_summary.csv", index=False)
    summary = {k2: {"pass": bool(val["pass"])} for k2, val in v.items()}
    report = {"checks": v, "summary": summary,
              "all_pass_V1_V10": all(x["pass"] for x in v.values())}
    if revised is not None:
        report["V10_revised"] = revised
        report["all_pass_V1_V9_V10_revised"] = revised["pass"] and all(
            x["pass"] for k2, x in v.items() if k2 != "V10_design_checkpoint")
    (args.out / "validation.json").write_text(
        json.dumps(report, indent=2, default=str))
    for k2, val in v.items():
        print(f"{k2}: {'PASS' if val['pass'] else 'FAIL'}")
    if revised is not None:
        print(f"V10_revised: {'PASS' if revised['pass'] else 'FAIL'} "
              f"(scan join {'ok' if revised['scan_join']['pass'] else 'FAILED'})")
        for c in revised["cells"]:
            print("  ", {k2: c[k2] for k2 in c if k2 not in ("method",)} | {"method": c["method"]})

if __name__ == "__main__":
    main()
