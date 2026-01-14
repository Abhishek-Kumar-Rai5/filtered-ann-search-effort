# Splits each query's recall loss into targets that are unreachable (U) and
# targets that were reachable but missed at that budget (N).

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

GT_MAGIC_BYTES = 8
GT_HEADER = 5


def read_gt(path: Path, condition_id: str, mask_hash: str):
    raw = path.read_bytes()
    h = np.frombuffer(raw[GT_MAGIC_BYTES:GT_MAGIC_BYTES + 8 * GT_HEADER], dtype=np.uint64)
    if f"{int(h[0]):016x}" != condition_id or f"{int(h[1]):016x}" != mask_hash:
        raise ValueError(f"ground-truth identity mismatch: {path}")
    nq, k = int(h[3]), int(h[4])
    off = GT_MAGIC_BYTES + 8 * GT_HEADER
    ids = np.frombuffer(raw[off:off + 8 * nq * k], dtype=np.int64).reshape(nq, k)
    return ids, f"{int(h[2]):016x}"


def read_raw(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    nq, k = (int(x) for x in np.frombuffer(raw[:16], dtype=np.uint64))
    return np.frombuffer(raw[16:16 + 8 * nq * k], dtype=np.int64).reshape(nq, k)


def bits_to_matrix(bits: np.ndarray, k: int) -> np.ndarray:
    b = np.asarray(bits, dtype=np.int64)[:, None]
    return ((b >> np.arange(k)) & 1).astype(bool)


def decompose(targets: np.ndarray, found: np.ndarray, reach: np.ndarray):
    k = targets.shape[1]
    hit = (targets[:, :, None] == found[:, None, :]).any(axis=2) & (targets >= 0)
    miss = ~hit
    unreach = ~reach
    return {"L": miss.sum(axis=1) / k,
            "U": unreach.sum(axis=1) / k,
            "N": (reach & miss).sum(axis=1) / k,
            "found_unreachable": (hit & unreach).sum(axis=1),
            "recall": hit.sum(axis=1) / k}


def bootstrap_mean_ci(x: np.ndarray, rng, resamples: int, ci: float):
    x = np.asarray(x, float)
    means = np.empty(resamples)
    for i in range(0, resamples, 250):
        n = min(250, resamples - i)
        means[i:i + n] = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [50 * (1 - ci), 50 * (1 + ci)])
    return float(x.mean()), float(lo), float(hi)


def reach_methods(cfg: dict) -> list:
    return sorted({m["reach"] for m in cfg["methods"].values() if m["reach"] != "pre"})


def run(cfg: dict) -> dict:
    man = yaml.safe_load(Path(cfg["manifest"]).read_text())
    k = cfg["k"]
    reach_dir = Path(cfg["reach_dir"])
    rows, val = [], {"recall_mismatch": 0, "identity_violations": 0,
                     "found_targets_unreachable": 0, "graph_not_subset_sem": 0,
                     "cap_not_subset_sem": 0, "seed_inconsistent": 0,
                     "pre_nonzero_U_or_N": 0, "gt_query_hash": set(), "rows": 0}
    comps = []
    for cond in man["conditions"]:
        name = cond["name"]
        targets_full, qhash = read_gt(Path(cond["gt"]), cond["condition_id"], cond["mask_hash"])
        val["gt_query_hash"].add(qhash)
        targets = targets_full[:, :k]
        for method, mcfg in cfg["methods"].items():
            sweep = Path(mcfg["sweep_dir"]) / name
            pq = pd.read_csv(sweep / "per_query.csv").sort_values(["budget", "query_id"])
            if mcfg["reach"] == "pre":
                reach = {"sem": np.ones_like(targets, bool)}
                reach["graph"] = reach["cap"] = reach["sem"]
            else:
                rc = pd.read_csv(reach_dir / mcfg["reach"] / f"{name}_reach.csv") \
                       .sort_values("query_id")
                reach = {s: bits_to_matrix(rc[s].to_numpy(), k)
                         for s in ("graph", "sem", "cap") if (rc[s] >= 0).all()}
                if "graph" in reach:
                    val["graph_not_subset_sem"] += int((reach["graph"] & ~reach["sem"]).sum())
                    val["cap_not_subset_sem"] += int((reach["cap"] & ~reach["sem"]).sum())
                if "seed0" in pq and mcfg["reach"] != "postfilter":
                    s0 = pq.groupby("query_id").seed0.agg(["nunique", "first"])
                    val["seed_inconsistent"] += int((s0["nunique"] != 1).sum())
                    val["seed_inconsistent"] += int(
                        (s0["first"].to_numpy() != rc.seed0.to_numpy()).sum())
            for b, g in pq.groupby("budget"):
                g = g.sort_values("query_id")
                found = read_raw(sweep / f"raw_b{b}.bin")
                d = decompose(targets, found, reach["sem"])
                val["recall_mismatch"] += int((~np.isclose(d["recall"], g.recall.to_numpy())).sum())
                val["identity_violations"] += int((~np.isclose(d["L"], d["U"] + d["N"])).sum())
                val["found_targets_unreachable"] += int(d["found_unreachable"].sum())
                extra = {}
                for s in ("graph", "cap"):
                    if s in reach:
                        extra[f"U_{s}"] = (~reach[s]).sum(axis=1) / k
                if mcfg["reach"] == "pre":
                    val["pre_nonzero_U_or_N"] += int(((d["U"] > 0) | (d["N"] > 0)).sum())
                rows.append(pd.DataFrame({
                    "method": method, "condition": name, "level": cond["level"],
                    "realization": cond["realization"], "s": cond["s"], "budget": b,
                    "query_id": g.query_id.to_numpy(), "L": d["L"], "U": d["U"],
                    "N": d["N"], **extra, "D": g.dist_exact.to_numpy(),
                    "F": g.n_scanned.to_numpy() if "n_scanned" in g else np.nan,
                    "tie": g.gt_tie_at_k.to_numpy()}))
                val["rows"] += len(g)
        for mname in reach_methods(cfg):
            rj = json.loads((reach_dir / mname / "reach.json").read_text())
            c = rj["conditions"][name]
            for sem in ("graph", "sem", "cap", "unfiltered"):
                if sem in c:
                    comps.append({"method": mname, "condition": name, "level": cond["level"],
                                  "realization": cond["realization"], "semantics": sem,
                                  "passing_clusters": cond["passing_clusters"],
                                  "distinct_seeds": c["distinct_seeds"],
                                  "seeds_failing_filter": c["seeds_failing_filter"], **c[sem]})
    val["gt_query_hash"] = sorted(val["gt_query_hash"])
    return {"rows": pd.concat(rows, ignore_index=True), "components": pd.DataFrame(comps),
            "validation": val}


def soundness(cfg: dict) -> dict:
    out = {}
    for mname in reach_methods(cfg):
        rj = json.loads((Path(cfg["reach_dir"]) / mname / "reach.json").read_text())
        out[mname] = {"checked": rj["soundness_checked"], "violations": rj["soundness_violations"],
                      "index_file_hash": rj["index_file_hash"]}
    return out


def aggregates(rows: pd.DataFrame, rng, resamples: int, ci: float) -> pd.DataFrame:
    out = []
    for (m, c, lvl, r, b), g in rows.groupby(["method", "condition", "level", "realization", "budget"]):
        rec = {"method": m, "condition": c, "level": lvl, "realization": r, "budget": b,
               "queries": len(g), "tie_fraction": g.tie.mean(), "D_mean": g.D.mean(),
               "F_mean": g.F.mean()}
        for col in ("L", "U", "N"):
            mu, lo, hi = bootstrap_mean_ci(g[col].to_numpy(), rng, resamples, ci)
            rec[col], rec[f"{col}_lo"], rec[f"{col}_hi"] = mu, lo, hi
        for col in ("U_graph", "U_cap"):
            rec[col] = g[col].mean() if col in g else np.nan
        rec["U_share_of_L"] = rec["U"] / rec["L"] if rec["L"] > 0 else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def figure(agg: pd.DataFrame, levels, methods, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e5e5e5", "grid.linewidth": 0.6})
    fig, axes = plt.subplots(len(methods), len(levels), figsize=(3 * len(levels), 2.4 * len(methods)),
                             sharex=True, sharey=True, squeeze=False)
    for i, m in enumerate(methods):
        for j, lvl in enumerate(levels):
            ax = axes[i, j]
            a = agg[(agg.method == m) & (agg.level == lvl)].groupby("budget")[["U", "N"]].mean()
            ax.stackplot(a.index, a.U, a.N, colors=["#eb6834", "#2a78d6"],
                         labels=["U: unreachable", "N: reachable, missed"])
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(lvl)
            if j == 0:
                ax.set_ylabel(f"{m}\nrecall loss")
            if i == len(methods) - 1:
                ax.set_xlabel("budget b")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Recall loss L = U + N at s = 0.01 (mean over realizations)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    res = run(cfg)
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    agg = aggregates(res["rows"], rng, cfg["bootstrap"]["resamples"], cfg["bootstrap"]["ci"])
    res["rows"].to_csv(out / "decomposition_rows.csv.gz", index=False)
    agg.to_csv(out / "aggregates.csv", index=False)
    res["components"].to_csv(out / "components.csv", index=False)
    v = res["validation"]
    v["soundness"] = soundness(cfg)
    v["checks"] = {
        "V-S1_identity": v["identity_violations"] == 0,
        "V-S2_soundness": all(x["violations"] == 0 for x in v["soundness"].values())
        and v["found_targets_unreachable"] == 0,
        "V-S3_pre_zero": v["pre_nonzero_U_or_N"] == 0,
        "V-S4_seed_consistent": v["seed_inconsistent"] == 0,
        "recall_consistent": v["recall_mismatch"] == 0,
        "subset_graph_cap_in_sem": v["graph_not_subset_sem"] == 0 and v["cap_not_subset_sem"] == 0,
        "single_query_hash": len(v["gt_query_hash"]) == 1}
    v["all_pass"] = all(v["checks"].values())
    (out / "validation.json").write_text(json.dumps(v, indent=2, default=str))
    levels = list(dict.fromkeys(c["level"] for c in yaml.safe_load(
        Path(cfg["manifest"]).read_text())["conditions"]))
    figure(agg, levels, [m for m in cfg["methods"] if m != "prefilter"],
           out / "figures" / "fig1_U_N_vs_budget.png")
    print(json.dumps(v["checks"], indent=1), "all_pass:", v["all_pass"])


if __name__ == "__main__":
    main(sys.argv[1])
