from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LEVELS = ["C100", "C1000", "C10000", "random"]
METHODS = ["prefilter", "postfilter", "acorn_1", "acorn_gamma"]
LAB = {"prefilter": "PRE", "postfilter": "POST", "acorn_1": "ACORN-1",
       "acorn_gamma": "ACORN-γ"}
COL = {"prefilter": "#555555", "postfilter": "#1baf7a", "acorn_1": "#2a78d6",
       "acorn_gamma": "#eb6834"}
N_BASE = 1_000_000


def t_ci(x, ci=0.95):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    h = stats.t.ppf(0.5 + ci / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def load(root: Path):
    aggs, comps, vals = [], [], {}
    for d in sorted(glob.glob(str(root / "analysis_s*"))):
        d = Path(d)
        if not (d / "aggregates.csv").exists():
            continue
        aggs.append(pd.read_csv(d / "aggregates.csv"))
        comps.append(pd.read_csv(d / "components.csv"))
        vals[d.name] = json.loads((d / "validation.json").read_text())
    agg = pd.concat(aggs, ignore_index=True)
    comp = pd.concat(comps, ignore_index=True)
    man = pd.concat(pd.read_csv(p) for p in [root / "conditions" / "conditions.csv"])
    agg = agg.merge(man[["name", "passing_clusters"]].rename(columns={"name": "condition"}),
                    on="condition", how="left")
    agg = agg.merge(man[["name", "s"]].rename(columns={"name": "condition"}),
                    on="condition", how="left")
    comp = comp.merge(man[["name", "s"]].rename(columns={"name": "condition"}),
                      on="condition", how="left")
    agg["recall"] = 1 - agg.L
    agg["U_share"] = agg.U / agg.L.where(agg.L > 0)
    return agg, comp, vals, man


def u_table(agg):
    u = agg[agg.budget == agg.groupby(["method", "condition"]).budget.transform("max")]
    u = u[["method", "s", "level", "realization", "condition", "passing_clusters",
           "U", "U_graph", "U_cap"]]
    rows = []
    for (m, s, lvl), g in u.groupby(["method", "s", "level"]):
        lo, hi = t_ci(g.U)
        rows.append({"method": m, "s": s, "level": lvl, "n": len(g), "U_mean": g.U.mean(),
                     "U_sd": g.U.std(ddof=1), "U_ci_lo": lo, "U_ci_hi": hi,
                     "U_min": g.U.min(), "U_max": g.U.max(),
                     "U_graph_mean": g.U_graph.mean(), "U_cap_mean": g.U_cap.mean(),
                     "passing_clusters_mean": g.passing_clusters.mean()})
    return u, pd.DataFrame(rows)


def paired_methods(u, root: Path):
    w = u.pivot_table(index=["s", "level", "realization", "condition"], columns="method",
                      values="U").reset_index()
    w["diff"] = w.acorn_gamma - w.acorn_1
    rows, qtests = [], []
    for (s, lvl), g in w.groupby(["s", "level"]):
        lo, hi = t_ci(g["diff"])
        rows.append({"s": s, "level": lvl, "n": len(g), "diff_mean": g["diff"].mean(),
                     "diff_ci_lo": lo, "diff_ci_hi": hi,
                     "gamma_higher": int((g["diff"] > 0).sum()),
                     "acorn1_higher": int((g["diff"] < 0).sum())})
    for d in sorted(glob.glob(str(root / "analysis_s*"))):
        r = pd.read_csv(Path(d) / "decomposition_rows.csv.gz",
                        usecols=["method", "condition", "budget", "query_id", "U"])
        r = r[r.method.isin(["acorn_1", "acorn_gamma"])]
        r = r[r.budget == r.budget.max()]
        p = r.pivot_table(index=["condition", "query_id"], columns="method", values="U")
        for cond, g in p.groupby(level=0):
            dd = g.acorn_gamma - g.acorn_1
            pv = stats.wilcoxon(dd).pvalue if (dd != 0).any() else 1.0
            qtests.append({"condition": cond, "queries_diff": int((dd != 0).sum()),
                           "gamma_worse_queries": int((dd > 0).sum()),
                           "acorn1_worse_queries": int((dd < 0).sum()), "p": pv})
    q = pd.DataFrame(qtests)
    q["p_holm"] = holm(q.p)
    return pd.DataFrame(rows), q


def fragmentation_tests(u):
    rank = {lvl: i for i, lvl in enumerate(LEVELS)}
    rows = []
    for (m, s), g in u.groupby(["method", "s"]):
        groups = [g[g.level == lvl].U.to_numpy() for lvl in LEVELS if (g.level == lvl).any()]
        allc = np.concatenate(groups)
        kw = stats.kruskal(*groups).pvalue if len(groups) > 1 and np.ptp(allc) > 0 else 1.0
        rho = stats.spearmanr(g.level.map(rank), g.U)
        rows.append({"method": m, "s": s, "kruskal_p": kw,
                     "spearman_rho_frag": rho.statistic, "spearman_p": rho.pvalue,
                     "argmax_level": g.groupby("level").U.mean().idxmax()})
    t = pd.DataFrame(rows)
    t["kruskal_p_holm"] = holm(t.kruskal_p.fillna(1.0))
    return t


def selectivity_tests(u):
    rows = []
    for (m, lvl), g in u.groupby(["method", "level"]):
        rho = stats.spearmanr(g.s, g.U) if np.ptp(g.U) > 0 else None
        rows.append({"method": m, "level": lvl,
                     **{f"U_s{s:g}": g[g.s == s].U.mean() for s in sorted(g.s.unique())},
                     "spearman_rho_s": rho.statistic if rho else np.nan,
                     "spearman_p": rho.pvalue if rho else np.nan})
    return pd.DataFrame(rows)


def baseline(root: Path, agg: pd.DataFrame, out: Path):
    b = pd.read_csv(root / "baseline" / "analysis" / "aggregates.csv")
    b["recall"] = 1 - b.L
    b["level"] = b.condition.map({"random_s0.0100": "random", "clustered_s0.0100": "C1000"})
    a = agg[(agg.realization == 0) & agg.level.isin(["random", "C1000"])].copy()
    t = pd.concat([b, a], ignore_index=True)[
        ["method", "level", "budget", "recall", "U", "N", "D_mean", "F_mean"]]
    t.to_csv(out / "table7_baselines_pre_post.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, lvl in zip(axes, ["C1000", "random"]):
        for m in METHODS:
            d = t[(t.method == m) & (t.level == lvl)].sort_values("budget")
            if m == "prefilter":
                ax.scatter(d.D_mean, d.recall, color=COL[m], marker="s", s=30, label=LAB[m], zorder=3)
            else:
                ax.plot(d.D_mean, d.recall, color=COL[m], lw=1.8, marker="o", ms=3, label=LAB[m])
        ax.set(xscale="log", title=f"s = 0.01, {lvl} (realization 0)",
               xlabel="mean exact distance computations D")
    axes[0].set_ylabel("Recall@10")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Recall vs effort: PRE / POST (Phase 4) vs ACORN-1 / ACORN-γ")
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig5_baselines_recall_vs_effort.png", dpi=150)
    plt.close(fig)
    return t


def figures(agg, comp, ut, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e5e5e5", "grid.linewidth": 0.6})
    out.mkdir(parents=True, exist_ok=True)
    ss = sorted(agg.s.unique())
    x = np.arange(len(LEVELS))
    u = agg[agg.budget == agg.groupby(["method", "condition"]).budget.transform("max")]

    fig, axes = plt.subplots(1, len(ss), figsize=(4.2 * len(ss), 3.6), sharey=True)
    for ax, s in zip(np.atleast_1d(axes), ss):
        for m in ("acorn_1", "acorn_gamma", "postfilter"):
            d = u[(u.method == m) & np.isclose(u.s, s)]
            off = {"acorn_1": -0.1, "acorn_gamma": 0.1, "postfilter": 0.0}[m]
            for i, lvl in enumerate(LEVELS):
                v = d[d.level == lvl].U
                ax.scatter(np.full(len(v), i) + off, v, color=COL[m], s=12, alpha=0.7)
            ax.plot(x, [d[d.level == l].U.mean() for l in LEVELS], color=COL[m], lw=2,
                    label=LAB[m])
        ax.set_xticks(x, LEVELS)
        ax.set_title(f"s = {s:g}")
        ax.set_xlabel("designed fragmentation (C; random = max)")
    np.atleast_1d(axes)[0].set_ylabel("U (unreachable fraction of 10 targets)")
    np.atleast_1d(axes)[0].legend(frameon=False)
    fig.suptitle("Topological loss U vs fragmentation (dots = realizations)")
    fig.tight_layout()
    fig.savefig(out / "fig1_U_vs_fragmentation.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(ss), len(LEVELS), figsize=(3.2 * len(LEVELS), 2.6 * len(ss)),
                             sharex=True, sharey=True, squeeze=False)
    for i, s in enumerate(ss):
        for j, lvl in enumerate(LEVELS):
            ax = axes[i, j]
            for m in ("postfilter", "acorn_1", "acorn_gamma"):
                d = agg[(agg.method == m) & np.isclose(agg.s, s) & (agg.level == lvl)]
                c = d.groupby("budget").recall.mean()
                ax.plot(c.index, c.values, color=COL[m], lw=1.8, marker="o", ms=2.5, label=LAB[m])
                ceil = 1 - d.U.mean()
                ax.axhline(ceil, color=COL[m], lw=0.8, ls=":")
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(lvl)
            if j == 0:
                ax.set_ylabel(f"s = {s:g}\nRecall@10")
            if i == len(ss) - 1:
                ax.set_xlabel("budget b")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Recall vs budget (mean over 5 realizations; dotted = ceiling 1 − U)")
    fig.tight_layout()
    fig.savefig(out / "fig2_recall_vs_budget.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(ss), figsize=(4.2 * len(ss), 3.4), sharey=True)
    for ax, s in zip(np.atleast_1d(axes), ss):
        for m, ls in (("acorn_1", "-"), ("acorn_gamma", "--")):
            for lvl, shade in zip(LEVELS, ["#cde2fb", "#6da7ec", "#256abf", "#104281"]):
                d = agg[(agg.method == m) & np.isclose(agg.s, s) & (agg.level == lvl)]
                c = d.groupby("budget").apply(lambda g: g.U.sum() / g.L.sum() if g.L.sum() > 0 else np.nan)
                ax.plot(c.index, c.values, color=shade if m == "acorn_1" else shade, ls=ls, lw=1.6,
                        label=f"{LAB[m]} {lvl}")
        ax.set(xscale="log", title=f"s = {s:g}", xlabel="budget b", ylim=(0, 1))
    np.atleast_1d(axes)[0].set_ylabel("U / L (topological share of loss)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=6, ncol=2)
    fig.suptitle("Share of recall loss that is unreachable (solid ACORN-1, dashed ACORN-γ)")
    fig.tight_layout()
    fig.savefig(out / "fig3_U_share_vs_budget.png", dpi=150)
    plt.close(fig)

    comp["lscc"] = comp.largest_scc / comp.nodes
    fig, axes = plt.subplots(1, len(ss), figsize=(4.2 * len(ss), 3.4), sharey=True)
    for ax, s in zip(np.atleast_1d(axes), ss):
        for m in ("acorn_1", "acorn_gamma"):
            for sem, ls in (("graph", ":"), ("sem", "-")):
                d = comp[(comp.method == m) & np.isclose(comp.s, s) & (comp.semantics == sem)]
                ax.plot(x, [d[d.level == l].lscc.mean() for l in LEVELS], color=COL[m], ls=ls,
                        lw=1.8, marker="o", ms=3,
                        label=f"{LAB[m]}, {'R_sem' if sem == 'sem' else 'plain graph'}")
        ax.set_xticks(x, LEVELS)
        ax.set_title(f"s = {s:g}")
    np.atleast_1d(axes)[0].set_ylabel("largest SCC / passing nodes")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Connectivity of the passing set: plain filtered graph vs ACORN semantics")
    fig.tight_layout()
    fig.savefig(out / "fig4_connectivity.png", dpi=150)
    plt.close(fig)


def main(root: str):
    root = Path(root)
    out = root / "final"
    out.mkdir(parents=True, exist_ok=True)
    agg, comp, vals, man = load(root)
    u, ut = u_table(agg)
    ut.to_csv(out / "table1_U_summary.csv", index=False)
    rb = agg.groupby(["method", "s", "level", "budget"]).agg(
        recall=("recall", "mean"), recall_sd=("recall", "std"), N=("N", "mean"),
        U=("U", "mean"), L=("L", "mean"), D=("D_mean", "mean"),
        F=("F_mean", "mean")).reset_index()
    rb["U_share"] = rb.U / rb.L.where(rb.L > 0)
    rb.to_csv(out / "table2_budget_curves.csv", index=False)
    cs = comp.groupby(["method", "s", "level", "semantics"]).agg(
        nodes=("nodes", "mean"), edges=("edges", "mean"), sccs=("sccs", "mean"),
        largest_scc_frac=("largest_scc", "mean"), wccs=("wccs", "mean"),
        distinct_seeds=("distinct_seeds", "mean")).reset_index()
    cs["largest_scc_frac"] = cs.largest_scc_frac / cs.nodes
    cs.to_csv(out / "table3_connectivity.csv", index=False)
    pm, q = paired_methods(u, root)
    pm.to_csv(out / "table4_acorn1_vs_gamma.csv", index=False)
    q.to_csv(out / "table4b_query_level_wilcoxon.csv", index=False)
    ft = fragmentation_tests(u[u.method.isin(["acorn_1", "acorn_gamma", "postfilter"])])
    ft.to_csv(out / "table5_fragmentation_tests.csv", index=False)
    st = pd.DataFrame()
    if agg.s.nunique() > 1:
        st = selectivity_tests(u[u.method.isin(["acorn_1", "acorn_gamma", "postfilter"])])
        st.to_csv(out / "table6_selectivity.csv", index=False)
    figures(agg, comp, ut, out / "figures")
    bt = baseline(root, agg, out) if (root / "baseline" / "analysis").exists() else None
    validation = {k: {"all_pass": v["all_pass"], "rows": v["rows"],
                      "identity_violations": v["identity_violations"],
                      "found_targets_unreachable": v["found_targets_unreachable"],
                      "recall_mismatch": v["recall_mismatch"],
                      "soundness": v["soundness"]} for k, v in vals.items()}
    (out / "validation_summary.json").write_text(json.dumps(validation, indent=2, default=str))
    pd.set_option("display.width", 250)
    print(json.dumps({k: v["all_pass"] for k, v in validation.items()}))
    print(ut.round(5).to_string(index=False))
    print(pm.round(5).to_string(index=False))
    print(ft.round(4).to_string(index=False))
    if len(st):
        print(st.round(5).to_string(index=False))
    if bt is not None:
        print(bt[bt.budget.isin([10, 40, 160, 640, 2560]) | (bt.method == "prefilter")]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
