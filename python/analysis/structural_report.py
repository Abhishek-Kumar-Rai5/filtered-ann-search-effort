"""Final structural report from the completed runs (no new experiments).

Usage:
  .venv/bin/python python/analysis/structural_report.py

Inputs (small summary/provenance files tracked in git; the per-query
reachability dumps are optional, see T3b):
  results/structural/pilot/decision/   decision experiment, s = 0.01:
      U and R_graph/R_sem connectivity for C100 x5, C1000 x5, random x2
      realizations (ACORN-1, ACORN-γ); decomposition (U, N(b), recall) on
      realization 0 of C100 / C1000 / random.
  results/structural/baseline/   Phase 4 PRE/POST at s = 0.01 (random;
      clustered = C1000 r0, identical masks) decomposed with POST's
      unfiltered-graph reachability.
Outputs: results/structural/report/{tables,figures,report.json}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

D = Path("results/structural/pilot/decision")
B = Path("results/structural/baseline")
OUT = Path("results/structural/report")
LEVELS = ["C100", "C1000", "random"]
LAB = {"prefilter": "PRE", "postfilter": "POST", "acorn_1": "ACORN-1",
       "acorn_gamma": "ACORN-γ"}
COL = {"prefilter": "#555555", "postfilter": "#1baf7a", "acorn_1": "#2a78d6",
       "acorn_gamma": "#eb6834"}


def t_ci(x, ci=0.95):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    h = stats.t.ppf(0.5 + ci / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return (x.mean() - h, x.mean() + h)


def holm(p):
    p = np.asarray(p, float)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def per_query_u(method, name):
    r = pd.read_csv(D / "reach" / method / f"{name}_reach.csv").sort_values("query_id")
    bits = r["sem"].to_numpy(np.int64)[:, None]
    return 1 - ((bits >> np.arange(10)) & 1).sum(axis=1) / 10


def main():
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    per = pd.read_csv(D / "decision_per_realization.csv")
    man = {c["name"]: c for c in
           __import__("yaml").safe_load((D / "manifest.yaml").read_text())["conditions"]}

    # T1: U across realizations (+ t-CI), plain-graph U, R_cap U
    t1 = []
    for (m, lvl), g in per.groupby(["method", "level"]):
        lo, hi = t_ci(g.U_sem)
        t1.append({"method": LAB[m], "level": lvl, "realizations": len(g),
                   "U_mean": g.U_sem.mean(), "U_ci95_lo": lo, "U_ci95_hi": hi,
                   "U_min": g.U_sem.min(), "U_max": g.U_sem.max(),
                   "queries_with_U>0": g.frac_q_U_pos.mean(),
                   "U_plain_graph": g.U_graph.mean(), "U_R_cap": g.U_cap.mean(),
                   "passing_clusters": g.passing_clusters.mean()})
    t1 = pd.DataFrame(t1)
    t1.to_csv(OUT / "tables" / "T1_U_by_method_fragmentation.csv", index=False)

    # T2: connectivity of the passing set, plain graph vs R_sem
    t2 = per.groupby(["method", "level"]).agg(
        largest_scc_plain=("lscc_graph", "mean"), largest_scc_Rsem=("lscc_sem", "mean"),
        sccs_plain=("sccs_graph", "mean"), sccs_Rsem=("sccs_sem", "mean"),
        wccs_plain=("wccs_graph", "mean"), wccs_Rsem=("wccs_sem", "mean"),
        distinct_level0_seeds=("distinct_seeds", "mean")).reset_index()
    t2["method"] = t2.method.map(LAB)
    t2.to_csv(OUT / "tables" / "T2_connectivity.csv", index=False)

    # T3: paired ACORN-γ − ACORN-1 (same predicate): realization-level and
    # query-level (Wilcoxon signed-rank per condition, Holm over 12)
    w = per.pivot_table(index=["level", "realization"], columns="method", values="U_sem")
    w["diff"] = w.acorn_gamma - w.acorn_1
    t3 = []
    for lvl, g in w.groupby(level=0):
        lo, hi = t_ci(g["diff"])
        t3.append({"level": lvl, "realizations": len(g), "diff_gamma_minus_1": g["diff"].mean(),
                   "ci95_lo": lo, "ci95_hi": hi,
                   "gamma_higher": int((g["diff"] > 0).sum()),
                   "acorn1_higher": int((g["diff"] < 0).sum()),
                   "sign_test_p": stats.binomtest(int((g["diff"] > 0).sum()),
                                                  int((g["diff"] != 0).sum()), 0.5).pvalue
                   if (g["diff"] != 0).any() else 1.0})
    t3 = pd.DataFrame(t3)
    t3.to_csv(OUT / "tables" / "T3_acorn1_vs_gamma_realization.csv", index=False)
    # Query-level test needs the per-query reachability dumps (not tracked in
    # git); without them, the committed table is reused unchanged.
    t3b = OUT / "tables" / "T3b_acorn1_vs_gamma_query_level.csv"
    have_dumps = all((D / "reach" / m / f"{n}_reach.csv").exists()
                     for m in ("acorn_1", "acorn_gamma") for n in man)
    q = []
    for name, c in (man.items() if have_dumps else []):
        u1, ug = per_query_u("acorn_1", name), per_query_u("acorn_gamma", name)
        d = ug - u1
        q.append({"condition": name, "level": c["level"], "realization": c["realization"],
                  "U_acorn1": u1.mean(), "U_gamma": ug.mean(),
                  "queries_gamma_worse": int((d > 0).sum()),
                  "queries_acorn1_worse": int((d < 0).sum()),
                  "wilcoxon_p": stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0})
    if have_dumps:
        q = pd.DataFrame(q).sort_values(["level", "realization"])
        q["p_holm"] = holm(q.wilcoxon_p)
        q.to_csv(t3b, index=False)
    else:
        print(f"note: per-query reach dumps absent; reusing committed {t3b}")
        q = pd.read_csv(t3b)

    # T4: fragmentation effect (realization-level Kruskal-Wallis, Spearman)
    rank = {l: i for i, l in enumerate(LEVELS)}
    t4 = []
    for m, g in per.groupby("method"):
        groups = [g[g.level == l].U_sem.to_numpy() for l in LEVELS]
        rho = stats.spearmanr(g.level.map(rank), g.U_sem)
        rc = stats.spearmanr(g.passing_clusters.clip(lower=1), g.U_sem) \
            if m == "acorn_1" else None
        t4.append({"method": LAB[m], "kruskal_wallis_p": stats.kruskal(*groups).pvalue,
                   "spearman_rho_fragmentation": rho.statistic, "spearman_p": rho.pvalue,
                   "worst_level": g.groupby("level").U_sem.mean().idxmax()})
    t4 = pd.DataFrame(t4)
    t4.to_csv(OUT / "tables" / "T4_fragmentation_tests.csv", index=False)
    # within C100: U vs number of passing clusters (ACORN-1)
    c100 = per[(per.level == "C100")][["method", "realization", "passing_clusters", "U_sem"]]
    c100.to_csv(OUT / "tables" / "T4b_C100_islands.csv", index=False)

    # T5: decomposition vs budget (realization 0) + baselines (Phase 4)
    a = pd.read_csv(D / "analysis_r0" / "aggregates.csv")
    b = pd.read_csv(B / "analysis" / "aggregates.csv")
    b["level"] = b.condition.map({"random_s0.0100": "random", "clustered_s0.0100": "C1000"})
    t5 = pd.concat([a, b], ignore_index=True)
    t5["recall"] = 1 - t5.L
    t5["U_share_of_loss"] = t5.U / t5.L.where(t5.L > 0)
    t5["method"] = t5.method.map(LAB)
    t5 = t5[["method", "level", "budget", "recall", "L", "U", "N", "U_share_of_loss",
             "D_mean", "F_mean", "L_lo", "L_hi"]].sort_values(["method", "level", "budget"])
    t5.to_csv(OUT / "tables" / "T5_decomposition_vs_budget.csv", index=False)

    validation = {
        "decision_r0": json.loads((D / "analysis_r0" / "validation.json").read_text()),
        "baseline_pre_post": json.loads((B / "analysis" / "validation.json").read_text()),
        "slice": json.loads(Path("results/structural/slice/analysis/validation.json").read_text()),
    }
    summary = {k: {"rows": v.get("rows"), "identity_violations": v.get("identity_violations"),
                   "found_targets_unreachable": v.get("found_targets_unreachable"),
                   "recall_mismatch": v.get("recall_mismatch"), "all_pass": v.get("all_pass")}
               for k, v in validation.items() if isinstance(v, dict) and "rows" in v}
    (OUT / "report.json").write_text(json.dumps({"validation": summary}, indent=2, default=str))

    figures(per, t5)
    pd.set_option("display.width", 250)
    for name, t in (("T1", t1), ("T2", t2), ("T3", t3), ("T4", t4)):
        print(f"== {name} ==\n{t.round(5).to_string(index=False)}\n")
    print("== T3b ==\n", q.round(5).to_string(index=False))
    print("\n== T5 (selected budgets) ==")
    print(t5[t5.budget.isin([0, 10, 40, 160, 640, 2560])].round(4).to_string(index=False))
    print(json.dumps(summary, indent=1))


def figures(per, t5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e5e5e5", "grid.linewidth": 0.6})
    x = np.arange(len(LEVELS))
    xl = ["C100\n(1–2 clusters)", "C1000\n(10–11)", "random\n(max)"]

    # F1: U vs fragmentation, realizations as dots (the main result)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in ("acorn_1", "acorn_gamma"):
        d = per[per.method == m]
        off = -0.08 if m == "acorn_1" else 0.08
        for ax, col in ((axes[0], "U_sem"), (axes[1], "U_graph")):
            for i, l in enumerate(LEVELS):
                v = d[d.level == l][col]
                ax.scatter(np.full(len(v), i) + off, v, color=COL[m], s=18, alpha=0.75)
            ax.plot(x, [d[d.level == l][col].mean() for l in LEVELS], color=COL[m], lw=2,
                    label=LAB[m])
            ax.set_xticks(x, xl)
    axes[0].set(title="U under ACORN's search semantics (R_sem)",
                ylabel="unreachable fraction of the 10 true targets")
    axes[1].set(title="U if only the plain filtered graph were traversable")
    axes[0].legend(frameon=False)
    fig.suptitle("s = 0.01: topological loss vs predicate fragmentation (dots = realizations)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "F1_U_vs_fragmentation.png", dpi=150)
    plt.close(fig)

    # F2: connectivity
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for m in ("acorn_1", "acorn_gamma"):
        d = per[per.method == m]
        for col, ls, nm in (("lscc_graph", ":", "plain graph"), ("lscc_sem", "-", "R_sem")):
            ax.plot(x, [d[d.level == l][col].mean() for l in LEVELS], color=COL[m], ls=ls,
                    lw=2, marker="o", ms=4, label=f"{LAB[m]}, {nm}")
    ax.set_xticks(x, xl)
    ax.set(ylabel="largest SCC / passing nodes", title="Passing-set connectivity, s = 0.01")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "F2_connectivity.png", dpi=150)
    plt.close(fig)

    # F3: L, N(b), U on log scale (realization 0), ACORN only
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    for i, m in enumerate(("ACORN-1", "ACORN-γ")):
        for j, l in enumerate(LEVELS):
            ax = axes[i, j]
            d = t5[(t5.method == m) & (t5.level == l)].sort_values("budget")
            ax.plot(d.budget, d.L, color="#555555", lw=2, marker="o", ms=3, label="L = 1 − Recall")
            ax.plot(d.budget, d.N, color="#2a78d6", lw=2, ls="--", label="N(b): reachable, missed")
            ax.plot(d.budget, d.U, color="#eb6834", lw=2, label="U: unreachable")
            ax.set(xscale="log", yscale="log", ylim=(5e-5, 1.2))
            if i == 0:
                ax.set_title(l)
            if j == 0:
                ax.set_ylabel(f"{m}\nfraction of 10 targets")
            if i == 1:
                ax.set_xlabel("budget b")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Recall loss L = U + N(b), s = 0.01, realization 0 (U is a budget-independent floor)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "F3_L_U_N_vs_budget.png", dpi=150)
    plt.close(fig)

    # F4: recall vs effort with PRE/POST baselines (Phase 4 conditions)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, l in zip(axes, ("C1000", "random")):
        for m in ("PRE", "POST", "ACORN-1", "ACORN-γ"):
            key = {v: k for k, v in LAB.items()}[m]
            d = t5[(t5.method == m) & (t5.level == l)].sort_values("budget")
            if m == "PRE":
                ax.scatter(d.D_mean, d.recall, color=COL[key], marker="s", s=36, label=m, zorder=3)
            else:
                ax.plot(d.D_mean, d.recall, color=COL[key], lw=1.8, marker="o", ms=3, label=m)
        ax.set(xscale="log", title=f"s = 0.01, {l}", xlabel="mean exact distance computations D")
    axes[0].set_ylabel("Recall@10")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Recall vs effort: PRE / POST (Phase 4) vs ACORN-1 / ACORN-γ")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "F4_recall_vs_effort_baselines.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
