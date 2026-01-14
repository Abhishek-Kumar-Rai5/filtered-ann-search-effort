import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

D = Path(sys.argv[1] if len(sys.argv) > 1 else "results/structural/pilot/decision")
LEVELS = ["C100", "C1000", "random"]
METHODS = ["acorn_1", "acorn_gamma"]
LAB = {"acorn_1": "ACORN-1", "acorn_gamma": "ACORN-γ"}
pd.set_option("display.width", 250)
man = yaml.safe_load((D / "manifest.yaml").read_text())["conditions"]


def popcount(a):
    return np.array([bin(int(x)).count("1") for x in a])


rows, comps = [], []
for m in METHODS:
    rj = json.loads((D / "reach" / m / "reach.json").read_text())["conditions"]
    for c in man:
        r = pd.read_csv(D / "reach" / m / f"{c['name']}_reach.csv")
        rec = {"method": m, "level": c["level"], "realization": c["realization"],
               "passing_clusters": c["passing_clusters"]}
        for s in ("graph", "sem", "cap"):
            u = (10 - popcount(r[s])) / 10
            rec[f"U_{s}"] = u.mean()
            if s == "sem":
                rec["frac_q_U_pos"] = (u > 0).mean()
                rec["U_q_p99"] = np.quantile(u, 0.99)
        j = rj[c["name"]]
        for s in ("graph", "sem"):
            rec[f"lscc_{s}"] = j[s]["largest_scc"] / j[s]["nodes"]
            rec[f"sccs_{s}"] = j[s]["sccs"]
            rec[f"wccs_{s}"] = j[s]["wccs"]
        rec["distinct_seeds"] = j["distinct_seeds"]
        rows.append(rec)
u = pd.DataFrame(rows)
u.to_csv(D / "decision_per_realization.csv", index=False)

summ = u.groupby(["method", "level"]).agg(
    n=("U_sem", "size"), U_mean=("U_sem", "mean"), U_sd=("U_sem", "std"),
    U_min=("U_sem", "min"), U_max=("U_sem", "max"),
    frac_q_U_pos=("frac_q_U_pos", "mean"), U_graph=("U_graph", "mean"),
    U_cap=("U_cap", "mean"), lscc_graph=("lscc_graph", "mean"),
    lscc_sem=("lscc_sem", "mean"), sccs_graph=("sccs_graph", "mean"),
    sccs_sem=("sccs_sem", "mean"))
summ.to_csv(D / "decision_U_summary.csv")
pw = u.pivot_table(index=["level", "realization"], columns="method", values="U_sem")
pw["diff"] = pw.acorn_gamma - pw.acorn_1
paired = pw.groupby("level")["diff"].agg(["mean", "min", "max"])
paired["gamma_higher"] = pw.groupby("level").apply(
    lambda g: f"{int((g.acorn_gamma > g.acorn_1).sum())}/{len(g)}")
paired.to_csv(D / "decision_paired.csv")

print("== U (R_sem, fraction of 10 targets) across realizations ==")
print(summ.round(5).to_string())
print("\n== per realization ==")
print(u[["method", "level", "realization", "passing_clusters", "U_sem", "U_graph",
         "lscc_graph", "lscc_sem", "sccs_sem"]].round(5).to_string(index=False))
print("\n== paired U(ACORN-γ) - U(ACORN-1), same predicate ==")
print(paired.round(5).to_string())

agg = pd.read_csv(D / "analysis_r0" / "aggregates.csv")
agg["recall"] = 1 - agg.L
agg["U_share"] = agg.U / agg.L.where(agg.L > 0)
print("\n== realization 0: recall / N(b) / U share of loss ==")
print(agg[agg.budget.isin([10, 40, 160, 640, 1280, 2560])].pivot_table(
    index=["method", "budget"], columns="level",
    values=["recall", "N", "U_share"]).round(4).to_string())

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e5e5e5"})
col = {"acorn_1": "#2a78d6", "acorn_gamma": "#eb6834"}
x = np.arange(len(LEVELS))
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for m in METHODS:
    d = u[u.method == m]
    off = 0.08 if m == "acorn_gamma" else -0.08
    for ax, c, t in ((axes[0], "U_sem", "U under ACORN search semantics (R_sem)"),
                     (axes[1], "U_graph", "U under the plain filtered graph")):
        for i, lvl in enumerate(LEVELS):
            v = d[d.level == lvl][c]
            ax.scatter(np.full(len(v), i) + off, v, color=col[m], s=16, alpha=0.75)
        ax.plot(x, [d[d.level == l][c].mean() for l in LEVELS], color=col[m], lw=2,
                label=LAB[m])
        ax.set_xticks(x, ["C100\n(1–2 clusters)", "C1000\n(10–11)", "random\n(max)"])
        ax.set_title(t)
    for sem, ls in (("graph", ":"), ("sem", "-")):
        axes[2].plot(x, [d[d.level == l][f"lscc_{sem}"].mean() for l in LEVELS],
                     color=col[m], ls=ls, lw=2, marker="o", ms=4,
                     label=f"{LAB[m]}, {'R_sem' if sem == 'sem' else 'plain graph'}")
axes[2].set_xticks(x, ["C100", "C1000", "random"])
axes[2].set_title("largest SCC / passing nodes")
axes[0].set_ylabel("fraction of the 10 true targets")
axes[0].legend(frameon=False)
axes[2].legend(frameon=False, fontsize=7)
fig.suptitle("s = 0.01: topological loss vs predicate fragmentation (dots = realizations)")
fig.tight_layout()
(D / "figures").mkdir(exist_ok=True)
fig.savefig(D / "figures" / "decision_U_vs_fragmentation.png", dpi=150)
