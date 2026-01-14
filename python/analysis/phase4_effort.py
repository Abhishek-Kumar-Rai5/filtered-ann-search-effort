"""Phase 4 effort-vs-selectivity, H3 cross-method and preliminary H2 analysis.

docs/phase4_matrix.md §17. Analysis only: reads the existing Phase 4 outputs
and writes into the configured output directory. Methodology (§16):
D = exact distance computations is the primary cross-method effort; ACORN's
F = n_scanned is supplementary and is never combined with D; latency is
secondary evidence. The oracle (per-query b*) is the frozen one from
phase4_matrix.compute_oracle. True local filtered density (design §8 measure
2) is a post-hoc measurement only and is never a predictor input.

Usage:
  .venv/bin/python python/analysis/phase4_effort.py configs/phase4/effort_analysis.yaml
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

_SPEC = importlib.util.spec_from_file_location(
    "phase4_matrix", Path(__file__).resolve().parent / "phase4_matrix.py")
pm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pm)

METHODS = ("prefilter", "postfilter", "acorn")
PAIRS = (("prefilter", "postfilter"), ("prefilter", "acorn"),
         ("postfilter", "acorn"))


# ------------------------------------------------------------- statistics --

def bootstrap_median_ci(x: np.ndarray, rng: np.random.Generator,
                        resamples: int, ci: float):
    """Median and percentile bootstrap CI over queries (+inf allowed)."""
    x = np.asarray(x, dtype=float)
    meds = np.empty(resamples)
    for i in range(0, resamples, 200):
        idx = rng.integers(0, len(x), size=(min(200, resamples - i), len(x)))
        meds[i:i + idx.shape[0]] = np.median(x[idx], axis=1)
    lo, hi = np.percentile(meds, [50 * (1 - ci), 50 * (1 + ci)])
    return float(np.median(x)), float(lo), float(hi)


def paired_log2_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """log2(a/b) per query; both censored (+inf) = 0 (same outcome)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log2(a / b)
    r[np.isinf(a) & np.isinf(b)] = 0.0
    return r


def sign_test(a: np.ndarray, b: np.ndarray):
    """Paired two-sided sign test of a vs b (ties dropped; +inf compares)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    less = int(np.sum(a < b))
    greater = int(np.sum(a > b))
    ties = len(a) - less - greater
    n = less + greater
    p = float(stats.binomtest(less, n, 0.5).pvalue) if n > 0 else 1.0
    return {"a_less": less, "a_greater": greater, "ties": ties, "p": p}


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho; NaN when either input is constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def cluster_bootstrap_spearman(frame: pd.DataFrame, effort: str,
                               predictors: list[str], rng: np.random.Generator,
                               resamples: int, ci: float):
    """Spearman(effort, predictor) pooled over the conditions in `frame`, with
    a bootstrap that resamples query ids (each query keeps all its condition
    rows, so the paired structure is preserved). Also the CI of
    |rho_a| - |rho_b| for every predictor pair (a, b)."""
    f = frame.sort_values(["query_id", "condition"])
    nq = f.query_id.nunique()
    if len(f) % nq or (f.groupby("query_id").size() != len(f) // nq).any():
        raise ValueError("every query needs the same number of condition rows")
    shape = (nq, len(f) // nq)
    e = f[effort].to_numpy(float).reshape(shape)
    cols = {p: f[p].to_numpy(float).reshape(shape) for p in predictors}
    point = {p: spearman(e.ravel(), cols[p].ravel()) for p in predictors}
    boot = {p: np.empty(resamples) for p in predictors}
    for i in range(resamples):
        pick = rng.integers(0, nq, size=nq)
        ee = e[pick].ravel()
        for p in predictors:
            boot[p][i] = spearman(ee, cols[p][pick].ravel())
    q = [50 * (1 - ci), 50 * (1 + ci)]
    out = {p: {"rho": point[p], "ci": np.nanpercentile(boot[p], q).tolist()}
           for p in predictors}
    diffs = {}
    for a in predictors:
        for b in predictors:
            if a < b:
                d = np.abs(boot[a]) - np.abs(boot[b])
                diffs[f"|{a}|-|{b}|"] = {
                    "diff": abs(point[a]) - abs(point[b]),
                    "ci": np.nanpercentile(d, q).tolist()}
    return out, diffs


# ------------------------------------------------------------------ data --

def load(cfg: dict):
    inp = cfg["inputs"]
    df = pm.load_rows(Path(inp["matrix"]))
    df, integrity = pm.join_scan(df, pm.load_rows(Path(inp["nscan"])))
    if not integrity["pass"]:
        raise RuntimeError(f"n_scanned join failed: {integrity}")
    oracle = pm.oracle_scan(pm.compute_oracle(df, cfg["oracle_target_recall"]), df)
    # attach the oracle-point counters/latency of each query (secondary);
    # pre-filter rows have budget 0 = its oracle budget
    pick = df[["method", "condition", "query_id", "budget", "latency_us",
               "filter_checks", "rounds"]].astype({"budget": float})
    oracle = oracle.merge(pick.rename(columns={"budget": "oracle_budget"}),
                          on=["method", "condition", "query_id", "oracle_budget"],
                          how="left")
    ld = pd.read_csv(inp["local_density"]).rename(columns={"query": "query_id"})
    ld = ld.pivot_table(index=["condition", "query_id"], columns="k_local",
                        values="local_density").reset_index()
    ld.columns = ["condition", "query_id"] + [f"rho{k}" for k in ld.columns[2:]]
    oracle = oracle.merge(ld, on=["condition", "query_id"], how="left")
    t = pd.read_csv(inp["timing_audit"])
    t = (t.groupby(["condition", "budget", "query_id"]).latency_us.median()
         .rename("ctrl_latency_us").reset_index().astype({"budget": float}))
    oracle = oracle.merge(
        t.rename(columns={"budget": "oracle_budget"}).assign(method="acorn"),
        on=["method", "condition", "query_id", "oracle_budget"], how="left")
    return df, oracle, integrity


# --------------------------------------------------------------- analyses --

def effort_curves(oracle: pd.DataFrame, rng, B: int, ci: float) -> pd.DataFrame:
    """Median oracle D (and ACORN F) per method x condition, bootstrap CIs,
    and the ratio to the same method/correlation at s = 1."""
    rows = []
    for (method, cond), d in oracle.groupby(["method", "condition"]):
        med, lo, hi = bootstrap_median_ci(d.oracle_dist.to_numpy(), rng, B, ci)
        r = {"method": method, "condition": cond,
             "correlation": d.correlation.iloc[0], "s": d.s_achieved.iloc[0],
             "queries": len(d), "censored_fraction": d.censored.mean(),
             "D_median": med, "D_ci_lo": lo, "D_ci_hi": hi,
             "oracle_budget_median": d.oracle_budget.median(),
             "latency_us_median_uncontrolled": d.latency_us.median()}
        if method == "acorn":
            fm, flo, fhi = bootstrap_median_ci(d.oracle_scan.to_numpy(), rng, B, ci)
            r.update(F_median=fm, F_ci_lo=flo, F_ci_hi=fhi,
                     ctrl_latency_us_median=d.ctrl_latency_us.median(),
                     ctrl_latency_queries=int(d.ctrl_latency_us.notna().sum()))
        rows.append(r)
    out = pd.DataFrame(rows).sort_values(["method", "correlation", "s"])
    s1 = out[np.isclose(out.s, 1.0)].set_index(["method", "correlation"])
    for col in ("D_median", "F_median"):
        if col in out:
            out[col.replace("_median", "_ratio_to_s1")] = [
                r[col] / s1.loc[(r.method, r.correlation), col]
                for _, r in out.iterrows()]
    return out


def fixed_budget_table(df: pd.DataFrame, fail_recall: float) -> pd.DataFrame:
    g = df.groupby(["method", "condition", "correlation", "s_achieved", "budget"])
    t = g.agg(queries=("recall", "size"), mean_recall=("recall", "mean"),
              failed_fraction=("recall", lambda x: float((x < fail_recall).mean())),
              D_median=("dist_exact", "median"), D_mean=("dist_exact", "mean"),
              latency_us_median_uncontrolled=("latency_us", "median"))
    a = df[df.method == "acorn"].groupby(
        ["method", "condition", "correlation", "s_achieved", "budget"]).n_scanned
    t = t.join(a.median().rename("F_median")).join(a.mean().rename("F_mean"))
    return t.reset_index()


def h3_paired(oracle: pd.DataFrame, rng, B: int, ci: float, alpha: float):
    """Per condition and method pair: paired per-query comparison of oracle D
    (sign test, Holm over the family) and the median log2 ratio with a
    bootstrap CI."""
    rows = []
    wide = oracle.pivot(index=["condition", "query_id"], columns="method",
                        values="oracle_dist").reset_index()
    meta = oracle.groupby("condition")[["correlation", "s_achieved"]].first()
    for cond, d in wide.groupby("condition"):
        for a, b in PAIRS:
            st = sign_test(d[a], d[b])
            lr = paired_log2_ratio(d[a], d[b])
            med, lo, hi = bootstrap_median_ci(lr, rng, B, ci)
            rows.append({"condition": cond, "correlation": meta.loc[cond, "correlation"],
                         "s": meta.loc[cond, "s_achieved"], "a": a, "b": b,
                         "queries": len(d), **st, "a_less_fraction": st["a_less"] / len(d),
                         "median_log2_a_over_b": med, "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.tolist())
    out["significant"] = out.p_holm < alpha
    return out.sort_values(["a", "b", "correlation", "s"])


def h3_secondary(oracle: pd.DataFrame) -> pd.DataFrame:
    """Secondary evidence per method x condition at the oracle point:
    uncontrolled matrix latency and each method's own counter. ACORN's
    controlled latency (timing audit, 500 queries) where available."""
    agg = {"latency_us_median_uncontrolled": ("latency_us", "median"),
           "filter_checks_median": ("filter_checks", "median"),
           "rounds_median": ("rounds", "median"),
           "F_median": ("oracle_scan", "median"),
           "ctrl_latency_us_median": ("ctrl_latency_us", "median")}
    return (oracle.groupby(["method", "correlation", "s_achieved"])
            .agg(**agg).reset_index())


def h2_structural(oracle: pd.DataFrame, rng, B: int, ci: float):
    """Preliminary H2 on effort (regret needs the Phase 5 predictor):
    (a) pooled over the six selectivities, per graph method and correlation
        subgroup and for both subgroups together: Spearman of oracle effort
        with global selectivity s vs true local density rho10 / rho100, with a
        query-cluster bootstrap CI of each |rho| difference;
    (b) within each condition (s constant): Spearman(effort, local density)
        — variation that global selectivity cannot explain by construction.
    Pre-filter is excluded: its effort is exactly T(s), a function of s."""
    preds = ["s_achieved", "rho10", "rho100"]
    pooled, within = [], []
    for method in ("postfilter", "acorn"):
        efforts = ["oracle_dist"] + (["oracle_scan"] if method == "acorn" else [])
        m = oracle[oracle.method == method]
        for effort in efforts:
            for subgroup, d in (("random", m[m.correlation == "random"]),
                                ("clustered", m[m.correlation == "clustered"]),
                                ("both", m)):
                res, diffs = cluster_bootstrap_spearman(d, effort, preds, rng, B, ci)
                pooled.append({"method": method, "effort": effort,
                               "subgroup": subgroup, "rows": len(d),
                               **{f"rho_{p}": res[p]["rho"] for p in preds},
                               **{f"rho_{p}_ci": res[p]["ci"] for p in preds},
                               **{k: v["diff"] for k, v in diffs.items()},
                               **{f"{k}_ci": v["ci"] for k, v in diffs.items()}})
            for cond, d in m.groupby("condition"):
                for k in ("rho10", "rho100"):
                    r = spearman(d[effort], d[k])
                    boot = np.empty(B)
                    e = d[effort].to_numpy(float)
                    x = d[k].to_numpy(float)
                    for i in range(B):
                        idx = rng.integers(0, len(d), size=len(d))
                        boot[i] = spearman(e[idx], x[idx])
                    lo, hi = (np.nanpercentile(boot, [50 * (1 - ci), 50 * (1 + ci)])
                              if np.isfinite(r) else (np.nan, np.nan))
                    within.append({"method": method, "effort": effort,
                                   "condition": cond,
                                   "correlation": d.correlation.iloc[0],
                                   "s": d.s_achieved.iloc[0], "density": k,
                                   "zero_density_fraction": float((x == 0).mean()),
                                   "rho": r, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(pooled), pd.DataFrame(within)


# ------------------------------------------------------------------- plots --

COLORS = {"prefilter": "#2a78d6", "postfilter": "#eb6834", "acorn": "#1baf7a"}
LABELS = {"prefilter": "pre-filter", "postfilter": "post-filter", "acorn": "ACORN"}


def plot_all(curves, fixed, within, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e5e5e5", "grid.linewidth": 0.6})
    out.mkdir(parents=True, exist_ok=True)

    # 1. oracle D vs s, per correlation (primary)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, corr in zip(axes, ("random", "clustered")):
        for method in METHODS:
            c = curves[(curves.method == method) & (curves.correlation == corr)]
            yerr = [c.D_median - c.D_ci_lo, c.D_ci_hi - c.D_median]
            ax.errorbar(c.s, c.D_median, yerr=yerr, marker="o", ms=4, lw=2,
                        capsize=2, color=COLORS[method], label=LABELS[method])
        ax.set(xscale="log", yscale="log", title=f"{corr} filters",
               xlabel="global selectivity s")
    axes[0].set_ylabel("median oracle D (distance computations)")
    axes[0].legend(frameon=False)
    fig.suptitle("Oracle effort D at Recall@10 ≥ 0.9 (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(out / "fig1_oracle_D_vs_selectivity.png", dpi=150)
    plt.close(fig)

    # 2. ACORN supplementary: D, F, controlled latency indexed to s = 1
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    series = (("D_median", "D (distance computations)", "#1baf7a"),
              ("F_median", "F (n_scanned)", "#4a3aa7"),
              ("ctrl_latency_us_median", "controlled latency (500 q)", "#e87ba4"))
    for ax, corr in zip(axes, ("random", "clustered")):
        c = curves[(curves.method == "acorn") & (curves.correlation == corr)]
        s1 = c[np.isclose(c.s, 1.0)].iloc[0]
        for col, lab, color in series:
            ax.plot(c.s, c[col] / s1[col], marker="o", ms=4, lw=2, color=color,
                    label=lab)
        ax.axhline(1.0, color="#999999", lw=0.8)
        ax.set(xscale="log", yscale="log", title=f"ACORN, {corr} filters",
               xlabel="global selectivity s")
    axes[0].set_ylabel("median at oracle budget, relative to s = 1")
    axes[0].legend(frameon=False)
    fig.suptitle("ACORN: D vs supplementary F vs controlled latency (indexed)")
    fig.tight_layout()
    fig.savefig(out / "fig2_acorn_D_F_latency_indexed.png", dpi=150)
    plt.close(fig)

    # 3. fixed-budget mean recall vs budget (graph methods), one line per s
    levels = sorted(fixed.s_achieved.unique())
    shades = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
    for i, method in enumerate(("postfilter", "acorn")):
        for j, corr in enumerate(("random", "clustered")):
            ax = axes[i, j]
            for s, shade in zip(levels, shades):
                c = fixed[(fixed.method == method) & (fixed.correlation == corr)
                          & np.isclose(fixed.s_achieved, s)]
                ax.plot(c.budget, c.mean_recall, marker="o", ms=3, lw=1.6,
                        color=shade, label=f"s = {s:.3g}")
            ax.axhline(0.9, color="#999999", lw=0.8, ls="--")
            ax.set(xscale="log", title=f"{LABELS[method]}, {corr}")
    for ax in axes[1]:
        ax.set_xlabel("candidate-list budget b")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean Recall@10")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Fixed-budget recall (dashed: oracle target 0.9)")
    fig.tight_layout()
    fig.savefig(out / "fig3_fixed_budget_recall.png", dpi=150)
    plt.close(fig)

    # 4. within-condition Spearman(D, rho10) per selectivity
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    w = within[(within.effort == "oracle_dist") & (within.density == "rho10")]
    for ax, method in zip(axes, ("postfilter", "acorn")):
        for corr, mk in (("random", "o"), ("clustered", "s")):
            c = w[(w.method == method) & (w.correlation == corr)].dropna(subset=["rho"])
            ax.errorbar(c.s, c.rho, yerr=[c.rho - c.ci_lo, c.ci_hi - c.rho],
                        marker=mk, ms=4, lw=1.6, capsize=2,
                        color=COLORS[method] if corr == "random" else "#555555",
                        label=corr)
        ax.axhline(0.0, color="#999999", lw=0.8)
        ax.set(xscale="log", title=LABELS[method], xlabel="global selectivity s")
    axes[0].set_ylabel("Spearman(oracle D, local density K=10)")
    axes[0].legend(frameon=False)
    fig.suptitle("Within-condition association (s fixed): effort vs true local density")
    fig.tight_layout()
    fig.savefig(out / "fig4_within_condition_spearman.png", dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------- main --

def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    bs = cfg["bootstrap"]
    rng = np.random.default_rng(bs["seed"])
    df, oracle, integrity = load(cfg)

    curves = effort_curves(oracle, rng, bs["resamples_median"], bs["ci"])
    fixed = fixed_budget_table(df, cfg["fixed_budget_fail_recall"])
    h3 = h3_paired(oracle, rng, bs["resamples_median"], bs["ci"], cfg["h3_family_alpha"])
    sec = h3_secondary(oracle)
    pooled, within = h2_structural(oracle, rng, bs["resamples_spearman"], bs["ci"])

    curves.to_csv(out / "effort_curves.csv", index=False)
    fixed.to_csv(out / "fixed_budget.csv", index=False)
    h3.to_csv(out / "h3_paired.csv", index=False)
    sec.to_csv(out / "h3_secondary.csv", index=False)
    pooled.to_csv(out / "h2_pooled.csv", index=False)
    within.to_csv(out / "h2_within_condition.csv", index=False)
    plot_all(curves, fixed, within, out / "figures")
    (out / "run.json").write_text(json.dumps(
        {"config": cfg, "nscan_join": integrity,
         "python": sys.version.split()[0],
         "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                      "scipy": __import__("scipy").__version__},
         "rows": len(df), "oracle_rows": len(oracle)}, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
