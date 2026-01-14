from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase5_lib as L  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "phase4_matrix", HERE.parent / "analysis" / "phase4_matrix.py")
pm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pm)


class MethodData:

    def __init__(self, rows: pd.DataFrame, budgets, feats: pd.DataFrame,
                 qfeat: pd.DataFrame, oracle: pd.DataFrame, success_recall: float):
        r = rows.sort_values(["condition", "query_id", "budget"])
        nb = len(budgets)
        if len(r) % nb:
            raise ValueError("incomplete budget grid")
        grid = r.budget.to_numpy().reshape(-1, nb)
        if not (grid == np.asarray(budgets)).all():
            raise ValueError("budget grid mismatch")
        first = r.iloc[::nb]
        self.condition = first.condition.to_numpy()
        self.query = first.query_id.to_numpy()
        self.s = first.s_achieved.to_numpy()
        self.correlation = first.correlation.to_numpy()
        self.D = r.dist_exact.to_numpy(float).reshape(-1, nb)
        self.succ = (r.recall.to_numpy() >= success_recall - 1e-12).reshape(-1, nb)
        self.F = (r.n_scanned.to_numpy(float).reshape(-1, nb)
                  if "n_scanned" in r and r.n_scanned.notna().all() else None)
        has = self.succ.any(axis=1)
        self.oracle_idx = np.where(has, self.succ.argmax(axis=1), -1)
        o = oracle.set_index(["condition", "query_id"])
        ob = o.loc[list(zip(self.condition, self.query)), "oracle_budget"].to_numpy()
        mine = np.where(self.oracle_idx >= 0,
                        np.asarray(budgets, float)[np.maximum(self.oracle_idx, 0)], np.nan)
        if not np.array_equal(np.nan_to_num(ob, nan=-1), np.nan_to_num(mine, nan=-1)):
            raise ValueError("oracle budget differs from phase4_matrix.compute_oracle")
        f = pd.DataFrame({"condition": self.condition, "query_id": self.query})
        f = f.merge(feats, on=["condition", "query_id"], how="left") \
             .merge(qfeat, on="query_id", how="left")
        f["s"] = self.s
        if f.isna().any().any():
            raise ValueError("missing features")
        self.features = f
        self.probe_cost = (f.probe_dc + f.centroid_dc).to_numpy(float)

    def x(self, names, mask):
        L.check_features(names)
        return self.features.loc[mask, list(names)].to_numpy(float)


def load(cfg):
    inp = cfg["inputs"]
    rows = pm.load_rows(Path(inp["matrix"]))
    rows, integrity = pm.join_scan(rows, pm.load_rows(Path(inp["nscan"])))
    if not integrity["pass"]:
        raise RuntimeError(integrity)
    oracle = pm.compute_oracle(rows, cfg["success_recall"])
    feats = pd.read_csv(Path(inp["features"]) / "proxy.csv")
    qfeat = pd.read_csv(Path(inp["features"]) / "query_features.csv")
    data = {m: MethodData(rows[rows.method == m], cfg["budgets"], feats, qfeat,
                          oracle[oracle.method == m], cfg["success_recall"])
            for m in cfg["methods"]}
    return data, integrity


def select_router(d: MethodData, tr, names, folds, cfg):
    x, succ, D, pc, q = d.x(names, tr), d.succ[tr], d.D[tr], d.probe_cost[tr], d.query[tr]
    oof = {fam: L.oof_probabilities(x, succ, q, folds, fam, cfg["models"])
           for fam in L.FAMILIES}
    sel = {}
    for s_star in cfg["s_star"]:
        cands = []
        for fam in L.FAMILIES:
            tau, sr, cost, feas = L.calibrate_tau(oof[fam], succ, D, pc, s_star,
                                                  cfg["tau_grid_step"])
            cands.append({"family": fam, "tau": tau, "oof_success": sr,
                          "oof_cost": cost, "feasible": feas})
        feas = [c for c in cands if c["feasible"]] or cands
        best = min(feas, key=lambda c: c["oof_cost"])
        sel[s_star] = {"chosen": best, "candidates": cands}
    return sel


def evaluate(cfg, data, train_q, test_q, rng):
    out = {"gate": [], "policies": [], "regret": [], "per_condition": [],
           "fixed_curve": [], "selection": {}, "b1": [], "h2_inputs": {}}
    folds = L.fold_of(train_q, cfg["cv"]["folds"], cfg["cv"]["seed"])
    u = dict(zip(map(int, np.concatenate([train_q, test_q])),
                 L.hash_unit(np.concatenate([train_q, test_q]), cfg["b1_hash_seed"])))
    bs = cfg["bootstrap"]
    for m, d in data.items():
        tr = np.isin(d.query, train_q)
        te = ~tr
        sel = {fs: select_router(d, tr, names, folds, cfg)
               for fs, names in cfg["feature_sets"].items()}
        b1 = {s: L.calibrate_b1(d.succ[tr], d.D[tr], d.query[tr], u, s)
              for s in cfg["s_star"]}
        out["selection"][m] = {fs: {str(s): v for s, v in x.items()} for fs, x in sel.items()}
        qte, succ, D = d.query[te], d.succ[te], d.D[te]
        pc, oidx = d.probe_cost[te], d.oracle_idx[te]
        decisions = {}
        for fs, names in cfg["feature_sets"].items():
            for s_star in cfg["s_star"]:
                ch = sel[fs][s_star]["chosen"]
                models = L.fit_budget_models(d.x(names, tr), d.succ[tr], ch["family"],
                                             cfg["models"])
                p = L.predict_monotone(models, d.x(names, te))
                decisions[(fs, s_star)] = (np.full(te.sum(), len(cfg["budgets"]) - 1)
                                           if not np.isfinite(ch["tau"]) else L.decide(p, ch["tau"]))
        for s_star in cfg["s_star"]:
            ib1 = L.b1_assign(qte, u, b1[s_star])
            b2 = L.calibrate_b1(succ, D, qte, u, s_star)
            ib2 = L.b1_assign(qte, u, b2)
            out["b1"].append({"method": m, "s_star": s_star, **b1[s_star],
                              "b2": b2, "b1_test_success": float(L.pick(succ, ib1).mean())})
            zero = np.zeros(te.sum())
            pol = {"B1": (ib1, zero), "B2": (ib2, zero),
                   "oracle": (np.maximum(oidx, 0), zero)}
            for fs in cfg["feature_sets"]:
                pol[f"router_{fs}"] = (decisions[(fs, s_star)], pc)
            cost_b1 = L.pick(D, ib1)
            ok_b1 = L.pick(succ, ib1)
            for name, (idx, cost0) in pol.items():
                cost = cost0 + L.pick(D, idx)
                ok = L.pick(succ, idx)
                row = {"method": m, "s_star": s_star, "policy": name,
                       "success": float(ok.mean()), "mean_cost": float(cost.mean()),
                       "mean_cost_free_probe": float(L.pick(D, idx).mean()),
                       "mean_budget": float(np.asarray(cfg["budgets"])[idx].mean())}
                if d.F is not None:
                    row["mean_F"] = float(L.pick(d.F[te], idx).mean())
                out["policies"].append(row)
                out["regret"].append({"method": m, "s_star": s_star, "policy": name,
                                      **L.regret_decomposition(idx, succ, D, cost0, oidx)})
                if name.startswith("router_"):
                    g = L.gate_stats(cost, cost_b1, ok, ok_b1, qte, rng,
                                     bs["resamples"], bs["ci"])
                    gf = L.gate_stats(L.pick(D, idx), cost_b1, ok, ok_b1, qte, rng,
                                      bs["resamples"], bs["ci"])
                    g2 = L.gate_stats(cost, L.pick(D, ib2), ok, L.pick(succ, ib2), qte,
                                      rng, bs["resamples"], bs["ci"])
                    rec = {"method": m, "s_star": s_star, "policy": name, **g,
                           "free_probe_saving": gf["saving"],
                           "free_probe_saving_ci": gf["saving_ci"],
                           "saving_vs_B2": g2["saving"], "saving_vs_B2_ci": g2["saving_ci"],
                           **L.gate_pass(g, cfg["gate"])}
                    if d.F is not None:
                        rec["F_saving"] = float(1 - L.pick(d.F[te], idx).mean()
                                                / L.pick(d.F[te], ib1).mean())
                    out["gate"].append(rec)
            idx = decisions[("full", s_star)]
            dfc = pd.DataFrame({"condition": d.condition[te],
                                "router_cost": pc + L.pick(D, idx),
                                "router_ok": L.pick(succ, idx),
                                "b1_cost": cost_b1, "b1_ok": ok_b1})
            agg = dfc.groupby("condition").mean().reset_index()
            agg.insert(0, "s_star", s_star)
            agg.insert(0, "method", m)
            out["per_condition"].append(agg)
            out["h2_inputs"][(m, s_star)] = {
                fs: L.per_pair_regret(decisions[(fs, s_star)], succ, D, pc, oidx)
                for fs in ("full", "s_only", "minus_proxy")}
        for j, b in enumerate(cfg["budgets"]):
            out["fixed_curve"].append({"method": m, "budget": b,
                                       "success": float(d.succ[te][:, j].mean()),
                                       "mean_cost": float(d.D[te][:, j].mean())})
        out["h2_inputs"][(m, "meta")] = {"condition": d.condition[te], "query": qte,
                                         "s": d.s[te], "correlation": d.correlation[te],
                                         "rho_hat": d.features.rho_hat.to_numpy()[te]}
    return out


def _h2_task(y, preds, groups, seed, resamples, ci):
    rng = np.random.default_rng(seed)
    return L.cluster_bootstrap_spearman(y, preds, groups, rng, resamples, ci)


def h2_analysis(cfg, out, density: pd.DataFrame):
    bs = cfg["bootstrap"]
    dens = density.pivot_table(index=["condition", "query"], columns="k_local",
                               values="local_density")
    tasks, keys = [], []
    seed = bs["seed"] + 1
    for m in cfg["methods"]:
        meta = out["h2_inputs"][(m, "meta")]
        rho = dens.loc[list(zip(meta["condition"], meta["query"]))]
        for s_star in cfg["s_star"]:
            for router, r in out["h2_inputs"][(m, s_star)].items():
                for sub in ("random", "clustered", "both"):
                    keep = ~np.isnan(r) & (np.ones_like(r, bool) if sub == "both"
                                           else meta["correlation"] == sub)
                    preds = {"s": meta["s"][keep], "rho10": rho[10].to_numpy()[keep],
                             "rho100": rho[100].to_numpy()[keep],
                             "rho_hat": meta["rho_hat"][keep]}
                    tasks.append(delayed(_h2_task)(r[keep], preds, meta["query"][keep],
                                                   seed, bs["resamples_spearman"], bs["ci"]))
                    keys.append((m, s_star, router, sub, int(keep.sum())))
                    seed += 1
    res = Parallel(n_jobs=16)(tasks)
    rows = []
    for (m, s_star, router, sub, n), (point, diffs) in zip(keys, res):
        row = {"method": m, "s_star": s_star, "router": router, "subgroup": sub, "pairs": n}
        for k, v in point.items():
            row[f"rho_{k}"], row[f"rho_{k}_ci"] = v["rho"], v["ci"]
        for k, v in diffs.items():
            row[k], row[f"{k}_ci"] = v["diff"], v["ci"]
        rows.append(row)
    pq = []
    for m in cfg["methods"][:1]:
        meta = out["h2_inputs"][(m, "meta")]
        rho = dens.loc[list(zip(meta["condition"], meta["query"]))]
        for cond in np.unique(meta["condition"]):
            k = meta["condition"] == cond
            pq.append({"condition": cond,
                       "spearman_rho_hat_rho10": L.spearman(meta["rho_hat"][k], rho[10].to_numpy()[k]),
                       "spearman_rho_hat_rho100": L.spearman(meta["rho_hat"][k], rho[100].to_numpy()[k]),
                       "mean_rho_hat": float(meta["rho_hat"][k].mean()),
                       "mean_rho100": float(rho[100].to_numpy()[k].mean())})
    return pd.DataFrame(rows), pd.DataFrame(pq)


def figures(cfg, out, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e5e5e5", "grid.linewidth": 0.6})
    outdir.mkdir(parents=True, exist_ok=True)
    pol = pd.DataFrame(out["policies"])
    curve = pd.DataFrame(out["fixed_curve"])
    gate = pd.DataFrame(out["gate"])
    colors = {"B1": "#555555", "router_full": "#2a78d6", "router_minus_proxy": "#eb6834",
              "router_s_only": "#1baf7a", "router_full_lid": "#4a3aa7"}
    markers = {0.90: "o", 0.95: "s", 0.99: "^"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m in zip(axes, cfg["methods"]):
        c = curve[curve.method == m]
        ax.plot(c.success, c.mean_cost, color="#999999", lw=1.2, marker=".",
                label="single fixed budgets")
        p = pol[pol.method == m]
        for name, col in colors.items():
            for s_star, mk in markers.items():
                r = p[(p.policy == name) & (p.s_star == s_star)]
                ax.scatter(r.success, r.mean_cost, color=col, marker=mk, s=28,
                           label=name if s_star == 0.95 else None, zorder=3)
        ax.set(yscale="log", title=m, xlabel="test success rate (Recall@10 ≥ 0.9)",
               xlim=(0.85, 1.0))
    axes[0].set_ylabel("mean cost D (router: + probe)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Test cost vs success (markers: S* 0.90 ○, 0.95 □, 0.99 △)")
    fig.tight_layout()
    fig.savefig(outdir / "fig1_cost_vs_success.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    names = [n for n in colors if n != "B1"]
    for ax, m in zip(axes, cfg["methods"]):
        g = gate[gate.method == m]
        for i, name in enumerate(names):
            r = g[g.policy == name]
            x = np.arange(len(cfg["s_star"])) + (i - 1.5) * 0.18
            ci = np.array(r.saving_ci.tolist())
            ax.errorbar(x, 100 * r.saving, yerr=[100 * (r.saving - ci[:, 0]),
                                                 100 * (ci[:, 1] - r.saving)],
                        fmt="o", ms=4, capsize=2, color=colors[name], label=name)
        ax.axhline(10, color="#e34948", lw=0.9, ls="--")
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xticks(range(len(cfg["s_star"])), [f"S*={s}" for s in cfg["s_star"]])
        ax.set_title(m)
    axes[0].set_ylabel("cost saving vs B1 (%), 95% CI")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Saving vs the fixed-budget baseline B1 (dashed: 10% gate)")
    fig.tight_layout()
    fig.savefig(outdir / "fig2_saving_vs_b1.png", dpi=150)
    plt.close(fig)

    pc = pd.concat(out["per_condition"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, m in zip(axes, cfg["methods"]):
        r = pc[(pc.method == m) & np.isclose(pc.s_star, 0.95)].sort_values("condition")
        x = np.arange(len(r))
        ax.bar(x, r.router_cost / r.b1_cost, color="#2a78d6", width=0.6)
        ax.axhline(1.0, color="#555555", lw=0.9)
        ax.set_xticks(x, r.condition, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{m}, S* = 0.95")
    axes[0].set_ylabel("router cost / B1 cost (per condition)")
    fig.suptitle("Per-condition cost ratio, full router vs B1 (reported, not gated)")
    fig.tight_layout()
    fig.savefig(outdir / "fig3_per_condition_cost_ratio.png", dpi=150)
    plt.close(fig)


def main(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    for names in cfg["feature_sets"].values():
        L.check_features(names)
    data, integrity = load(cfg)
    nq = len(np.unique(next(iter(data.values())).query))
    train_q, test_q = L.query_split(nq, cfg["split"]["seed"], cfg["split"]["train_queries"])
    split_check = {"train": len(train_q), "test": len(test_q),
                   "disjoint": not np.intersect1d(train_q, test_q).size,
                   "complete": len(np.union1d(train_q, test_q)) == nq}
    pd.DataFrame({"query_id": np.concatenate([train_q, test_q]),
                  "split": ["train"] * len(train_q) + ["test"] * len(test_q)}) \
        .sort_values("query_id").to_csv(outdir / "split.csv", index=False)
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    out = evaluate(cfg, data, train_q, test_q, rng)
    density = pd.read_csv(cfg["inputs"]["local_density"])
    h2, proxy_quality = h2_analysis(cfg, out, density)

    gate = pd.DataFrame(out["gate"])
    gate.to_csv(outdir / "gate.csv", index=False)
    pd.DataFrame(out["policies"]).to_csv(outdir / "policies.csv", index=False)
    pd.DataFrame(out["regret"]).to_csv(outdir / "regret.csv", index=False)
    pd.concat(out["per_condition"]).to_csv(outdir / "per_condition.csv", index=False)
    pd.DataFrame(out["fixed_curve"]).to_csv(outdir / "fixed_curve.csv", index=False)
    pd.DataFrame(out["b1"]).to_csv(outdir / "b1.csv", index=False)
    h2.to_csv(outdir / "h2_regret.csv", index=False)
    proxy_quality.to_csv(outdir / "proxy_quality.csv", index=False)
    full = gate[gate.policy == "router_full"]
    verdict = {m: {"per_s_star": {str(r.s_star): bool(r["pass"]) for _, r in g.iterrows()},
                   "phase5_gate": bool(g["pass"].all())}
               for m, g in full.groupby("method")}
    (outdir / "summary.json").write_text(json.dumps({
        "config": cfg, "split_check": split_check, "nscan_join": integrity,
        "selection": out["selection"], "verdict": verdict,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": __import__("sklearn").__version__},
        "features_json": json.loads((Path(cfg["inputs"]["features"]) / "features.json").read_text()),
    }, indent=2, default=str))
    figures(cfg, out, outdir / "figures")
    print(json.dumps({"split": split_check, "verdict": verdict}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
