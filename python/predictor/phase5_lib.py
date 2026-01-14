"""Phase 5 predictor library (docs/phase5_predictor.md).

Pure functions: query split, per-budget success router (Project 1 design),
tau calibration, the fixed-budget baseline B1, regret decomposition, gate
statistics. Nothing here reads files. Model/threshold/baseline selection
functions receive training arrays only; the runner (phase5.py) enforces that
test outcomes are touched exactly once, in evaluation.

Conventions: a "pair" is one (query, condition); budgets are the frozen grid
B, indexed 0..len(B)-1; succ[pair, j] = Recall@10 >= 0.9 at budget j;
D[pair, j] = exact distance computations at budget j.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ALLOWED_FEATURES = {"s", "rho_hat", "centroid_dist", "score_concentration", "lid"}
FAMILIES = ("lr", "tree2", "tree3", "tree4")  # simplest first (tie-break)
MASK64 = (1 << 64) - 1


# ------------------------------------------------------------------ split --

def query_split(n_queries: int, seed: int, n_train: int):
    """Seeded permutation of query ids; first n_train = training."""
    perm = np.random.default_rng(seed).permutation(n_queries)
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def fold_of(queries: np.ndarray, folds: int, seed: int) -> dict:
    """Grouped CV: every training query (with all its conditions) in one fold."""
    perm = np.random.default_rng(seed).permutation(np.sort(queries))
    return {int(q): i % folds for i, q in enumerate(perm)}


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def hash_unit(queries: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic per-query value in [0, 1) for the B1 mix assignment."""
    return np.array([splitmix64(int(q) ^ seed) / 2.0**64 for q in queries])


def check_features(names) -> None:
    """Leakage guard: only frozen live features may enter a model."""
    bad = set(names) - ALLOWED_FEATURES
    if bad:
        raise ValueError(f"non-live feature(s) in model input: {sorted(bad)}")


# ----------------------------------------------------------------- router --

def make_model(family: str, cfg: dict):
    if family == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=cfg["lr"]["C"], max_iter=cfg["lr"]["max_iter"]))
    depth = int(family.removeprefix("tree"))
    if depth not in cfg["tree_depths"]:
        raise ValueError(family)
    return DecisionTreeClassifier(max_depth=depth, random_state=0)


class ConstantModel:
    """A budget whose training labels are all one class."""

    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, x):
        return np.column_stack([np.full(len(x), 1 - self.p), np.full(len(x), self.p)])


def fit_budget_models(x: np.ndarray, succ: np.ndarray, family: str, cfg: dict):
    models = []
    for j in range(succ.shape[1]):
        y = succ[:, j].astype(int)
        if y.min() == y.max():
            models.append(ConstantModel(float(y[0])))
        else:
            models.append(make_model(family, cfg).fit(x, y))
    return models


def predict_monotone(models, x: np.ndarray) -> np.ndarray:
    """P(success at budget j); made non-decreasing in j (cumulative max)."""
    p = np.column_stack([m.predict_proba(x)[:, 1] for m in models])
    return np.maximum.accumulate(p, axis=1)


def decide(p: np.ndarray, tau: float) -> np.ndarray:
    """Cheapest budget index with P >= tau; fallback = largest budget."""
    return np.minimum((p < tau).sum(axis=1), p.shape[1] - 1)


def oof_probabilities(x, succ, query_of_pair, folds: dict, family: str, cfg: dict):
    """Out-of-fold P on the training pairs (folds grouped by query)."""
    fold = np.array([folds[int(q)] for q in query_of_pair])
    p = np.empty(succ.shape, dtype=float)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        p[te] = predict_monotone(fit_budget_models(x[tr], succ[tr], family, cfg), x[te])
    return p


def pick(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return arr[np.arange(len(idx)), idx]


def calibrate_tau(p, succ, D, probe_cost, s_star: float, step: float):
    """Training-only: the tau whose decisions reach success >= s_star with the
    lowest mean cost (probe included). Returns (tau, success, cost, feasible);
    infeasible -> tau = +inf (always the largest budget)."""
    best = None
    for tau in np.round(np.arange(0.0, 1.0 + step / 2, step), 10):
        idx = decide(p, tau)
        sr = pick(succ, idx).mean()
        if sr + 1e-12 < s_star:
            continue
        cost = (probe_cost + pick(D, idx)).mean()
        if best is None or cost < best[2] - 1e-9:
            best = (float(tau), float(sr), float(cost), True)
    if best is None:
        idx = np.full(len(p), p.shape[1] - 1)
        best = (float("inf"), float(pick(succ, idx).mean()),
                float((probe_cost + pick(D, idx)).mean()), False)
    return best


# --------------------------------------------------------------- baseline --

def calibrate_b1(succ, D, query_of_pair, u_of_query: dict, s_star: float):
    """Two-adjacent-budget mix, identical across conditions, calibrated to
    pooled success >= s_star on the given (training) pairs. Queries with
    hash value u < lam get the higher budget. Returns dict(lo, hi, lam,
    success, cost, feasible)."""
    rates = succ.mean(axis=0)
    ok = np.nonzero(rates + 1e-12 >= s_star)[0]
    nb = succ.shape[1]
    if len(ok) == 0:
        return dict(lo=nb - 1, hi=nb - 1, lam=1.0, success=float(rates[-1]),
                    cost=float(D[:, -1].mean()), feasible=False)
    hi = int(ok[0])
    if hi == 0:
        return dict(lo=0, hi=0, lam=1.0, success=float(rates[0]),
                    cost=float(D[:, 0].mean()), feasible=True)
    lo = hi - 1
    qs = np.unique(query_of_pair)
    u = np.array([u_of_query[int(q)] for q in qs])
    order = qs[np.argsort(u, kind="stable")]
    gain = {int(q): 0.0 for q in qs}
    for q, g in zip(query_of_pair, succ[:, hi].astype(float) - succ[:, lo]):
        gain[int(q)] += g
    need = s_star * len(succ) - succ[:, lo].sum()
    cum = np.cumsum([gain[int(q)] for q in order])
    n = int(np.nonzero(cum + 1e-9 >= need)[0][0]) + 1
    lam = 1.0 if n >= len(order) else float(u_of_query[int(order[n])])
    idx = b1_assign(query_of_pair, u_of_query, dict(lo=lo, hi=hi, lam=lam))
    return dict(lo=lo, hi=hi, lam=lam, success=float(pick(succ, idx).mean()),
                cost=float(pick(D, idx).mean()), feasible=True)


def b1_assign(query_of_pair, u_of_query: dict, b1: dict) -> np.ndarray:
    u = np.array([u_of_query[int(q)] for q in query_of_pair])
    return np.where(u < b1["lam"], b1["hi"], b1["lo"])


# ------------------------------------------------------------------ regret --

def regret_decomposition(idx, succ, D, probe_cost, oracle_idx):
    """Test pairs. Censored pairs (oracle_idx < 0) are unavoidable failures:
    excluded from regret, included in success/cost totals."""
    ok = pick(succ, idx).astype(bool)
    cost = probe_cost + pick(D, idx)
    cens = oracle_idx < 0
    ev = ~cens
    dstar = np.where(ev, D[np.arange(len(idx)), np.maximum(oracle_idx, 0)], np.nan)
    good = ev & ok
    avoid = ev & ~ok
    return {
        "pairs": int(len(idx)), "censored": int(cens.sum()),
        "success_rate": float(ok.mean()), "mean_cost": float(cost.mean()),
        "oracle_mean_cost_evaluable": float(np.nanmean(dstar)),
        "avoidable_failure_rate": float(avoid.sum() / ev.sum()),
        "overspend_on_successes": float((cost[good] - dstar[good]).mean()) if good.any() else 0.0,
        "underspend_on_failures": float((dstar[avoid] - pick(D, idx)[avoid]).mean()) if avoid.any() else 0.0,
        "probe_overhead": float(np.mean(probe_cost)),
    }


def per_pair_regret(idx, succ, D, probe_cost, oracle_idx):
    """log2(cost / D*) on successes; +inf on avoidable failures; NaN on
    censored pairs (excluded)."""
    ok = pick(succ, idx).astype(bool)
    cost = probe_cost + pick(D, idx)
    r = np.full(len(idx), np.nan)
    ev = oracle_idx >= 0
    dstar = D[np.arange(len(idx)), np.maximum(oracle_idx, 0)]
    r[ev & ok] = np.log2(cost[ev & ok] / dstar[ev & ok])
    r[ev & ~ok] = np.inf
    return r


# -------------------------------------------------------------- statistics --

def per_query_mean(values: np.ndarray, query_of_pair: np.ndarray):
    qs, inv = np.unique(query_of_pair, return_inverse=True)
    return qs, np.bincount(inv, weights=values) / np.bincount(inv)


def gate_stats(cost_r, cost_b, ok_r, ok_b, query_of_pair, rng, resamples, ci):
    """Paired query-cluster bootstrap of saving = 1 - mean(cost_r)/mean(cost_b)
    and of dsuccess = success_r - success_b; Wilcoxon signed-rank on the
    per-query mean cost difference."""
    qs, inv = np.unique(query_of_pair, return_inverse=True)
    cnt = np.bincount(inv)
    cr = np.bincount(inv, weights=cost_r)
    cb = np.bincount(inv, weights=cost_b)
    sr = np.bincount(inv, weights=ok_r.astype(float))
    sb = np.bincount(inv, weights=ok_b.astype(float))
    saving = 1 - cr.sum() / cb.sum()
    dsucc = (sr.sum() - sb.sum()) / cnt.sum()
    bs, bd = np.empty(resamples), np.empty(resamples)
    for i in range(resamples):
        w = np.bincount(rng.integers(0, len(qs), len(qs)), minlength=len(qs))
        bs[i] = 1 - (w * cr).sum() / (w * cb).sum()
        bd[i] = ((w * sr).sum() - (w * sb).sum()) / (w * cnt).sum()
    q = [50 * (1 - ci), 50 * (1 + ci)]
    diff = (cb - cr) / cnt
    p = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
    return {"saving": float(saving), "saving_ci": np.percentile(bs, q).tolist(),
            "dsuccess": float(dsucc), "dsuccess_ci": np.percentile(bd, q).tolist(),
            "wilcoxon_p": p, "median_query_cost_diff": float(np.median(diff))}


def gate_pass(g: dict, gate: dict) -> dict:
    c1 = g["saving"] >= gate["min_saving"] and g["saving_ci"][0] > 0 \
        and g["wilcoxon_p"] < gate["alpha"]
    c2 = g["dsuccess_ci"][0] >= gate["min_dsuccess_lb"]
    return {"saving_criterion": bool(c1), "success_criterion": bool(c2),
            "pass": bool(c1 and c2)}


def spearman(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def cluster_bootstrap_spearman(y, preds: dict, groups, rng, resamples, ci):
    """Spearman(y, each predictor) with a bootstrap that resamples groups
    (queries) with replacement; groups may have unequal sizes. Also CIs of
    |rho_a| - |rho_b| for each predictor pair."""
    y = np.asarray(y, float)
    gs, inv = np.unique(groups, return_inverse=True)
    members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    point = {k: spearman(y, v) for k, v in preds.items()}
    boot = {k: np.empty(resamples) for k in preds}
    for i in range(resamples):
        idx = np.concatenate([members[j] for j in rng.integers(0, len(gs), len(gs))])
        for k, v in preds.items():
            boot[k][i] = spearman(y[idx], np.asarray(v, float)[idx])
    q = [50 * (1 - ci), 50 * (1 + ci)]
    out = {k: {"rho": point[k], "ci": np.nanpercentile(boot[k], q).tolist()} for k in preds}
    names = list(preds)
    diffs = {f"|{a}|-|{b}|": {"diff": abs(point[a]) - abs(point[b]),
                              "ci": np.nanpercentile(np.abs(boot[a]) - np.abs(boot[b]), q).tolist()}
             for i, a in enumerate(names) for b in names[i + 1:]}
    return out, diffs
