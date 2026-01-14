import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "predictor"))
import phase5_lib as L  # noqa: E402

CFG = {"lr": {"C": 1.0, "max_iter": 2000}, "tree_depths": [2, 3, 4]}


def test_query_split_is_seeded_disjoint_and_complete():
    a_tr, a_te = L.query_split(100, 7, 50)
    b_tr, b_te = L.query_split(100, 7, 50)
    assert np.array_equal(a_tr, b_tr) and np.array_equal(a_te, b_te)
    assert len(a_tr) == len(a_te) == 50
    assert not np.intersect1d(a_tr, a_te).size
    assert np.array_equal(np.union1d(a_tr, a_te), np.arange(100))
    assert not np.array_equal(a_tr, L.query_split(100, 8, 50)[0])


def test_folds_group_queries_and_hash_is_deterministic():
    f = L.fold_of(np.arange(20), 5, 1)
    assert sorted(set(f.values())) == [0, 1, 2, 3, 4]
    assert all(list(f.values()).count(k) == 4 for k in range(5))
    u1 = L.hash_unit(np.arange(10), 3)
    assert np.array_equal(u1, L.hash_unit(np.arange(10), 3))
    assert ((u1 >= 0) & (u1 < 1)).all() and len(set(u1)) == 10


def test_leakage_guard_rejects_post_hoc_features():
    L.check_features(["s", "rho_hat", "centroid_dist", "score_concentration", "lid"])
    for bad in (["rho10"], ["s", "local_density"], ["oracle_budget"], ["n_scanned"]):
        with pytest.raises(ValueError):
            L.check_features(bad)


def test_decide_cheapest_budget_meeting_tau_with_fallback():
    p = np.maximum.accumulate(np.array([[0.1, 0.5, 0.9], [0.0, 0.2, 0.3],
                                        [0.95, 0.4, 0.99]]), axis=1)
    assert p[2, 1] == 0.95
    assert L.decide(p, 0.5).tolist() == [1, 2, 0]
    assert L.decide(p, 0.0).tolist() == [0, 0, 0]


def test_calibrate_tau_picks_cheapest_feasible_threshold():
    succ = np.array([[0, 1], [1, 1]], bool)
    D = np.array([[10.0, 100.0], [10.0, 100.0]])
    p = np.array([[0.3, 1.0], [0.8, 1.0]])
    tau, sr, cost, feas = L.calibrate_tau(p, succ, D, np.zeros(2), 1.0, 0.1)
    assert feas and sr == 1.0 and cost == 55.0 and 0.3 < tau <= 0.8
    tau, sr, cost, feas = L.calibrate_tau(p, succ, D, np.zeros(2), 0.5, 0.1)
    assert cost == 10.0 and sr == 0.5
    succ2 = np.array([[0, 0], [1, 1]], bool)
    tau, sr, cost, feas = L.calibrate_tau(p, succ2, D, np.zeros(2), 1.0, 0.1)
    assert not feas and np.isinf(tau) and cost == 100.0


def test_calibrate_b1_minimal_mix_identical_across_conditions():
    q = np.repeat(np.arange(4), 2)
    succ = np.array([[1, 1], [1, 1], [0, 1], [0, 1],
                     [1, 1], [0, 1], [0, 1], [1, 1]], bool)
    D = np.tile([10.0, 20.0], (8, 1))
    u = {0: 0.1, 1: 0.4, 2: 0.6, 3: 0.9}
    b = L.calibrate_b1(succ, D, q, u, 0.75)
    assert (b["lo"], b["hi"]) == (0, 1) and b["success"] >= 0.75
    idx = L.b1_assign(q, u, b)
    assert all(len(set(idx[q == k])) == 1 for k in range(4))
    assert b["lam"] == 0.6 and idx.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert L.calibrate_b1(succ, D, q, u, 0.5)["hi"] == 0


def test_regret_decomposition_and_per_pair_regret_with_censoring():
    succ = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0], [1, 1, 1]], bool)
    D = np.array([[1.0, 2, 4], [1, 2, 4], [1, 2, 4], [1, 2, 4]])
    oracle = np.array([1, 2, -1, 0])
    idx = np.array([2, 1, 2, 0])
    probe = np.full(4, 1.0)
    r = L.regret_decomposition(idx, succ, D, probe, oracle)
    assert r["censored"] == 1 and r["success_rate"] == 0.5
    assert r["avoidable_failure_rate"] == pytest.approx(1 / 3)
    assert r["overspend_on_successes"] == pytest.approx(((5 - 2) + (2 - 1)) / 2)
    assert r["underspend_on_failures"] == pytest.approx(4 - 2)
    pr = L.per_pair_regret(idx, succ, D, probe, oracle)
    assert pr[0] == pytest.approx(np.log2(5 / 2)) and np.isinf(pr[1])
    assert np.isnan(pr[2]) and pr[3] == pytest.approx(1.0)


def test_gate_stats_and_gate_rule():
    rng = np.random.default_rng(0)
    q = np.repeat(np.arange(400), 3)
    cost_b = np.full(len(q), 100.0)
    ok = np.ones(len(q), bool)
    g = L.gate_stats(cost_b * 0.8 + rng.normal(0, 1, len(q)), cost_b, ok, ok, q,
                     np.random.default_rng(1), 300, 0.95)
    assert g["saving"] == pytest.approx(0.2, abs=0.01)
    assert g["saving_ci"][0] > 0.18 and g["wilcoxon_p"] < 1e-10
    gate = {"min_saving": 0.10, "alpha": 0.01, "min_dsuccess_lb": -0.01}
    assert L.gate_pass(g, gate)["pass"]
    worse = ok.copy()
    worse[: len(q) // 10] = False
    g2 = L.gate_stats(cost_b * 0.8, cost_b, worse, ok, q, np.random.default_rng(1),
                      300, 0.95)
    assert g2["dsuccess"] == pytest.approx(-0.1, abs=0.01)
    res = L.gate_pass(g2, gate)
    assert res["saving_criterion"] and not res["success_criterion"] and not res["pass"]
    g3 = L.gate_stats(cost_b * 0.95, cost_b, ok, ok, q, np.random.default_rng(1), 300, 0.95)
    assert not L.gate_pass(g3, gate)["pass"]


def test_router_learns_a_separable_budget_rule():
    rng = np.random.default_rng(0)
    x = rng.random((600, 1))
    succ = np.column_stack([x[:, 0] < 0.5, np.ones(600, bool)])
    for fam in L.FAMILIES:
        p = L.predict_monotone(L.fit_budget_models(x, succ, fam, CFG), x)
        idx = L.decide(p, 0.5)
        assert (L.pick(succ, idx)).mean() > 0.95
        assert 0.4 < (idx == 0).mean() < 0.6


def test_cluster_bootstrap_spearman_unequal_groups():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(300), rng.integers(1, 5, 300))
    dens = rng.random(len(groups))
    y = -dens + 0.01 * rng.normal(size=len(groups))
    noise = rng.random(len(groups))
    res, diffs = L.cluster_bootstrap_spearman(y, {"d": dens, "n": noise}, groups,
                                              np.random.default_rng(1), 100, 0.95)
    assert res["d"]["rho"] < -0.99 and abs(res["n"]["rho"]) < 0.15
    assert diffs["|d|-|n|"]["ci"][0] > 0.7
