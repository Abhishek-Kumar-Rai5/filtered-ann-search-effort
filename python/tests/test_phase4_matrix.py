import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "phase4_matrix",
    Path(__file__).resolve().parents[1] / "analysis" / "phase4_matrix.py")
m = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(m)

BUDGETS = [10, 20, 40]


def frame(conds=("random_s0.1000", "clustered_s0.1000"), nq=3, s=0.1):
    rows = []
    for cond in conds:
        corr = cond.split("_")[0]
        for q in range(nq):
            rows.append(dict(query_id=q, condition=cond, correlation=corr,
                             s_achieved=s, method="prefilter", budget=0,
                             effective_list=0, recall=1.0, dist_exact=100,
                             dist_native=100, filter_checks=1000, rounds=0,
                             last_fetch=0, latency_us=5.0, n_valid=10,
                             filter_violations=0, distance_mismatches=0,
                             gt_tie_at_k=0))
            for i, b in enumerate(BUDGETS):
                for method in ("postfilter", "acorn"):
                    native = 50 + 10 * i + q
                    rows.append(dict(
                        query_id=q, condition=cond, correlation=corr,
                        s_achieved=s, method=method, budget=b,
                        effective_list=b, recall=min(1.0, 0.8 + 0.1 * i),
                        dist_exact=native + (1 if method == "acorn" else 0),
                        dist_native=native, filter_checks=20, rounds=1,
                        last_fetch=100, latency_us=3.0, n_valid=10,
                        filter_violations=0, distance_mismatches=0,
                        gt_tie_at_k=0))
    return pd.DataFrame(rows)


def test_completeness_detects_missing_duplicate_and_nan():
    df = frame()
    conds = sorted(df.condition.unique())
    assert m.check_completeness(df, conds, BUDGETS, 3)["pass"]
    assert not m.check_completeness(df.iloc[1:], conds, BUDGETS, 3)["pass"]
    dup = pd.concat([df, df.iloc[[0]]])
    assert m.check_completeness(dup, conds, BUDGETS, 3)["duplicates"] == 1
    bad = df.copy()
    bad.loc[0, "recall"] = np.nan
    assert m.check_completeness(bad, conds, BUDGETS, 3)["nonfinite_values"] == 1


def test_filter_and_accounting_checks():
    df = frame()
    assert m.check_filter(df)["pass"]
    thresholds = {c: 100 for c in df.condition.unique()}
    assert m.check_accounting(df, thresholds)["pass"]
    bad = df.copy()
    bad.loc[bad.method == "acorn", "dist_exact"] -= 1
    assert m.check_accounting(bad, thresholds)["acorn_not_native_plus_1"] > 0
    bad2 = df.copy()
    bad2.loc[0, "filter_violations"] = 1
    assert not m.check_filter(bad2)["pass"]
    assert not m.check_accounting(df, {c: 99 for c in thresholds})["pass"]


def test_monotone_check():
    df = frame()
    assert m.check_monotone(df)["pass"]
    bad = df.copy()
    sel = (bad.method == "acorn") & (bad.budget == 40)
    bad.loc[sel, "recall"] = 0.5
    assert ("acorn", "clustered_s0.1000", "recall") in \
        m.check_monotone(bad)["failures"]
    bad2 = df.copy()
    bad2.loc[bad2.method == "prefilter", "recall"] = 0.9
    assert not m.check_monotone(bad2)["pass"]


def test_s1_identity():
    df = frame(conds=("random_s1.0000", "clustered_s1.0000"), s=1.0)
    assert m.check_s1_identity(df, "random_s1.0000", "clustered_s1.0000")["pass"]
    bad = df.copy()
    bad.loc[(bad.condition == "clustered_s1.0000") & (bad.query_id == 1)
            & (bad.method == "acorn") & (bad.budget == 10), "dist_exact"] += 1
    assert not m.check_s1_identity(bad, "random_s1.0000",
                                   "clustered_s1.0000")["pass"]


def test_inert_budgets():
    df = frame(s=0.4)
    po = df.method == "postfilter"
    df.loc[po & (df.budget == 20), "recall"] = \
        df.loc[po & (df.budget == 10), "recall"].to_numpy()
    df.loc[po & (df.budget == 20), "dist_exact"] = \
        df.loc[po & (df.budget == 10), "dist_exact"].to_numpy()
    df.loc[po & (df.budget == 20), "dist_native"] = \
        df.loc[po & (df.budget == 10), "dist_native"].to_numpy()
    assert m.check_inert(df, 10)["pass"]
    df.loc[po & (df.budget == 20) & (df.query_id == 0), "dist_exact"] += 5
    assert not m.check_inert(df, 10)["pass"]


def test_oracle_first_budget_reaching_target_and_censoring():
    df = frame()
    o = m.compute_oracle(df, 0.9)
    acorn = o[(o.method == "acorn") & (o.condition == "random_s0.1000")]
    assert (acorn.oracle_budget == 20).all()
    assert list(acorn.oracle_dist) == [61.0, 62.0, 63.0]
    pre = o[o.method == "prefilter"]
    assert (pre.oracle_budget == 0).all() and (pre.oracle_dist == 100).all()
    never = df.copy()
    never.loc[never.method == "postfilter", "recall"] = 0.5
    o2 = m.compute_oracle(never, 0.9)
    post = o2[o2.method == "postfilter"]
    assert post.censored.all() and np.isinf(post.oracle_dist).all()


def test_v10_compares_low_selectivity_against_s1():
    lo = frame(s=0.01, conds=("random_s0.0100", "clustered_s0.0100"))
    hi = frame(s=1.0, conds=("random_s1.0000", "clustered_s1.0000"))
    hi["dist_exact"] = hi.dist_exact - 30
    o = m.compute_oracle(pd.concat([lo, hi]), 0.9)
    assert m.check_v10(o, 0.01, 1.0)["pass"]
    o.loc[np.isclose(o.s_achieved, 1.0), "oracle_dist"] = 1e9
    assert not m.check_v10(o, 0.01, 1.0)["pass"]


def test_determinism_check_on_subset():
    df = frame()
    sub = df[df.query_id.isin([0, 2])].copy()
    sub["latency_us"] = 999.0
    assert m.check_determinism(df, sub)["pass"]
    sub.loc[sub.index[0], "dist_exact"] += 1
    assert not m.check_determinism(df, sub)["pass"]


def test_read_neighbor_table_roundtrip(tmp_path):
    ids = np.arange(6, dtype=np.int64).reshape(3, 2)
    d = np.linspace(0, 1, 6, dtype=np.float32).reshape(3, 2)
    p = tmp_path / "t.bin"
    p.write_bytes(np.array([3, 2], dtype=np.uint64).tobytes() + ids.tobytes()
                  + d.tobytes())
    ri, rd = m.read_neighbor_table(p)
    assert np.array_equal(ri, ids) and np.array_equal(rd, d)


def test_aggregates_failed_fraction():
    a = m.aggregates(frame(), 0.9)
    row = a[(a.method == "acorn") & (a.budget == 10)].iloc[0]
    assert row.failed_fraction == pytest.approx(1.0)
    row = a[(a.method == "acorn") & (a.budget == 40)].iloc[0]
    assert row.failed_fraction == pytest.approx(0.0)


@pytest.mark.parametrize("d_lo,d_hi,f_lo,f_hi,expected", [
    (100, 200, 900, 300, True),
    (100, 200, 200, 300, False),
    (300, 200, 200, 300, True),
    (300, 200, 900, 300, True),
    (200, 200, 300, 300, False),
    (200, 200, 301, 300, True),
    (100, 200, 300, 300, False),
    (201, 200, 300, 300, True),
])
def test_not_dominated_rule(d_lo, d_hi, f_lo, f_hi, expected):
    assert m.not_dominated(d_lo, d_hi, f_lo, f_hi) is expected


def oracle_frame(acorn_lo, acorn_hi, post_lo=(500.0,) * 3, post_hi=(100.0,) * 3):
    rows = []
    for corr in ("random", "clustered"):
        for s, pairs in ((0.01, acorn_lo), (1.0, acorn_hi)):
            for q, p in enumerate(pairs):
                cens = p is None
                rows.append(dict(query_id=q, method="acorn", condition=f"{corr}_{s}",
                                 correlation=corr, s_achieved=s,
                                 oracle_budget=np.nan if cens else 10.0,
                                 oracle_dist=np.inf if cens else p[0],
                                 oracle_scan=np.inf if cens else p[1],
                                 censored=cens))
        for s, ds in ((0.01, post_lo), (1.0, post_hi)):
            for q, d in enumerate(ds):
                rows.append(dict(query_id=q, method="postfilter",
                                 condition=f"{corr}_{s}", correlation=corr,
                                 s_achieved=s, oracle_budget=10.0, oracle_dist=d,
                                 oracle_scan=np.nan, censored=False))
    return pd.DataFrame(rows)


def test_v10_revised_acorn_decided_by_pair_and_reports_d_decrease():
    o = oracle_frame([(100, 900)] * 3, [(200, 300)] * 3)
    r = m.check_v10_revised(o, 0.01, 1.0)
    assert r["pass"]
    acorn = [c for c in r["cells"] if c["method"] == "acorn"]
    assert all(c["D_decreases"] and c["median_F_low_s"] == 900 for c in acorn)
    o2 = oracle_frame([(100, 200)] * 3, [(200, 300)] * 3)
    assert not m.check_v10_revised(o2, 0.01, 1.0)["pass"]


def test_v10_revised_postfilter_keeps_strict_d_rule():
    ok = [(100, 900)] * 3
    assert not m.check_v10_revised(
        oracle_frame(ok, ok, post_lo=(100.0,) * 3, post_hi=(100.0,) * 3),
        0.01, 1.0)["pass"]


def test_v10_revised_censored_counts_as_infinite():
    o = oracle_frame([None, None, (1, 1)], [(200, 300)] * 3)
    r = m.check_v10_revised(o, 0.01, 1.0)
    acorn = [c for c in r["cells"] if c["method"] == "acorn"]
    assert all(np.isinf(c["median_F_low_s"]) for c in acorn) and r["pass"]
    o2 = oracle_frame([(100, 900)] * 3, [None, None, (1, 1)])
    assert not m.check_v10_revised(o2, 0.01, 1.0)["pass"]


def test_join_scan_and_oracle_scan():
    df = frame()
    a = df[df.method == "acorn"].copy()
    a["n_scanned"] = 1000 + a.budget
    joined, integrity = m.join_scan(df, a)
    assert integrity["pass"]
    assert joined.loc[joined.method != "acorn", "n_scanned"].isna().all()
    o = m.oracle_scan(m.compute_oracle(joined, 0.9), joined)
    acorn = o[o.method == "acorn"]
    assert (acorn.oracle_scan == 1000 + acorn.oracle_budget).all()
    assert o[o.method != "acorn"].oracle_scan.isna().all()
    bad = a.copy()
    bad.loc[bad.index[0], "dist_exact"] += 1
    assert not m.join_scan(df, bad)[1]["pass"]
    never = joined.copy()
    never.loc[never.method == "acorn", "recall"] = 0.5
    o2 = m.oracle_scan(m.compute_oracle(never, 0.9), never)
    assert np.isinf(o2[o2.method == "acorn"].oracle_scan).all()
