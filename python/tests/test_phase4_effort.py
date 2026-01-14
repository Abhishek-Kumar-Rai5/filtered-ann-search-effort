import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "phase4_effort",
    Path(__file__).resolve().parents[1] / "analysis" / "phase4_effort.py")
e = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(e)


def test_bootstrap_median_ci_is_seeded_and_brackets_median():
    x = np.arange(1, 1002, dtype=float)
    a = e.bootstrap_median_ci(x, np.random.default_rng(1), 500, 0.95)
    b = e.bootstrap_median_ci(x, np.random.default_rng(1), 500, 0.95)
    assert a == b
    med, lo, hi = a
    assert med == 501 and lo <= med <= hi and hi - lo < 100
    med2, _, _ = e.bootstrap_median_ci(np.array([1.0, np.inf, np.inf]),
                                       np.random.default_rng(1), 50, 0.95)
    assert np.isinf(med2)


def test_paired_log2_ratio_handles_censoring():
    r = e.paired_log2_ratio(np.array([4.0, np.inf, 2.0, np.inf]),
                            np.array([1.0, np.inf, np.inf, 8.0]))
    assert r[0] == 2.0 and r[1] == 0.0 and r[2] == -np.inf and r[3] == np.inf


def test_sign_test_counts_ties_and_infinite_values():
    t = e.sign_test(np.array([1, 2, 3, np.inf, 5.0]),
                    np.array([2, 2, 4, 1, np.inf]))
    assert (t["a_less"], t["a_greater"], t["ties"]) == (3, 1, 1)
    assert 0 < t["p"] <= 1
    t2 = e.sign_test(np.zeros(100), np.ones(100))
    assert t2["a_less"] == 100 and t2["p"] < 1e-20


def test_holm_matches_hand_computation():
    adj = e.holm([0.01, 0.04, 0.03, 0.5])
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])


def test_spearman_constant_input_is_nan():
    assert np.isnan(e.spearman(np.ones(5), np.arange(5)))
    assert e.spearman(np.arange(5), -np.arange(5.0)) == pytest.approx(-1.0)


def test_cluster_bootstrap_spearman_paired_by_query():
    rows = []
    rng = np.random.default_rng(0)
    for q in range(200):
        dens = rng.random()
        for c, s in enumerate((0.1, 1.0)):
            rows.append(dict(query_id=q, condition=f"c{c}", s_achieved=s,
                             rho10=dens, effort=10 - 5 * dens + 0.01 * c))
    f = pd.DataFrame(rows)
    res, diffs = e.cluster_bootstrap_spearman(f, "effort", ["s_achieved", "rho10"],
                                              np.random.default_rng(2), 200, 0.95)
    assert res["rho10"]["rho"] == pytest.approx(-1.0, abs=1e-3)
    d = diffs["|rho10|-|s_achieved|"]
    assert d["diff"] > 0.9 and d["ci"][0] > 0.8
    with pytest.raises(ValueError):
        e.cluster_bootstrap_spearman(f.iloc[1:], "effort", ["rho10"],
                                     np.random.default_rng(2), 10, 0.95)


def test_h3_paired_direction_and_holm():
    rows = []
    for q in range(300):
        for method, d in (("prefilter", 100.0), ("postfilter", 200.0 + q),
                          ("acorn", 50.0)):
            rows.append(dict(method=method, condition="random_s0.1000",
                             correlation="random", s_achieved=0.1, query_id=q,
                             oracle_dist=d))
    out = e.h3_paired(pd.DataFrame(rows), np.random.default_rng(3), 200, 0.95, 0.01)
    pre_post = out[(out.a == "prefilter") & (out.b == "postfilter")].iloc[0]
    assert pre_post.a_less == 300 and pre_post.significant
    assert pre_post.median_log2_a_over_b < 0
    post_acorn = out[(out.a == "postfilter") & (out.b == "acorn")].iloc[0]
    assert post_acorn.a_greater == 300 and post_acorn.median_log2_a_over_b > 0
