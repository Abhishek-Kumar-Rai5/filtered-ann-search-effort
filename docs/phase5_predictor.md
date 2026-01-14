# Phase 5 — Live effort predictor vs fixed-budget baseline

**Status (2026-10-06): run and evaluated. Phase 5 gate = FAIL for both
post-filter and ACORN (all three S\*).** This is a valid research outcome
under the frozen gate; the gate was not altered. Phase 6 not started.

Previous checkpoint: Phase 4 revised V10 PASS (docs/phase4_matrix.md §16;
ACORN F s=0.01 vs s=1: 70,576.5 vs 2,719 random, 18,648 vs 2,719 clustered;
post-filter D 12,407 / 108,654.5 vs 724), V1–V9, V11 PASS; matrix fingerprint
`79e1eebd582696aa` unchanged.

## 1. Frozen protocol (user decisions, 2026-10-06, before any fitting)
- **Scope:** one predictor per graph method (post-filter, ACORN), pooled over
  all 12 conditions. Pre-filter excluded (no effort knob; effort = T(s)).
  No new searches: a policy's cost/recall is a lookup of the Phase 4 matrix
  row at its chosen budget.
- **Split (decision 1):** the 10,000 Phase 4 queries, seeded permutation
  (seed 20261500): 5,000 train / 5,000 test, at query level (a query's 12
  condition pairs are on one side). `results/phase5/split.csv`. Known caveat:
  all 10K appeared in Phase 4's descriptive analysis (§17 there); nothing was
  selected from it.
- **Success:** per-query Recall@10 ≥ 0.9 (Phase 4 §1.6). Co-primary pooled
  success-rate targets **S\* ∈ {0.90, 0.95, 0.99}** (decision 6).
- **Effort tiers:** the frozen grid B = {10, …, 2560}. Oracle b\* = smallest
  successful b (censored if none) — cross-checked equal to Phase 4
  `compute_oracle` for every pair.
- **Cost:** D (exact distance computations). Router cost = probe D + 1
  (centroid distance) + D(chosen b), fully additive (decision 2). ACORN gates
  on D only; F (`n_scanned`) reported, never combined or gated (decision 5).
- **B1:** two adjacent grid budgets, per-query deterministic splitmix64 hash
  (seed 20261502) assigns the higher one when u < λ; identical in every
  condition; calibrated on **training** pairs to pooled success ≥ S\*.
  **B2:** the same form calibrated on test (hindsight reference, not gated).
- **Router (Project 1 design):** per budget, a model of P(success | features)
  — logistic regression (standardised, C = 1) or tree depth 2/3/4 — made
  non-decreasing across budgets (cumulative max); decision = cheapest b with
  P ≥ τ, fallback 2560. Grouped 5-fold CV on training (folds by query, seed
  20261501): τ (grid step 0.001) = lowest out-of-fold cost with OOF success
  ≥ S\*; model family = lowest OOF cost. Refit on all training; test once.
- **Features (live only; leakage guard in code):** full = {s, ρ̂,
  centroid_dist, score_concentration}; ablations minus-proxy, s-only,
  full + LID. True local density, the correlation label, D, F, recall, b\*
  are never inputs.
- **Gate (per method, decision 4; all three S\* must pass):** saving vs B1
  ≥ 10 % with paired query-cluster bootstrap 95 % CI > 0 and Wilcoxon
  signed-rank p < 0.01 (per-query mean cost difference), and lower CI bound
  of Δsuccess (router − B1) ≥ −0.01. Bootstrap 2,000 (seed 20261503).

## 2. Live features and probe (`fse_features`)
- One unfiltered probe per query on the **shared post-filter hnswlib index**
  (hash `2545153aadf3eee8`), **k = 20, ef = 20** (decisions 2–3), exact
  distance computations counted (thread-local counter).
- ρ̂ = fraction of the 20 probe results passing the condition's mask (mask
  regenerated from the hash-verified Phase 4 attributes; ground truth never
  read). score_concentration = d₁/d₁₀; LID = MLE over the 20 distances;
  centroid_dist = ‖q − base mean‖ (Project 1 definitions, ported to
  `cpp/feature_extraction/live_features.{h,cpp}` with Project 1's tests).
- Mean probe cost **553 dc/query** (min 169, max 1,158). Two runs
  byte-identical (features, proxy, probe table).
- **Limitation (decision 3):** for ACORN, the probe assumes a second
  (hnswlib) index alongside ACORN's graph; an ACORN-only deployment would
  probe its own graph instead (not evaluated).

Files: `cpp/cli/fse_features.cpp`, `configs/phase5/features.yaml`,
`results/phase5/features/{query_features.csv, proxy.csv, probe_k20_ef20.bin,
features.json}`.

## 3. Implementation and validation
- `python/predictor/phase5_lib.py` (pure functions), `python/predictor/phase5.py`
  (pipeline), config `configs/phase5/phase5.yaml`, run with the repo `.venv`.
- Selection (model family, τ, B1) receives training pairs only; test
  outcomes are used once, in evaluation; true density is read only in the
  post-hoc H2 step.
- Checks: split 5,000/5,000 disjoint and complete; oracle identical to
  Phase 4; leakage guard rejects any non-live feature; features complete and
  finite; ACORN F join 1,080,000 rows, 0 mismatches.
- Tests: C++ 73/73 (5 new `LiveFeatures.*`), Python 44/44 (10 new
  `test_phase5.py`); clang-tidy 0 findings, clang-format clean.

## 4. Results (test set, 5,000 queries × 12 conditions = 60,000 pairs per method)

### 4.1 Gate — full router vs B1

| Method | S\* | Saving vs B1 [95 % CI] | Wilcoxon p | Δsuccess [95 % CI] | Free-probe saving | Gate |
|---|---|---|---|---|---|---|
| post-filter | 0.90 | −1.8 % [−1.9, −1.7] | < 1e-4 | −0.0033 [−0.0061, −0.0004] | +0.1 % | FAIL (saving) |
| post-filter | 0.95 | −2.4 % [−2.7, −2.2] | 0.066 | −0.0014 [−0.0035, +0.0005] | −0.6 % | FAIL (saving) |
| post-filter | 0.99 | +4.9 % [+4.5, +5.4] | < 1e-4 | +0.0008 [+0.0002, +0.0015] | +6.1 % | FAIL (saving < 10 %) |
| ACORN | 0.90 | −34.6 % [−35.5, −33.7] | < 1e-4 | +0.0004 [−0.0025, +0.0032] | −5.7 % | FAIL (saving) |
| ACORN | 0.95 | −23.6 % [−24.6, −22.6] | < 1e-4 | +0.0025 [+0.0004, +0.0046] | −3.8 % | FAIL (saving) |
| ACORN | 0.99 | −13.8 % [−14.9, −12.6] | < 1e-4 | +0.0029 [+0.0020, +0.0039] | −3.8 % | FAIL (saving) |

**Phase 5 gate: post-filter FAIL, ACORN FAIL.** The success criterion
passes in every cell; the saving criterion fails in every cell.

Test mean cost (D): B1 post-filter 29,246 / 30,690 / 46,918; router
29,781 / 31,432 / 44,598. B1 ACORN 1,915 / 2,799 / 5,524; router 2,577 /
3,459 / 6,285 (S\* 0.90 / 0.95 / 0.99). B1 calibrated on training reached
test success 0.901 / 0.951 / 0.991 (post) and 0.900 / 0.950 / 0.989 (ACORN).
Versus B2 (hindsight) the savings differ from the B1 comparison by ≤ 3.7 pp (all still < 10 %).

### 4.2 Ablations (`gate.csv`, `policies.csv`; none passes)
- Post-filter: all variants within −2.1 % … +5.1 % of B1; minus-proxy
  +5.1 % at 0.99 (≈ full router).
- ACORN: s-only best (−27.1 / −29.6 / +0.3 %); with a free probe s-only
  reaches +1.8 / −9.9 / +10.3 % — the only cell ≥ 10 %, and only without
  probe cost. Adding ρ̂ (full vs minus-proxy) does not lower cost.
- ACORN F (supplementary, not gated): the full router uses less F than B1
  at 0.95 (−18 %) and 0.99 (−4 %), more at 0.90 (+6 %).

### 4.3 Regret decomposition (`regret.csv`)
Censored (unavoidable) pairs: post-filter 406, ACORN 24 of 60,000. At S\* =
0.95: post-filter router overspend on successes 2,514 dc (B1 1,841),
avoidable failures 4.4 % (B1 4.2 %), probe 553; ACORN router overspend
2,199 (B1 1,619), avoidable failures 4.7 % (B1 5.0 %), probe 553, against
an oracle mean of 1,500 dc.

### 4.4 Why the gate fails
- **ACORN:** the probe (553 dc) is 20–29 % of B1's mean cost (1,915–2,799 dc
  at 0.90–0.95). Even with a free probe the full router does not beat B1
  (−3.8 to −5.7 %): the live features carry little per-query information for
  ACORN (Phase 4 §17: within-condition |ρ(D, density)| ≤ 0.13 random,
  ≤ 0.39 clustered), so a single mixed budget is already near-optimal for
  pooled success.
- **Post-filter:** cost is dominated by the budget-inert k/s over-fetch
  (budgets ≤ ⌈k/s⌉ inert, Phase 4 V8); the budget the router controls moves
  a small fraction of total cost. Only at S\* = 0.99, where B1 must push all
  conditions to b = 1280/2560, does routing save 5 % (6 % probe-free) —
  below the 10 % threshold.

### 4.5 H2 on regret (post hoc; `h2_regret.csv`)
Per-pair regret r = log₂(cost / D\*) (success), +∞ (avoidable failure),
censored excluded; Spearman with query-cluster bootstrap (1,000).
- **Post-filter:** regret tracks global selectivity strongly (ρ_s ≈ 0.65–
  0.81). True density ρ₁₀₀ is about equal (full router, |ρ_s| − |ρ₁₀₀|:
  random +0.004 / +0.009 / +0.037; clustered −0.038 / −0.010 / +0.082 at
  S\* 0.90 / 0.95 / 0.99); ρ₁₀ is weaker than s everywhere (+0.07 … +0.21).
  **No consistent advantage of local density over selectivity.**
- **ACORN:** all correlations weak (|ρ| ≤ 0.38). True density correlates more
  strongly than s in most cells (both subgroups, full router: Δ = −0.055 /
  −0.051 at 0.95 / 0.99; s-only router −0.072 / −0.082), but not at S\* = 0.90
  (s stronger, +0.061) nor under random filters at 0.90–0.99 (s stronger,
  +0.014 … +0.025 for ρ₁₀₀). **Weak, condition-dependent support only.**
- Proxy quality (post hoc): Spearman(ρ̂, ρ₁₀) 0.52–0.55 (random), 0.73–0.90
  (clustered); ρ̂ correlates with regret about as well as ρ₁₀.
- **H2 verdict (preliminary): not supported as a general claim** — local
  density does not explain regret better than global selectivity
  consistently; at most a small advantage for ACORN regret at S\* ≥ 0.95.

## 5. Information use (§8 of the protocol)
Used: frozen Phase 4 definitions (grid, target, oracle, D/F methodology);
training-split rows and labels. Not used for any choice: test rows, test
b\*, Phase 4 aggregate tables, §17 per-query H2 results, true density (only
read after evaluation). Probe k/ef fixed a priori (not tuned).

## 6. Outputs
`results/phase5/`: `gate.csv`, `policies.csv`, `regret.csv`,
`per_condition.csv`, `fixed_curve.csv`, `b1.csv`, `h2_regret.csv`,
`proxy_quality.csv`, `split.csv`, `summary.json` (config, split check,
selection per method/feature set/S\*, verdict, versions), `figures/fig1–3`,
`features/`, `logs/phase5.log`.

## 7. Limitations
- Test queries were seen in Phase 4's descriptive analysis (decision 1).
- Single index configuration per method; SIFT1M only; D not latency.
- Shared hnswlib probe for ACORN assumes a second index (decision 3).
- Probe size fixed a priori (k = ef = 20); a cheaper or ACORN-native probe
  was not evaluated (would be a protocol change).
- Pooled S\*: per-condition success is reported, not guaranteed
  (`per_condition.csv`; ACORN router cost ≈ 2× B1 at s = 1).
