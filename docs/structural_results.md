# Structural Results — Topological vs Navigational Recall Loss (final)

Status: **final** (2026-10-07). Definitions are frozen in
`docs/structural_design.md`. This document reports only what was actually
run; the full factorial study in design §4 was **not** run (see §1.2).

## 1. What was run

### 1.1 Experiments that produced the reported results
All at selectivity **s = 0.01**, SIFT1M (1M base), the fixed 10,000 Phase 4
queries, k = 10, exact filtered ground truth, frozen budget grid
{10, 20, 40, 80, 160, 320, 640, 1280, 2560}, frozen indexes.

| Run | Conditions | Methods | Measured | Output |
|---|---|---|---|---|
| Vertical slice | random r0, clustered C=1000 r0 | ACORN-1, ACORN-γ | U, N(b), recall, connectivity, soundness | `results/structural/slice/` |
| **Decision experiment (main)** | C=100 × 5 realizations, C=1000 × 5, random × 2 | ACORN-1, ACORN-γ | U, R_graph/R_sem/R_cap connectivity for all 12 conditions; U, N(b), recall for realization 0 of each level | `results/structural/pilot/decision/` |
| Baselines | Phase 4 s = 0.01 random and clustered (= C1000 r0, identical masks) | PRE, POST (Phase 4 sweeps, not rerun) | U, N(b), recall, effort | `results/structural/baseline/` |

Indexes (hashes verified on every load): ACORN-γ γ = 100, M = 32, Mβ = 64
(Phase 4, `20b0dacd17007dd4`); ACORN-1 γ = 1, M = 32, Mβ = 64 (Phase 1 build A,
`d228d82803cf00cb`); POST HNSW M = 32, efc = 40 (Phase 4, `2545153aadf3eee8`).
Fragmentation is the designed k-means cluster count C (fixed partition per C,
realizations differ only in the cluster-order seed); random = maximal
fragmentation. Realization 0 at C = 1000 and random r0 reproduce the Phase 4
attributes exactly.

### 1.2 Planned but not run (stopped by decision)
- The full design (§4: s ∈ {0.01, 0.0625, 0.25} × C ∈ {100, 1000, 10000,
  random} × 5 realizations × 4 methods) was **not run**. The user stopped it
  in favour of the completed pilot evidence.
- C = 10,000 was **not analysed**: its conditions and ground truth were
  generated (`results/structural/final/conditions/`) but no ACORN sweep or
  reachability analysis was completed. Random has only 2 realizations.
- `results/structural/mvp/` (stopped during POST C100) and the partial
  sweeps in `results/structural/final/matrix*/` are incomplete and are not
  used.
- No other selectivity, dataset, latency study or predictor.

## 2. Validation (all pass)
- **Identity** 1 − Recall = U + N(b): 0 violations in 540,000 (decision),
  360,000 (slice) and 200,000 (baseline) per-query rows.
- **Soundness of R_sem**: every id ACORN returned, at every budget, is
  reachable from the instrumented level-0 seed — 0 violations in 3.6 M
  (slice) checks; 0 found targets outside reach in all decompositions. POST:
  0 violations in 1.8 M checks on the unfiltered graph.
- Recall recomputed from returned ids = recorded recall (0 mismatches);
  R_graph ⊆ R_sem and R_cap ⊆ R_sem everywhere; level-0 seed identical across
  budgets; instrumented ACORN build byte-identical to stored Phase 4 rows.
- Tests: C++ 79/79 (incl. soundness on real toy ACORN indexes, γ = 1 and
  γ = 4), Python 49/49 (incl. decomposition tests); clang-tidy 0 findings.

## 3. Results

Tables: `results/structural/report/tables/`; figures:
`results/structural/report/figures/` (produced by
`python/analysis/structural_report.py` from the runs above).

### 3.1 Topological loss U vs fragmentation (T1, F1)
U = fraction of the 10 true targets unreachable from the query's level-0
seed under ACORN's own traversal rules (R_sem). Mean over realizations
[min, max].

| Method | C = 100 (1–2 clusters), n = 5 | C = 1000 (10–11), n = 5 | random, n = 2 |
|---|---|---|---|
| ACORN-1 | **2.71 %** [0, 10.8 %] | 0.059 % [0.007, 0.112 %] | 0.040 % [0.039, 0.040 %] |
| ACORN-γ | 0.000 % [0, 0.001 %] | 0.005 % [0, 0.014 %] | **0.185 %** [0.138, 0.232 %] |
| U if only the plain filtered graph were traversable (ACORN-1 / ACORN-γ) | 15.6 % / 2.9 % | 22.1 % / 4.6 % | 99.2 % / 99.7 % |

- ACORN-γ's U rises with fragmentation (Spearman ρ = 0.75, p = 0.005;
  Kruskal–Wallis p = 0.034 across levels, realization level).
- ACORN-1's U is not monotone in C (Kruskal–Wallis p = 0.36): it is highest
  for the *least* fragmented predicates, and within C = 100 it is driven by
  the number of separate passing islands — realizations with 2 passing
  clusters: U = 0.54 %, 2.17 %, 10.8 %; with 1 cluster: 0.07 %, 0 %.

### 3.2 ACORN-1 vs ACORN-γ (T3, T3b)
Same predicate, paired:
- **C = 100 and C = 1000: ACORN-γ has lower U** in every realization where
  they differ (C100: 4/4, one tie; C1000: 5/5). Query level: ACORN-γ is
  better on 67–1,992 queries per affected C100 realization; Wilcoxon p ≈ 0
  after Holm in 8 of 10 clustered conditions (C1000 r0 n.s.; C100 r3 tie).
- **random: ACORN-1 has lower U** (2/2 realizations; 137–213 queries worse
  under ACORN-γ vs 30–39 under ACORN-1; Holm p ≈ 0).
- Realization-level intervals are wide (n = 5 / 2); the direction is
  consistent, the magnitude is realization-dependent.

### 3.3 Mechanism: connectivity of the passing set (T2, F2)
Largest strongly connected component / passing nodes (mean):

| | C100 plain → R_sem | C1000 plain → R_sem | random plain → R_sem |
|---|---|---|---|
| ACORN-1 | 0.950 → 0.977 (6.2 SCCs) | 0.464 → 0.999 | 0.002 → 0.998 (21 SCCs) |
| ACORN-γ | 0.992 → 1.000 (2.6 SCCs) | 0.621 → 1.000 | 0.002 → 0.966 (337 SCCs) |

- Fragmentation shatters the plain filtered graph (random: largest SCC
  ≈ 20 of 10,000 nodes). ACORN's search semantics (2-hop expansion through
  non-passing neighbours) repair almost all of it.
- **ACORN-1** expands every list position: it reconnects scattered
  predicates best, but its sparse graph (M = 32, no γ densification) cannot
  bridge *separate dense islands* — a query whose level-0 seed is in one
  island cannot reach targets in the other (C100).
- **ACORN-γ** has dense build-time lists (M·γ at upper levels, compressed
  level 0) that bridge islands, but expands only list positions ≥ Mβ = 64
  at level 0, which leaves a scattered predicate in ~337 SCCs with 3.4 % of
  passing nodes outside the largest.
- The 2M truncation is small topologically: R_cap adds ≤ 0.11 pp to U in
  every cell except ACORN-1 at C = 100 (+1.4 pp: 4.1 % vs 2.7 %).

### 3.4 U vs N over the budget (T5, F3; realization 0)
- At typical budgets the loss is almost entirely **navigational**: U is
  ≤ 1.7 % of the loss at b ≤ 40 in every cell and < 8 % at b ≤ 160.
- U is a **budget-independent floor that becomes the residual loss** at high
  budgets: at b = 2560, U / L = 89 % (ACORN-1, C100), 91 % (ACORN-1,
  random), 90 % (ACORN-γ, random). It sets the recall ceiling 1 − U (e.g.
  ACORN-1 C100 r0: 0.9946; ACORN-1 C100 r1: 0.892).
- Fragmentation also changes navigation: the concentrated C100 predicate is
  the hardest at small budgets (recall at b = 10: 0.38 / 0.37 vs 0.67 / 0.47
  for random, ACORN-1 / ACORN-γ).

### 3.5 Baselines (Phase 4, T5, F4)
At s = 0.01:
- **PRE** (exact scan of the passing set): recall 1, U = N = 0, D = 10,000
  distance computations.
- **POST** (unfiltered HNSW + filter, over-fetch k/s, ≤ 10 rounds): its
  unfiltered graph is one SCC of 999,999 / 1,000,000 nodes, so **U = 0**;
  all loss is navigational (N = 1.9–2.2 % at b ≤ 640, 0.9–1.5 % at
  b = 2560) and costs D = 17K–210K. Budgets ≤ ⌈k/s⌉ = 1000 are inert.
- **ACORN** (these two conditions) reaches recall ≥ 0.99 at D ≈ 2,200–2,600
  (b = 640): ~4× cheaper than PRE, and below POST's 17K–210K, which never
  reaches 0.99 on the clustered predicate (max 0.985). ACORN's loss is navigational except for its U floor.

## 4. Conclusions
1. **Under ACORN's own search semantics, filtered recall loss at s = 0.01
   is overwhelmingly navigational.** The plain filtered graph is shattered
   by fragmented predicates (up to 99 % of targets unreachable), but ACORN's
   2-hop traversal repairs nearly all of it; residual topological loss U is
   ≤ 0.2 % on average in every cell except ACORN-1 on island-like (C = 100)
   predicates (2.7 %, up to 10.8 % in one realization).
2. **U is small on average but structurally decisive at the top of the
   recall range.** It is budget-independent, becomes 89–91 % of the
   remaining loss at the largest budget, and fixes the achievable recall
   ceiling — no budget can recover it.
3. **ACORN-1 and ACORN-γ repair different failure modes (method ×
   fragmentation crossover).** ACORN-γ's densified graph bridges separate
   predicate islands (U ≈ 0 for C ≤ 1000) but its Mβ-limited 2-hop
   expansion leaves scattered predicates partly disconnected (U highest for
   random). ACORN-1's full 2-hop expansion handles scattered predicates
   (U = 0.04 %) but cannot bridge distant islands (U up to 10.8 % with two
   passing clusters).
4. **Fragmentation is not a single scalar.** What predicts ACORN-1's U is
   the number of separated islands; what predicts ACORN-γ's U is scatter.
   Global selectivity (fixed here) explains neither.
5. **Baselines behave as expected:** PRE has no topological or navigational
   loss at 4× ACORN's cost; POST has no topological loss (connected
   unfiltered graph) and pays all of its loss and cost in navigation.

## 5. Limitations
- One selectivity (s = 0.01), one dataset (SIFT1M), k = 10, one index per
  method (no build-to-build variation; ACORN-1 uses the Phase 1 driver's
  Mβ = 64, not the paper's Mβ = M).
- C = 10,000 not analysed; random has 2 realizations; N(b) and recall come
  from realization 0 only; realization-level confidence intervals are wide.
- R_sem is a cap-free, budget-free closure from the true level-0 seed: U is
  a lower bound on execution-level unreachability; truncation effects are
  counted as navigational (R_cap sensitivity: ≤ 0.11 pp, except +1.4 pp for
  ACORN-1 at C = 100).
- POST's level-0 seed is not observable (hnswlib black box); U_POST uses
  the unfiltered graph's giant SCC (999,999 / 1M nodes).
- Effort is exact distance computations D; ACORN's filter/traversal work
  (`n_scanned`, Phase 4 §15) is reported but not combined with D; no
  latency claims.
- The full factorial design (§1.2) was not run; conclusions about
  selectivity dependence are not supported.

## 6. Reproducibility artifacts
- Design / definitions: `docs/structural_design.md`.
- One-command reproduction of the reported results (after the Phase 1/4
  prerequisites): `scripts/reproduce_structural.sh`.
- Code: `cpp/reachability/` (edge rules, SCC/reach), `cpp/cli/fse_reach.cpp`,
  `cpp/cli/fse_fragment_conditions.cpp`, `cpp/cli/fse_matrix.cpp` (manifest
  conditions, seed0 column, POST inert-budget reuse, index override);
  ACORN instrumentation `patches/acorn_instrumentation.patch` (the single
  patch: tagged `n_scanned` + `acorn_level0_seed`; applies to
  `third_party/acorn` @ c259f11).
- Analysis: `python/analysis/decomposition.py` (L = U + N, validation),
  `python/analysis/structural_manifests.py` (decision / slice manifests),
  `python/analysis/structural_decision_summary.py` (decision summary),
  `python/analysis/structural_report.py` (final tables and figures).
- Tests: `cpp/tests/test_reachability.cpp`, `python/tests/test_decomposition.py`.
- Configs used: `configs/structural/{conditions_pilot,slice_*,decision_*,baseline_*}.yaml`.
  Configs/scripts of the stopped runs are in `configs/structural/archive/`
  and `scripts/archive/` (not used for any result).
- Tracked summary / provenance files (small): manifests with all hashes
  (`results/structural/pilot/decision/manifest*.yaml`,
  `results/structural/slice/manifest.yaml`), per-realization summaries
  (`results/structural/pilot/decision/decision_*.csv`), decomposition
  aggregates and validation (`*/analysis*/{aggregates,components}.csv`,
  `validation.json`), reachability statistics (`*/reach/*/reach.json`).
- Final tables and figures: `results/structural/report/`.
- Not tracked (large, regenerable): `data/` (SIFT1M, indexes, attribute and
  ground-truth caches) and raw outputs (`per_query.csv`, `raw_b*.bin`,
  `*_reach.csv`, `decomposition_rows.csv.gz`, logs). Without the per-query
  reachability dumps, `structural_report.py` reuses the committed query-level
  table T3b; every other table is rebuilt from the tracked summaries.
