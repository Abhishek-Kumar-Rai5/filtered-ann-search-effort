# Structural Design — Topological vs Navigational Recall Loss in Filtered ANN

**Status: FROZEN 2026-10-06 (definitions §3–§5 fixed before any
implementation or result).** This document supersedes `docs/design_doc.md`
as the source of truth for all work from this date. `docs/design_doc.md`
remains the record of Phases 0–5; Phase 5 (live effort predictor) is
**archived as a completed negative result** (`docs/phase5_predictor.md`) and
is not continued. Phases 1–4 results are reused unchanged.

## 1. Research question

At exactly matched global selectivity, what fraction of filtered-ANN recall
loss is caused by true targets being **unreachable** under the search's
traversal semantics, versus targets being **reachable but not found** within
the search budget? How does this balance change with **predicate
fragmentation**, and which component do ACORN's repair mechanisms (ACORN-1's
search-time 2-hop expansion; ACORN-γ's build-time neighbour densification
with compressed level-0 lists) affect?

Novelty position: filtered-ANN connectivity, selectivity difficulty,
fragmentation/percolation and ACORN's connectivity argument are prior work.
The hypothesis to be established experimentally is the exact per-query
decomposition below under controlled fragmentation; no "first" claim is
made.

## 2. Decomposition

For query q, method m, budget b, k = 10, exact filtered top-k targets T_q
(Phase 3/4 verified ground truth; id-based; ties at rank k flagged):

    L_q,m(b) = 1 − Recall_q,m(b) = U_q,m + N_q,m(b)
    U_q,m    = |T_q \ Reach_m(P, q)| / k              (topological)
    N_q,m(b) = |(T_q ∩ Reach_m(P, q)) \ Found_q,m(b)| / k   (navigational)

The two terms partition the missed targets, so the identity is exact.
U does not depend on the budget.

## 3. Reachability semantics (frozen)

Notation: level-0 adjacency lists Γ(u) in stored order (truncated at the first
−1); P = set of filter-passing base vectors; s₀(q) = the level-0 starting node
actually used by the method's search for q (§3.4). All reachability is
**directed** (search follows out-lists) and evaluated on level 0 (upper levels
only choose s₀; all results come from level 0).

### 3.1 ACORN search behaviour these definitions follow (code `c259f11`, `hybrid_search`)
Upper levels: greedy descent from the entry point, beam 1 → s₀(q) (may fail
the filter if no passing node was met). Level 0: bounded best-first search;
for each popped node u, scan Γ(u) in order; a passing, unvisited entry gets a
distance and becomes a candidate; non-passing nodes never become candidates.
**2-hop expansion**: an unvisited entry v at position j of Γ(u) has its own
list Γ(v) scanned (passing, unvisited entries added) when **j ≥ Mβ (γ > 1)**
or **always (γ = 1)**; v itself need not pass. **Truncation**: once the scan
of Γ(u) has counted 2M passing entries, it stops (state-dependent detail:
the stop is checked only after adding an unvisited entry).

### 3.2 Notions

| Notion | Edge u → w (w ∈ P) iff | Role |
|---|---|---|
| **R_sem (primary)** | w ∈ Γ(u), or w ∈ Γ(v) for some v ∈ Γ(u) at a position that is expanded (ACORN-γ: j ≥ Mβ; ACORN-1: any j) | ACORN's own step rules with **unlimited budget and no truncation** |
| R_graph (reference) | w ∈ Γ(u) | plain filtered subgraph (graph connectivity) |
| R_cap (sensitivity) | the entries a single scan of Γ(u) adds when simulated exactly with an empty visited set (including the 2M stop and the 2-hop rule) | the fixed part of the truncation rule |
| POST (unfiltered HNSW) | w ∈ Γ_HNSW(u), all nodes eligible | POST traverses the unfiltered graph, filters afterwards |
| PRE | Reach = P | exact scan: U = N = 0 by construction |

Reach_m(P, q) = nodes reachable from s₀(q) along these edges (s₀'s own
out-edges follow the same rule whether or not s₀ passes).

### 3.3 Decisions (user, 2026-10-06)
1. **R_sem is primary.** The 2M truncation is treated as **navigational**
   (a target lost only to truncation lands in N). R_cap is a sensitivity
   analysis only.
   Soundness argument: every node an actual search can add is reachable in
   R_sem (truncation and the bounded queue only remove edges; a passing
   neighbour whose 2-hop expansion is skipped because it was already visited
   is itself a candidate whose own list reaches the same nodes when popped).
   Hence U is a lower bound on unreachability under any budget, and a target
   outside R_sem cannot be found by any budget.
2. **Exact s₀(q)** from a minimal, tagged, additive ACORN instrumentation
   (`acorn_level0_seed`), behaviour preserved byte-for-byte.
3. **Fixed k-means partition per C**; realizations differ only in the
   cluster-order seed (which clusters pass). Varying the k-means seed is a
   later robustness check.
4. **Designed fragmentation level C** is the factor (passing set ≈ s·C whole
   clusters; random = maximal fragmentation). Measured component statistics
   (SCC/WCC counts, largest-component fraction under R_graph/R_sem) are
   reported separately as outcomes.
5. **POST**: unfiltered HNSW level-0 graph. If effectively strongly
   connected, U ≈ 0 is reported; otherwise the exposed fraction (targets
   outside the giant SCC's reach).
6. **ACORN-1 = the frozen Phase 1 index** (γ = 1, M = 32, Mβ = 64, build A);
   not rebuilt. **ACORN-γ = the frozen Phase 4 index** (γ = 100, M = 32,
   Mβ = 64). The Phase 1 γ = 12 index is available for a later γ
   dose-response.
7. **k = 10**; ties at rank k flagged as in Phase 4.

### 3.4 Starting node
ACORN: s₀(q) recorded per query by the instrumentation (independent of the
budget; cross-checked identical across budgets). POST: hnswlib cannot be
instrumented (black box); handled by §3.3 item 5.

## 4. Experimental factors (full study; NOT yet run)

Selectivity × fragmentation C (clustered; plus random) × realization (≥ 5
independent cluster-order / permutation seeds per cell) × method {PRE, POST,
ACORN-1, ACORN-γ} × budget (Phase 4 grid {10 … 2560}). Fixed: SIFT1M base,
the 10,000 Phase 4 queries, k = 10, exact filtered ground truth per
condition. Outcomes: U, N(b), L(b), recall-vs-budget, effort (D = exact
distance computations, ACORN F = `n_scanned`, as in Phase 4 §16), R_sem vs
R_graph (what the 2-hop rule repairs), component statistics. Uncertainty:
query-level bootstrap within realizations and realization-level resampling
across them. Conditional extensions (only if the core supports them): γ
dose-response (γ ∈ {1, 12, 100}), robustness to k-means seed / predicate
distribution. No learned predictor; latency is not a primary metric; no new
datasets yet.

## 5. Validation experiment (MVP — the only run authorised now)

s = 0.01; C ∈ {100, 1,000, 10,000} + random; 2 realizations each (8
conditions; realization 0 at C = 1,000 reproduces the Phase 4 attribute);
10,000 queries; PRE, POST, ACORN-γ, ACORN-1; Phase 4 budget grid.
Validation criteria (no hypothesis gate):
- V-S1 exact identity L = U + N for every row;
- V-S2 soundness: every returned passing id of ACORN is in R_sem(s₀) (0
  violations);
- V-S3 PRE: U = N = 0; POST: U reported from the unfiltered-graph SCCs;
- V-S4 s₀ identical across budgets per (condition, query); instrumented
  build reproduces stored Phase 4 rows byte-identically (except latency);
- V-S5 generator: exact T, nested levels, realizations distinct, fixed
  partition per C, realization 0 at C = 1,000 = Phase 4 attribute hash;
  hashes recorded and verified on load;
- V-S6 unit tests for the edge rules (hand graphs) and SCC/reach code;
  integration soundness test on a toy ACORN index (γ = 1 and γ > 1).

## 6. Implementation map

Reused unchanged: filter generator core, verified GT store, PRE/POST, ACORN
wrapper search, n3/`n_scanned` accounting, run metadata, Phase 4 analysis
patterns. New: `fse_fragment_conditions` (conditions + GT + manifest),
`cpp/reachability/` (edge rules, SCC/reach), `fse_reach`,
`python/analysis/decomposition.py`, configs under `configs/structural/`.
Modified (additive): ACORN wrapper (graph export, s₀ readout), HNSW wrapper
(level-0 export), `fse_matrix` (manifest conditions, `seed0` column, index
path override), ACORN source (one tagged s₀ record).

## 7. What was actually executed (added 2026-10-07)
The full study of §4 was **not** run. By user decision the project was
finished from the completed pilot evidence at s = 0.01 only:
the vertical slice (C1000 r0, random r0), the decision experiment (C = 100
× 5, C = 1000 × 5, random × 2 realizations; ACORN-1 and ACORN-γ) and the
Phase 4 PRE/POST baselines. C = 10,000 conditions were generated but not
analysed; the MVP grid and the final 60-condition / 20-condition runs were
stopped and their partial outputs are not used. Results, statistics,
conclusions and limitations: `docs/structural_results.md`.
