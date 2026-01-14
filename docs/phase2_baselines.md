# Phase 2 — Pre-filter and post-filter baselines

Status: **COMPLETE — checkpoint satisfied** (2026-10-05).

Design doc §21, Phase 2: *"Pre-filter and post-filter baselines — implement
both directly — correctness check against brute force on a small case —
checkpoint: both match exact brute-force results on a toy case."*

Previous checkpoint: Phase 1 complete for the ACORN-γ baseline
(`docs/phase1_acorn_repro.md` §10; Table 3 581.2 vs 611.0, −4.9 %; all
correctness/reproducibility checks pass). The ACORN-1 ≈ −10 % gap is an
accepted Phase 1 limitation; nothing from Phase 1 was changed or re-run.

## 1. Specification → implementation

| Design doc | Requirement | Implementation |
|---|---|---|
| §2.2 | Pre-filtering: identify every vector satisfying the filter, then search restricted to that subset | `PrefilterSearch` |
| §18 | `prefilter/` — *exact* brute-force-within-filter baseline | exact squared-L2 over passing vectors only |
| §2.2 | Post-filtering: ordinary ANN search on the full index, more candidates than needed, discard failures, *if too few survive, fetch more* | `PostfilterSearch`: over-fetch rounds |
| §18 | `postfilter/` — standard HNSW search + over-fetch logic | hnswlib (standard HNSW) via `HnswIndex` |
| §6.1 | Implemented directly, in-house; no codebase beyond ACORN/hnswlib | in-house logic on hnswlib |
| §13 | Identical query/filter conditions across methods | all methods take the same per-query filter mask as ACORN |
| §12 | Distance computations = primary effort measure | exact per-query counts, separate from filter checks |
| CLAUDE.md | Reuse Project 1 code where it fits | `HnswIndex` + `CountingL2Space` ported from Project 1 |

No Phase 3 filter generator, no predictor, no experimental matrix, no
change to ACORN or the Phase 1 configuration.

## 2. What was implemented

New project code (ACORN and hnswlib untouched; Phase 1 driver untouched):

| File | Purpose |
|---|---|
| `cpp/methods/filtered_result.h` | `FilteredSearchResult`: ids, distances, `distance_computations`, `filter_checks`, `rounds`, `last_fetch` — shared by both baselines |
| `cpp/methods/prefilter/prefilter.{h,cpp}` | pre-filter baseline |
| `cpp/methods/postfilter/hnsw_index.{h,cpp}` | hnswlib wrapper with exact counting, ported from Project 1 (`ars::HnswIndex`), trimmed; adds `SetEf`/`Ef` and a public const `SearchAtCurrentEf` |
| `cpp/methods/postfilter/postfilter.{h,cpp}` | post-filter baseline + `PostfilterFetchSize` (the fetch schedule) |
| `cpp/cli/fse_baselines_check.cpp` | Phase 2 validation driver (YAML-driven) |
| `configs/phase2/toy_validation.yaml` | validation configuration |
| `cpp/common/run_metadata.{h,cpp}` | added shared `BuildInfoJson()` / `HardwareJson()` (now also logs load average at start) for new drivers; existing functions unchanged |
| `cpp/tests/test_{prefilter,hnsw_index,postfilter}.cpp` | 22 new unit tests |

Integration choices:
- **One filter representation for all three methods.** Every method takes
  the query's filter as a mask row (N bytes, nonzero = passes) — exactly
  ACORN's `filter_id_map` row — so filter conditions are identical across
  methods by construction (§13).
- **The ground-truth reference is reused, not duplicated.** Validation
  compares against the existing `FilteredGroundTruthEquals` (verified in
  Phase 1 against FAISS and the official SIFT ground truth). The pre-filter
  shares only the `SquaredL2` kernel with it; its selection path (mask scan,
  per-query heap) is separate code, and a naive sort-everything reference is
  used in the unit tests as a second, independent check.
- **The Phase 1 ACORN wrapper (`AcornQueryResult`) is left as is** (Phase 1
  frozen). Unifying it with `FilteredSearchResult` is a Phase 4 integration
  task.

## 3. Pre-filter (exact)

`PrefilterSearch(base, query, k, filter_row)`:
1. Scan the mask over all N base vectors (N filter checks).
2. For each passing vector, compute exact squared L2 (one distance
   computation each) and keep the best k in a max-heap on (distance, id).
3. Return the k results in ascending (distance, id) order — exact ties go
   to the smaller id, the same convention as the ground truth; fewer than k
   passing vectors pad with id −1 / distance +∞.

Effort: `distance_computations` = number of passing vectors (=
selectivity × N), `filter_checks` = N. Exact by construction, so it has
**no search-budget knob** (design §18 specifies the exact variant).
Thread-safe.

## 4. Post-filter (over-fetch)

**Index.** hnswlib `HierarchicalNSW` through `HnswIndex`. hnswlib rather
than the HNSW inside ACORN's FAISS fork, because the fork's `HNSW` is
modified (`gamma` parameter, embedded metadata, `hybrid_search` path) and
is therefore not the "standard HNSW" §18 calls for; hnswlib is standard,
named in the design (§6.1, §19), black-box, and already validated in
Project 1, and the counting wrapper gives exact, thread-safe per-query
counts.

**Parameters** (`PostfilterParams`, all explicit, fixed per run):

| Parameter | Meaning | Validation value |
|---|---|---|
| `ef_search` | HNSW search budget (same knob as ACORN's efSearch) | grid {10, 40, 160, 640} |
| `overfetch_factor` α | round-1 fetch = ⌈α·k / s⌉ | 1.0 (the ACORN paper's K/s post-filter) |
| `growth_factor` g | each further round multiplies the fetch by g | 2.0 |
| `max_rounds` R | cap on rounds (1 = no refetch) | 10 |
| HNSW build | M, efConstruction, seed, threads | 32, 40, 42, 1 thread (provisional; full-scale values are a Phase 4 config) |

**Fetch schedule** (`PostfilterFetchSize`): round r fetches
`min(N, max(k, ⌈⌈α·k/s⌉ · g^(r−1)⌉))` candidates, where s is the filter's
**global selectivity** (design §8 measure 1: known by construction and
legitimately live/router-usable — not the local density, which never enters
any method).

**Algorithm** (`PostfilterSearch`), per query:
1. Round r: fresh top-`fetch_r` HNSW search at the index's ef (hnswlib
   searches with **max(ef, fetch)**); add its exact distance count.
2. Walk candidates in ascending distance, checking each against the filter
   (one filter check each), keeping the first k that pass.
3. Stop if k survived, or fetch reached N, or r = R; otherwise refetch.
   The last round's survivors are returned (possibly fewer than k, padded).
4. s = 0 returns no results without searching.

No per-query adaptation, prediction or routing: the only per-query input
besides the filter is the filter's global selectivity. The caller sets the
index's ef once (`SetEf`); `PostfilterSearch` checks it equals
`params.ef_search` and is then thread-safe across queries.

## 5. Distance-computation accounting

| Method | Definition | How it is guaranteed |
|---|---|---|
| Pre-filter | one per passing vector | counted in the loop; validated = passing count for every query |
| Post-filter | every distance evaluation hnswlib performs, all layers, all rounds | `CountingL2Space` wraps hnswlib's own L2 function with a thread-local counter (exact, per query, thread-safe); validated against an independent replay of the rounds and serial = parallel |
| ACORN (Phase 1) | ACORN's `acorn_stats.n3` (upper-level greedy + level 0) | ACORN's own counter |

Filter checks are counted separately and never added to distance
computations.

**Cross-method note for Phase 4 (not resolved here, by scope):** ACORN's n3
does not count the entry-point distance (`d_nearest = qdis(nearest)`,
`impl/ACORN.cpp` line 1531), whereas the hnswlib count is exhaustive — at
least a 1-per-query convention difference. Whether ACORN leaves any other
evaluation uncounted is unverified. Design §13 requires identical
effort-budget definitions across methods, so this must be measured (e.g.
with a counting storage passed to ACORN's `IndexACORN(Index* storage, …)`
constructor, which needs no change to ACORN) before the Phase 4 matrix.

## 6. Correctness validation (the Phase 2 checkpoint)

Command: `./build/cpp/fse_baselines_check configs/phase2/toy_validation.yaml`
Output: `results/phase2/phase2_toy_validation/{summary.json, per_query.csv,
config.yaml}` (3,600 per-query rows). Source fingerprint
`e27b408a28f1be42…`; 1-min load average at start 0.56 (idle VM).

Small case: SIFT1M prefix, 20,000 base vectors × 200 queries, k = 10.
Three **correctness conditions** (uniform integer attribute in [1, v],
`equals(y)`; not the Phase 3/4 grid — no correlation control, toy scale):
v = 1 (s = 1), v = 12 (s ≈ 0.0834), v = 100 (s ≈ 0.0100). One HNSW index
serves all conditions (construction is filter-blind).

Checks per query (all against the brute-force filtered ground truth):
recall@10; every returned id passes the filter; returned distance equals
recomputed squared L2; ascending order; effort accounting (pre-filter:
= passing count and N checks; post-filter: = independent replay of its
rounds); serial = parallel (8 threads) for post-filter results and effort;
pre-filter ids identical to ground truth (even tie-only differences count
as errors); post-filter at an **exhaustive setting** (ef = N, whole-index
fetch) ids identical to ground truth and recall exactly 1.

| Condition | Method / setting | mean recall | min recall | mean dist. comps | mean rounds (max) |
|---|---|---|---|---|---|
| s = 1 | pre-filter | **1.0000** | 1.00 | 20,000.0 | — |
| | post ef10 | 0.8535 | 0.40 | 256.7 | 1.00 (1) |
| | post ef40 | 0.9820 | 0.70 | 520.5 | 1.00 (1) |
| | post ef160 | 0.9980 | 0.80 | 1,308.8 | 1.00 (1) |
| | post ef640 | 0.9990 | 0.80 | 3,197.6 | 1.00 (1) |
| | post exhaustive | **1.0000** | 1.00 | 20,082.2 | 1.00 (1) |
| s ≈ 0.083 | pre-filter | **1.0000** | 1.00 | 1,667.4 | — |
| | post ef10 | 0.9890 | 0.80 | 1,926.9 | 1.49 (3) |
| | post ef40 | 0.9890 | 0.80 | 1,926.9 | 1.49 (3) |
| | post ef160 | 0.9915 | 0.80 | 2,153.0 | 1.49 (3) |
| | post ef640 | 0.9985 | 0.90 | 4,704.5 | 1.47 (3) |
| | post exhaustive | **1.0000** | 1.00 | 20,082.2 | 1.00 (1) |
| s ≈ 0.010 | pre-filter | **1.0000** | 1.00 | 199.2 | — |
| | post ef10/40/160/640 | 0.9985 | 0.90 | 6,723.3 | 1.43 (3) |
| | post exhaustive | **1.0000** | 1.00 | 20,082.2 | 1.00 (1) |

**Every check count is zero in every row**: 0 filter violations, 0
distance mismatches, 0 ordering errors, 0 accounting errors, 0
ground-truth id mismatches (pre-filter and exhaustive post-filter; also 0
tie-only differences), 0 serial/parallel mismatches. 0 padded slots.
`all_checks_pass: true`.

## 7. Behaviours observed (documented, not changed)

These are properties of the methods as specified, relevant to Phase 4;
no parameter was tuned in response.

1. **ef below k/s has no effect on post-filter.** hnswlib searches with
   max(ef, fetch) and the round-1 fetch is k/s (120 at s ≈ 0.083, 1,000 at
   s ≈ 0.01), so ef10 = ef40 at s ≈ 0.083 and the whole ef grid ≤ 640 is
   identical at s ≈ 0.01. For post-filter the effective budget at low
   selectivity is set by the over-fetch, not by efSearch — this bears on
   §13's "identical effort-budget definitions" and should be addressed
   explicitly when the Phase 4 budget grid is fixed.
2. **About half of filtered queries need a second round** (mean 1.43–1.49)
   because α = 1 makes the round-1 fetch expect exactly k survivors.
   Inherent to the paper's K/s rule; α is an explicit parameter.
3. **The exhaustive setting costs 20,082 > N = 20,000** distance
   computations: every node plus the upper-layer descent — consistent with
   exhaustive counting.
4. **Pre-filter has no budget knob** (exact): its fixed-effort policy is a
   single point per filter condition.

## 8. Unit tests (22 new; GoogleTest)

- `Prefilter.*` (5): exact match with the filtered ground-truth reference
  (ids and distances, recall 1); match with a naive sort-everything
  reference on integer data with many exact ties; only passing ids,
  ascending, exact distances; effort = passing count and N filter checks
  for 7 filters; padding, empty filter, k = 0 rejected.
- `CountingL2Space`/`HnswIndex.*` (8, ported from Project 1 + 1 new):
  counting matches reference distance; sorted results; recall → 1 at large
  ef and monotone; per-query counts reset and grow with ef; **new:**
  concurrent searches at a fixed ef equal serial (labels and counts);
  seeded single-thread build reproducible; save/load identical; label
  offsets.
- `PostfilterFetchSize`/`Postfilter.*` (9): fetch schedule incl. ceil, ≥ k
  and ≤ N caps, invalid inputs; **exhaustive budget equals brute-force
  filtered ground truth exactly** (60 queries); filter / order / distance
  checks for s ∈ {1, ¼, 1/50} × ef ∈ {10, 40, 160}; recall vs ground truth
  rises with budget, ≥ 0.99 at ef 800; distance count = sum of replayed
  rounds (refetch path exercised); refetch recovers survivors a single
  round misses, and a single round keeps its partial survivors (regression
  test for a bug caught in review before the first build: the last round's
  partial results were being discarded); unfiltered query = plain HNSW
  search (results and effort); empty filter / invalid inputs / ef mismatch
  rejected; concurrent queries = serial incl. effort.

## 9. Tests and checks run

| | Result |
|---|---|
| Build (`build/`, GCC 13, -O3 -march=native) | 0 warnings |
| C++ unit tests (`ctest`) | **50/50 pass** (28 existing + 22 new) |
| Python tests (`pytest python/tests`) | 5/5 pass (no Python changes this phase) |
| clang-tidy, production code (`build-tidy/`: library + all drivers) | **0 findings** (initial findings in the new driver fixed: function split, reserve, explicit bool conversions) |
| clang-tidy, new test files (manual run) | only `readability-function-cognitive-complexity` on GoogleTest bodies (macro-inflated; tests are not under the CMake clang-tidy hook, as in Phases 0–1); one `modernize-use-auto` fixed |
| clang-format | clean |
| Toy validation (`fse_baselines_check`) | **all checks pass** (§6) |

## 10. Deviations from the design

None. Where the design is silent, the following implementation decisions
were made and are recorded here rather than changing anything in the
design: exact (not approximate) pre-filter, per §18; hnswlib as the
post-filter's standard HNSW (§4); over-fetch rule ⌈α·k/s⌉ with geometric
refetch (α = 1, g = 2, R = 10 for validation); provisional HNSW build
parameters M = 32, efConstruction = 40 for the toy case. The full-scale
post-filter configuration (HNSW parameters, α, g, R, and how its budget
relates to ACORN's efSearch — see §7.1) is a Phase 4 configuration decision
and has not been fixed here.

## 11. Phase 2 checkpoint status

**Satisfied.** Both baselines match exact brute-force results on the toy
case: the pre-filter returns exactly the brute-force filtered top-k (ids,
distances, recall 1) for every query in every condition; the post-filter
reproduces it exactly at an exhaustive budget, and at every finite budget
returns only filter-passing results with correct distances and order, with
recall correctly measured against the filtered ground truth. Distance
accounting is exact and independently verified for both. No known
correctness issue is open.

Open items carried forward (not Phase 2 correctness issues): cross-method
counting convention vs ACORN (§5); post-filter budget semantics at low
selectivity (§7.1).

## 12. Follow-up A — distance-counting mismatch: resolved (2026-10-05)

**Where each counter increments.**

| Method | Counter | Evaluations it sees | Evaluations it misses |
|---|---|---|---|
| ACORN (γ and 1) | `acorn_stats.n3` (per-query delta) | upper-level greedy (`hybrid_greedy_update_nearest`: every `qdis(v)` / `qdis(v2)` is followed by `ndis += 1`), level-0 search (`hybrid_search_from_candidates`: every `qdis(v1)` / `qdis(v2)` is preceded by `ndis++`; the seed candidate's distance is reused, not recomputed) | the entry-point distance `d_nearest = qdis(nearest)` in `ACORN::hybrid_search` (`impl/ACORN.cpp:1531`) — always exactly one per query |
| Post-filter (hnswlib) | `CountingL2Space` thread-local count | every call of hnswlib's distance function, all layers, all rounds — including hnswlib's own second evaluation of the entry node when it seeds the base-layer search | none |
| Pre-filter | loop counter | one per passing vector | none |

ACORN-1's upper-level 2-hop expansion can evaluate the same node more than
once; each such evaluation is counted, so it is not a source of mismatch.

**Controlled experiment** (`cpp/tests/test_distance_accounting.cpp`, part
of the test suite). ACORN is given a counting vector store through its
public `IndexACORN(Index* storage, ...)` constructor — `IndexACORNFlat` is
exactly `IndexACORN(new IndexFlat(d), ...)`, so nothing in ACORN is
modified. For ACORN-γ (γ = 4) and ACORN-1, 3,000 Gaussian vectors, filters
of selectivity 1, ¼ and 1/25, efSearch 10, 40, 160, and 60 queries each
(1,080 query searches), single-threaded:
- the counting store does not change behaviour: returned ids and native n3
  equal those of the identically built standard `IndexACORNFlat`;
- **exact evaluations − native n3 = 1 for every one of the 1,080 queries**
  (min = max = 1 in all 18 cells).

hnswlib check: on a one-element index, the counter reports 2 for one search
(the entry node is evaluated in the descent setup and again when seeding
the base layer) — confirming that `CountingL2Space` counts real evaluations,
including the entry point.

**Conclusion.** The difference is a **fixed per-query offset of exactly
one evaluation**, not anything filter-, γ- or efSearch-dependent.

**Decision.**
- The project's cross-method effort measure is **distance evaluations
  actually performed by the implementation** ("exact distance
  computations").
- Native counters are preserved and always recorded. Pre-filter and
  post-filter natives are already exact. For ACORN, the exact count is
  `AcornExactDistanceComputations(n3) = n3 + 1`
  (`cpp/methods/ingraph/acorn_accounting.h`, `kAcornUncountedPerQuery = 1`),
  the single place this normalization lives; the audit test pins it.
- Phase 4 output records both `native_distance_computations` and
  `distance_computations` (exact) per query; every cross-method comparison
  (effort curves, oracle effort, regret) uses the exact count.
- Phase 1 numbers stay in ACORN's native convention — that is the
  convention of the paper being reproduced (Table 3), so they are correct
  as reported. The effect of the offset is +1 per query (≈ 0.2 % at
  ACORN-γ's ~580 evaluations at Recall 0.8).
- Implementation-specific evaluation patterns (e.g. hnswlib re-evaluating
  its base-layer entry node) are real costs of that implementation and are
  not normalized away.

## 13. Follow-up B — common effort-budget definition: resolved (2026-10-05)

**What the design says.** The design uses "effort" in two distinct roles:
- **Control** — what a policy sets: "search-budget values" applied by the
  fixed-effort baselines, identical across every filter condition (§5.3,
  §7.5) and across all three methods (§13); the predictor's output, "a
  discrete search-effort tier" (§9), i.e. "selected search budget" (§5.1).
  §2.4 names the effort parameters: "how large a candidate list to explore,
  how many rounds of over-fetching to allow, how far to search before
  giving up".
- **Measurement** — what is compared: "search effort (distance
  computations, latency as secondary)" (§7.11); "Distance computations —
  hardware-independent effort/cost measure" (§12); "distance computations
  treated as the primary effort measure" (§13). Oracle effort is "the
  minimum effort reaching a target recall" (§7.6) and regret is the "gap
  between predicted and oracle effort at matched recall" (§12).

**Ambiguity.** §13 requires "identical effort-budget definitions" across
pre-filter, post-filter and in-graph search, but the design never states
the budget control for the pre-filter (exact, no knob) or how the
post-filter's over-fetch relates to the candidate-list budget. Phase 2
showed this matters: with the ACORN-style fetch k/s, a nominal
candidate-list budget below k/s is inert for the post-filter.

**Options considered.**

| Option | Verdict |
|---|---|
| (a) HNSW candidate-list size (efSearch) as the control **and** as the comparison unit | Rejected as the comparison unit: one efSearch unit costs different numbers of evaluations in different methods (ACORN's filtered neighbour expansion; the post-filter's k/s fetch floor), so equal efSearch is not equal effort. Kept as the control (below). |
| (b) Total distance computations as the control (a per-query cap) | Rejected: neither ACORN nor hnswlib can be stopped at an evaluation cap without changing their search loops (forbidden for ACORN; a new search policy for both), and the exact pre-filter cannot be capped at all. |
| (c) Candidate/fetch budget (scale the post-filter fetch with the budget) | Rejected: a post-filter-only notion with no ACORN or pre-filter counterpart, and it would replace the ACORN-style fetch rule with a new, untested policy. |
| (d) **Common control = candidate-list budget; common measure = exact distance computations** | **Chosen** — the smallest interpretation that uses the design's own two definitions literally. |

**Decision (binding for Phase 4).**
1. **Common budget control.** One grid B of candidate-list budgets
   b (efSearch values), fixed before Phase 4 from training-split behaviour
   only (§13 hyperparameter-leakage rule), applied identically to ACORN and
   the post-filter in every filter condition. Each method interprets b by
   its own unchanged semantics:
   - ACORN: list size max(b, k) — native.
   - Post-filter: round-r list size max(b, fetch_r), fetch_r =
     min(N, max(k, ⌈⌈α·k/s⌉·g^(r−1)⌉)) — the ACORN-style fetch rule is kept;
     α, g, R are fixed method constants (not budget-dependent).
   - Pre-filter: budget-independent (exact); one effort value per filter
     condition, reported at every b.
2. **Common effort measure.** All cross-method comparisons — H1/H3 effort
   curves, oracle effort, regret — use **exact distance computations per
   query** (§12 of this file: native counts, ACORN + 1).
3. **Oracle.** For each query, method and condition: the minimum exact
   distance computations over budgets b ∈ B at which the query reaches the
   target recall (equivalently the cost at the smallest such b). Predictor
   tiers are budget levels b; regret is measured in exact distance
   computations.
4. **Inert budgets are recorded, not hidden.** Per query, Phase 4 records
   the nominal b, the effective list size (max(b, last fetch) for the
   post-filter, max(b, k) for ACORN), rounds, and exact distance
   computations. Budgets that are inert for a method map to the same
   realized point on its effort curve, so they cannot make a method look
   cheaper or dearer than it is: comparisons are in realized effort, not
   nominal b.

This changes no hypothesis, method or measure: the knob is the design's
"how large a candidate list to explore" (§2.4), the measure is the
design's distance computations (§12), and the post-filter's k/s floor at
low selectivity — its larger minimum effort — is a real property of the
method that H3 is meant to detect, not an artefact to be tuned away.

**Supporting evidence (existing Phase 2 data, §6).** At s ≈ 0.01 all of
b ∈ {10, 40, 160, 640} give the identical post-filter result (6,723.3
evaluations, recall 0.9985), i.e. a single realized point, while the
pre-filter's realized effort is 199.2 evaluations at recall 1 — the
comparison is meaningful only in realized evaluations, as decided.
