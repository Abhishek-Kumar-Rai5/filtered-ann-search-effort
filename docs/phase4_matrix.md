# Phase 4 — Core effort-selectivity matrix (fixed-effort policies)

Status: sections 1–5 are the **frozen design**, written 2026-10-05 before
any Phase 4 code was run at full scale and before any Phase 4 outcome was
observed. Later sections record execution and results. Changes to
sections 1–5 after this point require stopping and asking the user
(CLAUDE.md; user instruction for Phase 4). The Phase 3 post-hoc
correction of its check 2 (docs/phase3_filter_generator.md §9) is a
disclosed exception for Phase 3 only and is **not** a precedent here.

Design doc §21, Phase 4: *"Core effort-selectivity matrix, fixed budgets
only — full selectivity × correlation × method grid, fixed-effort policies
only — effort-vs-selectivity curves per method — checkpoint: curves behave
sensibly (effort rises as selectivity drops) before adding the
predictor."*

## 1. Frozen experimental configuration

### 1.1 Data and queries
- Base: SIFT1M, all **1,000,000** vectors (`data/sift/sift_base.fvecs`,
  SHA-256 verified by `scripts/download_sift1m.sh`).
- Queries: all **10,000** SIFT1M queries, file order, identical in every
  condition (query-set content hash recorded everywhere).
- k = **10**. Ground truth depth **11** (the 11th neighbour only detects
  exact ties at rank 10; recall uses the first 10).

### 1.2 Filter conditions (Phase 3 generator, unchanged)
- Selectivity levels: the six Phase 3 levels, s_i = 10^(−2 + 0.4 i):
  0.0100, 0.0251, 0.0631, 0.1585, 0.3981, 1.0000 → T = round(s·N) =
  10,000; 25,119; 63,096; 158,489; 398,107; 1,000,000.
- Correlation: random, clustered — 12 conditions, one filter per
  condition shared by all queries (Phase 3 §2).
- Generator seeds (Phase 3 values): random 20261010; k-means 20261011;
  cluster order 20261012; k-means 25 iterations, max 256 points/centroid.
- Cluster count C: chosen by the calibration in §2 (only structural
  generator statistics; no search outcome is computed before C is frozen).

### 1.3 Methods
| Method | Configuration |
|---|---|
| Pre-filter | Phase 2 `PrefilterSearch`: exact brute force over the passing set. Budget-independent: one evaluation per query and condition. Distance computations = T(s) exactly; filter checks = N. |
| Post-filter | Phase 2 `PostfilterSearch` on hnswlib (`HnswIndex`): **M = 32, efConstruction = 40** (the ACORN paper's HNSW post-filter setting, as validated in Phase 2), seed 42, **single-threaded build** (bit-reproducible). Over-fetch α = 1 (round-1 fetch ⌈k/s⌉), growth g = 2, max rounds R = 10; s = the condition's exact global selectivity T/N. |
| In-graph: ACORN-γ | **γ = 100, M = 32, M_β = 64, efConstruction = M·γ = 3,200** (ACORN default; Phase 1 resolution). γ follows the ACORN paper's own rule γ = 1/s_min (§5.2 "One simple choice for γ is 1/s_min"; its degree guarantee holds only for s > 1/γ), with s_min = 0.01 the lowest Phase 4 level. Phase 1's γ = 12 would leave 0.01, 0.025 and 0.063 outside ACORN's specified range (the paper's system would pre-filter there). Same code and build flags as Phase 1 (FAISS_OPT_LEVEL generic); built once with 16 threads (ACORN's parallel build is not bit-reproducible; Phase 1 measured build-to-build effects < 1 %); index file hash recorded. Measured cost probe (100K prefix): build 106 s vs 29 s for γ = 12; 0.14 GB. |

### 1.4 Budget grid and recall target (fixed a priori)
- **Budget grid B = {10, 20, 40, 80, 160, 320, 640, 1280, 2560}** =
  10·2^i, i = 0…8 — the common candidate-list budget b
  (docs/phase2_baselines.md §13), identical for ACORN and post-filter in
  every condition. ACORN interprets b as max(b, k); post-filter as
  max(b, fetch_r) per round (inert budgets recorded, not removed).
- **Target recall: per-query id-based Recall@10 ≥ 0.9** (at least 9 of the
  10 filtered ground-truth neighbours). Rationale: the ACORN paper's
  headline operating point and Phase 1's metric. The companion project's
  0.95 used a tie-aware recall; with this project's id-based Recall@10 a
  per-query 0.95 means 10/10, which exact distance ties can make
  unattainable.
- Both are fixed **a priori, from no data** — stricter than the design's
  rule that they be fixed from training-split behaviour only (§13).

### 1.5 Effort accounting (docs/phase2_baselines.md §12–13)
- Primary effort = **exact distance computations per query**: pre-filter
  native count (= T); post-filter `CountingL2Space` count (all rounds);
  ACORN `AcornExactDistanceComputations(n3) = n3 + 1`. Native counts are
  recorded alongside.
- Filter checks recorded separately, never added to distance computations.

### 1.6 Oracle (reference) effort
For each (query, method, condition): the smallest b ∈ B whose per-query
Recall@10 ≥ 0.9; oracle effort = that run's exact distance computations.
If no b ∈ B reaches the target the query is **censored** ("not reached
within B") and counted in the fraction of failed queries. Pre-filter:
oracle effort = its fixed exact effort (recall is always 1). Computed for
all 10K queries; Phase 5 will use only its training split's labels.

### 1.7 Latency (secondary measure)
- Every sweep records each query's single-threaded wall time. Sweeps run
  several single-threaded workers concurrently, so this is latency under
  uniform concurrent load (documented, as in Phase 1 D1).
- **Timing pass** (design §13: single-threaded, warm-up discarded,
  repeated, median): a seeded subsample of **500 queries** (seed
  20261200), one process, one thread, all 12 conditions × all methods ×
  all budgets, 1 warm-up + 3 timed repetitions, median per query; run only
  on an idle VM (1-min load < 1.0 at start, load logged every 30 s). If the
  VM is not idle it is deferred and reported, not run contended.

### 1.8 Seeds (all logged)
Generator 20261010 / 20261011 / 20261012; homophily sample 20261013;
independence null 20261100 + r; post-filter HNSW 42; ACORN internal RNGs
(fixed in ACORN: 12345, 789); timing subsample 20261200; determinism
re-run subsample 20261300.

### 1.9 Repetition policy
Recall and distance counts are deterministic given an index, so each
(query, condition, method, budget) is evaluated **once**. Reproducibility
is verified by re-running a pre-specified subset in a fresh process
(§3 V9). Indexes are built once and their file hashes recorded.

### 1.10 Output schema and run metadata
Per (method, condition) — `results/phase4/matrix/<method>/<condition>/`:
- `per_query.csv`: query_id, condition, condition_id, correlation,
  s_requested, s_achieved, method, budget, effective_list, recall,
  dist_exact, dist_native, filter_checks, rounds, last_fetch, latency_us,
  n_valid, filter_violations, distance_mismatches, gt_tie_at_k (+ n_scanned
  from §15 on: ACORN neighbour entries scanned; 0 = n/a for pre/post).
- `raw_b<budget>.bin`: returned ids and distances (all queries) — recall
  can be recomputed from these and the stored ground truth.
- `sweep.json`: git state, source fingerprint, build flags, hardware,
  load average, seeds, config, index file hashes, ground-truth identity
  (condition id, mask hash, query hash), wall time.
Ground truth: verified per-condition store (Phase 3 format; identity
checked on every load). Attributes: stored with content hashes, verified
on load.

## 2. Cluster granularity (frozen procedure, run before anything else)

The toy used C = 100 at N = 20,000 (≈ 200 vectors per cell), a choice
made for N = 20,000. At N = 10⁶ the cell size is a free design parameter.
Its meaning: clustered filters should create substantially more local
concentration than random filters at the same global selectivity (design
§5.2), and the clustered condition should reflect the clustering
mechanism rather than the idiosyncrasy of a single region.

**Candidates:** C ∈ {1,000; 2,000; 5,000} (≈ 1,000 / 500 / 200 vectors
per cell; 200 = the cell size validated on the toy).
**A-priori constraint:** at s_min = 0.01 the passing set must span at
least 10 cells (C ≥ 1,000), so that the 1 % clustered condition is a
union of several independent regions, not one region's peculiarities.
This excludes C = 100 (one cell at 1 %) before any statistic is computed.

**Calibration statistics** (structural only, full N, all 10K queries,
generator seeds as frozen), for each candidate and each level s < 1:
κ_h (chance-corrected homophily, §4.2 estimator), D(K = 100) of query
local density, f₀(K = 10), f₀(K = 100), cells spanned, partial cells.

**Selection rule (fixed now):**
1. Eligible: at every level s < 1, κ_h ≥ 0.20 and D(K = 100) ≥ 3 (the
   Phase 3 corrected clustered thresholds).
2. Among eligible candidates choose the one with the largest minimum κ_h
   over levels (strongest concentration); if two are within 0.01, prefer
   the larger C (more independent regions).
3. If no candidate is eligible: stop and report; do not widen the set.

No search is run, and no Phase 4 outcome is computed, before C is frozen.

## 3. Frozen validation criteria

### 3.1 Full-scale generator validation (gate before the pilot)
Phase 3's checks at full scale (N = 10⁶, 10K queries, 12 conditions), with
the Phase 3 **corrected** check 2 unchanged in every threshold:
- G1 selectivity: mask population = T exactly; |achieved − requested| ≤
  1/(2N); nested at every level.
- G2 same query set: one query hash in every condition record.
- G3 ground truth per condition (depth 11), stored with identity;
  stored = freshly computed (read-back).
- G4 independent verification: FAISS `IndexFlatL2` over each passing
  subset for **all 10K queries**: 0 mismatches (tie-only differences
  allowed and reported).
- G5 integrity: 0 ground-truth ids failing their own filter;
  wrong-identity loads rejected for every ordered pair (132/132).
- G6 correlation (Phase 3 §9.4 criteria, unchanged): random D(K) ∈
  [0.5, 2], random mean within 4σ, random |h − s| ≤ 0.1; clustered
  D(K = 100) ≥ 3 and ≥ 2× random; zero-inflation f₀(K = 10, 100) and κ_h
  vs a Monte Carlo independence null (R = 5,000, seeds 20261100 + r,
  Bonferroni α′ = 0.01/15), κ_h ≥ 0.20; random inside the null's
  two-sided interval; ≤ 1 partial cell per clustered level.
  **Estimator note (pre-specified, forced by scale):** exact homophily
  needs an N × N neighbour table (10¹² distances); at N = 10⁶ homophily is
  computed over a seeded sample of **20,000** base vectors (seed 20261013)
  with their exact 10 nearest neighbours in the full base, for the
  observed conditions and every null draw alike (≈ 200 contributing
  vectors at s = 0.01, as on the toy).
- G7 the **1 % condition with all 10K queries** is reported separately
  (f₀, κ_h, D, number of queries meeting the passing region).
- G8 determinism: two separate full runs give an identical digest.
Any G-failure: stop and report before the pilot; the generator is not
modified to pass.

### 3.2 Phase 4 matrix validation (V1–V11)
- V1 completeness: exactly 12 × 10,000 × (1 + 9 + 9) = **2,280,000**
  per-query rows; every (condition, method, budget, query) once; no
  missing or non-finite values.
- V2 same query set in every sweep (query hash).
- V3 ground truth: every sweep loads its condition's ground truth through
  the verified store (identity regenerated from the frozen generator); any
  mismatch aborts.
- V4 filter correctness: **0** filter violations and **0** distance
  mismatches (relative tolerance 1e−3) in all rows.
- V5 accounting: pre-filter dist_exact = T(s) in every row; ACORN
  dist_exact = dist_native + 1 in every row; post-filter dist_exact =
  dist_native; post-filter replay of its rounds equals its count for a
  seeded subsample of 200 queries per (condition, budget); counting-store
  audit of the +1 for the frozen ACORN configuration (γ = 100, M = 32,
  M_β = 64) on a 20K-vector prefix: offset = 1 for every query.
- V6 implementation sanity per (graph method, condition): mean recall and
  mean dist_exact are non-decreasing along B (tolerance 0.002 on recall;
  equality allowed); pre-filter recall = 1 for every query.
- V7 s = 1 identity: random and clustered have the same mask at s = 1, so
  every per-query result (except latency) must be identical across the two
  correlation labels.
- V8 inert budgets: for post-filter, per query, results and counts are
  identical for all budgets b ≤ ⌈k/s⌉.
- V9 determinism: a fresh-process re-run of a pre-specified subset
  (random s = 0.01 and clustered s = 0.0631; all methods and budgets;
  1,000 queries, seed 20261300) is identical except latency.
- V10 **design checkpoint ("effort rises as selectivity drops")**: for
  post-filter and ACORN separately, in each correlation condition, the
  median oracle effort over all 10K queries (censored = +∞) at s = 0.01
  is strictly greater than at s = 1. Pre-filter's effort is T(s) by
  construction (it falls as s drops); this is checked as an identity, not
  as V10.
  *[Post-hoc note: this original V10 is kept unchanged and was reported
  FAIL for ACORN (§12). A revised V10 was adopted afterwards; see §16.]*
- V11 no regressions: full C++ and Python test suites pass; clang-tidy
  reports 0 findings on production code; Phase 1–3 files unchanged.
Reported, not gated: fraction of failed queries per (method, condition,
budget); oracle distributions; latency.

**Decision rule.** Phase 4 is complete only if the full matrix has run
and V1–V11 all pass. Any failure: stop and report; criteria are not
changed after seeing results — an unforeseen implementation problem that
seems to require a change is brought to the user first.

## 4. Procedure and stopping points

1. Cluster calibration (§2) → freeze C (recorded in §6 before step 2).
2. Full-scale generator validation run ×2 (§3.1). Stop on failure.
3. Index builds (post-filter hnswlib, ACORN-γ).
4. Pilot (implementation/operational only): 1,000 queries; s ∈ {0.01,
   0.1585, 1.0} × both correlations; all three methods; b ∈ {10, 160,
   2560}; checks V3–V8 on the pilot rows; ACORN +1 audit; runtime and
   storage measurement. No research outcome is used to change anything.
5. Workload estimate from the pilot. If the projected full matrix exceeds
   12 h wall time or 50 GB storage, stop and ask before launching.
6. Full matrix, detached (2-hour background-task limit).
7. Oracle, validation V1–V11, timing pass (if the VM is idle).

## 5. Out of scope (unchanged)
No predictor, no model training, no adaptive routing, no Phase 7
hypothesis tests, no real filtered dataset, no ACORN or Phase 1–3 change,
no change of levels, query set or budget definition.
*[Scope note 2026-10-06: the user explicitly authorised the Phase 4 effort,
cross-method (H3) and preliminary structural (H2-on-effort) analysis on the
existing data — §17. Regret-based H2 still needs Phase 5.]*

## 6. Cluster granularity — calibration result (frozen 2026-10-05)

Run: `fse_filter_conditions configs/phase4/cluster_calibration.yaml`
(source fingerprint `9dd914e497ee809f…`, base hash `705d494ebd58ff4f`,
query hash `1581622a88a15dec`; `results/phase4/phase4_cluster_calibration/`).
Structural statistics only; no search was run.

| C | min κ_h (s < 1) | κ_h range | D(K=100) range | cells at s = 0.01 | partial cells / level | k-means time |
|---|---|---|---|---|---|---|
| **1,000** | **0.390** | 0.390–0.460 | 18.5–22.0 | 10 | 1 | 43 s |
| 2,000 | 0.303 | 0.303–0.360 | 8.6–16.9 | 22 | 1 | 154 s |
| 5,000 | 0.318 | 0.318–0.323 | 11.7–12.8 | 51 | 1 | 724 s |

All three are eligible (κ_h ≥ 0.20, D(K = 100) ≥ 3 at every level).
Largest minimum κ_h: C = 1,000 (0.390); no other candidate within 0.01.
**Selected and frozen: C = 1,000** (≈ 1,000 vectors per cell; the 1 %
condition is 10 cells).

## 7. Full-scale generator validation (G1–G8) — PASS (2026-10-05)

Config `configs/phase4/generator_full.yaml` (C = 1,000), two separate
runs (`scripts/run_phase4_generator.sh`): run 1 10:29–11:03 UTC, run 2
11:03–11:35 UTC; outputs `results/phase4/phase4_generator_full_run{1,2}/`;
ground truth (depth 11) and attributes in `data/cache/phase4/`. Base hash
`705d494ebd58ff4f`, query hash `1581622a88a15dec`; attribute hashes
random `d9ee351d3dd6101b`, clustered `1244cce0305ac7b6`; source
fingerprint `55955923a26c06a3…` (both runs).

| Gate | Result |
|---|---|
| G1 selectivity | all 12: mask count = T exactly (10,000 … 1,000,000); \|error\| ≤ 3.2e−7 (bound 5e−7); nested |
| G2 query set | one query hash in all 12 records |
| G3 ground truth | 12 separate computations (depth 11), stored with identity; read-back = computed |
| G4 independent check | FAISS `IndexFlatL2`, all 10K queries: **10,000/10,000 identical in all 12** (0 tie-only, 0 mismatched) |
| G5 integrity | 0 ground-truth ids failing their filter; wrong-identity loads rejected 132/132 |
| G6 correlation (Phase 3 §9.4 criteria) | **5/5 levels pass**; clustered κ_h 0.390–0.460 (z 134–207 vs null), all 15 clustered tests p = 2.0e−4 ≤ α′ = 6.7e−4; random κ_h ∈ [−0.002, 0.003] and inside its null at every level; clustered D(K=100) 18.5–22.0; one partial cell per level. (The superseded original Phase 3 criteria, logged for the record, fail 1/5 levels.) |
| G7 the 1 % condition, 10K queries | clustered f₀(K=10) 0.951 vs null 0.904 (99.93 %: 0.913, z = 15.3); f₀(K=100) 0.827 vs 0.366 (z = 78.6); κ_h 0.460 (z = 207); **490 queries meet a passing vector among their 10-NN (1,733 among their 100-NN)** — the toy's 200-query limitation (≈ 4 queries) is resolved |
| G8 determinism | run 1 = run 2: digest `3d3b6ac5076d5a4e`; conditions, local density, check-2, structure outputs and config byte-identical; summaries differ only in run id and start load |

## 8. Indexes and the accounting audit

| Item | Result |
|---|---|
| Post-filter hnswlib | M = 32, efConstruction = 40, seed 42, single-threaded; built in 589.7 s; 788,233,336 bytes; file hash `2545153aadf3eee8`, re-verified independently; hnswlib header: 1,000,000 elements, M = 32, maxM0 = 64, efConstruction = 40; loads and searches via `fse_matrix`/`HnswIndex` |
| ACORN +1 audit (V5) | frozen configuration (γ = 100, M = 32, M_β = 64), 20K prefix, 6 selectivities × 9 budgets × 50 queries = **2,700 searches: exact − native = 1 for every one; counting store = standard index (ids and native counts)** — `results/phase4/matrix/audit_acorn.json` |
| ACORN-γ index | γ = 100, M = 32, M_β = 64, efConstruction = 3,200 (default M·γ); 16 threads; built in **9,279.8 s (2.6 h)** under ≈ 16–18 external load (other users' jobs on the shared VM; 100K-prefix probe extrapolated ≈ 1.3–2 h); 1,384,928,270 bytes (in-memory 1,384,928,008); file hash `20b0dacd17007dd4`; average out-degree per level 83.3 (level 0, compressed; slots M_β + 1.5M = 112) / 3,200 (level 1, full M·γ lists) / 917.6 / 25.3 / 0 |

## 9. Pilot (implementation/operational only) — PASS

Config `configs/phase4/pilot.yaml`: first 1,000 queries; conditions s ∈
{0.0100, 0.1585, 1.0000} × {random, clustered}; all three methods;
budgets {10, 160, 2560}; same frozen settings and indexes. Outputs
`results/phase4/pilot/`, logs `results/phase4/logs/pilot/`.

| Check | Result |
|---|---|
| Rows (V1) | 42,000 = 6 × 1,000 × (1 + 3 + 3), complete, no duplicates, finite |
| Query set / indexes (V2) | one query hash `1581622a88a15dec`; ACORN `20b0dacd17007dd4`, post-filter `2545153aadf3eee8` |
| Ground truth (V3) | every sweep loaded its condition's verified store (identity check passed) |
| Filter (V4) | 0 filter violations, 0 distance mismatches |
| Accounting (V5) | pre-filter = T in every row; ACORN = native + 1 in every row; post-filter exact = native; post-filter round replay 3,600/3,600 equal |
| Monotonicity (V6) | mean recall and mean distance computations non-decreasing along the pilot budgets; pre-filter recall = 1 for all queries |
| s = 1 identity (V7) | random and clustered at s = 1: identical rows and byte-identical raw files |
| Inert budgets (V8) | post-filter results identical for budgets ≤ ⌈k/s⌉ |
| Behaviour | post-filter refetch reaches 9 rounds (fetch 256,000) on clustered 1 % — within the frozen cap R = 10; ACORN effective list = max(b, k) as specified; no crash |
| Not computed | the V10 outcome check (implementation pilot only; nothing was tuned) |

## 10. Workload estimate and preflight (before launch)

From the pilot (sweep wall times; 1,000 queries, 3 budgets) scaled to
10,000 queries and 9 budgets:
- searches: 12 × 10,000 × (1 + 9 + 9) = **2,280,000** per-query runs;
- pre-filter (16 threads, sequential conditions): ≈ 30 min;
- post-filter (16 threads): ≈ 2–3 h, dominated by clustered s ≤ 0.025
  (many refetch rounds: ≈ 43 s per 1,000 queries per budget at 1 %);
- ACORN (12 single-threaded processes in parallel): ≈ 10–20 min;
- **total ≈ 3–4 h (upper bound ≈ 5 h) < 12 h**;
- storage: pilot 9.7 MB for 42,000 rows → **≈ 0.53 GB** for the matrix
  (≈ 255 MB CSV + 274 MB raw), plus the V9 re-run subset — **< 50 GB**.
The timing pass (§1.7) is separate and secondary; it is not part of this
estimate.

Preflight: G1–G8 passed twice (§7); C = 1,000 frozen (§6); post-filter
index verified (§8); ACORN index built and recorded (§8); +1 audit passed
(§8); pilot passed (§9); estimate within limits; frozen configs
`configs/phase4/{matrix,rerun,pilot,generator_full,cluster_calibration}.yaml`.
**All gates passed → full matrix launched.**

## 11. Full matrix execution (2026-10-05)

Launched 15:29:00Z on an idle VM; `ALL DONE fail=0` at 17:01:11Z
(**1 h 32 min**, inside the 3–4 h estimate). 36/36 sweeps OK
(`results/phase4/logs/matrix_orchestrator.log`): pre-filter 15:29–15:55,
post-filter 15:57–16:47 (clustered s = 0.01 the heaviest, 30 min), ACORN
12 parallel processes 16:47–17:01. Nothing in the config, budgets,
selectivities, query set, C or thresholds was changed.

Outputs (raw, per query, kept):
- `results/phase4/matrix/<method>/<condition>/{per_query.csv, sweep.json,
  raw_b<b>.bin}` — 301 files, 513 MB; SHA-256 of the sorted per_query.csv
  hashes `79e1eebd582696aa…`; ACORN audit `results/phase4/matrix/audit_acorn.json`.
- V9 re-run: `results/phase4/rerun/` (8.8 MB; fingerprint `ad8d32b48faa40fb…`),
  log `results/phase4/logs/rerun_orchestrator.log` (all OK, 17:01–17:04Z).
- Oracle/validation: `results/phase4/analysis/{oracle.csv.gz,
  oracle_summary.csv, aggregates.csv, validation.json}` from
  `python3 python/analysis/phase4_matrix.py --matrix results/phase4/matrix
  --rerun results/phase4/rerun --audit results/phase4/matrix/audit_acorn.json
  --out results/phase4/analysis`.

## 12. Validation V1–V11

| Check | Result | Numbers |
|---|---|---|
| V1 completeness | PASS | 2,280,000 / 2,280,000 rows; 0 duplicates; 0 non-finite |
| V2 same query set | PASS | one query hash across all 36 sweeps |
| V3 ground-truth identity | PASS | 36 sweeps, all loaded through the verified store |
| V4 filter | PASS | 0 filter violations; 0 distance mismatches |
| V5 accounting | PASS | pre = T(s) everywhere; ACORN exact = native + 1 everywhere; post exact = native; post replay 21,600 checked / 0 mismatches; ACORN +1 audit 2,700 / 0 offset errors / 0 behaviour differences |
| V6 monotone | PASS | pre-filter recall = 1 for every query; graph methods monotone |
| V7 s = 1 identity | PASS | random_s1 ≡ clustered_s1 (all fields but latency) |
| V8 inert budgets | PASS | post-filter identical for b ≤ ⌈k/s⌉ |
| V9 determinism | PASS | 38,000 rows compared, identical (except latency) |
| **V10 design checkpoint** | **FAIL** | post-filter passes; **ACORN fails in both correlations** (below) |
| V11 no regressions | PASS | ctest 67/67; pytest 15/15; clang-tidy build 0 findings; Phase 3 toy re-run digest `0b49429d6104335f` with byte-identical data files; Phase 1–3 results/docs untouched (newest files predate Phase 4) |

**Decision rule (§3.2): V10 failed → Phase 4 is NOT complete. Stopped and
reported; the criterion was not changed.**

### 12.1 V10 detail — median oracle effort (exact distance computations; censored = +∞)

| Method | Corr. | s=0.01 | 0.0251 | 0.0631 | 0.1585 | 0.3981 | 1.0 | V10 |
|---|---|---|---|---|---|---|---|---|
| post-filter | random | 12,407 | 6,187 | 3,588.5 | 2,241 | 1,347 | 724 | PASS |
| post-filter | clustered | 108,654.5 | 34,128 | 13,575 | 3,962 | 1,745 | 724 | PASS |
| ACORN | random | 361 | 472.5 | 692 | 898 | 999 | 1,589.5 | **FAIL** |
| ACORN | clustered | 612 | 910 | 908 | 975.5 | 1,322.5 | 1,589.5 | **FAIL** |
| pre-filter | both | T(s) = 10,000 … 1,000,000 (identity, not V10) | | | | | | — |

Censored fractions ≤ 0.22 % (ACORN) and ≤ 2.2 % (post-filter, clustered 1 %).
The ACORN result is not an accounting artefact: V5 (n3 + 1 audit against
an independently counting storage, 2,700 searches) and the V7/V9 identities
pass. ACORN-γ evaluates distances only for predicate-passing neighbours
in its expanded (M·γ) lists, so at low s it searches a much smaller
filtered subgraph; its exact-distance effort **falls** monotonically as s
drops (random) or nearly so (clustered). Its predicate-evaluation work is
not exposed by ACORN (black box; `filter_checks` = 0 for ACORN), and
latency from the 12 parallel single-thread processes is not a clean
measure (timing pass §1.7 not run). The design doc's Phase 4 checkpoint
("effort rises as selectivity drops") therefore does not hold for ACORN
under the frozen effort measure. Whether that is a finding (H3) or calls
for a different checkpoint/effort definition for in-graph search is the
user's decision; nothing has been changed.

### 12.2 Deviations / notes
- ACORN index build took 2.6 h under external load (§8); no effect on results.
- Timing pass (§1.7) not run (secondary; latency fields in the matrix are
  operational only).
- No Phase 5 work, no predictor, no H2/H3 statistics.

## 13. Controlled ACORN timing audit (2026-10-06; audit only, V10 unchanged)

Approved, timing-stage-only change: `fse_matrix ... timing` with
`timing.mode: acorn_audit` (`TimingAcornAudit` in `cpp/cli/fse_matrix.cpp`;
default mode = the untouched §1.7 pass). Config
`configs/phase4/timing_acorn_audit.yaml` (matrix.yaml + mode/pin_core/output
only); runner `scripts/run_phase4_timing_audit.sh` (idle gate, 30-s load log).
Protocol: frozen 500-query subsample (seed 20261200), 12 ACORN conditions ×
9 budgets = 108 blocks in an order permuted by seed 20261200; per query 1
discarded warm-up + 3 timed calls, each recorded; one process pinned to core
3 (affinity read back = 3; all 162,000 timed calls on cpu 3), OMP threads 1
(process had 16 OS threads: idle runtime pools; load ≈ 1.0–1.4 throughout);
latency = the matrix's timer inside `AcornIndex::SearchOne`. Run
05:12:31–05:29:44Z, start load 0.84. Output `results/phase4/timing_audit/`
(timing_rows.csv, blocks.csv, timing.json; analysis/ with script + tables).
Validation PASS: 162,000 rows, 0 duplicates, 500 queries per
(condition, budget, repeat), every repeat identical to its warm-up, index /
query / condition hashes match, counted distances and recall identical to
the matrix rows (0 mismatches). Sweep path regression: a re-run V9 sweep was
identical to the stored one (except latency).

Result (per-query median of 3 repeats): at the V10 operating points latency
does not fall with s (random 318 µs at s=1 → 214–242 mid → 398 at s=0.01;
clustered 316 → 216–309) while counted distances fall to 0.23× / 0.45×; at
b = 2560 latency rises as s drops (random 16.6 → 30.8 ms, monotone;
clustered 16.6 → 21–22 ms) while distances fall 29,190 → ≈4,000; marginal
µs per counted distance rises 0.49 → 6.33 (random) / 4.22 (clustered).
Repeat noise: per-query CV median 3.5 %, cell spread median 5 %. ACORN's
n3 counts only filter-passing distance evaluations (`ACORN.cpp:1404`);
neighbour-list scans, filter lookups and the γ-expanded second hop are
uncounted. Classification: **evidence supports metric incompleteness**.
V10 and the effort metric are unchanged pending the user's decision.

## 14. ACORN scan counter `n_scanned` (2026-10-06; instrumentation only)

Approved, instrumentation-only change to the pinned ACORN source
(`third_party/acorn` @ c259f11; diff now in `patches/acorn_instrumentation.patch`
— the single ACORN patch, which also carries the later level-0 seed record —
every line tagged `// fse: n_scanned instrumentation`). New field
`ACORNStats::n_scanned`: every valid (v ≥ 0) neighbour-list entry read and
filter-checked during hybrid (filtered) search — level-0 scan and γ-expanded
second hop in `hybrid_search_from_candidates`, upper-level scan and its
γ = 1 expansion in `hybrid_greedy_update_nearest` (out-parameter), added to
the stats with `n3`, summed in the `IndexACORN` filtered-search reduction
and `ACORNStats::combine()`. `n3` and all search behaviour unchanged. The
wrapper exposes the per-query delta as `AcornQueryResult::entries_scanned`
(read like `n3`; same single-thread requirement). Not yet written by sweeps;
V10 and the effort metric unchanged.

Validation: tests 68/68 (new `ScanCounterIsDeterministicBoundsDistancesAndSumsToBatch`),
Python 15/15, clang-tidy 0, format clean. The V9 subset (1,000 queries,
ACORN random_s0.0100 and clustered_s0.0631, 9 budgets; 18,000 rows) re-run
with the instrumented build is identical to the stored Phase 4 rerun and
matrix rows (ids/distances raw files byte-identical; recall, n3 and exact
distances 0 differences).

## 15. `n_scanned` in ACORN sweeps — targeted re-run and analysis (2026-10-06)

Sweep output gains a trailing `n_scanned` column (set in `RunAcorn` beside
`dist_native`; 0 for pre/post-filter). Re-ran the 12 ACORN conditions × 9
budgets × 10,000 queries (`configs/phase4/nscan_acorn.yaml` = matrix.yaml with
a new output dir; `results/phase4/nscan/`, 06:18–06:34Z). Regression: all
1,080,000 rows identical to the matrix except latency; 108/108 raw files
byte-identical; 0 filter violations; same index and query hashes. Analysis
`results/phase4/nscan/analysis/` (script + tables).

At b = 2560 (recall 1.0): n3 falls as s drops (29,649 → 3,915 clustered /
4,272 random) while n_scanned rises (170,950 → 2.56 M clustered / 4.00 M
random). Controlled latency rises (1.31× clustered, 1.86× random) — same
direction as n_scanned but far smaller: ns per scanned entry falls from 97 to
8. Neither counter alone tracks latency across s (R² over all budgets and
conditions: n3 0.45, n_scanned 0.69); both together 0.96 (max per-condition
bias 29 % vs 171 % / 320 %). At the V10 operating points n_scanned rises as s
drops (random 2,719 → 70,574; clustered 2,719 → 18,625). Conclusion: ACORN
search work needs both counters (distance evaluations and list scanning);
no combination adopted; V10 unchanged pending the user's decision.

## 16. Post-hoc methodological revision: ACORN effort and V10 (2026-10-06)

**Status: post-hoc revision, adopted after seeing the Phase 4 results.** The
original V10 (§3.2; `n3`-based effort for every graph method) remains in the
record and is **not overwritten**: it was evaluated as frozen and **FAILED**
for ACORN in both correlations (§12). This section adds a revised criterion
beside it.

### 16.1 Why
- §12: ACORN's median oracle distance computations fall as s drops (random
  361 vs 1,589.5; clustered 612 vs 1,589.5 at s = 0.01 vs 1).
- §13 controlled timing audit (one pinned process, interleaved, 3 repeats):
  latency does not fall with s — it rises at b = 2560 (random 1.86×,
  clustered 1.31× of s = 1) while `n3` falls to 0.14×.
- §14–15: ACORN's `n3` counts only distance evaluations on filter-passing
  nodes; neighbour-list scanning and filter checks (incl. the γ-expanded
  second hop) were uncounted. The new counter `n_scanned` rises as s drops
  (23.3× random, 14.9× clustered at b = 2560) in the direction of latency.
  Latency R²: `n3` 0.45, `n_scanned` 0.69, both 0.96.
- Conclusion: `n3` is an incomplete measure of ACORN search effort.

### 16.2 Effort representation (adopted)
- **D = exact distance computations** (ACORN: `n3` + 1, as before) remains
  the **primary cross-method** hardware-independent effort metric.
- **F = `n_scanned`** is an **ACORN-specific supplementary** effort component
  (neighbour-list entries read and filter-checked).
- **No scalar combination** of D and F: their relative cost is
  hardware-dependent; no weight (including the fitted timing coefficients)
  is used.
- **Latency** is supporting/diagnostic evidence only, never a primary metric.
- The per-query **oracle budget b\*** (smallest budget with Recall@10 ≥ 0.9)
  is **unchanged**; D and F are both read at b\*.
- **Leakage:** `n_scanned` is a search outcome and must **never** be used as
  a predictor input.
- Cross-method comparisons use D. F is not comparable with pre-filter's
  filter checks (N mask reads) or post-filter's (fetched candidates; its
  graph scanning is largely inside D). Because D understates ACORN's work at
  low s, a claim that ACORN is cheaper than another method on D alone is not
  valid at low s unless D, F and controlled latency agree; disagreements are
  reported.

### 16.3 Revised V10 (exact rule)
For each graph method and correlation type, compare medians over all
10,000 queries at each query's oracle budget, censored queries = +∞:
- **Post-filter:** median D at s = 0.01 strictly greater than at s = 1.
- **ACORN:** the pair (D, F) at s = 0.01 must not be dominated by (D, F) at
  s = 1. It **fails only if** median D(0.01) ≤ median D(1) **and** median
  F(0.01) ≤ median F(1). Because ACORN's D falls, the check is effectively
  decided by F; both components are reported and the fall in D is shown.
- **Pre-filter:** keeps its D = T(s) identity check; the graph-method rule
  does not apply.

Implementation: `python/analysis/phase4_matrix.py --nscan results/phase4/nscan`
(`join_scan`, `oracle_scan`, `not_dominated`, `check_v10_revised`; tests in
`python/tests/test_phase4_matrix.py`). Output `results/phase4/analysis_v10r/`
(original `results/phase4/analysis/` untouched).

### 16.4 Result
Join: 1,080,000 ACORN rows, 0 unmatched, 0 D / recall mismatches with the
matrix. Oracle budgets, D and aggregates identical to §12.

| Method | Corr. | median D s=0.01 | median D s=1 | median F s=0.01 | median F s=1 | original V10 | revised V10 |
|---|---|---|---|---|---|---|---|
| post-filter | random | 12,407 | 724 | — | — | PASS | PASS |
| post-filter | clustered | 108,654.5 | 724 | — | — | PASS | PASS |
| ACORN | random | 361 (falls) | 1,589.5 | 70,576.5 | 2,719 | **FAIL** | PASS |
| ACORN | clustered | 612 (falls) | 1,589.5 | 18,648 | 2,719 | **FAIL** | PASS |

Original V10: FAIL (kept). Revised V10: PASS. V1–V9 and V11 unchanged
(PASS).

## 17. Effort-vs-selectivity, H3 and preliminary H2 analysis (2026-10-06)

User-authorised analysis of the existing Phase 4 data (no re-runs; §16
methodology). Code `python/analysis/phase4_effort.py` (tests
`python/tests/test_phase4_effort.py`), config
`configs/phase4/effort_analysis.yaml` (bootstrap seed 20261400; 2,000
resamples for medians/paired ratios, 1,000 query-cluster resamples for
correlations; 95 % percentile CIs; Holm over the 36 H3 tests, α = 0.01).
Run with the repo `.venv` (pinned versions). Outputs
`results/phase4/effort_analysis/` (CSV tables, `figures/fig1–4`, `run.json`).
D = oracle exact distance computations at each query's b\* (censored = +∞);
ACORN F = n_scanned at b\* (supplementary); latency secondary.

### 17.1 Effort vs selectivity (D primary; fig1, `effort_curves.csv`)
Median oracle D, s = 0.01 → 1 (95 % CI widths < 6 %):

| Method | random s=0.01 | random s=1 | clustered s=0.01 | clustered s=1 | censored (max) |
|---|---|---|---|---|---|
| pre-filter | 10,000 | 1,000,000 | 10,000 | 1,000,000 | 0 |
| post-filter | 12,407 (17.1×) | 724 | 108,654.5 (150×) | 724 | 2.2 % (clustered 0.01) |
| ACORN D | 361 (0.23×) | 1,589.5 | 612 (0.39×) | 1,589.5 | 0.2 % |
| ACORN F (supp.) | 70,576.5 (26×) | 2,719 | 18,648 (6.9×) | 2,719 | — |

- Pre-filter: D = T(s) exactly (falls linearly with s).
- Post-filter: degrades monotonically and sharply as s drops; clustered is
  far worse than random (8.8× at s = 0.01) — the k/s over-fetch misses
  when the passing vectors are concentrated away from the query.
- ACORN: D falls as s drops (§16: incomplete); F rises monotonically for
  random and up to s = 0.025 for clustered (7.5×, 6.9× at 0.01); controlled
  latency is roughly flat (0.67–1.25× of s = 1; fig2). ACORN's work grows
  gradually and far less than post-filter's.
- Fixed budgets (`fixed_budget.csv`, fig3): post-filter budgets b ≤ ⌈k/s⌉
  are inert (V8), so a fixed b does not cap its effort and its fixed-budget
  failure rate is low at low s (≤ 3 % at b = 10, s = 0.01) — fixed-budget
  recall is therefore not a cross-method effort comparison; the oracle is.
  ACORN needs b ≈ 40–80 for mean recall ≥ 0.9; its hardest levels are
  random s = 0.01 and s = 1.

### 17.2 H3 cross-method comparison (paired per query; `h3_paired.csv`)
All 36 paired sign tests are significant after Holm (n = 10,000 per test;
p ≈ 0), so effect sizes are what matter (median log2 ratio, 95 % CI):
- **Post-filter degrades first and sharpest** (D 17× / 150× of s = 1).
- **Pre-filter becomes competitive at very low s**: pre < post for 73 %
  (random) / 91 % (clustered) of queries at s = 0.01 (post/pre = 1.24× /
  10.9×). Crossover: random between s = 0.01 and 0.025 (at 0.025 post is
  4× cheaper); clustered between 0.025 and 0.063 (pre cheaper for 59 % at
  0.025).
- **ACORN vs post-filter (D):** ACORN cheaper for s ≤ 0.398 (post/ACORN =
  2^0.13–2^7.1, i.e. 1.1× → 136×); post-filter (plain HNSW) cheaper at s = 1
  (ACORN 2× post). **ACORN vs pre-filter (D):** ACORN 16–630× cheaper.
- **§16 rule for "ACORN cheaper" claims:** D understates ACORN at low s, and
  F has no cross-method counterpart. Controlled cross-method latency does
  **not exist** (the frozen §1.7 all-method timing pass was never run; only
  ACORN's controlled audit, §13). Uncontrolled matrix latency (pre/post: 16
  OpenMP threads, ACORN: 12 single-thread processes — different contention)
  agrees in direction: at s = 0.01, ACORN 706 / 335 µs vs pre 1,822 /
  1,846 µs vs post 14,737 / 92,703 µs (random / clustered). ACORN ≪ post is
  robust (D and latency differ by 20–280×); ACORN < pre is supported in
  direction only — D says 16–28×, uncontrolled latency 2.6–5.5×. A
  magnitude claim for ACORN vs pre needs the controlled cross-method timing.
- **H3 verdict (preliminary, D-based):** consistent with H3 — post-filter
  degrades sharply, pre-filter becomes relatively competitive at very low s,
  in-graph (ACORN) degrades gradually (F, latency) across the range.

### 17.3 Preliminary H2 on effort (`h2_pooled.csv`, `h2_within_condition.csv`, fig4)
True local filtered density ρ_K (design measure 2, K = 10 and 100, from the
verified full-scale generator run) is used **post hoc only**; it is never a
predictor input. Pre-filter is excluded (its effort is exactly T(s)).
**H2 as defined in the design (density vs predictor regret) cannot be tested
yet: regret requires the Phase 5 predictor.** What the data support is
density vs oracle effort:
- **Pooled over the six selectivities (Spearman, query-cluster bootstrap):**
  post-filter D: clustered |ρ| = 0.897 (ρ100) vs 0.806 (s), Δ = +0.092
  [0.089, 0.094]; random 0.818 vs 0.808, Δ = +0.010 [0.009, 0.011]; ρ10 is
  weaker than s in random (Δ = −0.074) because 52–90 % of K = 10
  neighbourhoods are empty at s ≤ 0.063. ACORN D: s correlates better than
  density (it carries the §16 inverted trend). ACORN F (supp.): s ≈ density
  (random s better by 0.010; clustered ρ100 better by 0.044 [0.040, 0.048]).
- **Within each condition (s fixed, so s explains nothing):** effort is
  negatively associated with local density in every condition (all CIs
  exclude 0); much more strongly under clustered filters: post-filter
  ρ100 up to −0.82 (clustered) vs ≤ −0.45 (random); ACORN D up to −0.39 vs
  −0.13, F up to −0.46 vs −0.19.
- **Reading:** under random filters local density ≈ s, so it adds little;
  under clustered filters it explains substantial per-query effort that
  global selectivity cannot (the design's disentangling condition works).
  Preliminary support for H2's mechanism on effort, strongest for
  post-filter; not a test of H2 as specified.

### 17.4 What the existing data cannot support
- Regret-based H2 and the live-proxy comparison (design §8 measure 3, §15
  item 5): need Phase 5 (predictor, live proxy, train/test split).
- Controlled cross-method latency: needs the frozen §1.7 timing pass for
  pre- and post-filter (not run).
- Cross-method comparison of F: no defined counterpart (§16).
- All analyses use all 10,000 queries; no train/test split exists yet. These
  descriptive results must not be used to choose Phase 5 features or
  thresholds (the feature set is fixed by the design doc).
- Fix during this work: `oracle_scan` (§16) also attached ACORN F to
  post-filter rows (merge without method); fixed, test added; revised V10
  result unchanged (it reads F for ACORN only); `analysis_v10r` regenerated.
