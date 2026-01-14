# Phase 3 — Synthetic filter generator and filtered ground truth

Status: sections 1–4 written **before implementation** (2026-10-05);
results in section 5+ after validation.

Design doc §21, Phase 3: *"Synthetic filter generator + filtered ground
truth — attribute generation with controlled selectivity/correlation;
filtered brute-force ground truth — generate a small set of test
conditions — checkpoint: selectivity and correlation behave as designed on
a verifiable toy case."*

## 1. What the design fixes

- Synthetic, controlled attributes; single-attribute filters (§6.1,
  CLAUDE.md).
- Selectivity = fraction of the index satisfying the filter, known exactly
  by construction (§8 measure 1); 6 log-spaced levels, roughly 1 %–100 %
  (§10).
- Correlation condition: *random* — "which vectors pass the filter is
  independent of their position in the vector space"; *clustered* —
  "filter-passing vectors are deliberately concentrated so that the same
  global selectivity can correspond to a very different local structure
  around any given query" (§5.2).
- Filter condition = selectivity level × correlation condition (§7.3);
  same fixed query set at every condition (§7.2, §13); exact filtered
  ground truth recomputed for every condition, never reused (§7.4, §13);
  all randomness seeded and logged (§13).
- Local filtered density (§8 measure 2) is post-hoc only — used here
  solely to *validate* that the correlation conditions behave as designed,
  never as a method or predictor input.

## 2. Ambiguities and the interpretation chosen

| Open point | Interpretation (smallest consistent with the design) |
|---|---|
| Is there one filter per condition, or a different predicate per query? | **One filter per condition, shared by all queries.** The design speaks of "the filter" of a condition, its global selectivity "the fraction of the index satisfying the filter", and ground truth "restricted to the filter-passing subset" (§7.4, §8). |
| Attribute and predicate form | **One integer attribute per base vector: a rank r(i) ∈ {0,…,N−1}** (a permutation), and the predicate **r(i) < T(s)** with T(s) = round(s·N). Selectivity is then exact by construction (T/N), and the levels of one correlation condition are **nested** (smaller s ⊂ larger s), which keeps the selectivity factor controlled. |
| How to make "random" | r = a seeded uniform random permutation → the passing set is a uniform random T-subset, independent of position. |
| How to make "clustered" | k-means (FAISS `Clustering`, standard, unmodified in ACORN's fork) with C clusters on the base vectors; every vector goes to its nearest centroid; clusters are taken in a seeded random order, vectors within a cluster in a seeded random order; r = position in that order. The passing set at any s is therefore a set of whole clusters plus at most one partial cluster — spatially concentrated, and chosen without reference to the queries. C is a generator parameter (toy: C = 100, ≈ 200 vectors per cluster). |
| Exact selectivity values | s_i = 10^(−2 + 0.4·i), i = 0…5: 0.0100, 0.0251, 0.0631, 0.1585, 0.3981, 1.0000. |
| s = 1 | Both correlation conditions are the full index there (identical masks) — expected, kept for a paired grid. |

## 3. Components

- `cpp/filter_generator/filter_conditions.{h,cpp}` — rank attributes
  (random, clustered), condition construction (mask in the same per-query
  filter format all three methods take; threshold; achieved selectivity;
  content hashes; a condition id derived from everything that defines it).
- `cpp/filter_generator/uniform_attributes` — gains `SeededPermutation`
  (same toolchain-independent unbiased sampling).
- `cpp/ground_truth/filtered_ground_truth` — gains a mask-filter variant
  of the existing brute-force kernel and a **verified ground-truth store**:
  each file carries the condition id, the mask hash and the query-set hash,
  and loading checks all three, so stale or swapped ground truth cannot be
  used silently.
- `cpp/metrics/local_density` — local filtered density (post-hoc measure 2),
  validation/analysis only.
- `cpp/cli/fse_filter_conditions.cpp` + `configs/phase3/toy_validation.yaml`
  — builds all conditions and runs every check below.

## 4. Validation (pre-specified checks)

Toy case: SIFT1M prefix, 20,000 base × 200 queries (as Phase 2), k = 10,
12 conditions (6 levels × {random, clustered}), C = 100.

1. **Selectivity:** achieved = T/N with |achieved − requested| ≤ 1/(2N);
   mask population = T exactly; nested across levels.
2. **Correlation behaves as intended** (post-hoc local density, K = 10 and
   K = 100 unfiltered true neighbours): random — mean ≈ s and
   variance ≈ binomial s(1−s)/K (dispersion index ≈ 1); clustered —
   dispersion index ≫ 1 (bimodal: many queries at 0, some near 1) for
   s < 1. Also base-level homophily (fraction of a passing base vector's 10
   nearest base neighbours that pass, 1,000-vector sample): ≈ s for
   random, ≫ s for clustered.
   **Numeric pass criteria (fixed before the first run), for every level
   with s < 1**, with dispersion index D = var / (s(1−s)/K):
   - random: 0.5 ≤ D ≤ 2.0 (K = 10 and 100); |mean − s| ≤
     4·√(s(1−s)/K / n_q); |homophily − s| ≤ 0.1;
   - clustered: D ≥ 3 (K = 10 and 100), and D ≥ 2 × the random D at the
     same s; homophily − s ≥ 0.3.
   (The D band for random allows for the strong skew of the sample variance
   at small s·K; s = 1 is the full index for both and is not judged.)
3. **Same query set:** query-set hash identical in every condition record;
   queries loaded once and never filtered per condition.
4. **Ground truth recomputed per condition:** computed fresh for each
   condition, stored with its condition id/mask hash/query hash.
5. **Independent verification:** every condition's ground truth vs FAISS
   `IndexFlatL2` over that condition's passing subset, all 200 queries
   (identical, or differing only by exact distance ties).
6. **No contamination / stale reuse:** every ground-truth id passes its own
   condition's filter; loading any condition's stored ground truth with
   another condition's identity (different mask) is rejected — every such
   pair; stored = freshly recomputed.
7. **Determinism:** generator twice in-process identical; the whole
   validation run twice in separate processes gives identical digests
   (masks, cluster assignments, ground truth).

## 5. Implementation (as built)

| File | Content |
|---|---|
| `cpp/filter_generator/filter_conditions.{h,cpp}` | `Correlation`, `RandomRankAttribute`, `ClusteredRankAttribute` (FAISS `Clustering`, standard), `MakeFilterCondition` (mask, T, achieved s, mask hash, condition id = hash of correlation, s, T, N, attribute content, base content), `LogSpacedSelectivities`, `SelectivityThreshold` |
| `cpp/filter_generator/uniform_attributes.{h,cpp}` | `SeededPermutation` added; internal unbiased draw factored out — `UniformIntAttributes` output unchanged (pinned by a test to the Phase 1 draw-A values) |
| `cpp/ground_truth/filtered_ground_truth.{h,cpp}` | `FilteredGroundTruthMask` (same brute-force kernel); verified store `WriteConditionGroundTruth` / `ReadConditionGroundTruth` with `GroundTruthIdentity` {condition id, mask hash, query hash} checked on load |
| `cpp/metrics/local_density.{h,cpp}` | `LocalFilteredDensity` (measure 2, post-hoc only; leakage note in the header) |
| `cpp/common/run_metadata.{h,cpp}` | `Fnv1a64Bytes` content hash |
| `cpp/cli/fse_filter_conditions.cpp`, `configs/phase3/toy_validation.yaml` | validation driver and config |
| `cpp/tests/test_filter_conditions.cpp` | 9 tests (below) |

## 6. Results (toy case, final code fingerprint `93d62b008f82f563…`)

Output: `results/phase3/phase3_toy_validation/{summary.json, conditions.csv,
local_density.csv, config.yaml}`; ground-truth store `data/cache/phase3/`.

**Checks 1, 3–7: all pass.**

| Check | Result |
|---|---|
| 1. Selectivity | all 12 conditions: mask population = T exactly; \|achieved − requested\| ≤ 1.9e−5 (bound 2.5e−5); nested at every level (both correlations) |
| 3. Same query set | query-set hash identical in all 12 condition records (`query_hash` column) |
| 4. GT recomputed per condition | 12 separate brute-force computations; stored = fresh = recomputed-again for all 12 |
| 5. Independent verification | FAISS `IndexFlatL2` on each passing subset: **200/200 identical in all 12 conditions** (0 tie-only, 0 mismatched) |
| 6. No contamination / stale reuse | 0 ground-truth ids failing their own filter (all 12); loading a condition's ground truth under another condition's identity rejected **132/132** ordered pairs (incl. random vs clustered at s = 1, whose masks are equal but identities differ) |
| 7. Determinism | attributes regenerated in-process identical; two separate processes: identical determinism digest `0b49429d6104335f`, identical attribute hashes, byte-identical `conditions.csv`; digest unchanged after the clang-tidy refactor of the driver |

**Check 2 (correlation behaves as intended): 3 of 5 judged levels FAIL the
pre-specified numeric criteria.**

| s | homophily random (×s) | homophily clustered (×s) | D, K=10 rand / clu | D, K=100 rand / clu | queries with 0 passing among 10-NN rand / clu | pre-specified verdict |
|---|---|---|---|---|---|---|
| 0.0100 | 0.000 (0.0×) | 0.256 (25.6×) | 0.61 / **0.35** | 1.19 / 5.27 | 93.5 % / 98.0 % | **FAIL** (clu D₁₀ < 3; clu homophily − s < 0.3) |
| 0.0251 | 0.032 (1.3×) | 0.358 (14.3×) | 0.71 / **0.46** | 0.89 / 4.34 | 84.0 % / 94.0 % | **FAIL** (clu D₁₀ < 3) |
| 0.0631 | 0.079 (1.3×) | 0.445 (7.1×) | 0.95 / 3.09 | 1.16 / 12.37 | 53.5 % / 80.0 % | pass |
| 0.1585 | 0.163 (1.0×) | 0.554 (3.5×) | 1.02 / 4.22 | 0.98 / 25.21 | 22.5 % / 55.5 % | pass |
| 0.3981 | 0.389 (1.0×) | 0.696 (1.75×) | 1.06 / 4.39 | 1.03 / 23.02 | 1.5 % / 19.5 % | **FAIL** (clu homophily − s = 0.298 < 0.3) |

The **random** condition meets every random-side criterion at every level
(homophily ≈ s, D within [0.5, 2], mean ≈ s).

## 7. Diagnosis of the check-2 failures (no parameter or criterion changed)

1. **D at K = 10 is ill-posed for small s with 200 toy queries.** At
   s = 0.01 the clustered passing set (two k-means cells, 200 vectors)
   touches the 10-NN of only **4 of 200 queries** (max density 0.2); at
   s = 0.025, 12 of 200. Clustering therefore shows up as *zero-inflation*
   (98 % vs 93.5 % of queries with no passing neighbour; mean local density
   0.0025 vs 0.0065 at s = 0.01) rather than as variance, and a variance
   statistic over 200 queries cannot see it. At K = 100 the same
   conditions give D = 5.3 and 4.3 (pass). At full scale (10K queries)
   ≈ s·n_q queries fall in the passing region instead of ≈ 2–5.
2. **The absolute homophily margin (+0.3) exceeds what k-means cells on
   SIFT can deliver.** Exploratory diagnostic (scratchpad tool, *not* part
   of the validation; same seeds): the share of a SIFT vector's 10-NN that
   lies in its own k-means cell is 0.443 (C = 200), **0.493 (C = 100, the
   configured value)**, 0.553 (C = 50), 0.667 (C = 20) — SIFT's cells are
   leaky at every granularity. Clustered homophily is nonetheless
   1.75–25.6× s at every level, versus ≈ 1× for random.

Interpretation: the generator does what the design asks — passing sets
are spatially concentrated (whole k-means cells), and the same global
selectivity produces very different local structure (strong zero-inflation
at small s, high dispersion at K = 100 and at larger s) — but my
pre-specified thresholds were miscalibrated for (a) a 200-query toy and
(b) SIFT's weak cluster structure. That is my judgement, not the
pre-registered verdict.

## 8. Phase 3 checkpoint status

**Not established.** Mechanics (selectivity, per-condition ground truth,
independent verification, contamination protection, determinism) are
validated; the correlation check failed 3 of 5 levels against the
pre-specified thresholds. Per CLAUDE.md, a failed checkpoint is flagged,
not patched around: neither the thresholds nor the generator (C, cluster
construction) were changed after seeing the results.

Decision needed (user):
- **A.** Keep the generator; replace the two miscalibrated sub-criteria by
  well-posed ones, explicitly labelled as revised after seeing results
  (e.g. zero-inflation for small s; homophily as a ratio to s), and
  re-judge.
- **B.** Change the clustered construction's strength (e.g. fewer, larger
  cells; or C defined by cell size so it scales to N = 1M), then re-run the
  pre-specified validation unchanged.
- **C.** Re-run the toy validation with more queries (e.g. all 10K SIFT
  queries on the 20K prefix) under the unchanged criteria, which addresses
  failure 1 but not failure 2.

No Phase 4 work has started.

## 9. Methodological correction of check 2 (2026-10-05, user-approved Option A)

Sections 1–8 are left exactly as written, including the original verdict
(3 of 5 levels FAIL). This section corrects the *validation method* of
check 2 only. The generator, cluster construction, selectivity levels,
query set, seeds and conditions are unchanged; the condition digest
(`0b49429d6104335f`, which covers every attribute, mask and ground-truth
table) must be reproduced exactly after the correction, which proves it.
This section (9.1–9.4) is written **before** any of the replacement
statistics were computed; results follow in 9.5.

### 9.1 What check 2 must establish (from the design)

Design §5.2: under *random* assignment, passing "is independent of
position in the vector space"; under *clustered* assignment, passing
vectors are "deliberately concentrated so that the same global
selectivity can correspond to a very different local structure around any
given query" — the property that lets global selectivity and local density
be disentangled (§5.2, §6.1, §13). So check 2 must show, at every level
s < 1:
- (P1) random is consistent with independence (local structure ≈ global s);
- (P2) clustered passing sets are spatially concentrated, far beyond
  independence, and meaningfully so;
- (P3) this changes the local structure seen by the queries.

### 9.2 The failed sub-criteria

| Original sub-criterion | Intended to establish | Why unreliable here |
|---|---|---|
| (a) clustered dispersion D = var/(s(1−s)/K) ≥ 3 at **K = 10** (failed at s = 0.010, 0.025) | P3: query-level local density is overdispersed vs independence | A variance over 200 queries only sees concentration if enough queries fall *inside* the passing region. At s ≤ 0.025 the clustered passing set (2–5 cells) touches the 10-NN of only 4–12 of the 200 queries; concentration then shows up almost entirely as **extra zeros** (98 % vs 93.5 % zero-density queries at s = 0.01), which *lowers* the variance. So D can fall *below* the random value precisely because the passing vectors are concentrated away from most queries: the statistic can point the wrong way for the property it is meant to detect. |
| (b) clustered homophily − s ≥ **0.3** (failed at s = 0.010 and, by 0.002, at 0.398) | P2: passing vectors' neighbours pass far more often than chance | (i) An absolute margin is not comparable across levels: homophily is bounded by 1, so +0.3 means a 31× lift at s = 0.01 but only 1.75× at s = 0.398, and at s ≥ 0.7 it would be unattainable even for perfect clustering. (ii) The required level exceeded what cell-based concentration on SIFT can reach: only 49 % of a SIFT vector's 10-NN share its k-means cell at C = 100 (§7). (iii) The estimator used a 1,000-vector sample, so at s = 0.01 it averaged over only the ~10 sampled vectors that pass — too few for a stable value. |

### 9.3 Could the failures indicate a real generator defect?

Possible defects and how each is checked (all reported in 9.5):
- **Clusters not spatially coherent** (k-means failing): cell purity
  (share of 10-NN in own cell) must be ≫ the chance value 1/C. Measured
  0.493 vs 0.01 at C = 100 (§7).
- **Passing sets not made of whole clusters**: on the actual toy
  conditions, at most one cluster may be partially included at each level
  (added to the driver as an integrity check; previously only unit-tested).
- **Concentration absent or too weak**: tested directly by the replacement
  statistics against an explicit independence null (9.4).
- **Bad luck in which cells were drawn at small s** (e.g. far from the
  queries): not a defect — cluster order is seeded and independent of the
  queries, as the design requires — but it limits how many queries see the
  concentrated region at s = 0.01 on a 200-query toy. Reported as a
  limitation, not hidden.

### 9.4 Replacement criteria (fixed before computing them)

**Independence null by Monte Carlo.** The reference distribution for every
statistic is generated, not assumed: R = 5,000 additional random-condition
draws (`SeededPermutation` with validation-only seeds null_seed + r,
logged), each giving nested masks at the same 6 levels with exactly the
same T, evaluated on the same 200 queries and the same base vectors. This
captures every source of sampling dependence (overlapping neighbourhoods,
finite N, hypergeometric draws) without a distributional assumption.
Empirical one-sided p = (1 + #{null ≥ observed}) / (R + 1). Family-wise
α = 0.01 with Bonferroni over the 15 clustered significance tests below →
per test α' = 0.01/15 ≈ 6.7e−4 (R = 5,000 resolves p down to 2.0e−4).

**Estimator change for homophily (precision, not a threshold change).**
Homophily is computed exactly over **all** passing base vectors and their
10 nearest base neighbours (self excluded), instead of over a 1,000-vector
sample (feasible at toy scale: 20,000 × 20,000 brute force).

| Replaces | Replacement | Justification |
|---|---|---|
| (a) clustered D at K = 10 | **Z: zero-inflation of query local density**, f₀(K) = fraction of the 200 queries with no passing vector among their K true unfiltered neighbours, for K = 10 and K = 100. Clustered must exceed the independence null: p(f₀) ≤ α' at every level s < 1 (10 tests). | Concentrating a fixed number of passing vectors into a few regions necessarily leaves more query neighbourhoods empty than independent placement does; f₀ measures exactly that, and it is well-defined and most sensitive in the many-zeros regime where D fails. It is judged against the generated null, so it needs no threshold chosen from the data. |
| (b) clustered homophily − s ≥ 0.3 | **H: chance-corrected homophily** κ_h = (h − s)/(1 − s), h = exact homophily (0 under independence, 1 when every neighbour of a passing vector passes — the Cohen-kappa form for a proportion against its chance value). Clustered must satisfy, at every level s < 1, **both** p(κ_h) ≤ α' vs the null (5 tests) **and** κ_h ≥ 0.20. | κ_h is comparable across levels and its range is [−s/(1−s), 1] at every s, unlike an absolute margin. The significance part shows concentration is real; the 0.20 floor requires it to be *meaningful*: 0.20 is the conventional boundary between "slight" and "fair" agreement for kappa-type statistics (Landis & Koch 1977), taken from convention, not from these data. Disclosure: the old 1,000-sample homophily values were known when this floor was chosen (they would give κ_h ≈ 0.25–0.50); the exact values are not yet known. The sensitivity of the verdict to the floor is reported in 9.5. |

**Unchanged sub-criteria** (they passed and remain well-posed):
random 0.5 ≤ D ≤ 2 (K = 10, 100); random |mean − s| ≤ 4·√(s(1−s)/K/n_q);
random |h − s| ≤ 0.1 (now with the exact h); clustered D(K = 100) ≥ 3 and
≥ 2 × random D(K = 100).

**Added consistency checks for the random condition (P1):** the
experimental random draw (seed 20261010) must lie inside the null's
two-sided 1 − α' interval for f₀(10), f₀(100) and κ_h at every level — it
is one draw from that null, so failing this would indicate an error in the
null or the statistic.

**Added integrity check:** at most one partially included cluster per
clustered level.

**Decision rule.** Check 2 passes iff, at every level s < 1, all unchanged
sub-criteria, all replacement criteria, the random consistency checks and
the integrity check pass. If any fails, the checkpoint stays NOT
established and the failure is reported as found.

### 9.5 Results under the corrected check 2

Run: `./build/cpp/fse_filter_conditions configs/phase3/toy_validation.yaml`,
source fingerprint `1dd80e578638f90b…`, R = 5,000 null draws (seeds
20261100 + r), α′ = 0.01/15 = 6.7e−4. Per-statistic detail:
`results/phase3/phase3_toy_validation/check2_corrected.csv`.

**Generator unchanged (verified):** condition digest `0b49429d6104335f`
reproduced; `conditions.csv` byte-identical to the original
(pre-correction) run.

| s | κ_h random | κ_h clustered (z vs null) | f₀(K=10) random / clustered / null 99.93 % | f₀(K=100) random / clustered / null 99.93 % | partial clusters | verdict |
|---|---|---|---|---|---|---|
| 0.0100 | 0.005 | **0.376** (z = 138) | 0.935 / **0.980** / 0.960 | 0.290 / **0.915** / 0.490 | 1 | PASS |
| 0.0251 | 0.001 | **0.362** (z = 127) | 0.840 / **0.940** / 0.870 | 0.055 / **0.785** / 0.160 | 1 | PASS |
| 0.0631 | 0.000 | **0.410** (z = 132) | 0.535 / **0.800** / 0.640 | 0.005 / **0.415** / 0.020 | 1 | PASS |
| 0.1585 | −0.003 | **0.488** (z = 133) | 0.225 / **0.555** / 0.270 | 0.000 / **0.160** / 0.000 | 1 | PASS |
| 0.3981 | −0.005 | **0.481** (z = 89) | 0.015 / **0.195** / 0.030 | 0.000 / **0.020** / 0.000 | 1 | PASS |

- **Clustered (P2, P3):** in all 15 replacement tests the clustered value
  exceeds every one of the 5,000 null draws (p = 1/5001 = 2.0e−4 ≤ α′).
  κ_h ≥ 0.362 at every level (floor 0.20). All unchanged sub-criteria pass
  (clustered D(K=100) = 4.3–25.2 vs random 0.89–1.19).
- **Random (P1):** inside the null's two-sided 1 − α′ interval for every
  statistic at every level (smallest two-sided tail p = 0.021); κ_h ∈
  [−0.005, 0.005]; exact homophily within 0.005 of s; all unchanged random
  sub-criteria pass.
- **Integrity:** exactly one partially included cluster per level; nested;
  selectivity exact.
- **Weakest evidence:** f₀(K=10) at s = 0.01 — clustered 0.980 vs null
  99.93rd percentile 0.960 (z = 3.4). It passes, but this is the
  least-powered comparison: only 200 queries, of which ≈ 2–4 can fall in
  two cells. At s = 0.01 the K = 100 zero fraction (0.915 vs 0.490) and
  κ_h (z = 138) are far stronger.
- **Exact vs sampled homophily:** the original 1,000-sample estimator gave
  0.256 at s = 0.01; the exact value over all 200 passing vectors is 0.382
  (κ_h 0.376) — the original value was dominated by sampling noise
  (≈ 10 contributing vectors), as diagnosed in 9.2.
- **Sensitivity to the κ floor:** the verdict is unchanged for any floor
  ≤ 0.36 and would fail at least one level for floors above 0.362. For
  s ≥ 0.16 clustered κ_h (0.48–0.49) reaches the construction's own
  ceiling — the k-means cell purity of 0.493 (section 7) — i.e. the
  concentration is as strong as whole-cell assignment on SIFT permits.
- **Original criteria:** still 3/5 levels FAIL, reported in every run
  (`original_check2_levels_failed: 3` in `summary.json`) — the record is
  not rewritten.

**Could a failure have indicated a defect?** No defect was found: clusters
are coherent (purity 0.493 vs chance 0.01), every level is whole cells
plus one partial cell, concentration is detected at every level against an
explicit null with large margins, and the random condition is
indistinguishable from independence. The three original failures are
fully explained by the statistic (variance at small s·n_q), the
estimator (≈ 10-vector sample), and an absolute margin that is not
comparable across s.

### 9.6 Remaining limitations

1. **Small-s, query-level power on the toy.** With 200 queries, the
   query-level evidence at s = 0.01 rests on few queries (z = 3.4 at
   K = 10). At full scale (10K queries) ≈ 50× more queries meet the
   concentrated region; Phase 4 should re-run check 2 there (exact
   homophily needs N × N neighbours and must be computed on a large base
   sample at N = 1M).
2. **Strength of concentration is bounded by SIFT geometry.** k-means
   cells on SIFT are leaky (≈ 49 % 10-NN purity at ~200 vectors/cell), so
   clustered conditions are strongly but not perfectly concentrated
   (κ_h 0.36–0.49). This is a property of the data; it is reported, not
   tuned.
3. **One realization per condition.** Which cells pass at small s is one
   seeded draw; at s = 0.01 two cells. The full-scale cluster granularity
   (C, or cells per level) is still an open Phase 4 configuration decision.
4. **The κ floor (0.20)** is a conventional benchmark chosen with the old
   sample-based values known; its sensitivity is reported above.
5. **The replacement criteria were defined after the original results
   were seen** (Option A). They are justified from the purpose of check 2
   and judged against a generated null, but they are post-hoc by
   construction; the original verdict remains on record (section 6).

## 10. Phase 3 checkpoint status (after the correction)

**PASS — established under the corrected, documented validation method
(section 9).** Selectivity is exact and nested; ground truth is recomputed
per condition, verified independently (FAISS, 200/200 in all 12
conditions) and protected against reuse (132/132 wrong-identity loads
rejected); output is deterministic across processes (digest
`0b49429d6104335f`); and random vs clustered produce clearly and
significantly different local structure at every selectivity level below
1, with random consistent with independence. The pass rests on replacement
criteria adopted after the original check failed; this is recorded as a
methodological correction, with the original verdict (3/5 FAIL) preserved
in sections 6–8.

Phase 4 has not started.
