# Filter Selectivity and Per-Query Search Effort in ANN Search

**Research Design Document — Project 2**

---

## 1. Overview

**Core research question:** Within a single filtered approximate nearest-neighbour (ANN) search method, how do filter selectivity, the local structure of the filtered subgraph around a query, and query difficulty jointly determine the search effort needed to reach a target recall — and does a *local* measure of filter effect predict required effort better than *global* selectivity alone?

This document specifies what the project investigates, why it matters, how the system and experiments will be built, and how results will be judged. It is written to stand on its own, without reference to any prior discussion, and assumes the reader understands general computer science and machine learning but not ANN or filtered-search internals.

---

## 2. Background

### 2.1 Filtered approximate nearest-neighbour search

Many real retrieval systems do not ask "find the nearest vectors to this query" in isolation — they ask "find the nearest vectors to this query **that also satisfy some attribute constraint**," such as a category, a date range, or an access-permission flag. This is **filtered ANN search**: the result must simultaneously be geometrically close to the query and pass a predicate (filter) over associated metadata.

### 2.2 Why filtering is not a trivial addition to ANN search

A standard ANN index, such as HNSW, is built to navigate efficiently toward the geometrically nearest vectors. It has no inherent notion of which vectors satisfy an arbitrary filter. Three broad strategies exist to combine filtering with ANN search:

- **Pre-filtering**: first identify every vector that satisfies the filter, then run an exact or approximate search restricted to that subset.
- **Post-filtering (over-fetch)**: run ordinary ANN search on the full index, retrieving more candidates than needed, then discard any that fail the filter — and if too few survive, fetch more.
- **In-graph filtering**: traverse the ANN graph as normal, but only consider filter-passing vectors as candidates during the search itself, so the filter is applied *during* traversal rather than before or after it.

Each strategy has a different failure mode. Pre-filtering can become slow if the filtered subset is large (little benefit from the index) or require an expensive full scan if the subset is small but unindexed. Post-filtering can waste enormous effort if the filter is highly selective, since most retrieved candidates may be discarded, possibly requiring many rounds of over-fetching. In-graph filtering avoids wasted retrieval but can struggle if the filter makes the graph *locally* sparse, even where it is not sparse overall, since the traversal may struggle to find a connected path through surviving candidates.

### 2.3 Selectivity and why it is not the whole story

**Selectivity** is the fraction of the index that satisfies a filter — a single global number, easy to compute and easy to report. It is a natural first guess at how hard a filtered query will be: a highly selective filter (few matches) seems like it should make search harder, since there is less to find.

But a global number can conceal a great deal. Two filters with identical global selectivity can behave very differently depending on *where* their matching vectors are located relative to a given query. If a query's true nearest neighbours happen to pass the filter at a high rate, that query may be nearly as easy as unfiltered search, however low the filter's overall selectivity is. If a query's true nearest neighbours rarely pass the filter — even when the filter's global selectivity looks moderate — that query may be unexpectedly difficult, because the *locally* reachable filtered neighbourhood is much sparser than the global number would suggest. This local/global distinction is the central object of study in this project.

### 2.4 Search effort under filtering

As with unfiltered ANN search, filtered search has effort parameters — how large a candidate list to explore, how many rounds of over-fetching to allow, how far to search before giving up. How much effort a given query needs to reach good recall under a filter is not obviously predictable from the filter's selectivity alone, which motivates treating this, like unfiltered per-query search effort, as a query-level prediction problem rather than a single global setting.

---

## 3. Research Motivation

This project investigates **how filter characteristics and local vector structure jointly determine per-query search effort**, within one filtering strategy at a time, and **how different filtering strategies compare as selectivity and local structure vary**.

This is deliberately not the same question as "which filtered-ANN method should be used for this query" — a question of **routing between methods** based on an offline benchmark of recall/QPS per method, which is closer to recent published work on query-aware routing for filtered ANN search. This project instead asks a question about **effort within one method**, conditioned on filter and local structure, which is complementary to a between-method router rather than a replica of one: a between-method router could, in principle, consult the within-method effort model developed here to decide not just *which* method to use but *how hard* to run it.

Nor is this project an attempt to design a new filtered-ANN algorithm. The three filtering strategies described in Section 2.2 are implemented or reused as existing, well-established techniques; the research contribution is in characterising their effort behaviour under controlled selectivity and structural conditions, not in inventing a fourth strategy.

---

## 4. Research Questions and Hypotheses

**Primary research question:** How do filter selectivity, local filtered density around a query, and the choice of filtering strategy jointly determine the search effort needed to reach a target recall — and does a local, per-query measure of filter effect predict required effort better than global selectivity alone?

**Secondary questions:**
- Do the three filtering strategies (pre-filter, post-filter, in-graph) have qualitatively different effort-versus-selectivity relationships, or scaled versions of the same relationship?
- Can a lightweight, live-computable signal estimate required effort per query without first running an exhaustive or expensive probe?
- Does the amount of search effort used under a filter affect how much information a query result reveals about index membership (an optional extension, not core)?

**Hypotheses**, stated as testable claims, not assumed conclusions:

- **H1 — Non-linear effort-selectivity relationship.** Required search effort increases as global selectivity decreases, but not smoothly; a threshold effect is expected once the filtered subgraph becomes locally sparse enough that connectivity, not just size, becomes the limiting factor.
- **H2 — Local density outperforms global selectivity.** A query's *local* filtered density — the fraction of its true nearest neighbours that pass the filter — predicts required search effort better than the filter's global selectivity does. This is the project's central, structural claim, directly analogous to the staleness-versus-update-count comparison in the companion project on index evolution.
- **H3 — Strategy-dependent degradation.** The three filtering strategies degrade differently as selectivity drops: post-filtering is expected to degrade sharply at low selectivity (most fetched candidates discarded); pre-filtering is expected to become relatively more competitive at very low selectivity (the matching subset itself becomes small enough to search directly); in-graph filtering is expected to degrade more gradually across the middle of the range.
- **H4 — A lightweight live predictor is useful.** A small model using global selectivity and a cheap local-density proxy estimates required effort well enough to beat a fixed-effort baseline at matched recall, without needing an expensive exhaustive probe.
- **H5 (optional, extension) — Effort affects privacy leakage.** The amount of search effort used under a filter changes the success rate of a simple membership-inference attack against the filtered index, independent of the filter itself.

**What a meaningful negative result looks like.** If H2 fails and global selectivity predicts effort just as well as any local measure, this is a useful simplification: it would mean practitioners do not need the more expensive local computation, and a single global number is sufficient. If H3 fails and all three strategies degrade in essentially the same way as selectivity changes, that is still informative — it would suggest that, within the conditions tested, strategy choice matters less than the filter's selectivity and structure, regardless of implementation.

---

## 5. System Concept

### 5.1 Base pipeline (single filter condition)

```
Dataset + synthetic attributes → vector index (HNSW / ACORN)
   → query + filter predicate
   → filtered-ground-truth computation (recomputed per filter condition)
   → effort predictor → selected search budget
   → filtered search (pre-filter / post-filter / in-graph) → retrieved neighbours
   → evaluation (recall, effort) against that condition's ground truth
```

### 5.2 Across selectivity and correlation conditions

```
Filter condition grid: selectivity level × correlation condition (random vs. clustered)
   → same fixed query set, every condition
   → filtered ground truth recomputed for every condition
   → effort predictor applied per query, per condition
   → compare against fixed-effort baselines and oracle-per-condition effort
```

The **correlation condition** is a controlled experimental factor: under "random" assignment, which vectors pass the filter is independent of their position in the vector space; under "clustered" assignment, filter-passing vectors are deliberately concentrated so that the same global selectivity can correspond to a very different local structure around any given query. This crossing is what allows global selectivity and local density to be disentangled as predictors, rather than assumed to move together.

### 5.3 Points of comparison

- **Fixed-effort baselines** — constant search-budget values, identical across every filter condition, representing the simplest alternative.
- **Oracle effort** — the minimum effort reaching a target recall for a given query under a given filter condition, found by direct search against that condition's own ground truth.
- **Live effort predictor** — the project's primary object of study: a model using only cheaply, live-computable signals.
- **Three filtering strategies**, each evaluated under all of the above, to test H3.

---

## 6. Research Scope

### 6.1 In scope, and why

| Choice | Reason |
|---|---|
| **Reproduce ACORN (SIGMOD 2024) first**, before building anything new | Validates a correct in-graph filtered-search baseline against independently published numbers before trusting any further results built on it — the same validate-first discipline used for ground truth in the companion project. If ACORN's current public repository is not cleanly buildable against the available FAISS version, in-graph filtering falls back to hnswlib's native filter-callback support, with the change explicitly logged rather than silently substituted. |
| **Pre-filter and post-filter implemented directly** | Both are simple enough to implement correctly in-house, and doing so avoids depending on a second external codebase beyond ACORN/hnswlib. |
| **Synthetic, controlled attributes for the core experiments** | Makes selectivity and the local/global correlation condition exactly known, controllable independent variables, rather than uncontrolled properties of whatever real metadata happens to be available. |
| **Selectivity swept log-spaced from roughly 1% to 100%** | Covers the range from highly restrictive to effectively unfiltered search. |
| **Correlation condition (random vs. clustered)** | The direct experimental lever for testing H2 — without it, global selectivity and local density cannot be disentangled. |
| **Single-attribute filters as the core case** | Keeps the experimental design and its interpretation tractable; this is the setting in which the central local-versus-global question can be most cleanly isolated. |
| **~1M-vector scale**, matching the companion project's convention | Large enough for selectivity levels to be meaningful, small enough for ground truth to be recomputed repeatedly within reasonable time. |
| **Feature reuse from the companion staleness-routing project** (centroid distance, score concentration, local intrinsic dimensionality as an ablation) | Keeps the research signals consistent across the portfolio rather than inventing an unrelated feature set, and allows the one genuinely new signal here — a live local-filtered-density proxy — to be evaluated as an addition rather than a replacement. |

### 6.2 Deliberately out of scope

- **Designing a new filtered-ANN algorithm.** The three strategies are used as established techniques; the contribution is in characterising their behaviour, not inventing a fourth.
- **Complex multi-attribute boolean filter planning.** Two-attribute AND filters are an optional extension at most; general boolean filter query planning is a distinct, larger problem.
- **GPU/CUDA.**
- **A full database query planner or production vector database system.**
- **The membership-inference extension (H5)**, beyond an optional, clearly time-boxed addition attempted only once the core project is complete.
- **Routing between filtered-ANN methods based on an offline benchmark table** — this is the subject of separate, recent published work and is explicitly not reproduced here; this project's router/predictor operates *within* one chosen method.

---

## 7. Research Design

1. **Index construction.** A single base HNSW/ACORN index is built once over the base dataset, with synthetic attribute values attached to each vector.
2. **Query set.** A single set of query vectors is drawn once and never changed across filter conditions.
3. **Filter condition grid.** Selectivity levels crossed with correlation conditions (random vs. clustered) define the experimental grid of filter conditions.
4. **Filtered ground truth.** For every filter condition, exact filtered nearest neighbours are computed by brute force restricted to the filter-passing subset, recomputed fresh for each condition.
5. **Fixed-effort baselines.** A small number of constant search-budget values, applied identically across every filter condition.
6. **Oracle effort.** For each query, under each filter condition, the minimum effort reaching a target recall, found against that condition's own ground truth.
7. **Feature extraction.** Global selectivity (known by construction), a live local-filtered-density proxy (a cheap shallow probe), centroid distance, score concentration, and LID as an ablation-only addition.
8. **Predictor training.** Fit once on a training split of filter conditions and queries, using oracle-effort labels from that same training split only.
9. **Predictor evaluation.** Applied to held-out filter conditions and the held-out query split, compared against fixed-effort baselines and the oracle.
10. **Cross-method comparison.** All of the above repeated identically across the three filtering strategies, to test H3.
11. **Measurement.** Recall, search effort (distance computations, latency as secondary), and predictor regret (gap between predicted and oracle effort at matched recall) are recorded for every query, filter condition, and method.

**Why ground truth must be recomputed per filter condition.** Changing the filter changes which vectors are valid answers at all; reusing one condition's ground truth for another would silently evaluate recall against the wrong answer set, invalidating every downstream result — the same class of error, and the same severity, as reusing a stale index state's ground truth in the companion project.

**Why the correlation condition is essential, not optional.** Without deliberately varying whether filter-passing vectors are randomly scattered or locally clustered, global selectivity and local density would be confounded in any real or naturally-arising dataset, making it impossible to attribute predictive power to one or the other. The synthetic, controlled construction is what makes this separation possible.

---

## 8. Selectivity / Local-Structure Model

Four measures are defined, distinguished by whether they are usable as live router inputs or only as post-hoc research measurements:

1. **Global selectivity** — the fraction of the index satisfying the filter. Known exactly by construction in the synthetic setting. The naive baseline that any structural measure must be compared against. **Live/router-usable.**
2. **Local filtered density** — the fraction of a query's *true* nearest neighbours (in the full, unfiltered index) that pass the filter. Requires ground truth to compute exactly, and therefore directly encodes information about the outcome being predicted. **Post-hoc research measurement only — never a router input.**
3. **Live local-density proxy** — a cheap, shallow-probe estimate of the same quantity: a small unfiltered probe is run near the query, and the fraction of those nearby candidates passing the filter is used as an approximation. This does not require ground truth and is the one genuinely deployable version of the local-structure signal. **Live/router-usable.**
4. **Correlation condition** (random vs. clustered filter assignment) — a controlled experimental factor describing how the filter condition was constructed, not a per-query measurement. Used to design the experimental grid, not as a router feature.

**Leakage note, stated explicitly.** Measure 2 requires knowing the true filtered nearest-neighbour set — the very thing the project is trying to predict the difficulty of finding — so using it as a predictor input would leak the answer into the question. It is used only to validate how well the live proxy (measure 3) approximates the true local density, and in post-hoc analysis.

---

## 9. Effort Predictor

**Inputs:** global selectivity, the live local-density proxy, centroid distance, score concentration, and local intrinsic dimensionality as a second-tier ablation addition — deliberately the same minimal-feature philosophy as the companion staleness-routing project, extended with the one new filter-specific signal.

**Output:** a discrete search-effort tier, not a continuous value, for the same reasons given in the companion project: robustness to noisy oracle labels, ease of calibration, and closeness to how such a system would plausibly be deployed.

**Model:** a small decision tree or logistic regression, not a neural model — chosen so that, if the predictor's decisions are found to be inaccurate under some filter condition, the failure can be traced back to a specific feature, directly supporting the failure analysis in Section 14.

**Relationship to the companion project's router.** This predictor reuses most of the same feature family and router design philosophy deliberately, as a matter of portfolio coherence, but is trained and evaluated on a separate research question (filter-conditioned effort within one method, rather than index-evolution-conditioned effort for a fixed filter). The two are complementary studies of per-query effort allocation under different sources of difficulty, not duplicate work.

---

## 10. Experimental Matrix

**Core experiments** (SIFT1M or modern-embedding set, synthetic attributes):

| Dimension | Levels |
|---|---|
| Selectivity | 6 log-spaced levels, roughly 1%–100% |
| Correlation condition | Random, clustered |
| Filtering strategy | Pre-filter, post-filter (over-fetch), in-graph |
| Policy | Fixed-effort baselines, live predictor, oracle |
| Query set | Fixed test set, held constant across all conditions |
| Metrics | Recall@k, distance computations, latency, predictor regret |

**Secondary experiments:** a reduced grid (fewer selectivity levels, random-correlation only) on a real filtered-attribute dataset (for example, a subset of the Big-ANN filtered-search benchmark track), testing whether findings generalise beyond synthetic, exactly-controlled conditions.

**Optional extensions:** two-attribute AND filters; the membership-inference extension (H5); cross-checking a subset of results against hnswlib's native filter support as an alternative to ACORN.

The matrix is deliberately bounded to the two dimensions — selectivity and correlation condition — most directly implicated in the primary research question, crossed with the three filtering strategies needed to test H3, rather than expanding into every combination of possible filter types and attribute structures.

---

## 11. Datasets

**SIFT1M or the modern high-dimensional embedding set** (reused from the companion project): base vectors for the core synthetic-attribute experiments, chosen for the same reasons given there — one well-understood, fast-iterating dataset, validated against a higher-dimensional, more modern embedding regime.

**Synthetic attributes**, generated directly rather than drawn from any real metadata, are essential to this project specifically: they are what makes selectivity and the local/global correlation condition exactly known and independently controllable, which is a requirement for cleanly testing H2.

**One real filtered-attribute dataset** (secondary validation only): confirms the synthetic-condition findings are not an artifact of how the synthetic attributes were constructed.

No further datasets are added, since each of the three serves a distinct, necessary purpose.

---

## 12. Metrics

| Metric | Plain-language meaning | Research question it answers |
|---|---|---|
| **Recall@k** | Fraction of true filtered nearest neighbours found | Basic search-quality outcome, per query per condition |
| **Distance computations** | Number of vector-distance evaluations | Hardware-independent effort/cost measure |
| **Latency / QPS** | Wall-clock cost | Secondary, practical throughput measure |
| **Predictor regret** | Gap between predicted and oracle effort at matched recall | Core quality measure for the effort predictor; tests H4 |
| **Local-density-vs-regret correlation** | Spearman correlation between local filtered density and predictor regret | Tests H2 against global selectivity as the comparison baseline |
| **Cross-method effort curves** | Effort-vs-selectivity curves compared across the three strategies | Tests H3 |
| **Fraction of failed queries** | Proportion of queries below the recall target, per condition and method | Whether degradation under low selectivity is widespread or concentrated |

---

## 13. Experimental Validity

| Risk | Safeguard |
|---|---|
| Predictor train/test leakage | Training and evaluation use disjoint filter-condition/query splits |
| Oracle-label leakage | Oracle labels for predictor training come only from the designated training split |
| Unfair baseline comparisons | Fixed-effort values identical across all filter conditions and all three methods; oracle honestly recomputed per condition |
| Incorrect ground truth after filter change | Ground truth recomputed fresh for every filter condition; never reused across selectivity levels or correlation conditions |
| Changing query sets across conditions | Forbidden — the same fixed query set is used at every filter condition, enabling paired comparison |
| Confounded selectivity and local structure | Addressed directly by the random-vs-clustered correlation condition (Section 5.2); without this, the two could not be disentangled at all |
| Unfair cross-method comparison | Identical effort-budget definitions and identical query/filter conditions used across pre-filter, post-filter, and in-graph methods |
| ACORN reproduction mismatch | Reproduction against ACORN's own published benchmark is a required Phase 1 checkpoint; results are not trusted until this is verified |
| Timing/warm-cache effects | Single-threaded timing, discarded warm-up queries, repeated runs, median reported; distance computations treated as the primary effort measure |
| Random-seed issues | All randomness (synthetic attribute assignment, query sampling) seeded and logged |
| Hyperparameter leakage | Fixed-effort tier values and the recall target are fixed from training-split behaviour only |

---

## 14. Statistical Analysis

Per-query, paired analysis across filter conditions is central, analogous to the companion project's per-query paired analysis across index states: since the same query set is used at every filter condition, each query's regret can be directly compared across selectivity levels and correlation conditions. **Spearman correlation** between local filtered density and predictor regret is the core test of H2, compared directly against the correlation of global selectivity and regret. **Bootstrap confidence intervals** accompany all aggregate metrics. Cross-method comparisons (H3) use paired tests across the identical query/filter-condition grid. Subgroup analysis by selectivity level and by correlation condition is planned from the experimental design itself, not fitted after the fact.

---

## 15. Failure Analysis

Investigated as a core part of the experimental process:

1. **Queries where the predictor is accurate under random correlation but fails under clustered correlation**, or vice versa — directly diagnostic of whether the predictor is actually using local structure or only global selectivity.
2. **Systematic over-estimation versus under-estimation of required effort**, reported separately, as in the companion project.
3. **Which filtering strategy fails first as selectivity drops**, and whether this matches the expectations in H3.
4. **Cases where global selectivity is moderate but local filtered density is very low for a specific query** — the direct, concrete test of whether the local measure catches something the global number misses.
5. **Whether predictor regret correlates more strongly with the live local-density proxy or with the (leakage-restricted) true local density** — informative about how much approximation cost the live proxy incurs relative to the ideal, unusable signal.

---

## 16. Possible Results

| Pattern | Scientific meaning |
|---|---|
| **A. Local density clearly outperforms global selectivity** as a predictor | The project's highest-novelty possible finding — the direct analogue of structural staleness outperforming update count |
| **B. Global selectivity is sufficient** | A useful simplification — local structure need not be computed in practice |
| **C. Strategies degrade identically** regardless of selectivity or correlation | Suggests effort allocation matters more than strategy choice, within the conditions tested |
| **D. Strategies degrade distinctly**, matching H3's predicted pattern | Directly actionable for choosing a filtering strategy based on expected selectivity |
| **E. The live proxy closely approximates the true local density** | Supports deploying the predictor in practice without needing ground truth |
| **F. The live proxy is a poor approximation of true local density** | A genuine limitation to report honestly, motivating future work on cheaper structural estimators |

All outcomes are treated as valid, reportable findings; none is assumed in advance.

---

## 17. Potential Contribution

- **The research question itself** — whether local filtered structure predicts effort better than global selectivity, and how filtering strategies compare under controlled selectivity/structure conditions — is the contribution regardless of outcome.
- **Possible findings**, contingent on results, include: evidence for or against local density as a superior predictor; a characterisation of strategy-specific degradation patterns; an estimate of how well a cheap live proxy approximates an expensive true structural measure.
- **Future extensions**, explicitly deferred: a combined predictor that routes both between methods (as in recent published work) and within a chosen method's effort level (as studied here); the membership-inference connection (H5) as a fuller study in its own right; multi-attribute boolean filter planning.

No claim of novelty is made in advance; this section describes where a contribution could emerge if the experiments support it.

---

## 18. Implementation Architecture

```
cpp/
  filter_generator/     — synthetic attribute generation; selectivity and correlation control
  methods/
    prefilter/            — exact brute-force-within-filter baseline
    postfilter/            — standard HNSW search + over-fetch logic
    ingraph/                — ACORN wrapper, or hnswlib native filter-callback path
  feature_extraction/      — reused companion-project features + live local-density proxy
  ground_truth/            — filtered brute-force, recomputed per filter condition
  metrics/                  — recall, distance computations, latency
  cli/                       — drives a single experiment run from a configuration file

python/
  predictor/                — effort-predictor training and evaluation
  analysis/                   — statistics, correlation tests, subgroup analysis
  plotting/                    — all figures
  orchestration/                — sweeps the filter-condition grid via the C++ CLI
```

**Why this split.** As in the companion project: index construction, filtered search execution, ground-truth computation, and feature extraction are performance- and correctness-critical, and are implemented in C++. Predictor fitting, statistics, and plotting are implemented in Python, where iteration is faster and correctness risk is lower.

---

## 19. Technology Stack

- **C++20**, matching the companion project's confirmed toolchain decision.
- **FAISS**, required as ACORN's dependency; also usable directly for the pre-filter and post-filter baselines.
- **ACORN** (SIGMOD 2024, public repository), reproduced as the in-graph filtering baseline; **hnswlib's native filter-callback support** as the documented fallback if ACORN proves difficult to build against the current FAISS version, with the fallback decision logged explicitly rather than made silently.
- **CMake**, **OpenMP**, **GoogleTest** — same roles as in the companion project.
- **clang-format / clang-tidy** — same open item as the companion project: not yet installed on the development VM as of this writing; to be installed and enabled before any code here is considered final.
- **Python** — NumPy, Pandas, scikit-learn, SciPy/statsmodels, Matplotlib, matching the companion project's stack.

**CUDA is not introduced**, for the same reason as the companion project: nothing in the current research design requires GPU acceleration.

---

## 20. Reproducibility

Every experimental condition (selectivity level, correlation condition, filtering strategy, policy) is defined by a YAML configuration file. Each run logs its configuration, Git commit hash, random seeds, and build flags alongside its results. Results are stored as structured, row-level output (one row per query, per filter condition, per method, per policy). A smoke-test configuration runs the full pipeline end-to-end on a small subsampled dataset in under a minute. The README documents build instructions, the ACORN build/fallback decision actually used, how to run the smoke test, and how to reproduce the full matrix from configuration alone. A technical report documents the research question, methodology, results, and limitations in full — matching the companion project's report structure, so the two read as a connected pair.

---

## 21. Phased Development Plan

| Phase | Objective | Implementation | Experiment | Expected output | Validation checkpoint | Condition to proceed |
|---|---|---|---|---|---|---|
| **0** | Environment and repository | CMake, FAISS/ACORN dependency setup, GoogleTest wired in, directory skeleton | — | Building, empty project | Build succeeds; empty tests pass | Clean build achieved |
| **1** | **Reproduce ACORN's published numbers** | ACORN build against current FAISS, or confirmed fallback to hnswlib native filtering | Reproduce one of ACORN's own published benchmark configurations | Matched (or closely matched) recall/QPS numbers | Numbers match within reasonable tolerance, or the fallback decision is made and logged with reasons | **Do not proceed without understanding any mismatch** |
| **2** | Pre-filter and post-filter baselines | Implement both directly | Correctness check against brute force on a small case | Verified-correct baseline implementations | Both match exact brute-force results on a toy case | Baselines trustworthy before use in the main matrix |
| **3** | Synthetic filter generator + filtered ground truth | Attribute generation with controlled selectivity/correlation; filtered brute-force ground truth | Generate a small set of test conditions | Filter conditions with known selectivity and correlation | Selectivity and correlation behave as designed on a verifiable toy case | Filter generator trustworthy before scaling up |
| **4** | Core effort-selectivity matrix, fixed budgets only | — | Full selectivity × correlation × method grid, fixed-effort policies only | Effort-vs-selectivity curves per method | Curves behave sensibly (effort rises as selectivity drops) before adding the predictor | Base matrix validated before predictor work begins |
| **5** | Feature extraction + predictor | Live local-density proxy + reused features; predictor training | Fit and evaluate predictor on training split | Trained predictor, held-out performance | Predictor clearly beats the best fixed-effort baseline at matched recall | **Must succeed before proceeding** |
| **6** | Cross-method comparison | — | Full matrix across all three strategies, all policies | Complete, serialised results | All conditions present, none missing | Full dataset available before analysis |
| **7** | Structural analysis | — | Local-density-vs-regret correlation, compared against global-selectivity-vs-regret | Preliminary answer to H2 | A clear correlation result for each candidate measure | — |
| **8** | Failure analysis and statistics | — | Section 15 breakdowns; full statistical treatment | Documented failure cases; statistically supported findings for H1–H4 | At least several individually inspected, explained failure cases exist | — |
| **9** | Secondary real-filter dataset | — | Reduced matrix on a real filtered-attribute dataset | Generalisation check | Synthetic-condition findings compared against real-data findings, whether they replicate or diverge | Generalisation question explicitly answered |
| **10** | Reproducibility and final report | README, reproduction scripts, technical report | — | Complete, reproducible repository and report | Clean checkout regenerates at least one headline figure | — |

---

## 22. First Implementation Milestone

The first concrete deliverable is **Phase 1 in full**: get ACORN building against the available FAISS version and reproduce its reported recall/QPS numbers on one of its own published benchmark configurations — or, if ACORN proves impractical to build cleanly, confirm and explicitly log the fallback to hnswlib's native filter-callback support as the in-graph baseline instead.

This milestone exists because every later phase depends on having a correct, trustworthy in-graph filtering baseline; building new experiments on top of an unverified or incorrectly reproduced baseline would put every downstream result at risk.

**Explicitly not included in this milestone:** pre-filter or post-filter implementations, the synthetic filter generator, feature extraction, the effort predictor, or any part of the core experimental matrix. These begin only once Phase 1 is validated and Phase 2 is under way.
