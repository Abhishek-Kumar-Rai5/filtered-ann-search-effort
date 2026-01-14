# Engineering notes / decision log

## Phase 0 (2026-10-02)

- **ACORN is a FAISS fork, not a FAISS add-on.** `third_party/acorn` (pinned
  `c259f11`, 2025-03-06) is FAISS **1.7.3** with `IndexACORN` added. There is
  no "build ACORN against the current FAISS": its `libfaiss` *is* this
  project's FAISS. Linking a second, upstream FAISS alongside it would give
  conflicting `faiss::` symbols, so any direct FAISS use (pre/post-filter
  baselines) goes through ACORN's fork.
- **ACORN is built as an isolated `ExternalProject`** (`cmake/acorn.cmake`)
  with the exact flags from its README (`GPU=OFF PYTHON=OFF SHARED=ON
  Release`, `FAISS_OPT_LEVEL=generic`), not `add_subdirectory`: its top-level
  CMake forces C++11, `include(CTest)`, and FetchContents nlohmann_json at
  configure time (needs network on first configure). Its sources are never
  modified. ACORN's own benchmark driver is `demos/test_acorn.cpp` (Phase 1).
- **Submodule pins** match Project 1 for consistency: googletest `52eb810`,
  hnswlib `3f34296`, yaml-cpp `0.8.0`.
- **clang-format / clang-tidy 18 are now installed.** clang-tidy cannot check
  files that include `omp.h`: clang rejects GCC's `omp.h`
  (`__malloc__ (omp_free)` attribute) -- verified, not assumed. Needs
  `libomp-18-dev`. As in Project 1: `build/` = experiments (tidy OFF,
  identical object code), `build-tidy/` = static analysis.
- **BLAS/LAPACK**: `libopenblas-dev` 0.3.26 and `libomp-18-dev` 18.1.3
  installed by the user (sudo). With libomp-18-dev, clang-tidy now parses
  all project sources cleanly, including the OpenMP ones.
- **ACORN build fix (configure-only, no source change).** As an
  ExternalProject, ACORN's generate step fails: it adds its FetchContent'd
  nlohmann_json dir to `faiss`'s PUBLIC include dirs, and CMake rejects a
  build-tree path there. Fixed by passing
  `-DFETCHCONTENT_BASE_DIR=<build>/acorn_deps` (outside ACORN's own build
  tree). Not yet checked whether ACORN's own README in-tree build hits this.
- ACORN's sources emit 3 `-Wc++17-extensions` warnings (structured bindings
  in `faiss/impl/HNSW.cpp` under its forced C++11 standard). Upstream code
  built as upstream builds it; left as is. ACORN (`libfaiss.so`) full
  build: ~75 s on 16 cores.

### Phase 0 checkpoint -- MET (2026-10-02)
- `build/` (ACORN ON, tidy OFF): builds, 5/5 tests pass, incl.
  `AcornSmoke.FilteredSearchReturnsOnlyPassingIds`. `fse_tests` links
  `build/acorn/faiss/libfaiss.so` + libopenblas + libgomp (checked via ldd).
- `build-tidy/` (ACORN ON, tidy ON): builds with zero project clang-tidy
  findings, 5/5 tests pass. clang-format: clean.
- Python venv at `.venv/` (system python3.12), pins in
  `python/requirements.txt` = Project 1's versions.

## Phase 1 (2026-10-02 → 2026-10-03) — ACORN reproduction

Full record (pre-registration, configuration, results, deviations):
`docs/phase1_acorn_repro.md`. Key engineering facts for later phases:

- **Benchmark reproduced:** ACORN paper §7.3.1 SIFT1M LCPS (random int
  1–12, `equals(y)`, γ = 12, M = 32, Mβ = 64, K = 10); ACORN-γ and ACORN-1.
- **efConstruction is M·γ (ACORN's default), not 40.** The paper text says
  efc = 40 but Table 6's full level-1 lists (384) are impossible with it;
  the authors' driver never sets it. ACORN-1 therefore uses efc = 32.
- **ACORN's distance counter** (`faiss::acorn_stats.n3`, upper-level greedy +
  level 0) is a process-wide global updated without synchronisation →
  per-query counts need single-threaded `search(n=1)` calls; parallelism
  must come from separate processes. `stats.ndis` is never set on the
  filtered path.
- **ACORN keeps a raw pointer to the caller's attribute vector** (read during
  construction); `read_index` leaves it dangling. `fse::AcornIndex` owns the
  attributes and re-attaches them on load.
- **Construction is predicate-agnostic** (attribute values are read but
  unused), so one index serves any attribute draw — used for draw B.
- **The VM is shared.** Timing results are only valid when no other heavy
  job is running; Project 1 jobs contaminated the first QPS stage by ~4.4×
  (deviation D3). Timing scripts now log /proc/loadavg every 30 s.
- **Runtime reality on 16 vCPUs:** ACORN-γ build ~22–24 min; per-query
  sweep (203 efs × 10K queries, single thread) 1.2 h (γ) / ~4 h (ACORN-1);
  timing stage (16 efs × 51 batches) ~2 h per method on an idle VM.
  The harness's 2-hour background limit means long runs go through
  `nohup setsid` scripts.
- **Batch search scales 8.3× on 16 threads** (idle VM, efs 10); per-thread
  speed matches the paper's per-vCPU throughput.
- Post-run: clang-tidy findings in the new code fixed (behaviour-preserving;
  re-verified, see `docs/phase1_acorn_repro.md` §11).

## Phase 2 (2026-10-05) — pre-filter and post-filter baselines

Full record: `docs/phase2_baselines.md`. Engineering facts for later phases:

- **All methods take the same filter input**: one char mask row per query
  (N bytes), identical to ACORN's `filter_id_map` row.
- **Post-filter runs on hnswlib**, not the HNSW in ACORN's FAISS fork (that
  one is modified: `gamma`, metadata, `hybrid_search`).
- `fse::HnswIndex` is Project 1's wrapper; `CountingL2Space` gives exact,
  thread-local per-query distance counts including upper layers, so
  post-filter sweeps can run multi-threaded (unlike ACORN's global counter).
- `PostfilterSearch` requires `index.SetEf(params.ef_search)` beforehand and
  is then thread-safe; hnswlib uses max(ef, fetch), so ef < k/s is inert.
- ACORN's n3 skips the entry-point distance (`impl/ACORN.cpp:1531`);
  hnswlib's count does not — a counting convention difference to measure
  before Phase 4.
- **Resolved (2026-10-05):** ACORN's n3 misses exactly one evaluation per
  query (entry point); verified offset = 1 on 1,080 queries
  (`test_distance_accounting.cpp`). Cross-method effort = exact evaluations
  = native for pre/post-filter, `AcornExactDistanceComputations(n3)` = n3 + 1
  for ACORN (`methods/ingraph/acorn_accounting.h`). Natives always recorded.
- **Resolved (2026-10-05):** common budget control = candidate-list budget b
  (efSearch grid, identical across methods/conditions; each method's native
  semantics; post-filter keeps the k/s fetch rule); common effort measure =
  exact distance computations; oracle/regret in exact evaluations. Details:
  `docs/phase2_baselines.md` §13.

## Phase 3 (2026-10-05) — filter generator (checkpoint pending)

Full record: `docs/phase3_filter_generator.md`.
- One filter per condition: rank attribute r(i) (permutation) and
  predicate r(i) < round(s·N); exact selectivity, nested levels.
  Random = seeded permutation; clustered = FAISS k-means cells in seeded
  random order.
- Condition ground truth is stored with {condition id, mask hash, query
  hash} and refused on mismatch (`ReadConditionGroundTruth`).
- FAISS k-means here is deterministic across processes (same digest).
- SIFT k-means cells are leaky: only ~49 % of a vector's 10-NN share its
  cell at ~200 vectors/cell (67 % at ~1,000/cell).
- `UniformIntAttributes` refactored; outputs pinned to the Phase 1 draw.
- **Check-2 correction (2026-10-05):** original correlation criteria failed
  3/5 (D at K = 10 is blind to zero-inflation at small s·n_q; absolute
  homophily margin not comparable across s; 1,000-sample homophily too
  noisy). Replaced (documented as a methodological correction) by
  zero-inflation f₀(K) and κ_h = (h − s)/(1 − s) with exact homophily, each
  vs a 5,000-draw Monte Carlo independence null (Bonferroni α′ = 6.7e−4),
  plus κ_h ≥ 0.20. Passes 5/5; generator unchanged (same digest).
  `metrics/structure_stats` holds the statistics.
