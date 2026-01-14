#pragma once

#include <cstdint>

namespace fse {

// Distance-computation accounting for ACORN (docs/phase2_baselines.md
// section 12; audited by cpp/tests/test_distance_accounting.cpp).
//
// ACORN's native counter (faiss::acorn_stats.n3, delta per query) counts
// every query-to-vector distance evaluated by its filtered search except
// one: the entry-point distance `d_nearest = qdis(nearest)` in
// ACORN::hybrid_search (impl/ACORN.cpp:1531). Every other evaluation in the
// upper-level greedy descent and the level-0 search is paired with an
// `ndis` increment. The difference is therefore a fixed offset of exactly
// one evaluation per query (verified for ACORN-gamma and ACORN-1 across
// filters and efSearch values).
//
// The project's cross-method effort measure is the number of distance
// evaluations actually performed. Pre-filter and post-filter (hnswlib with
// CountingL2Space) report that natively; ACORN's native n3 is kept as is
// (it is the convention of the ACORN paper, used for the Phase 1
// reproduction) and converted only for cross-method comparison.
inline constexpr std::uint64_t kAcornUncountedPerQuery = 1;

inline constexpr std::uint64_t AcornExactDistanceComputations(
    std::uint64_t native_n3_per_query) {
  return native_n3_per_query + kAcornUncountedPerQuery;
}

}  // namespace fse
