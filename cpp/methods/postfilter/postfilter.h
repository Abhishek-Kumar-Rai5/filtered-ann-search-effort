#pragma once

#include <cstddef>
#include <cstdint>

#include "methods/filtered_result.h"
#include "methods/postfilter/hnsw_index.h"

namespace fse {

// Over-fetch / search-budget parameters of the post-filter baseline. Fixed
// constants for a run (no per-query adaptation beyond the filter's global
// selectivity, which is known by construction - design doc section 8,
// measure 1). See PostfilterSearch for how they are used.
struct PostfilterParams {
  std::size_t ef_search = 0;      // HNSW search budget (same knob as ACORN)
  double overfetch_factor = 1.0;  // round-1 fetch = ceil(factor * k / s)
  double growth_factor = 2.0;     // fetch multiplier for each further round
  std::uint32_t max_rounds = 1;   // >= 1; 1 = single over-fetch, no refetch
};

// Fetch size of round `round` (1-based), capped at `n`:
//   min(n, ceil(overfetch_factor * k / s) * growth_factor^(round - 1)),
// and at least k. Exposed for testing and for recording the schedule.
std::size_t PostfilterFetchSize(std::size_t k, double global_selectivity,
                                const PostfilterParams& p, std::uint32_t round,
                                std::size_t n);

// Post-filter (over-fetch) baseline (design doc sections 2.2 and 18):
// ordinary HNSW search on the full index for more candidates than needed,
// discard candidates failing the filter, and if fewer than k survive, fetch
// more. Each round is a fresh top-`fetch` HNSW search at the index's current
// ef (hnswlib uses max(ef, fetch)); rounds stop as soon as k candidates
// survive, the fetch size reaches the index size, or max_rounds is reached.
//
// Effort: distance_computations is the exact sum over all rounds (every
// distance evaluation, upper layers included); filter_checks is the number
// of candidates checked against the filter.
//
// global_selectivity: fraction of the index passing this query's filter, in
// [0, 1]. 0 returns no results without searching (nothing can pass).
// filter_row: index.Size() bytes, nonzero = passes (same as ACORN's input).
// The index's ef must already equal params.ef_search (index.SetEf); the call
// is then thread-safe across queries.
FilteredSearchResult PostfilterSearch(const HnswIndex& index,
                                      const float* query, std::size_t k,
                                      const char* filter_row,
                                      double global_selectivity,
                                      const PostfilterParams& params);

}  // namespace fse
