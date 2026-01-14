#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fse {

// Result of one filtered k-NN query, shared by the pre-filter and post-filter
// baselines. Effort is reported as exact counts, kept separate by kind:
// distance computations (vector-to-vector distance evaluations, the
// project's primary effort measure) and filter checks (predicate
// evaluations), which cost far less and are never mixed into the former.
struct FilteredSearchResult {
  std::vector<std::int64_t> ids;  // k entries, ascending distance; -1 = none
  std::vector<float> distances;   // squared L2; +inf where ids is -1
  std::uint64_t distance_computations = 0;
  std::uint64_t filter_checks = 0;
  // Post-filter only (0 for pre-filter): over-fetch rounds run and the
  // candidate count requested in the last round.
  std::uint32_t rounds = 0;
  std::size_t last_fetch = 0;
};

}  // namespace fse
