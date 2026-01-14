#include "methods/postfilter/postfilter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fse {

std::size_t PostfilterFetchSize(std::size_t k, double global_selectivity,
                                const PostfilterParams& p, std::uint32_t round,
                                std::size_t n) {
  if (round == 0 || global_selectivity <= 0.0 || global_selectivity > 1.0) {
    throw std::invalid_argument("PostfilterFetchSize: bad round/selectivity");
  }
  const double base = std::ceil(p.overfetch_factor * static_cast<double>(k) /
                                global_selectivity);
  const double fetch = base * std::pow(p.growth_factor, round - 1);
  const double capped =
      std::min(static_cast<double>(n),
               std::max(static_cast<double>(k), std::ceil(fetch)));
  return static_cast<std::size_t>(capped);
}

FilteredSearchResult PostfilterSearch(const HnswIndex& index,
                                      const float* query, std::size_t k,
                                      const char* filter_row,
                                      double global_selectivity,
                                      const PostfilterParams& params) {
  if (k == 0) {
    throw std::invalid_argument("PostfilterSearch: k must be > 0");
  }
  // growth_factor must exceed 1 when refetching, or rounds would repeat the
  // identical search.
  if (params.max_rounds == 0 || params.overfetch_factor <= 0.0 ||
      params.growth_factor < 1.0 ||
      (params.max_rounds > 1 && params.growth_factor <= 1.0)) {
    throw std::invalid_argument("PostfilterSearch: bad over-fetch params");
  }
  if (global_selectivity < 0.0 || global_selectivity > 1.0) {
    throw std::invalid_argument("PostfilterSearch: selectivity not in [0,1]");
  }
  if (index.Ef() != params.ef_search) {
    throw std::invalid_argument(
        "PostfilterSearch: index ef != params.ef_search (call SetEf first)");
  }
  FilteredSearchResult r;
  r.ids.assign(k, -1);
  r.distances.assign(k, std::numeric_limits<float>::infinity());
  if (global_selectivity == 0.0) {
    return r;  // nothing passes; no search needed
  }

  const std::size_t n = index.Size();
  for (std::uint32_t round = 1; round <= params.max_rounds; ++round) {
    const std::size_t fetch =
        PostfilterFetchSize(k, global_selectivity, params, round, n);
    const HnswSearchResult s = index.SearchAtCurrentEf(query, fetch);
    r.distance_computations += s.distance_computations;
    r.rounds = round;
    r.last_fetch = fetch;

    // Candidates arrive in ascending distance; keep the first k survivors.
    // Each round re-searches from scratch, so survivors are rebuilt; the
    // last round's (possibly partial) survivors are what is returned.
    std::fill(r.ids.begin(), r.ids.end(), -1);
    std::fill(r.distances.begin(), r.distances.end(),
              std::numeric_limits<float>::infinity());
    std::size_t kept = 0;
    for (std::size_t i = 0; i < s.labels.size() && kept < k; ++i) {
      ++r.filter_checks;
      if (filter_row[s.labels[i]] != 0) {
        r.ids[kept] = static_cast<std::int64_t>(s.labels[i]);
        r.distances[kept] = s.dists[i];
        ++kept;
      }
    }
    if (kept == k || fetch >= n) {
      break;  // enough survivors, or nothing more to fetch
    }
  }
  return r;
}

}  // namespace fse
