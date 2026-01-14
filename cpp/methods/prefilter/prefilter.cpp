#include "methods/prefilter/prefilter.h"

#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

#include "ground_truth/filtered_ground_truth.h"

namespace fse {

FilteredSearchResult PrefilterSearch(const FloatMatrix& base,
                                     const float* query, std::size_t k,
                                     const char* filter_row) {
  if (k == 0) {
    throw std::invalid_argument("PrefilterSearch: k must be > 0");
  }
  FilteredSearchResult r;
  r.ids.assign(k, -1);
  r.distances.assign(k, std::numeric_limits<float>::infinity());

  using Candidate = std::pair<float, std::int64_t>;
  std::priority_queue<Candidate> heap;
  for (std::size_t i = 0; i < base.rows; ++i) {
    ++r.filter_checks;
    if (filter_row[i] == 0) {
      continue;
    }
    ++r.distance_computations;
    const Candidate c{SquaredL2(query, base.Row(i), base.dim),
                      static_cast<std::int64_t>(i)};
    if (heap.size() < k) {
      heap.push(c);
    } else if (c < heap.top()) {
      heap.pop();
      heap.push(c);
    }
  }
  for (std::size_t j = heap.size(); j-- > 0;) {
    r.distances[j] = heap.top().first;
    r.ids[j] = heap.top().second;
    heap.pop();
  }
  return r;
}

}
