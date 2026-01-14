#pragma once

#include <cstddef>

#include "common/vecs_io.h"
#include "methods/filtered_result.h"

namespace fse {

// Pre-filter baseline (design doc sections 2.2 and 18): identify every base
// vector that passes the filter, then exact brute-force k-NN restricted to
// that subset. Exact by construction, so it has no search-budget knob; its
// effort is fixed by the filter: distance_computations = number of passing
// vectors, filter_checks = base.rows.
//
// filter_row: base.rows bytes, nonzero = passes (the same per-query filter
// representation ACORN takes). Results are ordered by (distance, id), so
// exact ties go to the smaller id; fewer than k passing vectors pad with
// id -1 / distance +inf. Thread-safe.
FilteredSearchResult PrefilterSearch(const FloatMatrix& base,
                                     const float* query, std::size_t k,
                                     const char* filter_row);

}  // namespace fse
