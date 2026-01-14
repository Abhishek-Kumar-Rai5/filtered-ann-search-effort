#pragma once

#include <cstddef>
#include <vector>

#include "ground_truth/filtered_ground_truth.h"

namespace fse {

// Local filtered density (design doc section 8, measure 2): for each query,
// the fraction of its K true nearest neighbours in the full, unfiltered
// index that pass the filter.
//
// LEAKAGE RULE (CLAUDE.md): this requires ground truth and is a post-hoc
// research measurement only -- never a method or predictor input.
//
// unfiltered_gt must hold at least K neighbours per query.
std::vector<double> LocalFilteredDensity(const NeighborTable& unfiltered_gt,
                                         const std::vector<char>& mask,
                                         std::size_t k_local);

}  // namespace fse
