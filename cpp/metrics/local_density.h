#pragma once

#include <cstddef>
#include <vector>

#include "ground_truth/filtered_ground_truth.h"

namespace fse {

std::vector<double> LocalFilteredDensity(const NeighborTable& unfiltered_gt,
                                         const std::vector<char>& mask,
                                         std::size_t k_local);

}
