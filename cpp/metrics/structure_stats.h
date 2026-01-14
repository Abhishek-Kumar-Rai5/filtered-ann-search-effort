#pragma once

#include <cstddef>
#include <vector>

#include "ground_truth/filtered_ground_truth.h"

namespace fse {

double ZeroFraction(const NeighborTable& unfiltered,
                    const std::vector<char>& mask, std::size_t k);

double ExactHomophily(const NeighborTable& base_knn,
                      const std::vector<char>& mask, std::size_t k);

double SampledHomophily(const NeighborTable& sample_knn,
                        const std::vector<std::int32_t>& sample,
                        const std::vector<char>& mask, std::size_t k);

double ChanceCorrected(double observed, double chance);

double EmpiricalPUpper(const std::vector<double>& null, double observed);

double EmpiricalPLower(const std::vector<double>& null, double observed);

}
