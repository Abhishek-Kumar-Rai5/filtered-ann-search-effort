#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/vecs_io.h"

namespace fse {

// Only things a live system can compute at query time. The true local
// density needs ground truth, so it is deliberately not here.
std::vector<double> ComputeCentroid(const FloatMatrix& base);

double CentroidDistance(const float* q, const std::vector<double>& centroid);

double ScoreConcentration(const std::vector<float>& sq_dists, std::size_t k);

double LidMle(const std::vector<float>& sq_dists, std::size_t k);

double PassingFraction(const std::vector<std::size_t>& labels,
                       const std::vector<char>& mask);

}
