#pragma once

// Phase 5 live predictor features (docs/design_doc.md §9; frozen in
// docs/phase5_predictor.md §2). Definitions are ported from Project 1
// (ann_router_staleness, src/features.cpp) for portfolio coherence.
//
// Every feature here is computable at query time without ground truth. The
// true local filtered density (metrics/local_density.h, design §8 measure 2)
// is deliberately NOT here: it is a post-hoc measurement and must never be a
// predictor input.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/vecs_io.h"

namespace fse {

// Mean of all base vectors (double accumulation).
std::vector<double> ComputeCentroid(const FloatMatrix& base);

// Euclidean distance from the query to the base centroid.
double CentroidDistance(const float* q, const std::vector<double>& centroid);

// d_1 / d_k over the first k ascending squared distances (Euclidean ratio);
// 1 when d_k == 0.
double ScoreConcentration(const std::vector<float>& sq_dists, std::size_t k);

// Levina-Bickel MLE of local intrinsic dimensionality from the first k
// ascending squared distances: -(k-1) / sum_{i<k} ln(d_i / d_k). Conventions
// (Project 1): 0 if d_k == 0 or some d_i == 0; +inf if all equal.
double LidMle(const std::vector<float>& sq_dists, std::size_t k);

// Live local-density proxy: fraction of the probe's labels whose filter-mask
// byte is nonzero. Reads only the filter mask (never ground truth).
double PassingFraction(const std::vector<std::size_t>& labels,
                       const std::vector<char>& mask);

}  // namespace fse
