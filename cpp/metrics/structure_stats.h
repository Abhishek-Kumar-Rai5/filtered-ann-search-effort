#pragma once

#include <cstddef>
#include <vector>

#include "ground_truth/filtered_ground_truth.h"

namespace fse {

// Spatial-structure statistics used to validate the Phase 3 correlation
// conditions (docs/phase3_filter_generator.md section 9). Post-hoc,
// ground-truth-derived measurements: never method or predictor inputs.

// Fraction of queries with no passing vector among their first k true
// unfiltered neighbours (zero-inflation of local filtered density).
double ZeroFraction(const NeighborTable& unfiltered,
                    const std::vector<char>& mask, std::size_t k);

// Exact homophily: over ALL passing base vectors, the mean fraction of their
// k nearest base neighbours (self excluded) that also pass. base_knn row i
// must be base vector i's neighbour list (queries = the base set itself),
// with at least k + 1 entries. Returns 0 if nothing passes.
double ExactHomophily(const NeighborTable& base_knn,
                      const std::vector<char>& mask, std::size_t k);

// Sampled homophily: over the sampled base vectors that pass, the mean
// fraction of their k nearest base neighbours (self excluded) that also
// pass. sample_knn row r holds the neighbours of base vector sample[r] in
// the full base (at least k + 1 entries). Returns 0 if no sampled vector
// passes. Used where the exact N x N table is infeasible (N = 1e6).
double SampledHomophily(const NeighborTable& sample_knn,
                        const std::vector<std::int32_t>& sample,
                        const std::vector<char>& mask, std::size_t k);

// Chance-corrected proportion (Cohen-kappa form): (observed - chance) /
// (1 - chance); 0 at chance, 1 at the maximum. Requires chance < 1.
double ChanceCorrected(double observed, double chance);

// Empirical one-sided p-value of `observed` in the upper tail of `null`:
// (1 + #{x >= observed}) / (R + 1).
double EmpiricalPUpper(const std::vector<double>& null, double observed);
// Same, lower tail: (1 + #{x <= observed}) / (R + 1).
double EmpiricalPLower(const std::vector<double>& null, double observed);

}  // namespace fse
