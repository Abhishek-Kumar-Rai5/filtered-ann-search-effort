#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/vecs_io.h"

namespace fse {

// Phase 3 synthetic filter generator (docs/phase3_filter_generator.md).
//
// Each correlation condition defines one integer attribute per base vector,
// a rank r(i) that is a permutation of 0..N-1. A filter condition at
// selectivity s is the single predicate r(i) < T(s), T(s) = round(s * N),
// shared by all queries: its selectivity is exactly T/N by construction,
// and the levels of one correlation condition are nested.

enum class Correlation { kRandom, kClustered };
std::string CorrelationName(Correlation c);
Correlation ParseCorrelation(const std::string& name);

struct ClusteredParams {
  std::size_t num_clusters = 0;       // k-means C
  int kmeans_iterations = 25;         // FAISS Clustering niter
  std::uint64_t kmeans_seed = 0;      // FAISS Clustering seed (sampling, init)
  int max_points_per_centroid = 256;  // FAISS training subsample cap
  std::uint64_t order_seed = 0;       // cluster order and within-cluster order
};

struct RankAttribute {
  Correlation correlation = Correlation::kRandom;
  std::vector<std::int32_t> rank;  // permutation of 0..N-1
  // Clustered only (empty for random): nearest-centroid cluster per vector
  // and the order in which clusters receive ranks.
  std::vector<std::int32_t> cluster;
  std::vector<std::int32_t> cluster_order;
  std::uint64_t content_hash = 0;  // hash of rank and cluster arrays
};

// Content hash of an attribute (correlation, rank, cluster, cluster order).
std::uint64_t RankAttributeHash(const RankAttribute& a);

// Binary persistence, so that experiment processes reuse the exact generated
// attribute (e.g. without re-running k-means). Read recomputes the content
// hash and throws if it differs from the stored one or from
// `expected_hash` (when nonzero).
void WriteRankAttribute(const std::string& path, const RankAttribute& a);
RankAttribute ReadRankAttribute(const std::string& path,
                                std::uint64_t expected_hash = 0);

// Random: a seeded uniform permutation -- passing sets are uniform random
// subsets, independent of position.
RankAttribute RandomRankAttribute(std::size_t n, std::uint64_t seed);

// Clustered: FAISS k-means (standard, unmodified) on `base`; each vector
// joins its nearest centroid; clusters are ranked in a seeded random order
// and vectors within a cluster in a seeded random order. Every passing set
// is whole clusters plus at most one partial cluster.
RankAttribute ClusteredRankAttribute(const FloatMatrix& base,
                                     const ClusteredParams& p);

struct FilterCondition {
  std::string name;  // e.g. "clustered_s0.0631"
  Correlation correlation = Correlation::kRandom;
  double requested_selectivity = 0.0;
  std::size_t threshold = 0;          // T: base vector i passes iff rank[i] < T
  double achieved_selectivity = 0.0;  // T / N
  std::vector<char> mask;  // N bytes, nonzero = passes (all methods' format)
  std::uint64_t mask_hash = 0;
  // Hash of everything defining the condition: correlation, requested s, T,
  // N, the attribute's content and the base dataset's content.
  std::uint64_t condition_id = 0;
};

// T = round(s * n). Throws unless 0 < s <= 1 and T >= 1.
std::size_t SelectivityThreshold(std::size_t n, double s);

// `levels` values log-spaced from lo to hi inclusive (design: 6, 0.01..1).
std::vector<double> LogSpacedSelectivities(double lo, double hi, int levels);

FilterCondition MakeFilterCondition(const RankAttribute& attr, double s,
                                    std::uint64_t base_hash);

}  // namespace fse
