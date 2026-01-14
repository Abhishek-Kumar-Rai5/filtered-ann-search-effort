#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/vecs_io.h"

namespace fse {

enum class Correlation { kRandom, kClustered };
std::string CorrelationName(Correlation c);
Correlation ParseCorrelation(const std::string& name);

struct ClusteredParams {
  std::size_t num_clusters = 0;
  int kmeans_iterations = 25;
  std::uint64_t kmeans_seed = 0;
  int max_points_per_centroid = 256;
  std::uint64_t order_seed = 0;
};

struct RankAttribute {
  Correlation correlation = Correlation::kRandom;
  std::vector<std::int32_t> rank;

  std::vector<std::int32_t> cluster;
  std::vector<std::int32_t> cluster_order;
  std::uint64_t content_hash = 0;
};

std::uint64_t RankAttributeHash(const RankAttribute& a);

void WriteRankAttribute(const std::string& path, const RankAttribute& a);
RankAttribute ReadRankAttribute(const std::string& path,
                                std::uint64_t expected_hash = 0);

RankAttribute RandomRankAttribute(std::size_t n, std::uint64_t seed);

RankAttribute ClusteredRankAttribute(const FloatMatrix& base,
                                     const ClusteredParams& p);

struct FilterCondition {
  std::string name;
  Correlation correlation = Correlation::kRandom;
  double requested_selectivity = 0.0;
  std::size_t threshold = 0;
  double achieved_selectivity = 0.0;
  std::vector<char> mask;
  std::uint64_t mask_hash = 0;

  std::uint64_t condition_id = 0;
};

std::size_t SelectivityThreshold(std::size_t n, double s);

std::vector<double> LogSpacedSelectivities(double lo, double hi, int levels);

FilterCondition MakeFilterCondition(const RankAttribute& attr, double s,
                                    std::uint64_t base_hash);

}
