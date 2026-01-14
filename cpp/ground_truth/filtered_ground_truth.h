#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/vecs_io.h"

namespace fse {

struct NeighborTable {
  std::size_t nq = 0;
  std::size_t k = 0;
  std::vector<std::int64_t> ids;
  std::vector<float> distances;

  [[nodiscard]] const std::int64_t* Ids(std::size_t q) const {
    return ids.data() + q * k;
  }
  [[nodiscard]] const float* Distances(std::size_t q) const {
    return distances.data() + q * k;
  }
};

float SquaredL2(const float* a, const float* b, std::size_t dim);

NeighborTable FilteredGroundTruthEquals(
    const FloatMatrix& base, const FloatMatrix& queries,
    const std::vector<std::int32_t>& base_attr,
    const std::vector<std::int32_t>& query_attr, std::size_t k);

NeighborTable FilteredGroundTruthMask(const FloatMatrix& base,
                                      const FloatMatrix& queries,
                                      const std::vector<char>& mask,
                                      std::size_t k);

NeighborTable UnfilteredGroundTruth(const FloatMatrix& base,
                                    const FloatMatrix& queries, std::size_t k);

void WriteNeighborTable(const std::string& path, const NeighborTable& t);
NeighborTable ReadNeighborTable(const std::string& path);

struct GroundTruthIdentity {
  std::uint64_t condition_id = 0;
  std::uint64_t mask_hash = 0;
  std::uint64_t query_hash = 0;

  bool operator==(const GroundTruthIdentity&) const = default;
};

void WriteConditionGroundTruth(const std::string& path,
                               const GroundTruthIdentity& id,
                               const NeighborTable& t);

NeighborTable ReadConditionGroundTruth(const std::string& path,
                                       const GroundTruthIdentity& expected);

}
