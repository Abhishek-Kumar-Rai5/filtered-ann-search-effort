#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/vecs_io.h"

namespace fse {

// Exact k nearest neighbours per query, restricted to base vectors passing
// that query's filter. Row-major nq x k. Rows with fewer than k passing
// vectors are padded with id -1 / distance +inf.
struct NeighborTable {
  std::size_t nq = 0;
  std::size_t k = 0;
  std::vector<std::int64_t> ids;
  std::vector<float> distances;  // squared L2

  [[nodiscard]] const std::int64_t* Ids(std::size_t q) const {
    return ids.data() + q * k;
  }
  [[nodiscard]] const float* Distances(std::size_t q) const {
    return distances.data() + q * k;
  }
};

// Squared L2 distance. Vectorised with an OpenMP SIMD reduction; for
// integer-valued data with exact float partial sums (e.g. SIFT, where every
// sum stays below 2^24) the result is exact regardless of summation order.
float SquaredL2(const float* a, const float* b, std::size_t dim);

// Brute-force filtered ground truth for an equality predicate: base vector i
// passes query q iff base_attr[i] == query_attr[q]. Results are sorted by
// (distance, id) ascending, so ties are broken by the smaller id. Recomputed
// per filter condition (CLAUDE.md); parallel over queries with OpenMP.
NeighborTable FilteredGroundTruthEquals(
    const FloatMatrix& base, const FloatMatrix& queries,
    const std::vector<std::int32_t>& base_attr,
    const std::vector<std::int32_t>& query_attr, std::size_t k);

// Brute-force filtered ground truth for one filter shared by all queries:
// base vector i passes iff mask[i] != 0 (the per-condition filter of the
// Phase 3 generator). Same kernel and (distance, id) ordering as above.
NeighborTable FilteredGroundTruthMask(const FloatMatrix& base,
                                      const FloatMatrix& queries,
                                      const std::vector<char>& mask,
                                      std::size_t k);

// Unfiltered exact ground truth (every base vector passes), same ordering.
NeighborTable UnfilteredGroundTruth(const FloatMatrix& base,
                                    const FloatMatrix& queries, std::size_t k);

// Binary cache format: header (nq, k) as uint64, then ids (int64) and
// distances (float32), row-major.
void WriteNeighborTable(const std::string& path, const NeighborTable& t);
NeighborTable ReadNeighborTable(const std::string& path);

// Identity of the filter condition a ground-truth table was computed for.
// Stored in the file header and checked on load, so ground truth can never
// be reused for a different condition, filter or query set by mistake.
struct GroundTruthIdentity {
  std::uint64_t condition_id = 0;  // hash of everything defining the condition
  std::uint64_t mask_hash = 0;     // content hash of the filter mask
  std::uint64_t query_hash = 0;    // content hash of the query set

  bool operator==(const GroundTruthIdentity&) const = default;
};

// Verified store: magic + identity + NeighborTable payload.
void WriteConditionGroundTruth(const std::string& path,
                               const GroundTruthIdentity& id,
                               const NeighborTable& t);
// Throws std::runtime_error if the file's identity differs from `expected`
// (stale or wrong-condition ground truth) or the file is malformed.
NeighborTable ReadConditionGroundTruth(const std::string& path,
                                       const GroundTruthIdentity& expected);

}  // namespace fse
