#include "ground_truth/filtered_ground_truth.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace fse {
namespace {

using Candidate = std::pair<float, std::int64_t>;  // (distance, id)

// Shared brute-force kernel: `passes(q, i)` decides membership.
template <typename Pass>
NeighborTable BruteForceTopK(const FloatMatrix& base,
                             const FloatMatrix& queries, std::size_t k,
                             Pass passes) {
  if (base.dim != queries.dim) {
    throw std::invalid_argument("ground truth: base/query dim mismatch");
  }
  if (k == 0) {
    throw std::invalid_argument("ground truth: k must be > 0");
  }
  NeighborTable t;
  t.nq = queries.rows;
  t.k = k;
  t.ids.assign(t.nq * k, -1);
  t.distances.assign(t.nq * k, std::numeric_limits<float>::infinity());

  const auto nq = static_cast<std::int64_t>(queries.rows);
#pragma omp parallel for schedule(dynamic, 16)
  for (std::int64_t q = 0; q < nq; ++q) {
    const auto qu = static_cast<std::size_t>(q);
    // Max-heap on (distance, id): the top is the current worst candidate,
    // and lexicographic order makes the smaller id win exact ties.
    std::priority_queue<Candidate> heap;
    for (std::size_t i = 0; i < base.rows; ++i) {
      if (!passes(qu, i)) {
        continue;
      }
      const Candidate c{SquaredL2(queries.Row(qu), base.Row(i), base.dim),
                        static_cast<std::int64_t>(i)};
      if (heap.size() < k) {
        heap.push(c);
      } else if (c < heap.top()) {
        heap.pop();
        heap.push(c);
      }
    }
    for (std::size_t j = heap.size(); j-- > 0;) {
      t.ids[qu * k + j] = heap.top().second;
      t.distances[qu * k + j] = heap.top().first;
      heap.pop();
    }
  }
  return t;
}

}  // namespace

float SquaredL2(const float* a, const float* b, std::size_t dim) {
  float acc = 0.0F;
#pragma omp simd reduction(+ : acc)
  for (std::size_t j = 0; j < dim; ++j) {
    const float d = a[j] - b[j];
    acc += d * d;
  }
  return acc;
}

NeighborTable FilteredGroundTruthEquals(
    const FloatMatrix& base, const FloatMatrix& queries,
    const std::vector<std::int32_t>& base_attr,
    const std::vector<std::int32_t>& query_attr, std::size_t k) {
  if (base_attr.size() != base.rows || query_attr.size() != queries.rows) {
    throw std::invalid_argument("ground truth: attribute count mismatch");
  }
  return BruteForceTopK(base, queries, k, [&](std::size_t q, std::size_t i) {
    return base_attr[i] == query_attr[q];
  });
}

NeighborTable FilteredGroundTruthMask(const FloatMatrix& base,
                                      const FloatMatrix& queries,
                                      const std::vector<char>& mask,
                                      std::size_t k) {
  if (mask.size() != base.rows) {
    throw std::invalid_argument("ground truth: mask size mismatch");
  }
  return BruteForceTopK(base, queries, k, [&](std::size_t, std::size_t i) {
    return mask[i] != 0;
  });
}

NeighborTable UnfilteredGroundTruth(const FloatMatrix& base,
                                    const FloatMatrix& queries, std::size_t k) {
  return BruteForceTopK(base, queries, k,
                        [](std::size_t, std::size_t) { return true; });
}

void WriteNeighborTable(const std::string& path, const NeighborTable& t) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cannot open for writing: " + path);
  }
  const std::array<std::uint64_t, 2> header = {t.nq, t.k};
  out.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
  out.write(reinterpret_cast<const char*>(t.ids.data()),
            static_cast<std::streamsize>(t.ids.size() * sizeof(std::int64_t)));
  out.write(reinterpret_cast<const char*>(t.distances.data()),
            static_cast<std::streamsize>(t.distances.size() * sizeof(float)));
  if (!out) {
    throw std::runtime_error("write failed: " + path);
  }
}

NeighborTable ReadNeighborTable(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  std::array<std::uint64_t, 2> header = {0, 0};
  in.read(reinterpret_cast<char*>(header.data()), sizeof(header));
  NeighborTable t;
  t.nq = header[0];
  t.k = header[1];
  t.ids.resize(t.nq * t.k);
  t.distances.resize(t.nq * t.k);
  in.read(reinterpret_cast<char*>(t.ids.data()),
          static_cast<std::streamsize>(t.ids.size() * sizeof(std::int64_t)));
  in.read(reinterpret_cast<char*>(t.distances.data()),
          static_cast<std::streamsize>(t.distances.size() * sizeof(float)));
  if (!in) {
    throw std::runtime_error("truncated neighbour table: " + path);
  }
  return t;
}

namespace {
constexpr std::array<char, 8> kGtMagic = {'F', 'S', 'E', 'G',
                                          'T', 'v', '1', '\0'};
}  // namespace

void WriteConditionGroundTruth(const std::string& path,
                               const GroundTruthIdentity& id,
                               const NeighborTable& t) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cannot open for writing: " + path);
  }
  const std::array<std::uint64_t, 5> header = {id.condition_id, id.mask_hash,
                                               id.query_hash, t.nq, t.k};
  out.write(kGtMagic.data(), kGtMagic.size());
  out.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
  out.write(reinterpret_cast<const char*>(t.ids.data()),
            static_cast<std::streamsize>(t.ids.size() * sizeof(std::int64_t)));
  out.write(reinterpret_cast<const char*>(t.distances.data()),
            static_cast<std::streamsize>(t.distances.size() * sizeof(float)));
  if (!out) {
    throw std::runtime_error("write failed: " + path);
  }
}

NeighborTable ReadConditionGroundTruth(const std::string& path,
                                       const GroundTruthIdentity& expected) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  std::array<char, 8> magic{};
  std::array<std::uint64_t, 5> header{};
  in.read(magic.data(), magic.size());
  in.read(reinterpret_cast<char*>(header.data()), sizeof(header));
  if (!in || magic != kGtMagic) {
    throw std::runtime_error("not a condition ground-truth file: " + path);
  }
  const GroundTruthIdentity found{header[0], header[1], header[2]};
  if (!(found == expected)) {
    throw std::runtime_error(
        "ground truth identity mismatch (stale or wrong condition): " + path);
  }
  NeighborTable t;
  t.nq = header[3];
  t.k = header[4];
  t.ids.resize(t.nq * t.k);
  t.distances.resize(t.nq * t.k);
  in.read(reinterpret_cast<char*>(t.ids.data()),
          static_cast<std::streamsize>(t.ids.size() * sizeof(std::int64_t)));
  in.read(reinterpret_cast<char*>(t.distances.data()),
          static_cast<std::streamsize>(t.distances.size() * sizeof(float)));
  if (!in) {
    throw std::runtime_error("truncated condition ground truth: " + path);
  }
  return t;
}

}  // namespace fse
