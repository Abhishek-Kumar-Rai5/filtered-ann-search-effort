#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "reachability/csr_graph.h"

namespace faiss {
struct IndexACORNFlat;
}

namespace fse {

struct AcornParams {
  int m = 32;
  int gamma = 1;
  int m_beta = 64;
};

struct AcornQueryResult {
  std::vector<std::int64_t> ids;
  std::vector<float> distances;
  std::uint64_t distance_computations = 0;
  std::uint64_t entries_scanned = 0;
  std::int64_t level0_seed = -1;
  double seconds = 0.0;
};

class AcornIndex {
 public:
  AcornIndex(int dim, const AcornParams& params,
             std::vector<std::int32_t> attributes);
  ~AcornIndex();
  AcornIndex(const AcornIndex&) = delete;
  AcornIndex& operator=(const AcornIndex&) = delete;
  AcornIndex(AcornIndex&&) noexcept;
  AcornIndex& operator=(AcornIndex&&) noexcept;

  static AcornIndex Load(const std::string& path,
                         std::vector<std::int32_t> attributes);
  void Save(const std::string& path) const;

  double Add(std::size_t n, const float* x);

  void SetEfSearch(int ef_search);
  [[nodiscard]] int EfSearch() const;
  [[nodiscard]] int EfConstruction() const;
  [[nodiscard]] std::size_t Size() const;
  [[nodiscard]] int Dim() const;

  [[nodiscard]] int Gamma() const;
  [[nodiscard]] int M() const;
  [[nodiscard]] int MBeta() const;

  [[nodiscard]] CsrGraph Level0Graph() const;

  void SearchBatch(std::size_t nq, const float* queries, std::size_t k,
                   const char* filter_map, std::int64_t* ids,
                   float* distances) const;

  // ACORN keeps its counters and the level-0 seed in process-wide globals,
  // so call this from one thread at a time.
  AcornQueryResult SearchOne(const float* query, std::size_t k,
                             const char* filter_row) const;

  [[nodiscard]] std::vector<double> AverageOutDegreePerLevel() const;

  [[nodiscard]] std::vector<std::size_t> NodesPerLevel() const;

  [[nodiscard]] std::size_t MemoryBytes() const;

 private:
  AcornIndex() = default;
  void AttachAttributes();

  std::vector<int> attributes_;
  std::unique_ptr<faiss::IndexACORNFlat> index_;
};

}
