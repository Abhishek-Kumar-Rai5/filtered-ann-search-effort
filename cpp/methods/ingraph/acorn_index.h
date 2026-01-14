#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "reachability/csr_graph.h"

namespace faiss {
struct IndexACORNFlat;
}  // namespace faiss

namespace fse {

struct AcornParams {
  int m = 32;
  int gamma = 1;
  int m_beta = 64;
  // efConstruction is left at ACORN's own default (M * gamma); see
  // docs/phase1_acorn_repro.md section 3.3 for why.
};

// Result of one filtered query, including ACORN's own effort counter.
struct AcornQueryResult {
  std::vector<std::int64_t> ids;            // k entries, -1 = no result
  std::vector<float> distances;             // squared L2
  std::uint64_t distance_computations = 0;  // delta of acorn_stats.n3
  std::uint64_t entries_scanned = 0;        // delta of acorn_stats.n_scanned
  std::int64_t level0_seed = -1;  // faiss::acorn_level0_seed (structural §3.4)
  double seconds = 0.0;
};

// Thin, behaviour-preserving wrapper around faiss::IndexACORNFlat (ACORN is
// a black box, CLAUDE.md). It exists for three project-side reasons:
//  1. ACORN stores a raw pointer to the caller's attribute array (read during
//     construction) and read_index leaves that pointer dangling; the wrapper
//     owns the array and (re)attaches it.
//  2. Per-query distance counts are only exposed through the global
//     faiss::acorn_stats; the wrapper reads its delta around a single-query
//     search.
//  3. Structural statistics (out-degree per level, memory) computed from
//     ACORN's public fields.
class AcornIndex {
 public:
  // Empty index. `attributes` must have one entry per vector later added.
  AcornIndex(int dim, const AcornParams& params,
             std::vector<std::int32_t> attributes);
  ~AcornIndex();
  AcornIndex(const AcornIndex&) = delete;
  AcornIndex& operator=(const AcornIndex&) = delete;
  AcornIndex(AcornIndex&&) noexcept;
  AcornIndex& operator=(AcornIndex&&) noexcept;

  // Loads an index written by Save(), attaching `attributes`.
  static AcornIndex Load(const std::string& path,
                         std::vector<std::int32_t> attributes);
  void Save(const std::string& path) const;

  // index.add(n, x); returns wall seconds. Uses the current OpenMP threads.
  double Add(std::size_t n, const float* x);

  void SetEfSearch(int ef_search);
  [[nodiscard]] int EfSearch() const;
  [[nodiscard]] int EfConstruction() const;
  [[nodiscard]] std::size_t Size() const;
  [[nodiscard]] int Dim() const;
  // Frozen build parameters as stored in the index.
  [[nodiscard]] int Gamma() const;
  [[nodiscard]] int M() const;
  [[nodiscard]] int MBeta() const;
  // Read-only copy of the level-0 neighbour lists in stored order (-1
  // padding dropped); node ids = base ids (docs/structural_design.md §3).
  [[nodiscard]] CsrGraph Level0Graph() const;

  // ACORN's batch filtered search. filter_map is nq x Size() bytes,
  // row-major [query][base], nonzero = passes. ids/distances: nq x k.
  void SearchBatch(std::size_t nq, const float* queries, std::size_t k,
                   const char* filter_map, std::int64_t* ids,
                   float* distances) const;

  // One query (n = 1 call), with ACORN's distance-computation counter delta.
  // The counter is a global updated without synchronisation: call this from
  // one thread only.
  AcornQueryResult SearchOne(const float* query, std::size_t k,
                             const char* filter_row) const;

  // Average number of distinct valid neighbours per node, for nodes present
  // at each level (ACORN's print_neighbor_stats definition), level 0..max.
  [[nodiscard]] std::vector<double> AverageOutDegreePerLevel() const;
  // Number of nodes present at each level.
  [[nodiscard]] std::vector<std::size_t> NodesPerLevel() const;
  // Bytes of vector storage plus graph arrays (neighbors, offsets, levels).
  [[nodiscard]] std::size_t MemoryBytes() const;

 private:
  AcornIndex() = default;
  void AttachAttributes();

  // Declared before index_ so it outlives every use by the index.
  std::vector<int> attributes_;
  std::unique_ptr<faiss::IndexACORNFlat> index_;
};

}  // namespace fse
