#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/vecs_io.h"
#include "hnswlib/hnswlib.h"
#include "reachability/csr_graph.h"

namespace fse {

// Ported from Project 1 (ann_router_staleness, ars::HnswIndex), trimmed to
// what the post-filter baseline needs.
//
// L2 space that delegates to hnswlib's own (SIMD) L2 distance and counts
// every distance evaluation into a thread-local counter. hnswlib's built-in
// metric_distance_computations is a shared atomic that also counts neighbours
// skipped as already-visited, so it is neither per-query nor exact; this
// wrapper is both, without modifying hnswlib. Upper-layer distances count.
class CountingL2Space : public hnswlib::SpaceInterface<float> {
 public:
  explicit CountingL2Space(std::size_t dim);

  std::size_t get_data_size() override;
  hnswlib::DISTFUNC<float> get_dist_func() override;
  void* get_dist_func_param() override;

  // Distance evaluations performed by the calling thread since its last reset.
  static std::uint64_t& ThreadCount();

 private:
  struct Param {
    hnswlib::DISTFUNC<float> inner;
    void* inner_param;
  };
  static float CountingDistance(const void* a, const void* b, const void* p);

  hnswlib::L2Space inner_;
  Param param_{};
};

struct HnswParams {
  std::size_t m = 16;
  std::size_t ef_construction = 200;
  // Seeds hnswlib's level-assignment RNG. The build is only bit-reproducible
  // for a fixed seed when inserted single-threaded (see Add()).
  std::uint64_t seed = 0;
};

struct HnswSearchResult {
  std::vector<std::size_t> labels;  // ascending distance
  std::vector<float> dists;         // squared L2
  std::uint64_t distance_computations = 0;
};

// Thin owner of an hnswlib index. hnswlib is used strictly as a black box:
// only its public insert / search / save / load API is called.
class HnswIndex {
 public:
  HnswIndex(std::size_t dim, std::size_t max_elements, const HnswParams& p);
  static HnswIndex Load(const std::string& path, std::size_t dim);

  // Inserts vectors.Row(i) with label first_label + i. num_threads > 1 is
  // faster but makes the graph depend on thread scheduling (non-reproducible).
  void Add(const FloatMatrix& vectors, std::size_t first_label,
           int num_threads);
  void Save(const std::string& path);

  // Index-wide efSearch. hnswlib searches with max(ef, k). Not thread-safe
  // with concurrent searches; set once, then search.
  void SetEf(std::size_t ef);
  [[nodiscard]] std::size_t Ef() const;

  // Top-k search at the current ef, with the exact number of distance
  // evaluations of this search (upper layers included). Thread-safe:
  // concurrent calls are fine while ef is not being changed.
  [[nodiscard]] HnswSearchResult SearchAtCurrentEf(const float* query,
                                                   std::size_t k) const;
  // Convenience: SetEf(ef) then SearchAtCurrentEf (single-threaded use).
  HnswSearchResult Search(const float* query, std::size_t k, std::size_t ef);

  [[nodiscard]] std::size_t Size() const;
  // Read-only copy of the level-0 lists, indexed and labelled by external
  // label (= base id), stored order (docs/structural_design.md §3.2 POST).
  [[nodiscard]] CsrGraph Level0Graph() const;
  [[nodiscard]] std::size_t Dim() const { return dim_; }

 private:
  HnswIndex() = default;

  std::size_t dim_ = 0;
  std::unique_ptr<CountingL2Space> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_;
};

}  // namespace fse
