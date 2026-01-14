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

class CountingL2Space : public hnswlib::SpaceInterface<float> {
 public:
  explicit CountingL2Space(std::size_t dim);

  std::size_t get_data_size() override;
  hnswlib::DISTFUNC<float> get_dist_func() override;
  void* get_dist_func_param() override;

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

  std::uint64_t seed = 0;
};

struct HnswSearchResult {
  std::vector<std::size_t> labels;
  std::vector<float> dists;
  std::uint64_t distance_computations = 0;
};

class HnswIndex {
 public:
  HnswIndex(std::size_t dim, std::size_t max_elements, const HnswParams& p);
  static HnswIndex Load(const std::string& path, std::size_t dim);

  void Add(const FloatMatrix& vectors, std::size_t first_label,
           int num_threads);
  void Save(const std::string& path);

  void SetEf(std::size_t ef);
  [[nodiscard]] std::size_t Ef() const;

  [[nodiscard]] HnswSearchResult SearchAtCurrentEf(const float* query,
                                                   std::size_t k) const;

  HnswSearchResult Search(const float* query, std::size_t k, std::size_t ef);

  [[nodiscard]] std::size_t Size() const;

  [[nodiscard]] CsrGraph Level0Graph() const;
  [[nodiscard]] std::size_t Dim() const { return dim_; }

 private:
  HnswIndex() = default;

  std::size_t dim_ = 0;
  std::unique_ptr<CountingL2Space> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_;
};

}
