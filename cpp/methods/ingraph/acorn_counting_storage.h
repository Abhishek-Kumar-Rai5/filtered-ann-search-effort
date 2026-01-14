#pragma once

#include <faiss/IndexFlat.h>
#include <faiss/impl/DistanceComputer.h>

#include <cstdint>
#include <memory>

namespace fse {

struct CountingDistanceComputer : faiss::DistanceComputer {
  CountingDistanceComputer(faiss::DistanceComputer* inner, std::uint64_t* count)
      : inner(inner), count(count) {}
  void set_query(const float* x) override { inner->set_query(x); }
  float operator()(faiss::idx_t i) override {
    ++*count;
    return (*inner)(i);
  }
  float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
    ++*count;
    return inner->symmetric_dis(i, j);
  }
  std::unique_ptr<faiss::DistanceComputer> inner;
  std::uint64_t* count;
};

struct CountingFlatL2 : faiss::IndexFlat {
  explicit CountingFlatL2(faiss::idx_t d) : faiss::IndexFlat(d) {}
  [[nodiscard]] faiss::DistanceComputer* get_distance_computer()
      const override {
    return new CountingDistanceComputer(
        faiss::IndexFlat::get_distance_computer(), &count);
  }
  mutable std::uint64_t count = 0;
};

}
