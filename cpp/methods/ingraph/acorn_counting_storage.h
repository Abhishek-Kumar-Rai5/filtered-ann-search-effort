#pragma once

#include <faiss/IndexFlat.h>
#include <faiss/impl/DistanceComputer.h>

#include <cstdint>
#include <memory>

namespace fse {

// Independent distance counter for ACORN audits (docs/phase2_baselines.md
// section 12; docs/phase4_matrix.md V5): an IndexFlat (L2) whose distance
// computers count every query-to-vector and vector-to-vector evaluation.
// Passed to ACORN through its public IndexACORN(Index* storage, ...)
// constructor -- IndexACORNFlat is exactly IndexACORN(new IndexFlat(d), ...)
// -- so ACORN itself is not modified. Single-threaded use only.
// (Same construction as the helper in cpp/tests/test_distance_accounting.cpp,
// which is kept unchanged as part of the Phase 2 record.)
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

}  // namespace fse
