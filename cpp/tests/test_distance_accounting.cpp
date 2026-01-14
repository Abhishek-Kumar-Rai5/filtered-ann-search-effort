#include <faiss/IndexACORN.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/ACORN.h>
#include <faiss/impl/DistanceComputer.h>
#include <gtest/gtest.h>
#include <omp.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "common/vecs_io.h"
#include "filter_generator/uniform_attributes.h"
#include "methods/ingraph/acorn_accounting.h"
#include "methods/postfilter/hnsw_index.h"

namespace {

struct CountingDistanceComputer : faiss::DistanceComputer {
  explicit CountingDistanceComputer(faiss::DistanceComputer* inner,
                                    std::uint64_t* count)
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
  faiss::DistanceComputer* get_distance_computer() const override {
    return new CountingDistanceComputer(
        faiss::IndexFlat::get_distance_computer(), &count);
  }
  mutable std::uint64_t count = 0;
};

constexpr int kDim = 16;
constexpr faiss::idx_t kN = 3000;
constexpr std::size_t kNq = 60;
constexpr faiss::idx_t kK = 10;

std::vector<float> Gaussian(std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0F, 1.0F);
  std::vector<float> v(n * kDim);
  for (float& x : v) {
    x = d(rng);
  }
  return v;
}

struct Built {
  std::vector<int> metadata;
  std::unique_ptr<CountingFlatL2> storage;
  std::unique_ptr<faiss::IndexACORN> counted;
  std::unique_ptr<faiss::IndexACORNFlat> plain;
};

void BuildBoth(int gamma, const std::vector<float>& base, Built* b) {
  b->metadata.assign(kN, 0);
  omp_set_num_threads(1);
  b->storage = std::make_unique<CountingFlatL2>(kDim);
  b->counted = std::make_unique<faiss::IndexACORN>(b->storage.get(), 16, gamma,
                                                   b->metadata, 32);
  b->counted->add(kN, base.data());
  b->plain =
      std::make_unique<faiss::IndexACORNFlat>(kDim, 16, gamma, b->metadata, 32);
  b->plain->add(kN, base.data());
}

TEST(DistanceAccounting, AcornUncountsExactlyTheEntryPointPerQuery) {
  const auto base = Gaussian(kN, 1);
  const auto queries = Gaussian(kNq, 2);
  std::size_t checked = 0;
  for (const int gamma : {4, 1}) {
    Built b;
    BuildBoth(gamma, base, &b);
    ASSERT_EQ(b.counted->ntotal, kN);
    for (const int values : {1, 4, 25}) {
      const auto attr = fse::UniformIntAttributes(kN, 1, values, 3);
      for (const int efs : {10, 40, 160}) {
        b.counted->acorn.efSearch = efs;
        b.plain->acorn.efSearch = efs;
        std::uint64_t offset_min = UINT64_MAX;
        std::uint64_t offset_max = 0;
        for (std::size_t q = 0; q < kNq; ++q) {
          std::vector<char> row(kN);
          const auto target = static_cast<int>(1 + q % values);
          for (faiss::idx_t i = 0; i < kN; ++i) {
            row[i] = static_cast<char>(attr[i] == target);
          }
          const float* qv = &queries[q * kDim];
          std::vector<faiss::idx_t> ids_c(kK);
          std::vector<faiss::idx_t> ids_p(kK);
          std::vector<float> d_c(kK);
          std::vector<float> d_p(kK);

          b.storage->count = 0;
          const std::size_t n3_c0 = faiss::acorn_stats.n3;
          b.counted->search(1, qv, kK, d_c.data(), ids_c.data(), row.data());
          const std::uint64_t native_c = faiss::acorn_stats.n3 - n3_c0;
          const std::uint64_t exact = b.storage->count;

          const std::size_t n3_p0 = faiss::acorn_stats.n3;
          b.plain->search(1, qv, kK, d_p.data(), ids_p.data(), row.data());
          const std::uint64_t native_p = faiss::acorn_stats.n3 - n3_p0;

          ASSERT_EQ(ids_c, ids_p);
          ASSERT_EQ(native_c, native_p);

          ASSERT_EQ(exact - native_c, fse::kAcornUncountedPerQuery)
              << "gamma=" << gamma << " values=" << values << " efs=" << efs
              << " q=" << q << " native=" << native_c << " exact=" << exact;
          ASSERT_EQ(fse::AcornExactDistanceComputations(native_c), exact);
          offset_min = std::min(offset_min, exact - native_c);
          offset_max = std::max(offset_max, exact - native_c);
          ++checked;
        }
        std::cout << "[audit] ACORN gamma=" << gamma << " s=1/" << values
                  << " efs=" << efs << ": exact - native in [" << offset_min
                  << ", " << offset_max << "] over " << kNq << " queries\n";
      }
    }
  }
  EXPECT_EQ(checked, std::size_t{2} * 3 * 3 * kNq);
}

TEST(DistanceAccounting, HnswlibCounterIncludesEntryPointEvaluations) {
  const auto one = Gaussian(1, 4);
  fse::FloatMatrix base{one, 1, kDim};
  fse::HnswIndex index(kDim, 1, fse::HnswParams{.m = 16, .seed = 1});
  index.Add(base, 0, 1);
  const auto q = Gaussian(1, 5);
  const auto r = index.Search(q.data(), 1, 10);
  ASSERT_EQ(r.labels.size(), 1U);
  EXPECT_EQ(r.distance_computations, 2U);
}

}
