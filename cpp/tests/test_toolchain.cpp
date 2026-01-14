#include <gtest/gtest.h>
#include <omp.h>
#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <random>
#include <vector>

#include "common/build_info.h"
#include "hnswlib/hnswlib.h"

#ifdef FSE_WITH_ACORN
#include <faiss/IndexACORN.h>
#endif

namespace {

std::vector<float> RandomVectors(std::size_t n, int dim, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> data(n * dim);
  for (float& x : data) {
    x = dist(rng);
  }
  return data;
}

TEST(Toolchain, BuildInfoIsPopulated) {
  const fse::BuildInfo info = fse::GetBuildInfo();
  EXPECT_FALSE(info.compiler_id.empty());
  EXPECT_FALSE(info.compiler_version.empty());
  EXPECT_GE(info.cxx_standard, 202002L);
  EXPECT_GT(info.openmp_version, 0);
  EXPECT_FALSE(info.hnswlib_commit.empty());
  EXPECT_FALSE(info.acorn_commit.empty());
}

TEST(Toolchain, OpenMPRunsParallelRegion) {
  int sum = 0;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < 1000; ++i) {
    sum += 1;
  }
  EXPECT_EQ(sum, 1000);
  EXPECT_GE(omp_get_max_threads(), 1);
}

TEST(Toolchain, YamlCppParses) {
  const YAML::Node node = YAML::Load("selectivity: [0.01, 1.0]\nseed: 7");
  EXPECT_EQ(node["seed"].as<int>(), 7);
  EXPECT_EQ(node["selectivity"].size(), 2U);
}

class EvenLabelsOnly : public hnswlib::BaseFilterFunctor {
 public:
  bool operator()(hnswlib::labeltype id) override { return id % 2 == 0; }
};

TEST(HnswlibSmoke, FilterCallbackReturnsOnlyAllowedLabels) {
  constexpr int kDim = 16;
  constexpr std::size_t kN = 500;
  constexpr std::size_t kK = 10;
  const std::vector<float> data = RandomVectors(kN, kDim, 42);

  hnswlib::L2Space space(kDim);
  hnswlib::HierarchicalNSW<float> index(&space, kN, 16, 200, 42);
  for (std::size_t i = 0; i < kN; ++i) {
    index.addPoint(&data[i * kDim], i);
  }
  index.setEf(100);

  EvenLabelsOnly filter;
  for (std::size_t q = 0; q < 20; ++q) {
    auto result = index.searchKnn(&data[q * kDim], kK, &filter);
    ASSERT_EQ(result.size(), kK);
    while (!result.empty()) {
      EXPECT_EQ(result.top().second % 2, 0U);
      result.pop();
    }
  }
}

#ifdef FSE_WITH_ACORN

TEST(AcornSmoke, FilteredSearchReturnsOnlyPassingIds) {
  constexpr int kDim = 16;
  constexpr faiss::idx_t kN = 500;
  constexpr faiss::idx_t kNq = 5;
  constexpr faiss::idx_t kK = 5;
  const std::vector<float> data = RandomVectors(kN, kDim, 42);

  std::vector<int> metadata(kN);
  for (faiss::idx_t i = 0; i < kN; ++i) {
    metadata[i] = static_cast<int>(i % 2);
  }
  faiss::IndexACORNFlat index(kDim, 16, 2, metadata, 16);
  index.add(kN, data.data());

  std::vector<char> filter_id_map(kNq * kN);
  for (faiss::idx_t q = 0; q < kNq; ++q) {
    for (faiss::idx_t i = 0; i < kN; ++i) {
      filter_id_map[q * kN + i] = static_cast<char>(i % 2 == 0);
    }
  }
  std::vector<float> distances(kNq * kK);
  std::vector<faiss::idx_t> labels(kNq * kK);
  index.search(kNq, data.data(), kK, distances.data(), labels.data(),
               filter_id_map.data());

  for (const faiss::idx_t label : labels) {
    ASSERT_GE(label, 0);
    EXPECT_EQ(label % 2, 0);
  }
}
#endif

}
