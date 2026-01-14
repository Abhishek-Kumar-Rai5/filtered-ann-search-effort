// hnswlib wrapper (ported from Project 1): search correctness against brute
// force, exact per-query distance counting, thread safety at a fixed ef,
// save/load round trip, and seeded reproducibility.

#include <gtest/gtest.h>
#include <omp.h>

#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "common/vecs_io.h"
#include "ground_truth/filtered_ground_truth.h"
#include "methods/postfilter/hnsw_index.h"
#include "metrics/recall.h"

namespace {

fse::FloatMatrix Gaussian(std::size_t rows, std::size_t dim,
                          std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  fse::FloatMatrix m{std::vector<float>(rows * dim), rows, dim};
  for (float& x : m.data) {
    x = normal(rng);
  }
  return m;
}

double MeanRecall(fse::HnswIndex& index, const fse::FloatMatrix& queries,
                  const fse::NeighborTable& gt, std::size_t k, std::size_t ef) {
  double sum = 0.0;
  for (std::size_t q = 0; q < queries.rows; ++q) {
    const auto r = index.Search(queries.Row(q), k, ef);
    const std::vector<std::int64_t> ids(r.labels.begin(), r.labels.end());
    sum += fse::RecallAtK(ids.data(), gt.Ids(q), k);
  }
  return sum / static_cast<double>(queries.rows);
}

constexpr fse::HnswParams kParams{.m = 16, .ef_construction = 200, .seed = 7};

TEST(CountingL2Space, MatchesReferenceDistanceAndCounts) {
  const auto data = Gaussian(2, 32, 1);
  fse::CountingL2Space space(32);
  auto fn = space.get_dist_func();
  fse::CountingL2Space::ThreadCount() = 0;
  const float d = fn(data.Row(0), data.Row(1), space.get_dist_func_param());
  EXPECT_NEAR(d, fse::SquaredL2(data.Row(0), data.Row(1), 32), 1e-4);
  (void)fn(data.Row(0), data.Row(0), space.get_dist_func_param());
  EXPECT_EQ(fse::CountingL2Space::ThreadCount(), 2U);
}

TEST(HnswIndex, ReturnsSortedResultsWithConsistentDistances) {
  const auto base = Gaussian(2000, 16, 2);
  const auto queries = Gaussian(20, 16, 3);
  fse::HnswIndex index(16, base.rows, kParams);
  index.Add(base, 0, 1);
  ASSERT_EQ(index.Size(), base.rows);
  for (std::size_t q = 0; q < queries.rows; ++q) {
    const auto r = index.Search(queries.Row(q), 10, 50);
    ASSERT_EQ(r.labels.size(), 10U);
    for (std::size_t i = 0; i < r.labels.size(); ++i) {
      EXPECT_NEAR(r.dists[i],
                  fse::SquaredL2(queries.Row(q), base.Row(r.labels[i]), 16),
                  1e-3);
      if (i > 0) {
        EXPECT_LE(r.dists[i - 1], r.dists[i]);
      }
    }
  }
}

TEST(HnswIndex, RecallReachesOneAtLargeEfAndIsMonotoneInEf) {
  const auto base = Gaussian(5000, 16, 4);
  const auto queries = Gaussian(200, 16, 5);
  const auto gt = fse::UnfilteredGroundTruth(base, queries, 10);
  fse::HnswIndex index(16, base.rows, kParams);
  index.Add(base, 0, 1);
  const double low = MeanRecall(index, queries, gt, 10, 10);
  const double high = MeanRecall(index, queries, gt, 10, 500);
  EXPECT_LT(low, high);
  EXPECT_GE(high, 0.999);
}

TEST(HnswIndex, DistanceCountIsPerQueryAndGrowsWithEf) {
  const auto base = Gaussian(5000, 16, 6);
  const auto queries = Gaussian(50, 16, 7);
  fse::HnswIndex index(16, base.rows, kParams);
  index.Add(base, 0, 1);
  for (std::size_t q = 0; q < queries.rows; ++q) {
    const auto small = index.Search(queries.Row(q), 10, 10);
    const auto large = index.Search(queries.Row(q), 10, 400);
    EXPECT_GT(small.distance_computations, 0U);
    EXPECT_LT(small.distance_computations, base.rows);
    EXPECT_GT(large.distance_computations, small.distance_computations);
    // Counter is reset per search: repeating gives the identical count.
    EXPECT_EQ(index.Search(queries.Row(q), 10, 10).distance_computations,
              small.distance_computations);
  }
}

TEST(HnswIndex, ConcurrentSearchesAtFixedEfMatchSerial) {
  const auto base = Gaussian(3000, 16, 13);
  const auto queries = Gaussian(64, 16, 14);
  fse::HnswIndex index(16, base.rows, kParams);
  index.Add(base, 0, 1);
  index.SetEf(60);
  EXPECT_EQ(index.Ef(), 60U);
  std::vector<fse::HnswSearchResult> serial(queries.rows);
  for (std::size_t q = 0; q < queries.rows; ++q) {
    serial[q] = index.SearchAtCurrentEf(queries.Row(q), 25);
  }
  std::vector<fse::HnswSearchResult> parallel(queries.rows);
#pragma omp parallel for num_threads(8)
  for (std::int64_t q = 0; q < static_cast<std::int64_t>(queries.rows); ++q) {
    parallel[q] = index.SearchAtCurrentEf(queries.Row(q), 25);
  }
  for (std::size_t q = 0; q < queries.rows; ++q) {
    EXPECT_EQ(parallel[q].labels, serial[q].labels);
    EXPECT_EQ(parallel[q].distance_computations,
              serial[q].distance_computations);
  }
}

TEST(HnswIndex, SingleThreadedBuildIsReproducibleForFixedSeed) {
  const auto base = Gaussian(3000, 16, 8);
  const auto queries = Gaussian(50, 16, 9);
  fse::HnswIndex a(16, base.rows, kParams);
  fse::HnswIndex b(16, base.rows, kParams);
  a.Add(base, 0, 1);
  b.Add(base, 0, 1);
  for (std::size_t q = 0; q < queries.rows; ++q) {
    const auto ra = a.Search(queries.Row(q), 10, 20);
    const auto rb = b.Search(queries.Row(q), 10, 20);
    EXPECT_EQ(ra.labels, rb.labels);
    EXPECT_EQ(ra.distance_computations, rb.distance_computations);
  }
}

TEST(HnswIndex, SaveLoadRoundTripGivesIdenticalSearches) {
  const auto base = Gaussian(2000, 16, 10);
  const auto queries = Gaussian(30, 16, 11);
  fse::HnswIndex index(16, base.rows, kParams);
  index.Add(base, 0, 4);
  const std::string path = ::testing::TempDir() + "/fse_hnsw_roundtrip.bin";
  index.Save(path);
  fse::HnswIndex loaded = fse::HnswIndex::Load(path, 16);
  std::filesystem::remove(path);
  ASSERT_EQ(loaded.Size(), index.Size());
  for (std::size_t q = 0; q < queries.rows; ++q) {
    const auto r1 = index.Search(queries.Row(q), 10, 40);
    const auto r2 = loaded.Search(queries.Row(q), 10, 40);
    EXPECT_EQ(r1.labels, r2.labels);
    EXPECT_EQ(r1.dists, r2.dists);
    EXPECT_EQ(r1.distance_computations, r2.distance_computations);
  }
}

TEST(HnswIndex, LabelsHonourFirstLabelOffset) {
  const auto base = Gaussian(100, 8, 12);
  fse::HnswIndex index(8, base.rows, kParams);
  index.Add(base, 1000, 1);
  const auto r = index.Search(base.Row(17), 1, 50);
  ASSERT_EQ(r.labels.size(), 1U);
  EXPECT_EQ(r.labels[0], 1017U);
}

}  // namespace
