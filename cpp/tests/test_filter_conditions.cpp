#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/vecs_io.h"
#include "filter_generator/filter_conditions.h"
#include "filter_generator/uniform_attributes.h"
#include "ground_truth/filtered_ground_truth.h"
#include "metrics/local_density.h"

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

fse::FloatMatrix Blobs(std::size_t per_blob, std::size_t blobs, std::size_t dim,
                       std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  fse::FloatMatrix m{std::vector<float>(per_blob * blobs * dim),
                     per_blob * blobs, dim};
  for (std::size_t b = 0; b < blobs; ++b) {
    const std::size_t tier = 1 + (b / dim);
    const float offset = 40.0F * static_cast<float>(tier);
    for (std::size_t i = 0; i < per_blob; ++i) {
      float* row = m.Row(b * per_blob + i);
      for (std::size_t j = 0; j < dim; ++j) {
        row[j] = normal(rng) + (j == b % dim ? offset : 0.0F);
      }
    }
  }
  return m;
}

std::size_t Count(const std::vector<char>& mask) {
  return static_cast<std::size_t>(
      std::count_if(mask.begin(), mask.end(), [](char c) { return c != 0; }));
}

TEST(SeededPermutation, IsADeterministicSeedDependentPermutation) {
  const auto a = fse::SeededPermutation(1000, 5);
  const auto b = fse::SeededPermutation(1000, 5);
  const auto c = fse::SeededPermutation(1000, 6);
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
  std::vector<std::int32_t> sorted = a;
  std::sort(sorted.begin(), sorted.end());
  std::vector<std::int32_t> iota(1000);
  std::iota(iota.begin(), iota.end(), 0);
  EXPECT_EQ(sorted, iota);
  EXPECT_TRUE(fse::SeededPermutation(0, 1).empty());
}

TEST(UniformAttributes, UnchangedByRefactorPinnedToPhase1Draw) {
  const auto v = fse::UniformIntAttributes(8, 1, 12, 20261003);
  EXPECT_EQ(v, (std::vector<std::int32_t>{12, 8, 6, 7, 1, 7, 7, 6}));
}

TEST(FilterConditions, SelectivityIsExactAndLevelsAreNested) {
  const std::size_t n = 20000;
  const auto attr = fse::RandomRankAttribute(n, 11);
  const auto levels = fse::LogSpacedSelectivities(0.01, 1.0, 6);
  ASSERT_EQ(levels.size(), 6U);
  EXPECT_NEAR(levels[0], 0.01, 1e-12);
  EXPECT_NEAR(levels[2], std::pow(10.0, -1.2), 1e-12);
  EXPECT_EQ(levels[5], 1.0);
  std::vector<char> prev(n, 0);
  for (const double s : levels) {
    const auto f = fse::MakeFilterCondition(attr, s, 99);
    EXPECT_EQ(f.threshold, static_cast<std::size_t>(std::llround(s * n)));
    EXPECT_EQ(Count(f.mask), f.threshold);
    EXPECT_LE(std::abs(f.achieved_selectivity - s), 0.5 / n);
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_TRUE(prev[i] == 0 || f.mask[i] != 0);
    }
    prev = f.mask;
  }
  EXPECT_THROW(fse::SelectivityThreshold(n, 0.0), std::invalid_argument);
  EXPECT_THROW(fse::SelectivityThreshold(n, 1.5), std::invalid_argument);
  EXPECT_THROW(fse::SelectivityThreshold(10, 0.01), std::invalid_argument);
  EXPECT_THROW(fse::SelectivityThreshold(n, std::nan("")),
               std::invalid_argument);
}

TEST(FilterConditions, ConditionIdentityDependsOnEveryDefiningInput) {
  const std::size_t n = 5000;
  const auto a = fse::RandomRankAttribute(n, 1);
  const auto b = fse::RandomRankAttribute(n, 2);
  const auto f = fse::MakeFilterCondition(a, 0.1, 7);
  EXPECT_EQ(f.condition_id, fse::MakeFilterCondition(a, 0.1, 7).condition_id);
  EXPECT_EQ(f.mask_hash, fse::MakeFilterCondition(a, 0.1, 7).mask_hash);
  EXPECT_NE(f.condition_id, fse::MakeFilterCondition(a, 0.2, 7).condition_id);
  EXPECT_NE(f.condition_id, fse::MakeFilterCondition(b, 0.1, 7).condition_id);
  EXPECT_NE(f.condition_id, fse::MakeFilterCondition(a, 0.1, 8).condition_id);
  EXPECT_NE(f.mask_hash, fse::MakeFilterCondition(b, 0.1, 7).mask_hash);
  EXPECT_EQ(f.name, "random_s0.1000");
}

TEST(FilterConditions, ClusteredIsDeterministicAndTakesWholeClusters) {
  const auto base = Gaussian(4000, 8, 21);
  const fse::ClusteredParams p{.num_clusters = 40,
                               .kmeans_iterations = 15,
                               .kmeans_seed = 3,
                               .max_points_per_centroid = 256,
                               .order_seed = 9};
  const auto a = fse::ClusteredRankAttribute(base, p);
  const auto b = fse::ClusteredRankAttribute(base, p);
  EXPECT_EQ(a.rank, b.rank);
  EXPECT_EQ(a.cluster, b.cluster);
  EXPECT_EQ(a.content_hash, b.content_hash);

  std::vector<std::int32_t> sorted = a.rank;
  std::sort(sorted.begin(), sorted.end());
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    ASSERT_EQ(sorted[i], static_cast<std::int32_t>(i));
  }

  for (const double s : {0.03, 0.1, 0.5}) {
    const auto f = fse::MakeFilterCondition(a, s, 1);
    std::map<std::int32_t, std::pair<std::size_t, std::size_t>> in_total;
    for (std::size_t i = 0; i < base.rows; ++i) {
      auto& e = in_total[a.cluster[i]];
      e.first += f.mask[i] != 0 ? 1 : 0;
      ++e.second;
    }
    std::size_t partial = 0;
    for (const auto& [c, e] : in_total) {
      partial += (e.first > 0 && e.first < e.second) ? 1 : 0;
    }
    EXPECT_LE(partial, 1U) << "s=" << s;
  }

  fse::ClusteredParams q = p;
  q.order_seed = 10;
  EXPECT_NE(fse::ClusteredRankAttribute(base, q).content_hash, a.content_hash);
}

TEST(FilterConditions, ClusteredConcentratesLocallyRandomDoesNot) {
  const auto base = Blobs(200, 20, 16, 31);
  const auto queries = Blobs(10, 20, 16, 32);
  const auto unfiltered = fse::UnfilteredGroundTruth(base, queries, 10);
  const fse::ClusteredParams p{.num_clusters = 20,
                               .kmeans_iterations = 20,
                               .kmeans_seed = 1,
                               .max_points_per_centroid = 256,
                               .order_seed = 2};
  const auto clustered =
      fse::MakeFilterCondition(fse::ClusteredRankAttribute(base, p), 0.1, 0);
  const auto random =
      fse::MakeFilterCondition(fse::RandomRankAttribute(base.rows, 3), 0.1, 0);
  const auto dc = fse::LocalFilteredDensity(unfiltered, clustered.mask, 10);
  const auto dr = fse::LocalFilteredDensity(unfiltered, random.mask, 10);
  auto mean_var = [](const std::vector<double>& v) {
    double m = 0.0;
    for (const double x : v) {
      m += x / static_cast<double>(v.size());
    }
    double var = 0.0;
    for (const double x : v) {
      var += (x - m) * (x - m) / static_cast<double>(v.size());
    }
    return std::pair<double, double>{m, var};
  };
  std::size_t c_extreme = 0;
  for (const double x : dc) {
    c_extreme += (x == 0.0 || x == 1.0) ? 1 : 0;
  }
  const auto [r_mean, r_var] = mean_var(dr);
  const auto [c_mean, c_var] = mean_var(dc);
  EXPECT_NEAR(r_mean, 0.1, 0.05);
  EXPECT_LT(r_var, 0.02);
  EXPECT_GT(c_var, 5.0 * r_var);
  EXPECT_GE(c_extreme, dc.size() * 9 / 10);
  (void)c_mean;
}

TEST(FilterConditions, RankAttributeRoundTripIsVerified) {
  const auto base = Gaussian(2000, 8, 51);
  const fse::ClusteredParams p{.num_clusters = 20,
                               .kmeans_iterations = 10,
                               .kmeans_seed = 1,
                               .max_points_per_centroid = 256,
                               .order_seed = 2};
  const auto a = fse::ClusteredRankAttribute(base, p);
  EXPECT_EQ(fse::RankAttributeHash(a), a.content_hash);
  const std::string path = ::testing::TempDir() + "/fse_attr.bin";
  fse::WriteRankAttribute(path, a);
  const auto r = fse::ReadRankAttribute(path, a.content_hash);
  EXPECT_EQ(r.rank, a.rank);
  EXPECT_EQ(r.cluster, a.cluster);
  EXPECT_EQ(r.cluster_order, a.cluster_order);
  EXPECT_EQ(r.correlation, fse::Correlation::kClustered);
  EXPECT_THROW(fse::ReadRankAttribute(path, a.content_hash + 1),
               std::runtime_error);

  EXPECT_EQ(fse::MakeFilterCondition(r, 0.1, 5).condition_id,
            fse::MakeFilterCondition(a, 0.1, 5).condition_id);

  {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(40);
    const char x = 0x55;
    f.write(&x, 1);
  }
  EXPECT_THROW(fse::ReadRankAttribute(path), std::runtime_error);
  const auto rnd = fse::RandomRankAttribute(500, 9);
  fse::WriteRankAttribute(path, rnd);
  EXPECT_EQ(fse::ReadRankAttribute(path).rank, rnd.rank);
  std::filesystem::remove(path);
}

TEST(GroundTruth, MaskVariantMatchesEqualityReference) {
  const auto base = Gaussian(3000, 12, 41);
  const auto queries = Gaussian(40, 12, 42);
  const auto f =
      fse::MakeFilterCondition(fse::RandomRankAttribute(3000, 4), 0.07, 0);
  std::vector<std::int32_t> attr(base.rows);
  for (std::size_t i = 0; i < base.rows; ++i) {
    attr[i] = f.mask[i] != 0 ? 1 : 0;
  }
  const auto a = fse::FilteredGroundTruthMask(base, queries, f.mask, 10);
  const auto b = fse::FilteredGroundTruthEquals(
      base, queries, attr, std::vector<std::int32_t>(queries.rows, 1), 10);
  EXPECT_EQ(a.ids, b.ids);
  EXPECT_EQ(a.distances, b.distances);
  EXPECT_THROW(fse::FilteredGroundTruthMask(base, queries, {1, 0}, 10),
               std::invalid_argument);
}

TEST(GroundTruth, StoreRejectsStaleOrWrongConditionGroundTruth) {
  const fse::NeighborTable t{2, 2, {1, 2, 3, -1}, {0.5F, 1, 2, 3}};
  const fse::GroundTruthIdentity id{
      .condition_id = 1, .mask_hash = 2, .query_hash = 3};
  const std::string path = ::testing::TempDir() + "/fse_cond_gt.bin";
  fse::WriteConditionGroundTruth(path, id, t);
  const auto r = fse::ReadConditionGroundTruth(path, id);
  EXPECT_EQ(r.ids, t.ids);
  EXPECT_EQ(r.distances, t.distances);
  for (const fse::GroundTruthIdentity& bad :
       {fse::GroundTruthIdentity{9, 2, 3}, fse::GroundTruthIdentity{1, 9, 3},
        fse::GroundTruthIdentity{1, 2, 9}}) {
    EXPECT_THROW(fse::ReadConditionGroundTruth(path, bad), std::runtime_error);
  }

  fse::WriteNeighborTable(path, t);
  EXPECT_THROW(fse::ReadConditionGroundTruth(path, id), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(LocalDensity, FractionOfTrueNeighboursPassing) {
  const fse::NeighborTable gt{
      2, 4, {0, 1, 2, 3, 4, 5, 6, 7}, std::vector<float>(8, 0.0F)};
  const std::vector<char> mask = {1, 0, 1, 0, 0, 0, 0, 1};
  EXPECT_EQ(fse::LocalFilteredDensity(gt, mask, 4),
            (std::vector<double>{0.5, 0.25}));
  EXPECT_EQ(fse::LocalFilteredDensity(gt, mask, 2),
            (std::vector<double>{0.5, 0.0}));
  EXPECT_THROW(fse::LocalFilteredDensity(gt, mask, 5), std::invalid_argument);
  EXPECT_THROW(fse::LocalFilteredDensity(gt, mask, 0), std::invalid_argument);
}

}
