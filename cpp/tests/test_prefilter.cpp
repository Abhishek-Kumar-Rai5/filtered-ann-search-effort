#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common/vecs_io.h"
#include "filter_generator/uniform_attributes.h"
#include "ground_truth/filtered_ground_truth.h"
#include "methods/prefilter/prefilter.h"
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

fse::FloatMatrix SmallIntegers(std::size_t rows, std::size_t dim,
                               std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> u(0, 2);
  fse::FloatMatrix m{std::vector<float>(rows * dim), rows, dim};
  for (float& x : m.data) {
    x = static_cast<float>(u(rng));
  }
  return m;
}

std::vector<char> MaskFor(const std::vector<std::int32_t>& base_attr,
                          std::int32_t query_attr) {
  std::vector<char> m(base_attr.size());
  for (std::size_t i = 0; i < base_attr.size(); ++i) {
    m[i] = static_cast<char>(base_attr[i] == query_attr);
  }
  return m;
}

TEST(Prefilter, MatchesFilteredGroundTruthExactly) {
  const auto base = Gaussian(3000, 16, 1);
  const auto queries = Gaussian(100, 16, 2);
  const auto base_attr = fse::UniformIntAttributes(base.rows, 1, 4, 3);
  const auto query_attr = fse::UniformIntAttributes(queries.rows, 1, 4, 4);
  constexpr std::size_t kK = 10;
  const auto gt =
      fse::FilteredGroundTruthEquals(base, queries, base_attr, query_attr, kK);
  for (std::size_t q = 0; q < queries.rows; ++q) {
    const auto mask = MaskFor(base_attr, query_attr[q]);
    const auto r = fse::PrefilterSearch(base, queries.Row(q), kK, mask.data());
    EXPECT_EQ(r.ids, std::vector<std::int64_t>(gt.Ids(q), gt.Ids(q) + kK));
    EXPECT_EQ(r.distances,
              std::vector<float>(gt.Distances(q), gt.Distances(q) + kK));
    EXPECT_DOUBLE_EQ(fse::RecallAtK(r.ids.data(), gt.Ids(q), kK), 1.0);
  }
}

TEST(Prefilter, MatchesNaiveReferenceWithExactTies) {
  const auto base = SmallIntegers(1500, 6, 5);
  const auto queries = SmallIntegers(40, 6, 6);
  const auto base_attr = fse::UniformIntAttributes(base.rows, 1, 3, 7);
  constexpr std::size_t kK = 12;
  for (std::size_t q = 0; q < queries.rows; ++q) {
    const auto qa = static_cast<std::int32_t>(1 + q % 3);
    const auto mask = MaskFor(base_attr, qa);
    std::vector<std::pair<float, std::int64_t>> all;
    for (std::size_t i = 0; i < base.rows; ++i) {
      if (mask[i] != 0) {
        float d = 0;
        for (std::size_t j = 0; j < base.dim; ++j) {
          const float diff = queries.Row(q)[j] - base.Row(i)[j];
          d += diff * diff;
        }
        all.emplace_back(d, static_cast<std::int64_t>(i));
      }
    }
    std::sort(all.begin(), all.end());
    const auto r = fse::PrefilterSearch(base, queries.Row(q), kK, mask.data());
    for (std::size_t j = 0; j < kK; ++j) {
      EXPECT_EQ(r.ids[j], all[j].second) << "q=" << q << " j=" << j;
      EXPECT_EQ(r.distances[j], all[j].first);
    }
  }
}

TEST(Prefilter, ReturnsOnlyPassingIdsInAscendingDistance) {
  const auto base = Gaussian(2000, 8, 8);
  const auto queries = Gaussian(30, 8, 9);
  const auto base_attr = fse::UniformIntAttributes(base.rows, 1, 20, 10);
  for (std::size_t q = 0; q < queries.rows; ++q) {
    const auto mask = MaskFor(base_attr, static_cast<std::int32_t>(1 + q % 20));
    const auto r = fse::PrefilterSearch(base, queries.Row(q), 10, mask.data());
    for (std::size_t j = 0; j < r.ids.size(); ++j) {
      ASSERT_GE(r.ids[j], 0);
      EXPECT_NE(mask[r.ids[j]], 0);
      EXPECT_EQ(r.distances[j],
                fse::SquaredL2(queries.Row(q), base.Row(r.ids[j]), 8));
      if (j > 0) {
        EXPECT_LE(r.distances[j - 1], r.distances[j]);
      }
    }
  }
}

TEST(Prefilter, EffortIsExactlyPassingCountAndFilterChecksIsN) {
  const auto base = Gaussian(2500, 8, 11);
  const auto queries = Gaussian(10, 8, 12);
  const auto base_attr = fse::UniformIntAttributes(base.rows, 1, 7, 13);
  for (std::int32_t a = 1; a <= 7; ++a) {
    const auto mask = MaskFor(base_attr, a);
    const auto passing = static_cast<std::uint64_t>(
        std::count(mask.begin(), mask.end(), static_cast<char>(1)));
    const auto r = fse::PrefilterSearch(base, queries.Row(0), 10, mask.data());
    EXPECT_EQ(r.distance_computations, passing);
    EXPECT_EQ(r.filter_checks, base.rows);
    EXPECT_EQ(r.rounds, 0U);
  }
}

TEST(Prefilter, PadsWhenFewerThanKPassAndHandlesEmptyFilter) {
  const auto base = Gaussian(50, 4, 14);
  std::vector<char> mask(base.rows, 0);
  mask[3] = 1;
  mask[40] = 1;
  const auto r = fse::PrefilterSearch(base, base.Row(40), 5, mask.data());
  EXPECT_EQ(r.ids[0], 40);
  EXPECT_EQ(r.distances[0], 0.0F);
  EXPECT_EQ(r.ids[1], 3);
  for (std::size_t j = 2; j < 5; ++j) {
    EXPECT_EQ(r.ids[j], -1);
    EXPECT_EQ(r.distances[j], std::numeric_limits<float>::infinity());
  }
  EXPECT_EQ(r.distance_computations, 2U);

  const std::vector<char> none(base.rows, 0);
  const auto e = fse::PrefilterSearch(base, base.Row(0), 3, none.data());
  EXPECT_EQ(e.ids, std::vector<std::int64_t>(3, -1));
  EXPECT_EQ(e.distance_computations, 0U);
  EXPECT_THROW(fse::PrefilterSearch(base, base.Row(0), 0, none.data()),
               std::invalid_argument);
}

}
