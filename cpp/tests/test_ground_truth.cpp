// Filtered brute-force ground truth: everything downstream depends on it.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "common/vecs_io.h"
#include "ground_truth/filtered_ground_truth.h"

namespace {

fse::FloatMatrix IntegerVectors(std::size_t n, std::size_t dim, unsigned seed,
                                int max_value) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> u(0, max_value);
  fse::FloatMatrix m;
  m.rows = n;
  m.dim = dim;
  m.data.resize(n * dim);
  for (float& x : m.data) {
    x = static_cast<float>(u(rng));
  }
  return m;
}

// Naive reference: sort all passing (distance, id) pairs, take k.
std::vector<std::pair<float, std::int64_t>> Naive(
    const fse::FloatMatrix& base, const float* q,
    const std::vector<std::int32_t>& attr, std::int32_t qa, std::size_t k) {
  std::vector<std::pair<float, std::int64_t>> all;
  for (std::size_t i = 0; i < base.rows; ++i) {
    if (attr[i] != qa) {
      continue;
    }
    float d = 0;
    for (std::size_t j = 0; j < base.dim; ++j) {
      const float diff = q[j] - base.Row(i)[j];
      d += diff * diff;
    }
    all.emplace_back(d, static_cast<std::int64_t>(i));
  }
  std::sort(all.begin(), all.end());
  all.resize(std::min(all.size(), k));
  return all;
}

TEST(GroundTruth, SquaredL2MatchesDefinition) {
  const std::vector<float> a = {1, 2, 3, 4, 5};
  const std::vector<float> b = {0, 2, 5, 1, 5};
  EXPECT_FLOAT_EQ(fse::SquaredL2(a.data(), b.data(), 5), 1 + 0 + 4 + 9 + 0);
}

TEST(GroundTruth, FilteredMatchesNaiveReferenceIncludingTies) {
  // Small integer range forces many exact distance ties.
  const auto base = IntegerVectors(2000, 8, 1, 3);
  const auto queries = IntegerVectors(50, 8, 2, 3);
  std::mt19937 rng(3);
  std::uniform_int_distribution<int> u(1, 4);
  std::vector<std::int32_t> base_attr(base.rows);
  std::vector<std::int32_t> query_attr(queries.rows);
  for (auto& a : base_attr) {
    a = u(rng);
  }
  for (auto& a : query_attr) {
    a = u(rng);
  }
  constexpr std::size_t kK = 11;
  const fse::NeighborTable t =
      fse::FilteredGroundTruthEquals(base, queries, base_attr, query_attr, kK);
  ASSERT_EQ(t.nq, queries.rows);
  ASSERT_EQ(t.k, kK);
  for (std::size_t q = 0; q < queries.rows; ++q) {
    const auto ref = Naive(base, queries.Row(q), base_attr, query_attr[q], kK);
    ASSERT_EQ(ref.size(), kK);
    for (std::size_t j = 0; j < kK; ++j) {
      EXPECT_EQ(t.Ids(q)[j], ref[j].second) << "q=" << q << " j=" << j;
      EXPECT_EQ(t.Distances(q)[j], ref[j].first);
      EXPECT_EQ(base_attr[t.Ids(q)[j]], query_attr[q]);
    }
  }
}

TEST(GroundTruth, ExactTieBrokenBySmallerId) {
  fse::FloatMatrix base{{5, 1, 1, 0, 1, 2}, 6, 1};  // ids 1,2,4 at d=0 to q=1
  fse::FloatMatrix q{{1}, 1, 1};
  const std::vector<std::int32_t> attr(6, 1);
  const auto t = fse::FilteredGroundTruthEquals(base, q, attr, {1}, 3);
  EXPECT_EQ(std::vector<std::int64_t>(t.Ids(0), t.Ids(0) + 3),
            (std::vector<std::int64_t>{1, 2, 4}));
}

TEST(GroundTruth, PadsWhenFewerThanKPassAndRespectsFilter) {
  fse::FloatMatrix base{{0, 1, 2, 3, 4}, 5, 1};
  fse::FloatMatrix q{{0}, 1, 1};
  const std::vector<std::int32_t> attr = {1, 2, 1, 2, 2};
  const auto t = fse::FilteredGroundTruthEquals(base, q, attr, {1}, 3);
  EXPECT_EQ(t.Ids(0)[0], 0);
  EXPECT_EQ(t.Ids(0)[1], 2);
  EXPECT_EQ(t.Ids(0)[2], -1);
  EXPECT_EQ(t.Distances(0)[2], std::numeric_limits<float>::infinity());
}

TEST(GroundTruth, UnfilteredEqualsAllPassFiltered) {
  const auto base = IntegerVectors(500, 4, 4, 10);
  const auto queries = IntegerVectors(20, 4, 5, 10);
  const auto a = fse::UnfilteredGroundTruth(base, queries, 10);
  const auto b = fse::FilteredGroundTruthEquals(
      base, queries, std::vector<std::int32_t>(500, 0),
      std::vector<std::int32_t>(20, 0), 10);
  EXPECT_EQ(a.ids, b.ids);
  EXPECT_EQ(a.distances, b.distances);
}

TEST(GroundTruth, RejectsMismatchedInputs) {
  fse::FloatMatrix base{{0, 1}, 2, 1};
  fse::FloatMatrix q{{0, 0}, 1, 2};
  EXPECT_THROW(fse::UnfilteredGroundTruth(base, q, 1), std::invalid_argument);
  fse::FloatMatrix q1{{0}, 1, 1};
  EXPECT_THROW(fse::FilteredGroundTruthEquals(base, q1, {1}, {1}, 1),
               std::invalid_argument);
  EXPECT_THROW(fse::UnfilteredGroundTruth(base, q1, 0), std::invalid_argument);
}

TEST(GroundTruth, NeighborTableRoundTrip) {
  fse::NeighborTable t{2, 2, {1, 2, 3, -1}, {0.5F, 1, 2, 3}};
  const std::string path = ::testing::TempDir() + "/nt.bin";
  fse::WriteNeighborTable(path, t);
  const auto r = fse::ReadNeighborTable(path);
  EXPECT_EQ(r.nq, 2U);
  EXPECT_EQ(r.k, 2U);
  EXPECT_EQ(r.ids, t.ids);
  EXPECT_EQ(r.distances, t.distances);
}

}  // namespace
