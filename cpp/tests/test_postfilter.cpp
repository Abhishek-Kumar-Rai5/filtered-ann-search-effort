// Post-filter (over-fetch) baseline: fetch schedule, exact agreement with
// brute-force filtered ground truth at an exhaustive budget, filter
// correctness, recall, refetch rounds, exact distance accounting, and
// thread safety.

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/vecs_io.h"
#include "filter_generator/uniform_attributes.h"
#include "ground_truth/filtered_ground_truth.h"
#include "methods/postfilter/hnsw_index.h"
#include "methods/postfilter/postfilter.h"
#include "metrics/recall.h"

namespace {

constexpr std::size_t kDim = 16;
constexpr std::size_t kK = 10;
constexpr fse::HnswParams kParams{.m = 16, .ef_construction = 100, .seed = 3};

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

// A toy filtered workload: Gaussian vectors, uniform integer attributes in
// [1, values], equality predicate; masks and exact global selectivities.
struct Toy {
  fse::FloatMatrix base;
  fse::FloatMatrix queries;
  std::vector<std::int32_t> base_attr;
  std::vector<std::int32_t> query_attr;
  std::vector<std::vector<char>> masks;
  std::vector<double> selectivity;
  fse::NeighborTable gt;
};

Toy MakeToy(std::size_t n, std::size_t nq, std::int32_t values,
            std::uint32_t seed) {
  Toy t;
  t.base = Gaussian(n, kDim, seed);
  t.queries = Gaussian(nq, kDim, seed + 1);
  t.base_attr = fse::UniformIntAttributes(n, 1, values, seed + 2);
  t.query_attr = fse::UniformIntAttributes(nq, 1, values, seed + 3);
  for (std::size_t q = 0; q < nq; ++q) {
    std::vector<char> m(n);
    std::size_t pass = 0;
    for (std::size_t i = 0; i < n; ++i) {
      m[i] = static_cast<char>(t.base_attr[i] == t.query_attr[q]);
      pass += m[i] != 0 ? 1 : 0;
    }
    t.masks.push_back(std::move(m));
    t.selectivity.push_back(static_cast<double>(pass) / static_cast<double>(n));
  }
  t.gt = fse::FilteredGroundTruthEquals(t.base, t.queries, t.base_attr,
                                        t.query_attr, kK);
  return t;
}

fse::HnswIndex Build(const fse::FloatMatrix& base) {
  fse::HnswIndex index(kDim, base.rows, kParams);
  index.Add(base, 0, 1);
  return index;
}

TEST(PostfilterFetchSize, FollowsOverfetchAndGrowthScheduleWithCaps) {
  const fse::PostfilterParams p{.ef_search = 10,
                                .overfetch_factor = 1.0,
                                .growth_factor = 2.0,
                                .max_rounds = 5};
  EXPECT_EQ(fse::PostfilterFetchSize(10, 0.1, p, 1, 100000), 100U);
  EXPECT_EQ(fse::PostfilterFetchSize(10, 0.1, p, 2, 100000), 200U);
  EXPECT_EQ(fse::PostfilterFetchSize(10, 0.1, p, 3, 100000), 400U);
  EXPECT_EQ(fse::PostfilterFetchSize(10, 0.1, p, 3, 250), 250U);  // cap n
  EXPECT_EQ(fse::PostfilterFetchSize(10, 1.0, p, 1, 1000), 10U);  // k / 1
  EXPECT_EQ(fse::PostfilterFetchSize(10, 0.3, p, 1, 1000), 34U);  // ceil
  const fse::PostfilterParams half{.ef_search = 10,
                                   .overfetch_factor = 0.5,
                                   .growth_factor = 2.0,
                                   .max_rounds = 1};
  EXPECT_EQ(fse::PostfilterFetchSize(10, 1.0, half, 1, 1000), 10U);  // >= k
  EXPECT_THROW(fse::PostfilterFetchSize(10, 0.0, p, 1, 1000),
               std::invalid_argument);
  EXPECT_THROW(fse::PostfilterFetchSize(10, 0.5, p, 0, 1000),
               std::invalid_argument);
}

TEST(Postfilter, ExhaustiveBudgetMatchesBruteForceFilteredGroundTruth) {
  // ef >= N and a fetch of the whole index: HNSW visits every reachable node,
  // so post-filtering must return exactly the brute-force filtered top-k.
  const Toy t = MakeToy(2000, 60, 5, 100);
  fse::HnswIndex index = Build(t.base);
  const fse::PostfilterParams p{.ef_search = 2000,
                                .overfetch_factor = 1e9,
                                .growth_factor = 2.0,
                                .max_rounds = 1};
  index.SetEf(p.ef_search);
  for (std::size_t q = 0; q < t.queries.rows; ++q) {
    const auto r = fse::PostfilterSearch(
        index, t.queries.Row(q), kK, t.masks[q].data(), t.selectivity[q], p);
    EXPECT_EQ(r.last_fetch, t.base.rows);
    EXPECT_EQ(r.ids, std::vector<std::int64_t>(t.gt.Ids(q), t.gt.Ids(q) + kK))
        << "q=" << q;
    for (std::size_t j = 0; j < kK; ++j) {
      EXPECT_NEAR(r.distances[j], t.gt.Distances(q)[j],
                  1e-4 * std::max(1.0F, t.gt.Distances(q)[j]));
    }
    EXPECT_DOUBLE_EQ(fse::RecallAtK(r.ids.data(), t.gt.Ids(q), kK), 1.0);
  }
}

TEST(Postfilter, ResultsPassFilterAreSortedAndDistancesAreCorrect) {
  for (const std::int32_t values : {1, 4, 50}) {
    const Toy t = MakeToy(3000, 40, values, 200 + values);
    fse::HnswIndex index = Build(t.base);
    for (const std::size_t ef : {10, 40, 160}) {
      const fse::PostfilterParams p{.ef_search = ef,
                                    .overfetch_factor = 1.0,
                                    .growth_factor = 2.0,
                                    .max_rounds = 8};
      index.SetEf(ef);
      for (std::size_t q = 0; q < t.queries.rows; ++q) {
        const auto r =
            fse::PostfilterSearch(index, t.queries.Row(q), kK,
                                  t.masks[q].data(), t.selectivity[q], p);
        for (std::size_t j = 0; j < kK; ++j) {
          if (r.ids[j] < 0) {
            EXPECT_EQ(r.distances[j], std::numeric_limits<float>::infinity());
            continue;
          }
          EXPECT_NE(t.masks[q][r.ids[j]], 0);
          EXPECT_NEAR(
              r.distances[j],
              fse::SquaredL2(t.queries.Row(q), t.base.Row(r.ids[j]), kDim),
              1e-3);
          if (j > 0 && r.ids[j - 1] >= 0) {
            EXPECT_LE(r.distances[j - 1], r.distances[j]);
          }
        }
      }
    }
  }
}

TEST(Postfilter, RecallAgainstFilteredGroundTruthRisesWithBudget) {
  const Toy t = MakeToy(5000, 100, 10, 300);
  fse::HnswIndex index = Build(t.base);
  auto mean_recall = [&](std::size_t ef) {
    const fse::PostfilterParams p{.ef_search = ef,
                                  .overfetch_factor = 1.0,
                                  .growth_factor = 2.0,
                                  .max_rounds = 8};
    index.SetEf(ef);
    double sum = 0.0;
    for (std::size_t q = 0; q < t.queries.rows; ++q) {
      const auto r = fse::PostfilterSearch(
          index, t.queries.Row(q), kK, t.masks[q].data(), t.selectivity[q], p);
      sum += fse::RecallAtK(r.ids.data(), t.gt.Ids(q), kK);
    }
    return sum / static_cast<double>(t.queries.rows);
  };
  const double low = mean_recall(10);
  const double high = mean_recall(800);
  EXPECT_LE(low, high);
  EXPECT_GE(high, 0.99);
}

TEST(Postfilter, DistanceCountEqualsSumOfReplayedRounds) {
  const Toy t = MakeToy(4000, 50, 20, 400);
  fse::HnswIndex index = Build(t.base);
  // Under-estimated selectivity (s = 1) forces refetch rounds.
  const fse::PostfilterParams p{.ef_search = 20,
                                .overfetch_factor = 1.0,
                                .growth_factor = 2.0,
                                .max_rounds = 10};
  index.SetEf(p.ef_search);
  std::uint32_t max_rounds_seen = 0;
  for (std::size_t q = 0; q < t.queries.rows; ++q) {
    const auto r = fse::PostfilterSearch(index, t.queries.Row(q), kK,
                                         t.masks[q].data(), 1.0, p);
    std::uint64_t replay = 0;
    std::uint64_t fetched = 0;
    std::size_t fetch = 0;
    for (std::uint32_t round = 1; round <= r.rounds; ++round) {
      fetch = fse::PostfilterFetchSize(kK, 1.0, p, round, t.base.rows);
      const auto s = index.SearchAtCurrentEf(t.queries.Row(q), fetch);
      replay += s.distance_computations;
      fetched += s.labels.size();
    }
    EXPECT_EQ(r.distance_computations, replay) << "q=" << q;
    EXPECT_EQ(r.last_fetch, fetch);
    EXPECT_GT(r.filter_checks, 0U);
    EXPECT_LE(r.filter_checks, fetched);
    max_rounds_seen = std::max(max_rounds_seen, r.rounds);
  }
  EXPECT_GT(max_rounds_seen, 1U);  // the refetch path was exercised
}

TEST(Postfilter, RefetchRecoversSurvivorsThatOneRoundMisses) {
  const Toy t = MakeToy(4000, 50, 20, 500);
  fse::HnswIndex index = Build(t.base);
  const fse::PostfilterParams one{.ef_search = 20,
                                  .overfetch_factor = 1.0,
                                  .growth_factor = 2.0,
                                  .max_rounds = 1};
  const fse::PostfilterParams many{.ef_search = 20,
                                   .overfetch_factor = 1.0,
                                   .growth_factor = 2.0,
                                   .max_rounds = 12};
  index.SetEf(20);
  std::size_t partial = 0;
  for (std::size_t q = 0; q < t.queries.rows; ++q) {
    // s = 1 under-fetches (10 candidates for a ~5 % filter).
    const auto a = fse::PostfilterSearch(index, t.queries.Row(q), kK,
                                         t.masks[q].data(), 1.0, one);
    const auto b = fse::PostfilterSearch(index, t.queries.Row(q), kK,
                                         t.masks[q].data(), 1.0, many);
    const auto valid = [](const std::vector<std::int64_t>& ids) {
      return std::count_if(ids.begin(), ids.end(),
                           [](std::int64_t id) { return id >= 0; });
    };
    EXPECT_EQ(a.rounds, 1U);
    partial += valid(a.ids) < static_cast<std::ptrdiff_t>(kK) ? 1 : 0;
    // A single round keeps its partial survivors (padded with -1).
    for (std::size_t j = 0; j < kK; ++j) {
      if (a.ids[j] >= 0) {
        EXPECT_NE(t.masks[q][a.ids[j]], 0);
      }
    }
    EXPECT_EQ(valid(b.ids), static_cast<std::ptrdiff_t>(kK));
  }
  EXPECT_GT(partial, 0U);
}

TEST(Postfilter, UnfilteredQueryEqualsPlainHnswSearch) {
  const Toy t = MakeToy(2000, 30, 1, 600);  // one attribute value: all pass
  fse::HnswIndex index = Build(t.base);
  const fse::PostfilterParams p{.ef_search = 50,
                                .overfetch_factor = 1.0,
                                .growth_factor = 2.0,
                                .max_rounds = 4};
  index.SetEf(50);
  for (std::size_t q = 0; q < t.queries.rows; ++q) {
    ASSERT_DOUBLE_EQ(t.selectivity[q], 1.0);
    const auto r = fse::PostfilterSearch(index, t.queries.Row(q), kK,
                                         t.masks[q].data(), 1.0, p);
    const auto s = index.SearchAtCurrentEf(t.queries.Row(q), kK);
    EXPECT_EQ(r.rounds, 1U);
    EXPECT_EQ(r.distance_computations, s.distance_computations);
    for (std::size_t j = 0; j < kK; ++j) {
      EXPECT_EQ(r.ids[j], static_cast<std::int64_t>(s.labels[j]));
    }
  }
}

TEST(Postfilter, EmptyFilterAndInvalidInputs) {
  const Toy t = MakeToy(500, 2, 4, 700);
  fse::HnswIndex index = Build(t.base);
  fse::PostfilterParams p{.ef_search = 30,
                          .overfetch_factor = 1.0,
                          .growth_factor = 2.0,
                          .max_rounds = 3};
  index.SetEf(30);
  const std::vector<char> none(t.base.rows, 0);
  const auto r =
      fse::PostfilterSearch(index, t.queries.Row(0), kK, none.data(), 0.0, p);
  EXPECT_EQ(r.ids, std::vector<std::int64_t>(kK, -1));
  EXPECT_EQ(r.distance_computations, 0U);
  EXPECT_EQ(r.rounds, 0U);

  const char* m = t.masks[0].data();
  const float* qv = t.queries.Row(0);
  EXPECT_THROW(fse::PostfilterSearch(index, qv, 0, m, 0.5, p),
               std::invalid_argument);
  EXPECT_THROW(fse::PostfilterSearch(index, qv, kK, m, 1.5, p),
               std::invalid_argument);
  index.SetEf(31);  // ef no longer matches params
  EXPECT_THROW(fse::PostfilterSearch(index, qv, kK, m, 0.5, p),
               std::invalid_argument);
  index.SetEf(30);
  p.growth_factor = 1.0;  // refetching would repeat the identical search
  EXPECT_THROW(fse::PostfilterSearch(index, qv, kK, m, 0.5, p),
               std::invalid_argument);
  p.max_rounds = 0;
  EXPECT_THROW(fse::PostfilterSearch(index, qv, kK, m, 0.5, p),
               std::invalid_argument);
}

TEST(Postfilter, ConcurrentQueriesMatchSerialIncludingEffort) {
  const Toy t = MakeToy(3000, 64, 12, 800);
  fse::HnswIndex index = Build(t.base);
  const fse::PostfilterParams p{.ef_search = 40,
                                .overfetch_factor = 1.0,
                                .growth_factor = 2.0,
                                .max_rounds = 6};
  index.SetEf(40);
  const auto n = static_cast<std::int64_t>(t.queries.rows);
  std::vector<fse::FilteredSearchResult> serial(t.queries.rows);
  for (std::int64_t q = 0; q < n; ++q) {
    serial[q] = fse::PostfilterSearch(index, t.queries.Row(q), kK,
                                      t.masks[q].data(), t.selectivity[q], p);
  }
  std::vector<fse::FilteredSearchResult> parallel(t.queries.rows);
#pragma omp parallel for num_threads(8)
  for (std::int64_t q = 0; q < n; ++q) {
    parallel[q] = fse::PostfilterSearch(index, t.queries.Row(q), kK,
                                        t.masks[q].data(), t.selectivity[q], p);
  }
  for (std::int64_t q = 0; q < n; ++q) {
    EXPECT_EQ(parallel[q].ids, serial[q].ids);
    EXPECT_EQ(parallel[q].distance_computations,
              serial[q].distance_computations);
    EXPECT_EQ(parallel[q].filter_checks, serial[q].filter_checks);
    EXPECT_EQ(parallel[q].rounds, serial[q].rounds);
  }
}

}  // namespace
