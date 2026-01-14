// Project-side ACORN wrapper. ACORN itself is a black box; these tests check
// only what the wrapper adds (attribute ownership, counters, I/O, stats) and
// that filtered search respects the filter.

#include <faiss/impl/ACORN.h>
#include <gtest/gtest.h>
#include <omp.h>

#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "common/vecs_io.h"
#include "filter_generator/uniform_attributes.h"
#include "ground_truth/filtered_ground_truth.h"
#include "methods/ingraph/acorn_index.h"
#include "metrics/recall.h"

namespace {

constexpr int kDim = 16;
constexpr std::size_t kN = 3000;
constexpr std::size_t kNq = 50;
constexpr std::size_t kK = 10;

std::vector<float> RandomVectors(std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0F, 1.0F);
  std::vector<float> v(n * kDim);
  for (float& x : v) {
    x = d(rng);
  }
  return v;
}

struct Fixture {
  std::vector<float> base = RandomVectors(kN, 1);
  std::vector<float> queries = RandomVectors(kNq, 2);
  std::vector<std::int32_t> base_attr = fse::UniformIntAttributes(kN, 1, 4, 3);
  std::vector<std::int32_t> query_attr =
      fse::UniformIntAttributes(kNq, 1, 4, 4);
  std::vector<char> filter_map = [this] {
    std::vector<char> m(kNq * kN);
    for (std::size_t q = 0; q < kNq; ++q) {
      for (std::size_t i = 0; i < kN; ++i) {
        m[q * kN + i] = static_cast<char>(base_attr[i] == query_attr[q]);
      }
    }
    return m;
  }();
};

fse::AcornIndex Build(const Fixture& f, int gamma) {
  fse::AcornIndex index(kDim, fse::AcornParams{16, gamma, 32}, f.base_attr);
  index.Add(kN, f.base.data());
  return index;
}

TEST(AcornIndex, BuildsWithDefaultEfConstructionAndStructure) {
  const Fixture f;
  const fse::AcornIndex index = Build(f, 4);
  EXPECT_EQ(index.Size(), kN);
  EXPECT_EQ(index.Dim(), kDim);
  EXPECT_EQ(index.EfConstruction(), 16 * 4);  // ACORN default M * gamma
  const auto nodes = index.NodesPerLevel();
  ASSERT_FALSE(nodes.empty());
  EXPECT_EQ(nodes[0], kN);
  for (std::size_t l = 1; l < nodes.size(); ++l) {
    EXPECT_LE(nodes[l], nodes[l - 1]);
  }
  const auto deg = index.AverageOutDegreePerLevel();
  ASSERT_EQ(deg.size(), nodes.size());
  EXPECT_GT(deg[0], 0.0);
  EXPECT_LE(deg[0], 32 + 1.5 * 16);  // level-0 slots: M_beta + 1.5 M
  for (std::size_t l = 1; l < deg.size(); ++l) {
    EXPECT_LE(deg[l], 16.0 * 4);  // upper-level slots: M * gamma
  }
  EXPECT_GE(index.MemoryBytes(), kN * kDim * sizeof(float));
}

TEST(AcornIndex, FilteredSearchRespectsFilterAndFindsNeighbours) {
  const Fixture f;
  fse::AcornIndex index = Build(f, 4);
  index.SetEfSearch(200);
  fse::FloatMatrix base{f.base, kN, kDim};
  fse::FloatMatrix queries{f.queries, kNq, kDim};
  const auto gt = fse::FilteredGroundTruthEquals(base, queries, f.base_attr,
                                                 f.query_attr, kK);
  std::vector<std::int64_t> ids(kNq * kK);
  std::vector<float> d(kNq * kK);
  index.SearchBatch(kNq, f.queries.data(), kK, f.filter_map.data(), ids.data(),
                    d.data());
  double recall = 0;
  for (std::size_t q = 0; q < kNq; ++q) {
    for (std::size_t j = 0; j < kK; ++j) {
      const std::int64_t id = ids[q * kK + j];
      ASSERT_GE(id, 0);
      EXPECT_EQ(f.base_attr[id], f.query_attr[q]);
      EXPECT_NEAR(
          d[q * kK + j],
          fse::SquaredL2(&f.queries[q * kDim], &f.base[id * kDim], kDim), 1e-4);
    }
    recall += fse::RecallAtK(&ids[q * kK], gt.Ids(q), kK);
  }
  // Sanity only (ACORN is not under test): a generous ef on 3K points
  // should find almost all filtered neighbours.
  EXPECT_GT(recall / kNq, 0.9);
}

TEST(AcornIndex, SearchOneMatchesBatchAndCounterSumsToBatchDelta) {
  const Fixture f;
  fse::AcornIndex index = Build(f, 4);
  index.SetEfSearch(40);
  omp_set_num_threads(1);
  std::vector<std::int64_t> one_ids;
  std::size_t per_query_total = 0;
  for (std::size_t q = 0; q < kNq; ++q) {
    const auto r =
        index.SearchOne(&f.queries[q * kDim], kK, &f.filter_map[q * kN]);
    EXPECT_GT(r.distance_computations, 0U);
    per_query_total += r.distance_computations;
    one_ids.insert(one_ids.end(), r.ids.begin(), r.ids.end());
  }
  omp_set_num_threads(4);
  std::vector<std::int64_t> ids(kNq * kK);
  std::vector<float> d(kNq * kK);
  const std::size_t before = faiss::acorn_stats.n3;
  index.SearchBatch(kNq, f.queries.data(), kK, f.filter_map.data(), ids.data(),
                    d.data());
  EXPECT_EQ(faiss::acorn_stats.n3 - before, per_query_total);
  EXPECT_EQ(ids, one_ids);
}

// n_scanned (project instrumentation in ACORN, docs/phase4_matrix.md §14):
// deterministic per query, every counted distance is on a scanned entry, and
// the multi-threaded batch reduction equals the per-query sum.
TEST(AcornIndex, ScanCounterIsDeterministicBoundsDistancesAndSumsToBatch) {
  const Fixture f;
  fse::AcornIndex index = Build(f, 4);
  index.SetEfSearch(40);
  omp_set_num_threads(1);
  std::size_t per_query_total = 0;
  for (std::size_t q = 0; q < kNq; ++q) {
    const auto r =
        index.SearchOne(&f.queries[q * kDim], kK, &f.filter_map[q * kN]);
    const auto again =
        index.SearchOne(&f.queries[q * kDim], kK, &f.filter_map[q * kN]);
    EXPECT_GE(r.entries_scanned, r.distance_computations);
    EXPECT_EQ(again.entries_scanned, r.entries_scanned);
    EXPECT_EQ(again.distance_computations, r.distance_computations);
    per_query_total += r.entries_scanned;
  }
  omp_set_num_threads(4);
  std::vector<std::int64_t> ids(kNq * kK);
  std::vector<float> d(kNq * kK);
  const std::size_t before = faiss::acorn_stats.n_scanned;
  index.SearchBatch(kNq, f.queries.data(), kK, f.filter_map.data(), ids.data(),
                    d.data());
  EXPECT_EQ(faiss::acorn_stats.n_scanned - before, per_query_total);
}

TEST(AcornIndex, SaveLoadGivesIdenticalResults) {
  const Fixture f;
  fse::AcornIndex index = Build(f, 4);
  const std::string path = ::testing::TempDir() + "/acorn_test.index";
  index.Save(path);
  fse::AcornIndex loaded = fse::AcornIndex::Load(path, f.base_attr);
  EXPECT_EQ(loaded.Size(), kN);
  EXPECT_EQ(loaded.EfConstruction(), index.EfConstruction());
  EXPECT_EQ(loaded.AverageOutDegreePerLevel(),
            index.AverageOutDegreePerLevel());
  index.SetEfSearch(30);
  loaded.SetEfSearch(30);
  std::vector<std::int64_t> a(kNq * kK);
  std::vector<std::int64_t> b(kNq * kK);
  std::vector<float> da(kNq * kK);
  std::vector<float> db(kNq * kK);
  index.SearchBatch(kNq, f.queries.data(), kK, f.filter_map.data(), a.data(),
                    da.data());
  loaded.SearchBatch(kNq, f.queries.data(), kK, f.filter_map.data(), b.data(),
                     db.data());
  EXPECT_EQ(a, b);
  EXPECT_EQ(da, db);
  // Attribute count must match the index.
  EXPECT_THROW(fse::AcornIndex::Load(path, {1, 2, 3}), std::invalid_argument);
}

TEST(AcornIndex, AddRejectsMoreVectorsThanAttributes) {
  const Fixture f;
  fse::AcornIndex index(kDim, fse::AcornParams{16, 1, 32},
                        std::vector<std::int32_t>(10, 1));
  EXPECT_THROW(index.Add(kN, f.base.data()), std::invalid_argument);
}

}  // namespace
