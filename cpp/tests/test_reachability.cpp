// Structural reachability (docs/structural_design.md §3): hand-computed edge
// rules, SCCs against naive mutual reachability, the analyzer against a naive
// search, and soundness on real ACORN indexes (every returned id must be in
// R_sem from the instrumented level-0 seed).

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include "methods/postfilter/hnsw_index.h"
#include "reachability/acorn_semantics.h"
#include "reachability/reach.h"

#ifdef FSE_WITH_ACORN
#include "methods/ingraph/acorn_index.h"
#endif

namespace {

fse::CsrGraph Make(const std::vector<std::vector<std::int32_t>>& lists) {
  fse::CsrGraph g;
  g.offsets.push_back(0);
  for (const auto& l : lists) {
    g.ids.insert(g.ids.end(), l.begin(), l.end());
    g.offsets.push_back(g.ids.size());
  }
  return g;
}

std::set<std::int32_t> Succ(const fse::CsrGraph& g,
                            const std::vector<char>& mask, std::size_t u,
                            fse::SemanticsParams p) {
  fse::MarkSet marks(g.Nodes());
  std::vector<std::int32_t> out;
  fse::Successors(g, mask, u, p, &marks, &out);
  return {out.begin(), out.end()};
}

using S = std::set<std::int32_t>;

TEST(AcornSemantics, HandGraphRules) {
  // 0 -> [1, 2, 3]; 1 -> [4]; 2 -> [5]; 3 -> [6, 7]; passing {0, 4, 5, 6}
  const auto g = Make({{1, 2, 3}, {4}, {5}, {6, 7}, {}, {}, {}, {}});
  const std::vector<char> mask = {1, 0, 0, 0, 1, 1, 1, 0};
  const fse::SemanticsParams graph{fse::Semantics::kGraph, 4, 1, 2};
  const fse::SemanticsParams sem1{fse::Semantics::kSem, 1, 1, 2};
  const fse::SemanticsParams semg{fse::Semantics::kSem, 4, 1, 2};
  const fse::SemanticsParams unf{fse::Semantics::kUnfiltered, 4, 1, 2};
  EXPECT_EQ(Succ(g, mask, 0, graph), S{});
  EXPECT_EQ(Succ(g, mask, 0, sem1), (S{4, 5, 6}));  // γ = 1: every position
  EXPECT_EQ(Succ(g, mask, 0, semg), (S{6}));        // γ > 1: only j >= Mβ = 2
  EXPECT_EQ(Succ(g, mask, 0, unf), (S{1, 2, 3}));
  // γ = 1 cap with M = 1 (stop after 2 passing): the 2-hop rule still fires
  // for a non-passing entry after the stop of the inner scan (ACORN code).
  const fse::SemanticsParams cap1{fse::Semantics::kCap, 1, 1, 2};
  EXPECT_EQ(Succ(g, mask, 0, cap1), (S{4, 5, 6}));
}

TEST(AcornSemantics, CapStopsAfterTwoMPassingDirectEntries) {
  // 0 -> [1, 2, 3, 4], all passing; M = 1 => stop once 2 are counted.
  const auto g = Make({{1, 2, 3, 4}, {}, {}, {}, {}});
  const std::vector<char> mask = {1, 1, 1, 1, 1};
  EXPECT_EQ(Succ(g, mask, 0, {fse::Semantics::kCap, 4, 1, 10}), (S{1, 2}));
  EXPECT_EQ(Succ(g, mask, 0, {fse::Semantics::kSem, 4, 1, 10}),
            (S{1, 2, 3, 4}));
  // duplicates and self-loops are dropped
  const auto g2 = Make({{0, 1, 1}, {0}});
  EXPECT_EQ(Succ(g2, {1, 1}, 0, {fse::Semantics::kGraph, 4, 1, 10}), (S{1}));
}

fse::CsrGraph RandomGraph(std::size_t n, std::size_t deg, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<std::int32_t> pick(0, static_cast<int>(n) - 1);
  std::vector<std::vector<std::int32_t>> lists(n);
  for (auto& l : lists) {
    for (std::size_t j = 0; j < deg; ++j) {
      l.push_back(pick(rng));
    }
  }
  return Make(lists);
}

std::vector<char> NaiveReach(const fse::CsrGraph& g,
                             const std::vector<char>& mask, std::size_t seed,
                             fse::SemanticsParams p) {
  const bool all = p.kind == fse::Semantics::kUnfiltered;
  std::vector<char> seen(g.Nodes(), 0);
  std::vector<std::size_t> stack = {seed};
  seen[seed] = 1;
  while (!stack.empty()) {
    const std::size_t u = stack.back();
    stack.pop_back();
    for (const std::int32_t w : Succ(g, mask, u, p)) {
      const auto wu = static_cast<std::size_t>(w);
      if (seen[wu] == 0 && (all || mask[wu] != 0)) {
        seen[wu] = 1;
        stack.push_back(wu);
      }
    }
  }
  return seen;
}

TEST(Reach, SccMatchesNaiveMutualReachability) {
  const auto g = RandomGraph(120, 2, 7);
  std::size_t nc = 0;
  const auto comp = fse::StronglyConnected(g, &nc);
  const std::vector<char> all(g.Nodes(), 1);
  const fse::SemanticsParams unf{fse::Semantics::kUnfiltered, 4, 16, 32};
  std::vector<std::vector<char>> r;
  for (std::size_t v = 0; v < g.Nodes(); ++v) {
    r.push_back(NaiveReach(g, all, v, unf));
  }
  for (std::size_t a = 0; a < g.Nodes(); ++a) {
    for (std::size_t b = 0; b < g.Nodes(); ++b) {
      EXPECT_EQ(comp[a] == comp[b], r[a][b] != 0 && r[b][a] != 0);
    }
  }
}

TEST(Reach, AnalyzerMatchesNaiveSearchForEveryRule) {
  const auto g = RandomGraph(300, 6, 11);
  std::mt19937 rng(3);
  std::vector<char> mask(g.Nodes());
  for (auto& m : mask) {
    m = static_cast<char>(rng() % 4 == 0);
  }
  for (const auto kind : {fse::Semantics::kGraph, fse::Semantics::kSem,
                          fse::Semantics::kCap, fse::Semantics::kUnfiltered}) {
    for (const int gamma : {1, 4}) {
      const fse::SemanticsParams p{kind, gamma, 2, 3};
      fse::ReachAnalyzer ra(g, mask, p);
      for (std::size_t seed = 0; seed < 40; ++seed) {  // incl. ineligible seeds
        // naive: seed's successors first (seed may be ineligible)
        std::vector<char> want(g.Nodes(), 0);
        for (const std::int32_t w : Succ(g, mask, seed, p)) {
          if (kind == fse::Semantics::kUnfiltered || mask[w] != 0) {
            const auto r = NaiveReach(g, mask, static_cast<std::size_t>(w), p);
            for (std::size_t v = 0; v < g.Nodes(); ++v) {
              want[v] = static_cast<char>(want[v] | r[v]);
            }
          }
        }
        want[seed] = 1;
        for (std::size_t t = 0; t < g.Nodes(); ++t) {
          const bool elig = kind == fse::Semantics::kUnfiltered || mask[t] != 0;
          EXPECT_EQ(ra.Reachable(static_cast<std::int64_t>(seed),
                                 static_cast<std::int64_t>(t)),
                    (want[t] != 0 && (elig || t == seed)))
              << "kind " << static_cast<int>(kind) << " gamma " << gamma
              << " seed " << seed << " t " << t;
        }
      }
    }
  }
}

// hnswlib level-0 export: base-id indexed, labels in range, and (for an
// in-order single-threaded build) the unfiltered graph is one SCC that
// reaches every node, as POST's U = 0 assumption needs to be checked.
TEST(Reach, HnswLevel0ExportIsLabelIndexedAndConnected) {
  constexpr std::size_t kN = 2000;
  std::mt19937 rng(5);
  std::normal_distribution<float> d(0.0F, 1.0F);
  fse::FloatMatrix base{std::vector<float>(kN * 8), kN, 8};
  for (float& x : base.data) {
    x = d(rng);
  }
  fse::HnswIndex index(8, kN, {.m = 8, .ef_construction = 100, .seed = 3});
  index.Add(base, 0, 1);
  const fse::CsrGraph g = index.Level0Graph();
  ASSERT_EQ(g.Nodes(), kN);
  for (const std::int32_t v : g.ids) {
    ASSERT_GE(v, 0);
    ASSERT_LT(static_cast<std::size_t>(v), kN);
  }
  fse::ReachAnalyzer ra(g, {}, {fse::Semantics::kUnfiltered, 1, 8, 16});
  EXPECT_EQ(ra.Stats().nodes, kN);
  EXPECT_GE(ra.Stats().largest_scc, kN * 99 / 100);
}

#ifdef FSE_WITH_ACORN
std::vector<float> Gaussian(std::size_t n, int dim, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0F, 1.0F);
  std::vector<float> v(n * static_cast<std::size_t>(dim));
  for (float& x : v) {
    x = d(rng);
  }
  return v;
}

// Soundness of R_sem on real ACORN indexes: with a large budget, every id
// the search returns is reachable from the instrumented level-0 seed under
// R_sem, and R_graph is a subset of R_sem.
TEST(Reach, AcornSearchResultsAreInRsemFromInstrumentedSeed) {
  constexpr int kDim = 16;
  constexpr std::size_t kN = 4000;
  constexpr std::size_t kNq = 60;
  const auto base = Gaussian(kN, kDim, 1);
  const auto queries = Gaussian(kNq, kDim, 2);
  omp_set_num_threads(1);
  for (const int gamma : {1, 4}) {
    fse::AcornIndex index(kDim, fse::AcornParams{16, gamma, 32},
                          std::vector<std::int32_t>(kN, 0));
    index.Add(kN, base.data());
    index.SetEfSearch(300);
    const fse::CsrGraph g = index.Level0Graph();
    ASSERT_EQ(g.Nodes(), kN);
    for (const double s : {0.03, 0.2}) {
      std::mt19937 rng(static_cast<unsigned>(gamma * 1000 + s * 100));
      std::vector<char> mask(kN);
      for (auto& m : mask) {
        m = static_cast<char>(std::uniform_real_distribution<>(0, 1)(rng) < s);
      }
      fse::ReachAnalyzer sem(
          g, mask,
          {fse::Semantics::kSem, index.Gamma(), index.M(), index.MBeta()});
      fse::ReachAnalyzer graph(
          g, mask,
          {fse::Semantics::kGraph, index.Gamma(), index.M(), index.MBeta()});
      std::size_t checked = 0;
      for (std::size_t q = 0; q < kNq; ++q) {
        const auto r = index.SearchOne(&queries[q * kDim], 10, mask.data());
        ASSERT_GE(r.level0_seed, 0);
        const auto again = index.SearchOne(&queries[q * kDim], 10, mask.data());
        EXPECT_EQ(again.level0_seed, r.level0_seed);
        for (const std::int64_t id : r.ids) {
          if (id < 0 || mask[static_cast<std::size_t>(id)] == 0) {
            continue;
          }
          ++checked;
          EXPECT_TRUE(sem.Reachable(r.level0_seed, id))
              << "gamma " << gamma << " s " << s << " q " << q;
        }
        for (std::size_t t = 0; t < kN; t += 97) {
          if (graph.Reachable(r.level0_seed, static_cast<std::int64_t>(t))) {
            EXPECT_TRUE(
                sem.Reachable(r.level0_seed, static_cast<std::int64_t>(t)));
          }
        }
      }
      EXPECT_GT(checked, 0U);
    }
  }
}
#endif

}  // namespace
