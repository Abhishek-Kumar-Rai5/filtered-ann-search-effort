#pragma once

// Directed reachability over the graph induced by one edge rule
// (acorn_semantics.h) on the eligible nodes (passing nodes; all nodes for
// kUnfiltered). Edges are materialised once per (graph, filter, rule); SCCs
// (iterative Tarjan) and the condensation give memoised per-seed
// reachability (docs/structural_design.md §3).

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "reachability/acorn_semantics.h"
#include "reachability/csr_graph.h"

namespace fse {

struct ReachStats {
  std::size_t nodes = 0;  // eligible nodes
  std::size_t edges = 0;  // materialised (de-duplicated) edges
  std::size_t sccs = 0;
  std::size_t largest_scc = 0;
  std::size_t wccs = 0;  // weakly connected components
  std::size_t largest_wcc = 0;
};

// Strongly connected components of a CSR graph (iterative Tarjan).
// Returns comp[v] in [0, ncomp); components numbered in reverse topological
// order (Tarjan finish order).
std::vector<std::int32_t> StronglyConnected(const CsrGraph& g,
                                            std::size_t* ncomp);

class ReachAnalyzer {
 public:
  ReachAnalyzer(const CsrGraph& g, const std::vector<char>& mask,
                const SemanticsParams& p);

  // True if `target` (base id) is reachable from `seed` (base id). The seed
  // need not be eligible: then reachability starts from its successors.
  bool Reachable(std::int64_t seed, std::int64_t target);
  [[nodiscard]] const ReachStats& Stats() const { return stats_; }
  // A base id inside the largest SCC (-1 if the graph is empty).
  [[nodiscard]] std::int64_t LargestSccMember() const {
    return largest_member_;
  }
  [[nodiscard]] bool Eligible(std::int64_t v) const {
    return local_[static_cast<std::size_t>(v)] >= 0;
  }

 private:
  std::vector<std::int32_t> SelectEligible();
  void Materialise(const std::vector<std::int32_t>& base_of);
  void Condense();
  void ComputeStats(const std::vector<std::int32_t>& base_of);
  const std::vector<char>& ReachFromSeed(std::int64_t seed);

  const CsrGraph& g_;
  const std::vector<char>& mask_;
  SemanticsParams p_;
  std::vector<std::int32_t> local_;  // base id -> local id, -1 = ineligible
  CsrGraph h_;                       // induced graph, local ids
  std::vector<std::int32_t> comp_;   // local id -> SCC
  std::size_t ncomp_ = 0;
  CsrGraph dag_;  // condensation (SCC ids)
  std::unordered_map<std::int64_t, std::vector<char>> memo_;
  ReachStats stats_;
  std::int64_t largest_member_ = -1;
};

}  // namespace fse
