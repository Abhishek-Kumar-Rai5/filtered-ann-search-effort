#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "reachability/acorn_semantics.h"
#include "reachability/csr_graph.h"

namespace fse {

struct ReachStats {
  std::size_t nodes = 0;
  std::size_t edges = 0;
  std::size_t sccs = 0;
  std::size_t largest_scc = 0;
  std::size_t wccs = 0;
  std::size_t largest_wcc = 0;
};

std::vector<std::int32_t> StronglyConnected(const CsrGraph& g,
                                            std::size_t* ncomp);

// Builds the graph for one filter once, then answers: starting from this
// seed, could the search ever get to this target?
class ReachAnalyzer {
 public:
  ReachAnalyzer(const CsrGraph& g, const std::vector<char>& mask,
                const SemanticsParams& p);

  bool Reachable(std::int64_t seed, std::int64_t target);
  [[nodiscard]] const ReachStats& Stats() const { return stats_; }

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
  std::vector<std::int32_t> local_;
  CsrGraph h_;
  std::vector<std::int32_t> comp_;
  std::size_t ncomp_ = 0;
  CsrGraph dag_;
  std::unordered_map<std::int64_t, std::vector<char>> memo_;
  ReachStats stats_;
  std::int64_t largest_member_ = -1;
};

}
