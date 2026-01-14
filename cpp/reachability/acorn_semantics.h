#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "reachability/csr_graph.h"

namespace fse {

// How the search is allowed to step from one node to the next. kSem is
// ACORN's own rule (direct neighbours plus its 2-hop jump) with no budget
// limit, and it is the one we report. kGraph is the plain filtered graph,
// kCap replays one scan with ACORN's 2M cut-off, kUnfiltered is POST's graph.
enum class Semantics { kGraph, kSem, kCap, kUnfiltered };

struct SemanticsParams {
  Semantics kind = Semantics::kSem;
  int gamma = 1;
  int m = 32;
  int m_beta = 64;
};

class MarkSet {
 public:
  explicit MarkSet(std::size_t n) : stamp_(n, 0) {}
  void NewRound() { ++round_; }

  bool Mark(std::size_t i) {
    if (stamp_[i] == round_) {
      return false;
    }
    stamp_[i] = round_;
    return true;
  }
  [[nodiscard]] bool Marked(std::size_t i) const { return stamp_[i] == round_; }

 private:
  std::vector<std::uint32_t> stamp_;
  std::uint32_t round_ = 0;
};

void Successors(const CsrGraph& g, const std::vector<char>& mask, std::size_t u,
                const SemanticsParams& p, MarkSet* marks,
                std::vector<std::int32_t>* out);

}
