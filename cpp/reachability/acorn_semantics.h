#pragma once

// Edge rules for the reachability notions frozen in
// docs/structural_design.md §3.2. They mirror ACORN's level-0 scan in
// hybrid_search_from_candidates (ACORN c259f11) without modifying it.
//
//   kGraph      w in Γ(u), w passes                    (R_graph)
//   kSem        direct passing entries, plus passing entries of Γ(v) for
//               every v in Γ(u) at an expanded position (γ = 1: all; γ > 1:
//               j >= Mβ); unlimited budget, no truncation  (R_sem, primary)
//   kCap        the entries ONE scan of Γ(u) adds, simulated exactly with an
//               empty visited set (u itself visited), incl. the 2M stop
//               (R_cap, sensitivity)
//   kUnfiltered Γ(u), every node eligible            (POST: unfiltered HNSW)

#include <cstddef>
#include <cstdint>
#include <vector>

#include "reachability/csr_graph.h"

namespace fse {

enum class Semantics { kGraph, kSem, kCap, kUnfiltered };

struct SemanticsParams {
  Semantics kind = Semantics::kSem;
  int gamma = 1;    // ACORN γ (2-hop at every position when γ == 1)
  int m = 32;       // ACORN M (truncation after 2M passing entries)
  int m_beta = 64;  // ACORN Mβ (γ > 1: 2-hop for positions j >= Mβ)
};

// Reusable de-duplication marks (avoids O(N) clears per call).
class MarkSet {
 public:
  explicit MarkSet(std::size_t n) : stamp_(n, 0) {}
  void NewRound() { ++round_; }
  // true if newly marked in this round
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

// Successors of u (base ids) under `p`; `mask` = filter (ignored for
// kUnfiltered). Output is de-duplicated and excludes u. `marks` must be
// sized to the graph.
void Successors(const CsrGraph& g, const std::vector<char>& mask, std::size_t u,
                const SemanticsParams& p, MarkSet* marks,
                std::vector<std::int32_t>* out);

}  // namespace fse
