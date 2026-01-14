#include "reachability/acorn_semantics.h"

#include <stdexcept>

namespace fse {
namespace {

bool Passes(const std::vector<char>& mask, std::int32_t v) {
  return mask[static_cast<std::size_t>(v)] != 0;
}

bool Expanded(const SemanticsParams& p, std::size_t j) {
  return p.gamma == 1 || j >= static_cast<std::size_t>(p.m_beta);
}

void Push(std::int32_t v, MarkSet* marks, std::vector<std::int32_t>* out) {
  if (marks->Mark(static_cast<std::size_t>(v))) {
    out->push_back(v);
  }
}

// Passing entries of Γ(v) (R_sem 2-hop step).
void PushPassing(const CsrGraph& g, const std::vector<char>& mask,
                 std::int32_t v, MarkSet* marks,
                 std::vector<std::int32_t>* out) {
  const auto vu = static_cast<std::size_t>(v);
  const std::int32_t* list = g.Begin(vu);
  for (std::size_t j = 0; j < g.Degree(vu); ++j) {
    if (Passes(mask, list[j])) {
      Push(list[j], marks, out);
    }
  }
}

// State of one simulated level-0 scan (ACORN c259f11,
// hybrid_search_from_candidates).
struct CapState {
  std::size_t cap = 0;
  std::size_t num_found = 0;
  bool keep_expanding = true;
};

// The 2-hop expansion of v1 inside a scan: passing entries are counted,
// unvisited ones added; stops (keep_expanding = false) at the 2M count.
void CapExpand(const CsrGraph& g, const std::vector<char>& mask,
               std::int32_t v1, CapState* st, MarkSet* vis,
               std::vector<std::int32_t>* out) {
  const auto vu = static_cast<std::size_t>(v1);
  const std::int32_t* list2 = g.Begin(vu);
  for (std::size_t j2 = 0; j2 < g.Degree(vu); ++j2) {
    const std::int32_t v2 = list2[j2];
    if (!Passes(mask, v2)) {
      continue;
    }
    ++st->num_found;
    if (!vis->Mark(static_cast<std::size_t>(v2))) {
      continue;  // already visited
    }
    out->push_back(v2);
    if (st->num_found >= st->cap) {
      st->keep_expanding = false;
      return;
    }
  }
}

// Exact simulation of one level-0 scan of Γ(u), starting from a visited set
// {u} (R_cap).
void CapScan(const CsrGraph& g, const std::vector<char>& mask, std::size_t u,
             const SemanticsParams& p, MarkSet* vis,
             std::vector<std::int32_t>* out) {
  CapState st{2 * static_cast<std::size_t>(p.m)};
  const std::int32_t* list = g.Begin(u);
  for (std::size_t j = 0; j < g.Degree(u); ++j) {
    const std::int32_t v1 = list[j];
    const bool passes = Passes(mask, v1);
    st.num_found += passes ? 1 : 0;
    if (vis->Marked(static_cast<std::size_t>(v1))) {
      continue;
    }
    if (passes) {
      vis->Mark(static_cast<std::size_t>(v1));
      out->push_back(v1);
      if (st.num_found >= st.cap) {
        return;  // ACORN: keep_expanding = false; break
      }
    }
    if ((j >= static_cast<std::size_t>(p.m_beta) && st.keep_expanding) ||
        p.gamma == 1) {
      CapExpand(g, mask, v1, &st, vis, out);
    }
  }
}

}  // namespace

void Successors(const CsrGraph& g, const std::vector<char>& mask, std::size_t u,
                const SemanticsParams& p, MarkSet* marks,
                std::vector<std::int32_t>* out) {
  if (p.kind != Semantics::kUnfiltered && mask.size() != g.Nodes()) {
    throw std::invalid_argument("Successors: mask size != graph nodes");
  }
  out->clear();
  marks->NewRound();
  marks->Mark(u);  // excludes u from its own successors / visited in a scan
  if (p.kind == Semantics::kCap) {
    CapScan(g, mask, u, p, marks, out);
    return;
  }
  const std::int32_t* list = g.Begin(u);
  for (std::size_t j = 0; j < g.Degree(u); ++j) {
    const std::int32_t v = list[j];
    if (p.kind == Semantics::kUnfiltered || Passes(mask, v)) {
      Push(v, marks, out);
    }
    if (p.kind == Semantics::kSem && Expanded(p, j)) {
      PushPassing(g, mask, v, marks, out);
    }
  }
}

}  // namespace fse
