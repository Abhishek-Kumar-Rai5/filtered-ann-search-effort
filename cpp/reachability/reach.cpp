#include "reachability/reach.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace fse {
namespace {

std::size_t FindRoot(std::vector<std::int32_t>* parent, std::size_t x) {
  auto& p = *parent;
  while (static_cast<std::size_t>(p[x]) != x) {
    p[x] = p[static_cast<std::size_t>(p[x])];
    x = static_cast<std::size_t>(p[x]);
  }
  return x;
}

}

std::vector<std::int32_t> StronglyConnected(const CsrGraph& g,
                                            std::size_t* ncomp) {
  const std::size_t n = g.Nodes();
  constexpr std::int32_t kUnset = -1;
  std::vector<std::int32_t> index(n, kUnset);
  std::vector<std::int32_t> low(n, 0);
  std::vector<std::int32_t> comp(n, kUnset);
  std::vector<char> on_stack(n, 0);
  std::vector<std::int32_t> stack;
  std::vector<std::pair<std::int32_t, std::uint64_t>> call;
  std::int32_t counter = 0;
  std::int32_t nc = 0;
  for (std::size_t root = 0; root < n; ++root) {
    if (index[root] != kUnset) {
      continue;
    }
    call.emplace_back(static_cast<std::int32_t>(root), g.offsets[root]);
    index[root] = low[root] = counter++;
    stack.push_back(static_cast<std::int32_t>(root));
    on_stack[root] = 1;
    while (!call.empty()) {
      auto& [v, e] = call.back();
      const auto vu = static_cast<std::size_t>(v);
      if (e < g.offsets[vu + 1]) {
        const auto w = static_cast<std::size_t>(g.ids[e]);
        ++e;
        if (index[w] == kUnset) {
          index[w] = low[w] = counter++;
          stack.push_back(static_cast<std::int32_t>(w));
          on_stack[w] = 1;
          call.emplace_back(static_cast<std::int32_t>(w), g.offsets[w]);
        } else if (on_stack[w] != 0) {
          low[vu] = std::min(low[vu], index[w]);
        }
        continue;
      }
      if (low[vu] == index[vu]) {
        std::int32_t w = 0;
        do {
          w = stack.back();
          stack.pop_back();
          on_stack[static_cast<std::size_t>(w)] = 0;
          comp[static_cast<std::size_t>(w)] = nc;
        } while (w != v);
        ++nc;
      }
      const std::int32_t finished = v;
      call.pop_back();
      if (!call.empty()) {
        const auto p = static_cast<std::size_t>(call.back().first);
        low[p] = std::min(low[p], low[static_cast<std::size_t>(finished)]);
      }
    }
  }
  *ncomp = static_cast<std::size_t>(nc);
  return comp;
}

ReachAnalyzer::ReachAnalyzer(const CsrGraph& g, const std::vector<char>& mask,
                             const SemanticsParams& p)
    : g_(g), mask_(mask), p_(p), local_(g.Nodes(), -1) {
  if (p.kind != Semantics::kUnfiltered && mask.size() != g.Nodes()) {
    throw std::invalid_argument("ReachAnalyzer: mask size != graph nodes");
  }
  const std::vector<std::int32_t> base_of = SelectEligible();
  Materialise(base_of);
  comp_ = StronglyConnected(h_, &ncomp_);
  Condense();
  ComputeStats(base_of);
}

std::vector<std::int32_t> ReachAnalyzer::SelectEligible() {
  const bool all = p_.kind == Semantics::kUnfiltered;
  std::vector<std::int32_t> base_of;
  for (std::size_t v = 0; v < g_.Nodes(); ++v) {
    if (all || mask_[v] != 0) {
      local_[v] = static_cast<std::int32_t>(base_of.size());
      base_of.push_back(static_cast<std::int32_t>(v));
    }
  }
  return base_of;
}

void ReachAnalyzer::Materialise(const std::vector<std::int32_t>& base_of) {
  MarkSet marks(g_.Nodes());
  std::vector<std::int32_t> succ;
  h_.offsets.reserve(base_of.size() + 1);
  h_.offsets.push_back(0);
  for (const std::int32_t v : base_of) {
    Successors(g_, mask_, static_cast<std::size_t>(v), p_, &marks, &succ);
    for (const std::int32_t w : succ) {
      h_.ids.push_back(local_[static_cast<std::size_t>(w)]);
    }
    h_.offsets.push_back(h_.ids.size());
  }
}

void ReachAnalyzer::Condense() {
  std::vector<std::vector<std::int32_t>> out(ncomp_);
  for (std::size_t v = 0; v < h_.Nodes(); ++v) {
    for (std::uint64_t e = h_.offsets[v]; e < h_.offsets[v + 1]; ++e) {
      const std::int32_t a = comp_[v];
      const std::int32_t b = comp_[static_cast<std::size_t>(h_.ids[e])];
      if (a != b) {
        out[static_cast<std::size_t>(a)].push_back(b);
      }
    }
  }
  dag_.offsets.push_back(0);
  for (auto& o : out) {
    std::sort(o.begin(), o.end());
    o.erase(std::unique(o.begin(), o.end()), o.end());
    dag_.ids.insert(dag_.ids.end(), o.begin(), o.end());
    dag_.offsets.push_back(dag_.ids.size());
  }
}

void ReachAnalyzer::ComputeStats(const std::vector<std::int32_t>& base_of) {
  stats_.nodes = h_.Nodes();
  stats_.edges = h_.ids.size();
  stats_.sccs = ncomp_;
  std::vector<std::size_t> scc_size(ncomp_, 0);
  for (const std::int32_t c : comp_) {
    ++scc_size[static_cast<std::size_t>(c)];
  }
  if (!scc_size.empty()) {
    const auto big = std::max_element(scc_size.begin(), scc_size.end());
    stats_.largest_scc = *big;
    const auto big_id = static_cast<std::int32_t>(big - scc_size.begin());
    const auto it = std::find(comp_.begin(), comp_.end(), big_id);
    largest_member_ = base_of[static_cast<std::size_t>(it - comp_.begin())];
  }
  std::vector<std::int32_t> parent(h_.Nodes());
  std::iota(parent.begin(), parent.end(), 0);
  for (std::size_t v = 0; v < h_.Nodes(); ++v) {
    for (std::uint64_t e = h_.offsets[v]; e < h_.offsets[v + 1]; ++e) {
      const std::size_t a = FindRoot(&parent, v);
      const std::size_t b =
          FindRoot(&parent, static_cast<std::size_t>(h_.ids[e]));
      if (a != b) {
        parent[a] = static_cast<std::int32_t>(b);
      }
    }
  }
  std::unordered_map<std::size_t, std::size_t> wcc;
  for (std::size_t v = 0; v < h_.Nodes(); ++v) {
    ++wcc[FindRoot(&parent, v)];
  }
  stats_.wccs = wcc.size();
  for (const auto& kv : wcc) {
    stats_.largest_wcc = std::max(stats_.largest_wcc, kv.second);
  }
}

const std::vector<char>& ReachAnalyzer::ReachFromSeed(std::int64_t seed) {
  const bool eligible = Eligible(seed);

  const std::int64_t key = eligible
                               ? comp_[static_cast<std::size_t>(
                                     local_[static_cast<std::size_t>(seed)])]
                               : -(seed + 1);
  auto it = memo_.find(key);
  if (it != memo_.end()) {
    return it->second;
  }
  std::vector<char> seen(ncomp_, 0);
  std::vector<std::int32_t> frontier;
  auto visit = [&](std::int32_t c) {
    if (seen[static_cast<std::size_t>(c)] == 0) {
      seen[static_cast<std::size_t>(c)] = 1;
      frontier.push_back(c);
    }
  };
  if (eligible) {
    visit(static_cast<std::int32_t>(key));
  } else {
    MarkSet marks(g_.Nodes());
    std::vector<std::int32_t> succ;
    Successors(g_, mask_, static_cast<std::size_t>(seed), p_, &marks, &succ);
    for (const std::int32_t w : succ) {
      const std::int32_t lw = local_[static_cast<std::size_t>(w)];
      if (lw >= 0) {
        visit(comp_[static_cast<std::size_t>(lw)]);
      }
    }
  }
  while (!frontier.empty()) {
    const auto c = static_cast<std::size_t>(frontier.back());
    frontier.pop_back();
    for (std::uint64_t e = dag_.offsets[c]; e < dag_.offsets[c + 1]; ++e) {
      visit(dag_.ids[e]);
    }
  }
  return memo_.emplace(key, std::move(seen)).first->second;
}

bool ReachAnalyzer::Reachable(std::int64_t seed, std::int64_t target) {
  if (seed < 0 || static_cast<std::size_t>(seed) >= g_.Nodes() || target < 0 ||
      static_cast<std::size_t>(target) >= g_.Nodes()) {
    throw std::out_of_range("Reachable: id outside graph");
  }
  if (seed == target) {
    return true;
  }
  const std::int32_t lt = local_[static_cast<std::size_t>(target)];
  if (lt < 0) {
    return false;
  }
  return ReachFromSeed(seed)[static_cast<std::size_t>(
             comp_[static_cast<std::size_t>(lt)])] != 0;
}

}
