#pragma once

// Directed adjacency in compressed sparse row form, list order preserved
// (docs/structural_design.md §3: ACORN's 2-hop rule depends on list
// positions). Node ids are base-vector ids; lists hold no -1 padding.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fse {

struct CsrGraph {
  std::vector<std::uint64_t> offsets;  // size nodes + 1
  std::vector<std::int32_t> ids;       // concatenated neighbour lists

  [[nodiscard]] std::size_t Nodes() const {
    return offsets.empty() ? 0 : offsets.size() - 1;
  }
  [[nodiscard]] const std::int32_t* Begin(std::size_t u) const {
    return ids.data() + offsets[u];
  }
  [[nodiscard]] std::size_t Degree(std::size_t u) const {
    return offsets[u + 1] - offsets[u];
  }
};

}  // namespace fse
