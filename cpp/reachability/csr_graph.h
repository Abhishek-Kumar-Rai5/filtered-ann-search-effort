#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fse {

struct CsrGraph {
  std::vector<std::uint64_t> offsets;
  std::vector<std::int32_t> ids;

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

}
