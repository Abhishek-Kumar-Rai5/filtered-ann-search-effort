#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fse {

struct FilteredSearchResult {
  std::vector<std::int64_t> ids;
  std::vector<float> distances;
  std::uint64_t distance_computations = 0;
  std::uint64_t filter_checks = 0;

  std::uint32_t rounds = 0;
  std::size_t last_fetch = 0;
};

}
