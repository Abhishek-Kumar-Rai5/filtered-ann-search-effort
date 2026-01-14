#pragma once

#include <cstdint>

namespace fse {

// ACORN never counts the distance to its entry point, hence the +1.
inline constexpr std::uint64_t kAcornUncountedPerQuery = 1;

inline constexpr std::uint64_t AcornExactDistanceComputations(
    std::uint64_t native_n3_per_query) {
  return native_n3_per_query + kAcornUncountedPerQuery;
}

}
