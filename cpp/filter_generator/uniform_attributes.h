#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fse {

// n i.i.d. attributes, each uniform in [lo, hi], from a seeded
// std::mt19937_64 with unbiased rejection sampling. Not built on
// std::uniform_int_distribution, whose output is standard-library specific,
// so a seed gives the same draw on any toolchain. This is the "random
// integer attribute" model of the ACORN paper (section 7.1.1).
std::vector<std::int32_t> UniformIntAttributes(std::size_t n, std::int32_t lo,
                                               std::int32_t hi,
                                               std::uint64_t seed);

// A uniformly random permutation of 0..n-1 (Fisher-Yates with the same
// toolchain-independent unbiased draws). Deterministic for (n, seed).
std::vector<std::int32_t> SeededPermutation(std::size_t n, std::uint64_t seed);

}  // namespace fse
