#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fse {

std::vector<std::int32_t> UniformIntAttributes(std::size_t n, std::int32_t lo,
                                               std::int32_t hi,
                                               std::uint64_t seed);

std::vector<std::int32_t> SeededPermutation(std::size_t n, std::uint64_t seed);

}
