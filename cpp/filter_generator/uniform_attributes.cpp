#include "filter_generator/uniform_attributes.h"

#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace fse {
namespace {

// Unbiased uniform draw in [0, range) by rejection: draws at or above the
// largest multiple of `range` representable in 64 bits are rejected.
std::uint64_t BoundedDraw(std::mt19937_64& rng, std::uint64_t range) {
  constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t limit = kMax - ((kMax % range) + 1) % range;
  std::uint64_t draw = rng();
  while (draw > limit) {
    draw = rng();
  }
  return draw % range;
}

}  // namespace

std::vector<std::int32_t> UniformIntAttributes(std::size_t n, std::int32_t lo,
                                               std::int32_t hi,
                                               std::uint64_t seed) {
  if (hi < lo) {
    throw std::invalid_argument("UniformIntAttributes: hi < lo");
  }
  const auto range =
      static_cast<std::uint64_t>(static_cast<std::int64_t>(hi) - lo) + 1;
  std::mt19937_64 rng(seed);
  std::vector<std::int32_t> out(n);
  for (std::int32_t& v : out) {
    v = static_cast<std::int32_t>(
        lo + static_cast<std::int64_t>(BoundedDraw(rng, range)));
  }
  return out;
}

std::vector<std::int32_t> SeededPermutation(std::size_t n, std::uint64_t seed) {
  if (n > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::invalid_argument("SeededPermutation: n too large");
  }
  std::vector<std::int32_t> p(n);
  for (std::size_t i = 0; i < n; ++i) {
    p[i] = static_cast<std::int32_t>(i);
  }
  // Fisher-Yates from the back with unbiased bounded draws.
  std::mt19937_64 rng(seed);
  for (std::size_t i = n; i > 1; --i) {
    const auto j = static_cast<std::size_t>(BoundedDraw(rng, i));
    std::swap(p[i - 1], p[j]);
  }
  return p;
}

}  // namespace fse
