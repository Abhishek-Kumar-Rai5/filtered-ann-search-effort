#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>

#include "filter_generator/uniform_attributes.h"

namespace {

TEST(UniformAttributes, DeterministicForSeedAndDiffersAcrossSeeds) {
  const auto a = fse::UniformIntAttributes(1000, 1, 12, 7);
  const auto b = fse::UniformIntAttributes(1000, 1, 12, 7);
  const auto c = fse::UniformIntAttributes(1000, 1, 12, 8);
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

TEST(UniformAttributes, PrefixStableAcrossLengths) {
  const auto a = fse::UniformIntAttributes(100, 1, 12, 3);
  const auto b = fse::UniformIntAttributes(1000, 1, 12, 3);
  EXPECT_EQ(a, std::vector<std::int32_t>(b.begin(), b.begin() + 100));
}

TEST(UniformAttributes, ValuesInRangeAndApproximatelyUniform) {
  constexpr std::size_t kN = 1'200'000;
  const auto v = fse::UniformIntAttributes(kN, 1, 12, 20261002);
  std::map<std::int32_t, std::size_t> counts;
  for (const std::int32_t x : v) {
    ASSERT_GE(x, 1);
    ASSERT_LE(x, 12);
    ++counts[x];
  }
  ASSERT_EQ(counts.size(), 12U);

  for (const auto& [value, count] : counts) {
    EXPECT_NEAR(static_cast<double>(count), 100000.0, 1820.0) << value;
  }
}

TEST(UniformAttributes, DegenerateAndInvalidRanges) {
  const auto v = fse::UniformIntAttributes(10, 5, 5, 1);
  EXPECT_EQ(v, std::vector<std::int32_t>(10, 5));
  EXPECT_THROW(fse::UniformIntAttributes(10, 3, 2, 1), std::invalid_argument);
  EXPECT_TRUE(fse::UniformIntAttributes(0, 1, 12, 1).empty());
}

}
