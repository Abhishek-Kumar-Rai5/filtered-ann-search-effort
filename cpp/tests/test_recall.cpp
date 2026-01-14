#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "metrics/recall.h"

namespace {

double R(const std::vector<std::int64_t>& got,
         const std::vector<std::int64_t>& truth, std::size_t k) {
  return fse::RecallAtK(got.data(), truth.data(), k);
}

TEST(Recall, CountsIdOverlapRegardlessOfOrder) {
  EXPECT_DOUBLE_EQ(R({5, 3, 9, 1}, {5, 3, 9, 1}, 4), 1.0);
  EXPECT_DOUBLE_EQ(R({1, 9, 3, 5}, {5, 3, 9, 1}, 4), 1.0);
  EXPECT_DOUBLE_EQ(R({5, 3, 0, 2}, {5, 3, 9, 1}, 4), 0.5);
  EXPECT_DOUBLE_EQ(R({7, 8, 0, 2}, {5, 3, 9, 1}, 4), 0.0);
}

TEST(Recall, PaddingNeverMatchesAndDuplicatesCountOnce) {
  EXPECT_DOUBLE_EQ(R({5, -1, -1, -1}, {5, 3, 9, 1}, 4), 0.25);
  EXPECT_DOUBLE_EQ(R({5, 5, 5, 5}, {5, 3, 9, 1}, 4), 0.25);

  EXPECT_DOUBLE_EQ(R({-1, -1}, {-1, -1}, 2), 0.0);
}

TEST(Recall, OnlyFirstKOfEachListCount) {
  EXPECT_DOUBLE_EQ(R({0, 5, 3}, {5, 3, 9}, 2), 0.5);
  EXPECT_DOUBLE_EQ(R({}, {}, 0), 0.0);
}

TEST(Recall, IdBasedEquidistantSubstituteIsAMiss) {
  EXPECT_DOUBLE_EQ(R({1, 2, 99}, {1, 2, 3}, 3), 2.0 / 3.0);
}

}
