#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ground_truth/filtered_ground_truth.h"
#include "metrics/structure_stats.h"

namespace {

TEST(StructureStats, ZeroFractionCountsQueriesWithNoPassingNeighbour) {
  const fse::NeighborTable t{
      3, 3, {0, 1, 2, 3, 4, 5, 1, 3, 5}, std::vector<float>(9, 0.0F)};
  const std::vector<char> mask = {0, 0, 1, 0, 0, 0};
  EXPECT_DOUBLE_EQ(fse::ZeroFraction(t, mask, 3), 2.0 / 3.0);
  EXPECT_DOUBLE_EQ(fse::ZeroFraction(t, mask, 2), 1.0);
  EXPECT_THROW(fse::ZeroFraction(t, mask, 4), std::invalid_argument);
  EXPECT_THROW(fse::ZeroFraction(t, mask, 0), std::invalid_argument);
}

TEST(StructureStats, ExactHomophilyExcludesSelfAndAveragesPassingOnly) {
  const fse::NeighborTable knn{
      4, 3, {0, 1, 2, 1, 0, 3, 2, 3, 0, 3, 2, 1}, std::vector<float>(12, 0.0F)};
  const std::vector<char> mask = {1, 1, 0, 0};

  EXPECT_DOUBLE_EQ(fse::ExactHomophily(knn, mask, 2), 0.5);
  EXPECT_DOUBLE_EQ(fse::ExactHomophily(knn, {0, 0, 0, 0}, 2), 0.0);
  EXPECT_DOUBLE_EQ(fse::ExactHomophily(knn, {1, 1, 1, 1}, 2), 1.0);
  EXPECT_THROW(fse::ExactHomophily(knn, mask, 3), std::invalid_argument);
  EXPECT_THROW(fse::ExactHomophily(knn, {1, 0}, 2), std::invalid_argument);
}

TEST(StructureStats, ChanceCorrectedIsZeroAtChanceAndOneAtMaximum) {
  EXPECT_DOUBLE_EQ(fse::ChanceCorrected(0.1, 0.1), 0.0);
  EXPECT_DOUBLE_EQ(fse::ChanceCorrected(1.0, 0.4), 1.0);
  EXPECT_DOUBLE_EQ(fse::ChanceCorrected(0.55, 0.1), 0.5);
  EXPECT_LT(fse::ChanceCorrected(0.0, 0.2), 0.0);
  EXPECT_THROW(fse::ChanceCorrected(1.0, 1.0), std::invalid_argument);
}

TEST(StructureStats, EmpiricalPValuesCountTiesConservatively) {
  const std::vector<double> null = {1, 2, 3, 4};
  EXPECT_DOUBLE_EQ(fse::EmpiricalPUpper(null, 5), 1.0 / 5.0);
  EXPECT_DOUBLE_EQ(fse::EmpiricalPUpper(null, 4), 2.0 / 5.0);
  EXPECT_DOUBLE_EQ(fse::EmpiricalPUpper(null, 0), 5.0 / 5.0);
  EXPECT_DOUBLE_EQ(fse::EmpiricalPLower(null, 0), 1.0 / 5.0);
  EXPECT_DOUBLE_EQ(fse::EmpiricalPLower(null, 1), 2.0 / 5.0);
}

TEST(StructureStats, SampledHomophilyOverAllVectorsEqualsExact) {
  const fse::NeighborTable knn{
      4, 3, {0, 1, 2, 1, 0, 3, 2, 3, 0, 3, 2, 1}, std::vector<float>(12, 0.0F)};
  const std::vector<char> mask = {1, 1, 0, 1};
  const std::vector<std::int32_t> all = {0, 1, 2, 3};
  EXPECT_DOUBLE_EQ(fse::SampledHomophily(knn, all, mask, 2),
                   fse::ExactHomophily(knn, mask, 2));

  const fse::NeighborTable sub{
      2, 3, {1, 0, 3, 3, 2, 1}, std::vector<float>(6, 0.0F)};
  EXPECT_DOUBLE_EQ(fse::SampledHomophily(sub, {1, 3}, mask, 2), 0.75);
  EXPECT_THROW(fse::SampledHomophily(sub, {1}, mask, 2), std::invalid_argument);
}

}
