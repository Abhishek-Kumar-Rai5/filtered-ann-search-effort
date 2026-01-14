// Phase 5 live features: Project 1 definitions on hand-computed inputs (tests
// ported from ann_router_staleness tests/test_features.cpp), LID against a
// known-dimension analytic case, and the live density proxy.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "feature_extraction/live_features.h"

namespace {

std::vector<float> Sq(const std::vector<double>& d) {
  std::vector<float> out;
  for (const double x : d) {
    out.push_back(static_cast<float>(x * x));
  }
  return out;
}

TEST(LiveFeatures, ScoreConcentrationOnHandValues) {
  EXPECT_DOUBLE_EQ(fse::ScoreConcentration(Sq({1, 2, 4, 8}), 4), 1.0 / 8.0);
  EXPECT_DOUBLE_EQ(fse::ScoreConcentration(Sq({1, 2, 4, 8}), 2), 1.0 / 2.0);
  EXPECT_DOUBLE_EQ(fse::ScoreConcentration(Sq({3, 3, 3}), 3), 1.0);
  EXPECT_DOUBLE_EQ(fse::ScoreConcentration(Sq({0, 0}), 2), 1.0);
  EXPECT_THROW((void)fse::ScoreConcentration(Sq({1, 2}), 3),
               std::invalid_argument);
  EXPECT_THROW((void)fse::ScoreConcentration(Sq({1}), 0),
               std::invalid_argument);
}

TEST(LiveFeatures, LidMleOnHandValuesAndKnownDimension) {
  EXPECT_NEAR(fse::LidMle(Sq({1, 2, 4}), 3), 2.0 / std::log(8.0), 1e-6);
  EXPECT_NEAR(fse::LidMle(Sq({1, 2, 4, 100}), 3), 2.0 / std::log(8.0), 1e-6);
  for (const int dim : {2, 5, 12}) {
    std::vector<double> d;
    const int k = 2000;
    for (int i = 1; i <= k; ++i) {
      d.push_back(std::pow(static_cast<double>(i) / k, 1.0 / dim));
    }
    EXPECT_NEAR(fse::LidMle(Sq(d), k), dim, 0.05 * dim) << "D=" << dim;
  }
}

TEST(LiveFeatures, LidMleEdgeConventions) {
  EXPECT_DOUBLE_EQ(fse::LidMle(Sq({0, 0, 0}), 3), 0.0);
  EXPECT_DOUBLE_EQ(fse::LidMle(Sq({0, 1, 2}), 3), 0.0);
  EXPECT_EQ(fse::LidMle(Sq({2, 2, 2}), 3),
            std::numeric_limits<double>::infinity());
  EXPECT_THROW((void)fse::LidMle(Sq({1}), 1), std::invalid_argument);
  EXPECT_THROW((void)fse::LidMle(Sq({1, 2}), 3), std::invalid_argument);
}

TEST(LiveFeatures, CentroidAndCentroidDistance) {
  const fse::FloatMatrix base{{0, 0, 2, 4, 4, 8}, 3, 2};
  const auto c = fse::ComputeCentroid(base);
  ASSERT_EQ(c.size(), 2U);
  EXPECT_DOUBLE_EQ(c[0], 2.0);
  EXPECT_DOUBLE_EQ(c[1], 4.0);
  const std::vector<float> q = {5.0F, 8.0F};
  EXPECT_DOUBLE_EQ(fse::CentroidDistance(q.data(), c), 5.0);
  EXPECT_THROW((void)fse::ComputeCentroid(fse::FloatMatrix{}),
               std::invalid_argument);
}

TEST(LiveFeatures, PassingFractionReadsOnlyTheMask) {
  const std::vector<char> mask = {1, 0, 0, 1, 1};
  EXPECT_DOUBLE_EQ(fse::PassingFraction({0, 1, 2, 3}, mask), 0.5);
  EXPECT_DOUBLE_EQ(fse::PassingFraction({1, 2}, mask), 0.0);
  EXPECT_DOUBLE_EQ(fse::PassingFraction({4}, mask), 1.0);
  EXPECT_THROW((void)fse::PassingFraction({}, mask), std::invalid_argument);
  EXPECT_THROW((void)fse::PassingFraction({5}, mask), std::out_of_range);
}

}  // namespace
