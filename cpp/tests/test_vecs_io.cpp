#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/run_metadata.h"
#include "common/vecs_io.h"

namespace {

std::string TempPath(const std::string& name) {
  return ::testing::TempDir() + "/" + name;
}

TEST(VecsIo, FvecsAndIvecsRoundTrip) {
  const fse::FloatMatrix f{{1.5F, -2, 3, 4, 5, 6.25F}, 3, 2};
  const fse::IdMatrix i{{7, 8, 9, 10, 11, 12}, 2, 3};
  fse::WriteFvecs(TempPath("rt.fvecs"), f);
  fse::WriteIvecs(TempPath("rt.ivecs"), i);
  const auto f2 = fse::ReadFvecs(TempPath("rt.fvecs"));
  const auto i2 = fse::ReadIvecs(TempPath("rt.ivecs"));
  EXPECT_EQ(f2.rows, 3U);
  EXPECT_EQ(f2.dim, 2U);
  EXPECT_EQ(f2.data, f.data);
  EXPECT_EQ(i2.rows, 2U);
  EXPECT_EQ(i2.dim, 3U);
  EXPECT_EQ(i2.data, i.data);
  const auto head = fse::ReadFvecs(TempPath("rt.fvecs"), 2);
  EXPECT_EQ(head.rows, 2U);
  EXPECT_EQ(head.data, (std::vector<float>{1.5F, -2, 3, 4}));
}

TEST(VecsIo, RejectsTruncatedFileAndMissingFile) {
  const fse::FloatMatrix f{{1, 2, 3, 4}, 2, 2};
  const std::string path = TempPath("trunc.fvecs");
  fse::WriteFvecs(path, f);
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 2);
  EXPECT_THROW(fse::ReadFvecs(path), std::runtime_error);
  EXPECT_THROW(fse::ReadFvecs(TempPath("does_not_exist.fvecs")),
               std::runtime_error);
}

TEST(RunMetadata, HashIsStableAndJsonIsEscaped) {
  EXPECT_EQ(fse::Fnv1a64("abc"), fse::Fnv1a64("abc"));
  EXPECT_NE(fse::Fnv1a64("abc"), fse::Fnv1a64("abd"));
  EXPECT_EQ(fse::Hex64(255).size(), 16U);
  const std::string json =
      fse::JsonObject().Str("a", "x\"y\n").Int("b", 3).Bool("c", true).Render();
  EXPECT_EQ(json, R"({"a": "x\"y\n", "b": 3, "c": true})");
}

}
