#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace fse {

// Dense row-major matrix. Used for vectors (float) and neighbour ids (int32).
template <typename T>
struct DenseMatrix {
  std::vector<T> data;
  std::size_t rows = 0;
  std::size_t dim = 0;

  [[nodiscard]] const T* Row(std::size_t i) const {
    return data.data() + i * dim;
  }
  [[nodiscard]] T* Row(std::size_t i) { return data.data() + i * dim; }
};

using FloatMatrix = DenseMatrix<float>;
using IdMatrix = DenseMatrix<std::int32_t>;

inline constexpr std::size_t kAllRows = std::numeric_limits<std::size_t>::max();

// TEXMEX .fvecs / .ivecs format: each vector is a little-endian int32 dim
// header followed by dim 4-byte values. Every vector must have the same dim.
// Reads the first min(max_rows, total) vectors; throws on malformed input.
FloatMatrix ReadFvecs(const std::string& path, std::size_t max_rows = kAllRows);
IdMatrix ReadIvecs(const std::string& path, std::size_t max_rows = kAllRows);

void WriteFvecs(const std::string& path, const FloatMatrix& m);
void WriteIvecs(const std::string& path, const IdMatrix& m);

}  // namespace fse
