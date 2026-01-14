#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace fse {

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

FloatMatrix ReadFvecs(const std::string& path, std::size_t max_rows = kAllRows);
IdMatrix ReadIvecs(const std::string& path, std::size_t max_rows = kAllRows);

void WriteFvecs(const std::string& path, const FloatMatrix& m);
void WriteIvecs(const std::string& path, const IdMatrix& m);

}
