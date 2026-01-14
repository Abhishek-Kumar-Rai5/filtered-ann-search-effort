#include "common/vecs_io.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace fse {
namespace {

template <typename T>
DenseMatrix<T> ReadVecs(const std::string& path, std::size_t max_rows) {
  static_assert(sizeof(T) == 4);
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  std::int32_t dim = 0;
  if (!in.read(reinterpret_cast<char*>(&dim), sizeof(dim)) || dim <= 0) {
    throw std::runtime_error("bad or empty vecs header in " + path);
  }
  in.seekg(0, std::ios::end);
  const auto file_bytes = static_cast<std::size_t>(in.tellg());
  const std::size_t record_bytes = sizeof(std::int32_t) + dim * sizeof(T);
  if (file_bytes % record_bytes != 0) {
    throw std::runtime_error("file size not a multiple of record size: " +
                             path);
  }
  const std::size_t rows = std::min(max_rows, file_bytes / record_bytes);

  DenseMatrix<T> m;
  m.rows = rows;
  m.dim = static_cast<std::size_t>(dim);
  m.data.resize(rows * m.dim);
  in.seekg(0, std::ios::beg);
  for (std::size_t i = 0; i < rows; ++i) {
    std::int32_t row_dim = 0;
    in.read(reinterpret_cast<char*>(&row_dim), sizeof(row_dim));
    if (row_dim != dim) {
      throw std::runtime_error("inconsistent dim at row " + std::to_string(i) +
                               " in " + path);
    }
    in.read(reinterpret_cast<char*>(m.Row(i)),
            static_cast<std::streamsize>(m.dim * sizeof(T)));
    if (!in) {
      throw std::runtime_error("truncated read in " + path);
    }
  }
  return m;
}

template <typename T>
void WriteVecs(const std::string& path, const DenseMatrix<T>& m) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cannot open for writing: " + path);
  }
  const auto dim = static_cast<std::int32_t>(m.dim);
  for (std::size_t i = 0; i < m.rows; ++i) {
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(m.Row(i)),
              static_cast<std::streamsize>(m.dim * sizeof(T)));
  }
  if (!out) {
    throw std::runtime_error("write failed: " + path);
  }
}

}  // namespace

FloatMatrix ReadFvecs(const std::string& path, std::size_t max_rows) {
  return ReadVecs<float>(path, max_rows);
}
IdMatrix ReadIvecs(const std::string& path, std::size_t max_rows) {
  return ReadVecs<std::int32_t>(path, max_rows);
}
void WriteFvecs(const std::string& path, const FloatMatrix& m) {
  WriteVecs(path, m);
}
void WriteIvecs(const std::string& path, const IdMatrix& m) {
  WriteVecs(path, m);
}

}  // namespace fse
