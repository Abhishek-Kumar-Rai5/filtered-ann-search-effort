#include "metrics/structure_stats.h"

#include <cstdint>
#include <stdexcept>

namespace fse {

double ZeroFraction(const NeighborTable& unfiltered,
                    const std::vector<char>& mask, std::size_t k) {
  if (k == 0 || k > unfiltered.k || unfiltered.nq == 0) {
    throw std::invalid_argument("ZeroFraction: bad k or empty table");
  }
  std::size_t zeros = 0;
  for (std::size_t q = 0; q < unfiltered.nq; ++q) {
    bool any = false;
    for (std::size_t j = 0; j < k && !any; ++j) {
      const std::int64_t id = unfiltered.Ids(q)[j];
      if (id < 0 || static_cast<std::size_t>(id) >= mask.size()) {
        throw std::invalid_argument("ZeroFraction: bad neighbour id");
      }
      any = mask[static_cast<std::size_t>(id)] != 0;
    }
    zeros += any ? 0 : 1;
  }
  return static_cast<double>(zeros) / static_cast<double>(unfiltered.nq);
}

double ExactHomophily(const NeighborTable& base_knn,
                      const std::vector<char>& mask, std::size_t k) {
  if (base_knn.nq != mask.size() || k == 0 || base_knn.k < k + 1) {
    throw std::invalid_argument("ExactHomophily: bad table / mask / k");
  }
  double sum = 0.0;
  std::size_t passing = 0;
  for (std::size_t i = 0; i < base_knn.nq; ++i) {
    if (mask[i] == 0) {
      continue;
    }
    std::size_t seen = 0;
    std::size_t pass = 0;
    for (std::size_t j = 0; j < base_knn.k && seen < k; ++j) {
      const std::int64_t id = base_knn.Ids(i)[j];
      if (id == static_cast<std::int64_t>(i)) {
        continue;
      }
      ++seen;
      pass += mask[static_cast<std::size_t>(id)] != 0 ? 1 : 0;
    }
    sum += static_cast<double>(pass) / static_cast<double>(seen);
    ++passing;
  }
  return passing == 0 ? 0.0 : sum / static_cast<double>(passing);
}

double SampledHomophily(const NeighborTable& sample_knn,
                        const std::vector<std::int32_t>& sample,
                        const std::vector<char>& mask, std::size_t k) {
  if (sample_knn.nq != sample.size() || k == 0 || sample_knn.k < k + 1) {
    throw std::invalid_argument("SampledHomophily: bad table / sample / k");
  }
  double sum = 0.0;
  std::size_t count = 0;
  for (std::size_t r = 0; r < sample.size(); ++r) {
    const auto self = static_cast<std::int64_t>(sample[r]);
    if (mask[static_cast<std::size_t>(self)] == 0) {
      continue;
    }
    std::size_t seen = 0;
    std::size_t pass = 0;
    for (std::size_t j = 0; j < sample_knn.k && seen < k; ++j) {
      const std::int64_t id = sample_knn.Ids(r)[j];
      if (id == self) {
        continue;
      }
      ++seen;
      pass += mask[static_cast<std::size_t>(id)] != 0 ? 1 : 0;
    }
    sum += static_cast<double>(pass) / static_cast<double>(seen);
    ++count;
  }
  return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

double ChanceCorrected(double observed, double chance) {
  if (!(chance < 1.0)) {
    throw std::invalid_argument("ChanceCorrected: chance must be < 1");
  }
  return (observed - chance) / (1.0 - chance);
}

double EmpiricalPUpper(const std::vector<double>& null, double observed) {
  std::size_t ge = 0;
  for (const double x : null) {
    ge += x >= observed ? 1 : 0;
  }
  return static_cast<double>(1 + ge) / static_cast<double>(null.size() + 1);
}

double EmpiricalPLower(const std::vector<double>& null, double observed) {
  std::size_t le = 0;
  for (const double x : null) {
    le += x <= observed ? 1 : 0;
  }
  return static_cast<double>(1 + le) / static_cast<double>(null.size() + 1);
}

}
