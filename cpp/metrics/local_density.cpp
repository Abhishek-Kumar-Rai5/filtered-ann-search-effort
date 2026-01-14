#include "metrics/local_density.h"

#include <stdexcept>

namespace fse {

std::vector<double> LocalFilteredDensity(const NeighborTable& unfiltered_gt,
                                         const std::vector<char>& mask,
                                         std::size_t k_local) {
  if (k_local == 0 || k_local > unfiltered_gt.k) {
    throw std::invalid_argument("LocalFilteredDensity: bad k_local");
  }
  std::vector<double> out(unfiltered_gt.nq);
  for (std::size_t q = 0; q < unfiltered_gt.nq; ++q) {
    std::size_t pass = 0;
    for (std::size_t j = 0; j < k_local; ++j) {
      const std::int64_t id = unfiltered_gt.Ids(q)[j];
      if (id < 0 || static_cast<std::size_t>(id) >= mask.size()) {
        throw std::invalid_argument("LocalFilteredDensity: bad neighbour id");
      }
      pass += mask[static_cast<std::size_t>(id)] != 0 ? 1 : 0;
    }
    out[q] = static_cast<double>(pass) / static_cast<double>(k_local);
  }
  return out;
}

}  // namespace fse
