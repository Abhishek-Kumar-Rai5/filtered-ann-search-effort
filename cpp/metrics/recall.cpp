#include "metrics/recall.h"

#include <algorithm>
#include <vector>

namespace fse {

double RecallAtK(const std::int64_t* returned, const std::int64_t* truth,
                 std::size_t k) {
  if (k == 0) {
    return 0.0;
  }
  std::vector<std::int64_t> g(truth, truth + k);
  std::sort(g.begin(), g.end());
  g.erase(std::unique(g.begin(), g.end()), g.end());
  std::vector<std::int64_t> r(returned, returned + k);
  std::sort(r.begin(), r.end());
  r.erase(std::unique(r.begin(), r.end()), r.end());

  std::size_t hits = 0;
  for (const std::int64_t id : r) {
    if (id >= 0 && std::binary_search(g.begin(), g.end(), id)) {
      ++hits;
    }
  }
  return static_cast<double>(hits) / static_cast<double>(k);
}

}
