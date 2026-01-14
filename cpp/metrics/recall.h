#pragma once

#include <cstddef>
#include <cstdint>

namespace fse {

// Recall@k = |R ∩ G| / k, with R the first k returned ids and G the first k
// ground-truth ids (ACORN paper section 3; ACORN's compute_recall). Id-based:
// an equidistant substitute for a ground-truth neighbour counts as a miss.
// Negative ids (padding for "no result") never match.
double RecallAtK(const std::int64_t* returned, const std::int64_t* truth,
                 std::size_t k);

}  // namespace fse
