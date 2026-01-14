#pragma once

#include <cstddef>
#include <cstdint>

namespace fse {

double RecallAtK(const std::int64_t* returned, const std::int64_t* truth,
                 std::size_t k);

}
