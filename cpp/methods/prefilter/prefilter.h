#pragma once

#include <cstddef>

#include "common/vecs_io.h"
#include "methods/filtered_result.h"

namespace fse {

FilteredSearchResult PrefilterSearch(const FloatMatrix& base,
                                     const float* query, std::size_t k,
                                     const char* filter_row);

}
