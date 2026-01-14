#pragma once

#include <cstddef>
#include <cstdint>

#include "methods/filtered_result.h"
#include "methods/postfilter/hnsw_index.h"

namespace fse {

struct PostfilterParams {
  std::size_t ef_search = 0;
  double overfetch_factor = 1.0;
  double growth_factor = 2.0;
  std::uint32_t max_rounds = 1;
};

std::size_t PostfilterFetchSize(std::size_t k, double global_selectivity,
                                const PostfilterParams& p, std::uint32_t round,
                                std::size_t n);

FilteredSearchResult PostfilterSearch(const HnswIndex& index,
                                      const float* query, std::size_t k,
                                      const char* filter_row,
                                      double global_selectivity,
                                      const PostfilterParams& params);

}
