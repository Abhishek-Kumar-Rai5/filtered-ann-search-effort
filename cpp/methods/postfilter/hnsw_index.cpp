#include "methods/postfilter/hnsw_index.h"

#include <stdexcept>

namespace fse {

CountingL2Space::CountingL2Space(std::size_t dim) : inner_(dim) {
  param_.inner = inner_.get_dist_func();
  param_.inner_param = inner_.get_dist_func_param();
}

std::size_t CountingL2Space::get_data_size() { return inner_.get_data_size(); }

hnswlib::DISTFUNC<float> CountingL2Space::get_dist_func() {
  return &CountingDistance;
}

void* CountingL2Space::get_dist_func_param() { return &param_; }

std::uint64_t& CountingL2Space::ThreadCount() {
  thread_local std::uint64_t count = 0;
  return count;
}

float CountingL2Space::CountingDistance(const void* a, const void* b,
                                        const void* p) {
  ++ThreadCount();
  const auto* param = static_cast<const Param*>(p);
  return param->inner(a, b, param->inner_param);
}

HnswIndex::HnswIndex(std::size_t dim, std::size_t max_elements,
                     const HnswParams& p)
    : dim_(dim), space_(std::make_unique<CountingL2Space>(dim)) {
  hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      space_.get(), max_elements, p.m, p.ef_construction, p.seed);
}

HnswIndex HnswIndex::Load(const std::string& path, std::size_t dim) {
  HnswIndex index;
  index.dim_ = dim;
  index.space_ = std::make_unique<CountingL2Space>(dim);
  index.hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      index.space_.get(), path);
  return index;
}

void HnswIndex::Add(const FloatMatrix& vectors, std::size_t first_label,
                    int num_threads) {
  if (vectors.dim != dim_) {
    throw std::invalid_argument("HnswIndex::Add: dim mismatch");
  }
  const auto n = static_cast<std::int64_t>(vectors.rows);
  if (num_threads <= 1) {
    for (std::int64_t i = 0; i < n; ++i) {
      hnsw_->addPoint(vectors.Row(i), first_label + i);
    }
    return;
  }
#pragma omp parallel for schedule(dynamic, 256) num_threads(num_threads)
  for (std::int64_t i = 0; i < n; ++i) {
    hnsw_->addPoint(vectors.Row(i), first_label + i);
  }
}

void HnswIndex::Save(const std::string& path) { hnsw_->saveIndex(path); }

void HnswIndex::SetEf(std::size_t ef) { hnsw_->setEf(ef); }

std::size_t HnswIndex::Ef() const { return hnsw_->ef_; }

HnswSearchResult HnswIndex::Search(const float* query, std::size_t k,
                                   std::size_t ef) {
  SetEf(ef);
  return SearchAtCurrentEf(query, k);
}

HnswSearchResult HnswIndex::SearchAtCurrentEf(const float* query,
                                              std::size_t k) const {
  std::uint64_t& counter = CountingL2Space::ThreadCount();
  counter = 0;
  auto heap = hnsw_->searchKnn(query, k);

  HnswSearchResult r;
  r.distance_computations = counter;
  r.labels.resize(heap.size());
  r.dists.resize(heap.size());
  // hnswlib returns a max-heap; fill from the back to get ascending order.
  for (std::size_t i = heap.size(); i-- > 0;) {
    r.dists[i] = heap.top().first;
    r.labels[i] = heap.top().second;
    heap.pop();
  }
  return r;
}

std::size_t HnswIndex::Size() const { return hnsw_->getCurrentElementCount(); }

CsrGraph HnswIndex::Level0Graph() const {
  const std::size_t n = Size();
  std::vector<std::size_t> internal_of(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto label = static_cast<std::size_t>(
        hnsw_->getExternalLabel(static_cast<hnswlib::tableint>(i)));
    if (label >= n) {
      throw std::runtime_error("Level0Graph: label outside [0, n)");
    }
    internal_of[label] = i;
  }
  CsrGraph g;
  g.offsets.reserve(n + 1);
  g.offsets.push_back(0);
  for (std::size_t label = 0; label < n; ++label) {
    auto* list = hnsw_->get_linklist0(
        static_cast<hnswlib::tableint>(internal_of[label]));
    const std::size_t size = hnsw_->getListCount(list);
    const auto* data = reinterpret_cast<const hnswlib::tableint*>(list + 1);
    for (std::size_t j = 0; j < size; ++j) {
      g.ids.push_back(
          static_cast<std::int32_t>(hnsw_->getExternalLabel(data[j])));
    }
    g.offsets.push_back(g.ids.size());
  }
  return g;
}

}  // namespace fse
