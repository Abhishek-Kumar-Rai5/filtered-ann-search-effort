#include "methods/ingraph/acorn_index.h"

#include <faiss/IndexACORN.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/ACORN.h>
#include <faiss/index_io.h>

#include <chrono>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace fse {
namespace {

double SecondsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

}

AcornIndex::AcornIndex(int dim, const AcornParams& params,
                       std::vector<std::int32_t> attributes)
    : attributes_(attributes.begin(), attributes.end()) {
  index_ = std::make_unique<faiss::IndexACORNFlat>(dim, params.m, params.gamma,
                                                   attributes_, params.m_beta);
}

AcornIndex::~AcornIndex() = default;
AcornIndex::AcornIndex(AcornIndex&&) noexcept = default;
AcornIndex& AcornIndex::operator=(AcornIndex&&) noexcept = default;

AcornIndex AcornIndex::Load(const std::string& path,
                            std::vector<std::int32_t> attributes) {
  std::unique_ptr<faiss::Index> raw(faiss::read_index(path.c_str()));

  if (dynamic_cast<faiss::IndexACORNFlat*>(raw.get()) == nullptr) {
    throw std::runtime_error("not an ACORN index: " + path);
  }
  AcornIndex out;
  out.index_.reset(static_cast<faiss::IndexACORNFlat*>(raw.release()));
  out.attributes_.assign(attributes.begin(), attributes.end());
  if (out.attributes_.size() != out.Size()) {
    throw std::invalid_argument("AcornIndex::Load: attribute count " +
                                std::to_string(out.attributes_.size()) +
                                " != index size " + std::to_string(out.Size()));
  }
  out.AttachAttributes();
  return out;
}

void AcornIndex::AttachAttributes() {
  index_->acorn.metadata = attributes_.data();
}

void AcornIndex::Save(const std::string& path) const {
  faiss::write_index(index_.get(), path.c_str());
}

double AcornIndex::Add(std::size_t n, const float* x) {
  if (index_->ntotal + static_cast<faiss::idx_t>(n) >
      static_cast<faiss::idx_t>(attributes_.size())) {
    throw std::invalid_argument(
        "AcornIndex::Add: more vectors than attributes");
  }
  const auto t0 = std::chrono::steady_clock::now();
  index_->add(static_cast<faiss::idx_t>(n), x);
  return SecondsSince(t0);
}

void AcornIndex::SetEfSearch(int ef_search) {
  index_->acorn.efSearch = ef_search;
}
int AcornIndex::EfSearch() const { return index_->acorn.efSearch; }
int AcornIndex::EfConstruction() const { return index_->acorn.efConstruction; }
std::size_t AcornIndex::Size() const {
  return static_cast<std::size_t>(index_->ntotal);
}
int AcornIndex::Dim() const { return index_->d; }
int AcornIndex::Gamma() const { return index_->acorn.gamma; }
int AcornIndex::M() const { return index_->acorn.M; }
int AcornIndex::MBeta() const { return index_->acorn.M_beta; }

CsrGraph AcornIndex::Level0Graph() const {
  const faiss::ACORN& a = index_->acorn;
  CsrGraph g;
  g.offsets.reserve(Size() + 1);
  g.offsets.push_back(0);
  for (std::size_t i = 0; i < Size(); ++i) {
    std::size_t begin = 0;
    std::size_t end = 0;
    a.neighbor_range(static_cast<faiss::idx_t>(i), 0, &begin, &end);
    for (std::size_t j = begin; j < end && a.neighbors[j] >= 0; ++j) {
      g.ids.push_back(a.neighbors[j]);
    }
    g.offsets.push_back(g.ids.size());
  }
  return g;
}

void AcornIndex::SearchBatch(std::size_t nq, const float* queries,
                             std::size_t k, const char* filter_map,
                             std::int64_t* ids, float* distances) const {
  static_assert(sizeof(faiss::idx_t) == sizeof(std::int64_t));

  index_->search(static_cast<faiss::idx_t>(nq), queries,
                 static_cast<faiss::idx_t>(k), distances,
                 reinterpret_cast<faiss::idx_t*>(ids),
                 const_cast<char*>(filter_map));
}

AcornQueryResult AcornIndex::SearchOne(const float* query, std::size_t k,
                                       const char* filter_row) const {
  AcornQueryResult r;
  r.ids.assign(k, -1);
  r.distances.assign(k, 0.0F);
  const std::size_t n3_before = faiss::acorn_stats.n3;
  const std::size_t scanned_before = faiss::acorn_stats.n_scanned;
  const auto t0 = std::chrono::steady_clock::now();
  SearchBatch(1, query, k, filter_row, r.ids.data(), r.distances.data());
  r.seconds = SecondsSince(t0);
  r.distance_computations = faiss::acorn_stats.n3 - n3_before;
  r.entries_scanned = faiss::acorn_stats.n_scanned - scanned_before;
  r.level0_seed = faiss::acorn_level0_seed;
  return r;
}

std::vector<double> AcornIndex::AverageOutDegreePerLevel() const {
  const faiss::ACORN& a = index_->acorn;
  std::vector<double> out;
  for (int level = 0; level <= a.max_level; ++level) {
    std::size_t nodes = 0;
    std::size_t total = 0;
    for (std::size_t i = 0; i < a.levels.size(); ++i) {
      if (a.levels[i] <= level) {
        continue;
      }
      ++nodes;
      std::size_t begin = 0;
      std::size_t end = 0;
      a.neighbor_range(static_cast<faiss::idx_t>(i), level, &begin, &end);
      std::unordered_set<int> distinct;
      for (std::size_t j = begin; j < end && a.neighbors[j] >= 0; ++j) {
        distinct.insert(a.neighbors[j]);
      }
      total += distinct.size();
    }
    out.push_back(nodes == 0 ? 0.0
                             : static_cast<double>(total) /
                                   static_cast<double>(nodes));
  }
  return out;
}

std::vector<std::size_t> AcornIndex::NodesPerLevel() const {
  const faiss::ACORN& a = index_->acorn;
  std::vector<std::size_t> out(static_cast<std::size_t>(a.max_level) + 1, 0);
  for (const int lv : a.levels) {
    for (int level = 0; level < lv && level <= a.max_level; ++level) {
      ++out[static_cast<std::size_t>(level)];
    }
  }
  return out;
}

std::size_t AcornIndex::MemoryBytes() const {
  const faiss::ACORN& a = index_->acorn;
  const std::size_t storage =
      Size() * static_cast<std::size_t>(index_->d) * sizeof(float);
  return storage + a.neighbors.size() * sizeof(a.neighbors[0]) +
         a.offsets.size() * sizeof(a.offsets[0]) +
         a.levels.size() * sizeof(a.levels[0]);
}

}
