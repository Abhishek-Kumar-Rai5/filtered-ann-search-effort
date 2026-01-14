#include "filter_generator/filter_conditions.h"

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include "common/run_metadata.h"
#include "filter_generator/uniform_attributes.h"

namespace fse {
namespace {

std::uint64_t HashAttributeImpl(const RankAttribute& a) {
  const auto c = static_cast<std::int32_t>(a.correlation);
  std::uint64_t h = Fnv1a64Bytes(&c, sizeof(c));
  h = Fnv1a64Bytes(a.rank.data(), a.rank.size() * sizeof(std::int32_t), h);
  h = Fnv1a64Bytes(a.cluster.data(), a.cluster.size() * sizeof(std::int32_t),
                   h);
  return Fnv1a64Bytes(a.cluster_order.data(),
                      a.cluster_order.size() * sizeof(std::int32_t), h);
}

}

std::uint64_t RankAttributeHash(const RankAttribute& a) {
  return HashAttributeImpl(a);
}

namespace {

void WriteVec(std::ofstream& out, const std::vector<std::int32_t>& v) {
  const std::uint64_t n = v.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(n * sizeof(std::int32_t)));
}

std::vector<std::int32_t> ReadVec(std::ifstream& in) {
  std::uint64_t n = 0;
  in.read(reinterpret_cast<char*>(&n), sizeof(n));
  if (!in || n > (std::uint64_t{1} << 34)) {
    throw std::runtime_error("ReadRankAttribute: bad vector header");
  }
  std::vector<std::int32_t> v(n);
  in.read(reinterpret_cast<char*>(v.data()),
          static_cast<std::streamsize>(n * sizeof(std::int32_t)));
  return v;
}

constexpr std::array<char, 8> kAttrMagic = {'F', 'S', 'E', 'A',
                                            'T', 'v', '1', '\0'};

}

void WriteRankAttribute(const std::string& path, const RankAttribute& a) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cannot open for writing: " + path);
  }
  const auto corr = static_cast<std::int32_t>(a.correlation);
  out.write(kAttrMagic.data(), kAttrMagic.size());
  out.write(reinterpret_cast<const char*>(&corr), sizeof(corr));
  out.write(reinterpret_cast<const char*>(&a.content_hash),
            sizeof(a.content_hash));
  WriteVec(out, a.rank);
  WriteVec(out, a.cluster);
  WriteVec(out, a.cluster_order);
  if (!out) {
    throw std::runtime_error("write failed: " + path);
  }
}

RankAttribute ReadRankAttribute(const std::string& path,
                                std::uint64_t expected_hash) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  std::array<char, 8> magic{};
  std::int32_t corr = 0;
  std::uint64_t stored = 0;
  in.read(magic.data(), magic.size());
  in.read(reinterpret_cast<char*>(&corr), sizeof(corr));
  in.read(reinterpret_cast<char*>(&stored), sizeof(stored));
  if (!in || magic != kAttrMagic || (corr != 0 && corr != 1)) {
    throw std::runtime_error("not a rank attribute file: " + path);
  }
  RankAttribute a;
  a.correlation = corr == 0 ? Correlation::kRandom : Correlation::kClustered;
  a.rank = ReadVec(in);
  a.cluster = ReadVec(in);
  a.cluster_order = ReadVec(in);
  if (!in) {
    throw std::runtime_error("truncated rank attribute: " + path);
  }
  a.content_hash = HashAttributeImpl(a);
  if (a.content_hash != stored ||
      (expected_hash != 0 && a.content_hash != expected_hash)) {
    throw std::runtime_error("rank attribute hash mismatch: " + path);
  }
  return a;
}

std::string CorrelationName(Correlation c) {
  return c == Correlation::kRandom ? "random" : "clustered";
}

Correlation ParseCorrelation(const std::string& name) {
  if (name == "random") {
    return Correlation::kRandom;
  }
  if (name == "clustered") {
    return Correlation::kClustered;
  }
  throw std::invalid_argument("unknown correlation condition: " + name);
}

RankAttribute RandomRankAttribute(std::size_t n, std::uint64_t seed) {
  RankAttribute a;
  a.correlation = Correlation::kRandom;
  a.rank = SeededPermutation(n, seed);
  a.content_hash = HashAttributeImpl(a);
  return a;
}

RankAttribute ClusteredRankAttribute(const FloatMatrix& base,
                                     const ClusteredParams& p) {
  const std::size_t n = base.rows;
  if (p.num_clusters == 0 || p.num_clusters > n) {
    throw std::invalid_argument("ClusteredRankAttribute: bad num_clusters");
  }
  const auto d = static_cast<faiss::idx_t>(base.dim);
  const auto c = static_cast<faiss::idx_t>(p.num_clusters);

  faiss::ClusteringParameters cp;
  cp.niter = p.kmeans_iterations;
  cp.nredo = 1;
  cp.verbose = false;
  cp.seed = static_cast<int>(p.kmeans_seed);
  cp.max_points_per_centroid = p.max_points_per_centroid;
  faiss::Clustering km(static_cast<int>(d), static_cast<int>(c), cp);
  faiss::IndexFlatL2 train_index(d);
  km.train(static_cast<faiss::idx_t>(n), base.data.data(), train_index);

  faiss::IndexFlatL2 centroids(d);
  centroids.add(c, km.centroids.data());
  std::vector<float> dist(n);
  std::vector<faiss::idx_t> assign(n);
  centroids.search(static_cast<faiss::idx_t>(n), base.data.data(), 1,
                   dist.data(), assign.data());

  RankAttribute a;
  a.correlation = Correlation::kClustered;
  a.cluster.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    a.cluster[i] = static_cast<std::int32_t>(assign[i]);
  }

  a.cluster_order = SeededPermutation(p.num_clusters, p.order_seed);
  std::vector<std::int32_t> position(p.num_clusters);
  for (std::size_t j = 0; j < p.num_clusters; ++j) {
    position[static_cast<std::size_t>(a.cluster_order[j])] =
        static_cast<std::int32_t>(j);
  }
  const std::vector<std::int32_t> tiebreak =
      SeededPermutation(n, p.order_seed + 1);
  std::vector<std::int32_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::int32_t x, std::int32_t y) {
    const auto px = position[static_cast<std::size_t>(a.cluster[x])];
    const auto py = position[static_cast<std::size_t>(a.cluster[y])];
    return px != py ? px < py : tiebreak[x] < tiebreak[y];
  });
  a.rank.resize(n);
  for (std::size_t r = 0; r < n; ++r) {
    a.rank[static_cast<std::size_t>(order[r])] = static_cast<std::int32_t>(r);
  }
  a.content_hash = HashAttributeImpl(a);
  return a;
}

std::size_t SelectivityThreshold(std::size_t n, double s) {
  if (std::isnan(s) || s <= 0.0 || s > 1.0) {
    throw std::invalid_argument("selectivity must be in (0, 1]");
  }
  const auto t =
      static_cast<std::size_t>(std::llround(s * static_cast<double>(n)));
  if (t == 0) {
    throw std::invalid_argument("selectivity too small for N: T = 0");
  }
  return std::min(t, n);
}

std::vector<double> LogSpacedSelectivities(double lo, double hi, int levels) {
  if (levels < 2 || !(lo > 0.0) || !(hi <= 1.0) || !(lo < hi)) {
    throw std::invalid_argument("LogSpacedSelectivities: bad arguments");
  }
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(levels));
  const double a = std::log10(lo);
  const double b = std::log10(hi);
  for (int i = 0; i < levels; ++i) {
    out.push_back(std::pow(10.0, a + (b - a) * i / (levels - 1)));
  }
  out.back() = hi;
  return out;
}

FilterCondition MakeFilterCondition(const RankAttribute& attr, double s,
                                    std::uint64_t base_hash) {
  const std::size_t n = attr.rank.size();
  FilterCondition f;
  f.correlation = attr.correlation;
  f.requested_selectivity = s;
  f.threshold = SelectivityThreshold(n, s);
  f.achieved_selectivity =
      static_cast<double>(f.threshold) / static_cast<double>(n);
  std::array<char, 32> buf{};
  std::snprintf(buf.data(), buf.size(), "_s%.4f", s);
  f.name = CorrelationName(attr.correlation) + buf.data();
  f.mask.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    f.mask[i] =
        static_cast<char>(static_cast<std::size_t>(attr.rank[i]) < f.threshold);
  }
  f.mask_hash = Fnv1a64Bytes(f.mask.data(), f.mask.size());
  const auto corr = static_cast<std::int32_t>(attr.correlation);
  const std::array<std::uint64_t, 4> words = {
      static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(f.threshold),
      attr.content_hash, base_hash};
  std::uint64_t h = Fnv1a64Bytes(&corr, sizeof(corr));
  h = Fnv1a64Bytes(&s, sizeof(s), h);
  f.condition_id = Fnv1a64Bytes(words.data(), sizeof(words), h);
  return f;
}

}
