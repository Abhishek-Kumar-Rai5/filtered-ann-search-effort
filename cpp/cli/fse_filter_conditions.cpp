#include <faiss/IndexFlat.h>
#include <omp.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/build_info.h"
#include "common/run_metadata.h"
#include "common/vecs_io.h"
#include "filter_generator/filter_conditions.h"
#include "filter_generator/uniform_attributes.h"
#include "ground_truth/filtered_ground_truth.h"
#include "metrics/local_density.h"
#include "metrics/structure_stats.h"

namespace fs = std::filesystem;

namespace {

struct Config {
  std::string path;
  std::string text;
  std::string experiment_name;
  std::string base_path;
  std::string query_path;
  std::size_t max_base = 0;
  std::size_t max_queries = 0;
  std::size_t k = 0;
  double s_min = 0.0;
  double s_max = 0.0;
  int levels = 0;
  std::vector<std::string> correlations;
  std::uint64_t random_seed = 0;
  fse::ClusteredParams clustered;
  std::vector<std::size_t> local_density_k;
  std::size_t homophily_sample = 0;
  std::size_t homophily_k = 0;
  std::uint64_t homophily_seed = 0;

  std::size_t null_draws = 0;
  std::uint64_t null_seed = 0;
  double alpha_family = 0.0;
  double kappa_min = 0.0;
  std::string output_dir;
  std::string cache_dir;

  std::string mode = "validation";
  std::size_t gt_k = 0;
  bool homophily_exact = true;
  bool gt_recompute_check = true;
  std::vector<std::size_t> calibration_candidates;
};

Config LoadConfig(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open config " + path);
  }
  std::stringstream ss;
  ss << in.rdbuf();
  Config c;
  c.path = path;
  c.text = ss.str();
  const YAML::Node y = YAML::Load(c.text);
  c.experiment_name = y["experiment_name"].as<std::string>();
  c.base_path = y["dataset"]["base"].as<std::string>();
  c.query_path = y["dataset"]["queries"].as<std::string>();
  c.max_base = y["dataset"]["max_base"].as<std::size_t>();
  c.max_queries = y["dataset"]["max_queries"].as<std::size_t>();
  c.k = y["search"]["k"].as<std::size_t>();
  c.s_min = y["selectivity"]["min"].as<double>();
  c.s_max = y["selectivity"]["max"].as<double>();
  c.levels = y["selectivity"]["levels"].as<int>();
  c.correlations = y["correlations"].as<std::vector<std::string>>();
  const YAML::Node g = y["generator"];
  c.random_seed = g["random_seed"].as<std::uint64_t>();
  c.clustered.num_clusters = g["clustered"]["num_clusters"].as<std::size_t>();
  c.clustered.kmeans_iterations = g["clustered"]["kmeans_iterations"].as<int>();
  c.clustered.kmeans_seed = g["clustered"]["kmeans_seed"].as<std::uint64_t>();
  c.clustered.max_points_per_centroid =
      g["clustered"]["max_points_per_centroid"].as<int>();
  c.clustered.order_seed = g["clustered"]["order_seed"].as<std::uint64_t>();
  const YAML::Node v = y["validation"];
  c.local_density_k = v["local_density_k"].as<std::vector<std::size_t>>();
  c.homophily_sample = v["homophily_sample"].as<std::size_t>();
  c.homophily_k = v["homophily_k"].as<std::size_t>();
  c.homophily_seed = v["homophily_seed"].as<std::uint64_t>();
  c.null_draws = v["null_draws"].as<std::size_t>();
  c.null_seed = v["null_seed"].as<std::uint64_t>();
  c.alpha_family = v["alpha_family"].as<double>();
  c.kappa_min = v["kappa_min"].as<double>();
  c.output_dir = y["paths"]["output_dir"].as<std::string>();
  c.cache_dir = y["paths"]["cache_dir"].as<std::string>();
  if (y["mode"]) {
    c.mode = y["mode"].as<std::string>();
  }
  c.gt_k = y["ground_truth"] ? y["ground_truth"]["k"].as<std::size_t>() : c.k;
  if (v["homophily_mode"]) {
    const auto m = v["homophily_mode"].as<std::string>();
    if (m != "exact" && m != "sampled") {
      throw std::invalid_argument("homophily_mode must be exact or sampled");
    }
    c.homophily_exact = m == "exact";
  }
  if (v["gt_recompute_check"]) {
    c.gt_recompute_check = v["gt_recompute_check"].as<bool>();
  }
  if (y["calibration"]) {
    c.calibration_candidates =
        y["calibration"]["candidates"].as<std::vector<std::size_t>>();
  }
  return c;
}

void Log(const std::string& msg) {
  std::cout << "[fse_filter_conditions] " << msg << '\n' << std::flush;
}

std::uint64_t HashMatrix(const fse::FloatMatrix& m) {
  return fse::Fnv1a64Bytes(m.data.data(), m.data.size() * sizeof(float));
}

std::uint64_t HashTable(const fse::NeighborTable& t) {
  const std::uint64_t h =
      fse::Fnv1a64Bytes(t.ids.data(), t.ids.size() * sizeof(std::int64_t));
  return fse::Fnv1a64Bytes(t.distances.data(),
                           t.distances.size() * sizeof(float), h);
}

std::array<std::size_t, 3> FaissCrossCheck(const fse::FloatMatrix& base,
                                           const fse::FloatMatrix& queries,
                                           const std::vector<char>& mask,
                                           const fse::NeighborTable& gt) {
  std::vector<std::int64_t> subset;
  for (std::size_t i = 0; i < base.rows; ++i) {
    if (mask[i] != 0) {
      subset.push_back(static_cast<std::int64_t>(i));
    }
  }
  faiss::IndexFlatL2 flat(static_cast<faiss::idx_t>(base.dim));
  std::vector<float> sub(subset.size() * base.dim);
  for (std::size_t s = 0; s < subset.size(); ++s) {
    std::copy_n(base.Row(subset[s]), base.dim,
                sub.begin() + static_cast<std::ptrdiff_t>(s * base.dim));
  }
  flat.add(static_cast<faiss::idx_t>(subset.size()), sub.data());
  const std::size_t k = gt.k;
  std::vector<float> d(queries.rows * k);
  std::vector<faiss::idx_t> l(queries.rows * k);
  flat.search(static_cast<faiss::idx_t>(queries.rows), queries.data.data(),
              static_cast<faiss::idx_t>(k), d.data(), l.data());
  std::array<std::size_t, 3> out{};
  for (std::size_t q = 0; q < queries.rows; ++q) {
    bool ids_same = true;
    bool dist_same = true;
    for (std::size_t j = 0; j < k; ++j) {
      const faiss::idx_t lj = l[q * k + j];
      const std::int64_t id =
          lj < 0 ? -1 : subset[static_cast<std::size_t>(lj)];
      ids_same = ids_same && id == gt.Ids(q)[j];
      dist_same = dist_same && d[q * k + j] == gt.Distances(q)[j];
    }
    if (ids_same) {
      ++out[0];
    } else if (dist_same) {
      ++out[1];
    } else {
      ++out[2];
    }
  }
  return out;
}

struct DensityStats {
  double mean = 0.0;
  double var = 0.0;
  double binomial_var = 0.0;
  double frac_zero = 0.0;
  double frac_one = 0.0;
};

DensityStats Summarise(const std::vector<double>& v, double s,
                       std::size_t k_local) {
  DensityStats d;
  const auto n = static_cast<double>(v.size());
  for (const double x : v) {
    d.mean += x / n;
    d.frac_zero += x == 0.0 ? 1.0 / n : 0.0;
    d.frac_one += x == 1.0 ? 1.0 / n : 0.0;
  }
  for (const double x : v) {
    d.var += (x - d.mean) * (x - d.mean) / n;
  }
  d.binomial_var = s * (1.0 - s) / static_cast<double>(k_local);
  return d;
}

struct Built {
  fse::FilterCondition cond;
  fse::GroundTruthIdentity identity;
  std::string gt_path;
  std::uint64_t gt_hash = 0;
  double homophily = 0.0;
  std::vector<DensityStats> density;

  double homophily_exact = 0.0;
  double kappa_h = 0.0;
  std::vector<double> zero_fraction;
  std::size_t partial_clusters = 0;
  std::size_t cells_spanned = 0;
};

std::string CorrelationFailures(const Built& rnd, const Built& clu,
                                const std::vector<std::size_t>& ks,
                                std::size_t nq) {
  std::ostringstream f;
  const double s = rnd.cond.achieved_selectivity;
  for (std::size_t i = 0; i < ks.size(); ++i) {
    const DensityStats& r = rnd.density[i];
    const DensityStats& c = clu.density[i];
    const double dr = r.var / r.binomial_var;
    const double dc = c.var / c.binomial_var;
    if (dr < 0.5 || dr > 2.0) {
      f << " random D(K=" << ks[i] << ")=" << dr;
    }
    if (std::abs(r.mean - s) >
        4.0 * std::sqrt(r.binomial_var / static_cast<double>(nq))) {
      f << " random mean(K=" << ks[i] << ")=" << r.mean;
    }
    if (dc < 3.0 || dc < 2.0 * dr) {
      f << " clustered D(K=" << ks[i] << ")=" << dc;
    }
  }
  if (std::abs(rnd.homophily - s) > 0.1) {
    f << " random homophily=" << rnd.homophily;
  }
  if (clu.homophily - s < 0.3) {
    f << " clustered homophily=" << clu.homophily;
  }
  return f.str();
}

struct Data {
  fse::FloatMatrix base;
  fse::FloatMatrix queries;
  std::uint64_t base_hash = 0;
  std::uint64_t query_hash = 0;
  fse::NeighborTable unfiltered;
  std::vector<std::int32_t> sample;
  fse::NeighborTable base_knn;
  fse::NeighborTable base_knn_all;
};

template <typename Compute>
fse::NeighborTable CachedTable(const std::string& cache_dir,
                               const std::string& name, std::uint64_t base_hash,
                               std::uint64_t query_hash, Compute compute) {
  const fse::GroundTruthIdentity id{fse::Fnv1a64(name), base_hash, query_hash};
  const std::string path =
      cache_dir + "/" + name + "_" + fse::Hex64(base_hash ^ query_hash) + ".nt";
  if (fs::exists(path)) {
    Log("loading cached " + name);
    return fse::ReadConditionGroundTruth(path, id);
  }
  Log("computing " + name);
  fse::NeighborTable t = compute();
  fse::WriteConditionGroundTruth(path, id, t);
  return t;
}

Data LoadData(const Config& c) {
  Data d;
  d.base =
      fse::ReadFvecs(c.base_path, c.max_base == 0 ? fse::kAllRows : c.max_base);
  d.queries = fse::ReadFvecs(
      c.query_path, c.max_queries == 0 ? fse::kAllRows : c.max_queries);
  d.base_hash = HashMatrix(d.base);
  d.query_hash = HashMatrix(d.queries);
  const std::size_t n = d.base.rows;
  Log("base " + std::to_string(n) + " x " + std::to_string(d.base.dim) +
      ", queries " + std::to_string(d.queries.rows));

  const std::size_t k_loc_max =
      *std::max_element(c.local_density_k.begin(), c.local_density_k.end());
  d.unfiltered = CachedTable(
      c.cache_dir, "unfiltered_k" + std::to_string(k_loc_max), d.base_hash,
      d.query_hash,
      [&] { return fse::UnfilteredGroundTruth(d.base, d.queries, k_loc_max); });
  const auto perm = fse::SeededPermutation(n, c.homophily_seed);
  d.sample.assign(perm.begin(),
                  perm.begin() + static_cast<std::ptrdiff_t>(
                                     std::min(c.homophily_sample, n)));
  fse::FloatMatrix sample_vecs{std::vector<float>(d.sample.size() * d.base.dim),
                               d.sample.size(), d.base.dim};
  for (std::size_t r = 0; r < d.sample.size(); ++r) {
    std::copy_n(d.base.Row(static_cast<std::size_t>(d.sample[r])), d.base.dim,
                sample_vecs.Row(r));
  }
  d.base_knn = CachedTable(c.cache_dir,
                           "sample_knn_k" + std::to_string(c.homophily_k + 1) +
                               "_n" + std::to_string(d.sample.size()) +
                               "_seed" + std::to_string(c.homophily_seed),
                           d.base_hash, HashMatrix(sample_vecs), [&] {
                             return fse::UnfilteredGroundTruth(
                                 d.base, sample_vecs, c.homophily_k + 1);
                           });

  if (c.homophily_exact) {
    d.base_knn_all =
        fse::UnfilteredGroundTruth(d.base, d.base, c.homophily_k + 1);
  }
  return d;
}

fse::RankAttribute MakeAttribute(const Config& c, const Data& d,
                                 fse::Correlation corr) {
  return corr == fse::Correlation::kRandom
             ? fse::RandomRankAttribute(d.base.rows, c.random_seed)
             : fse::ClusteredRankAttribute(d.base, c.clustered);
}

std::vector<fse::RankAttribute> BuildAttributes(const Config& c,
                                                const Data& d) {
  std::vector<fse::RankAttribute> attrs;
  attrs.reserve(c.correlations.size());
  for (const std::string& name : c.correlations) {
    const fse::Correlation corr = fse::ParseCorrelation(name);
    attrs.push_back(MakeAttribute(c, d, corr));
    if (MakeAttribute(c, d, corr).content_hash != attrs.back().content_hash) {
      throw std::runtime_error("attribute generation not deterministic: " +
                               name);
    }
    Log(name + " attribute hash " + fse::Hex64(attrs.back().content_hash));
  }
  return attrs;
}

struct Outputs {
  std::ofstream conditions;
  std::ofstream local_density;
  std::string digest_text;
};

void WriteHeaders(const Config& c, Outputs* out) {
  out->conditions << "condition,correlation,requested_s,threshold,achieved_s,"
                     "selectivity_error,mask_count,nested_ok,mask_hash,"
                     "condition_id,query_hash,gt_hash,gt_violations,"
                     "gt_fresh_equals_stored,faiss_identical,faiss_tie_only,"
                     "faiss_mismatched,homophily";
  for (const std::size_t kl : c.local_density_k) {
    out->conditions << ",ld" << kl << "_mean,ld" << kl << "_var,ld" << kl
                    << "_binomial_var,ld" << kl << "_frac0,ld" << kl
                    << "_frac1";
  }
  out->conditions << "\n";
  out->local_density << "condition,query,k_local,local_density\n";
}

std::size_t CellsSpanned(const fse::RankAttribute& attr,
                         const std::vector<char>& mask) {
  std::vector<char> hit(attr.cluster_order.size(), 0);
  for (std::size_t i = 0; i < mask.size(); ++i) {
    if (mask[i] != 0) {
      hit[static_cast<std::size_t>(attr.cluster[i])] = 1;
    }
  }
  return static_cast<std::size_t>(std::count(hit.begin(), hit.end(), 1));
}

std::size_t PartialClusters(const fse::RankAttribute& attr,
                            const std::vector<char>& mask) {
  std::vector<std::size_t> in(attr.cluster_order.size(), 0);
  std::vector<std::size_t> total(attr.cluster_order.size(), 0);
  for (std::size_t i = 0; i < mask.size(); ++i) {
    const auto cl = static_cast<std::size_t>(attr.cluster[i]);
    ++total[cl];
    in[cl] += mask[i] != 0 ? 1 : 0;
  }
  std::size_t partial = 0;
  for (std::size_t cl = 0; cl < in.size(); ++cl) {
    partial += (in[cl] > 0 && in[cl] < total[cl]) ? 1 : 0;
  }
  return partial;
}

double HomophilyFor(const Config& c, const Data& d,
                    const std::vector<char>& mask) {
  return c.homophily_exact
             ? fse::ExactHomophily(d.base_knn_all, mask, c.homophily_k)
             : fse::SampledHomophily(d.base_knn, d.sample, mask, c.homophily_k);
}

void RecordCorrectedStructure(const Config& c, const Data& d,
                              const fse::RankAttribute& attr, double s,
                              Built* b) {
  const std::vector<char>& mask = b->cond.mask;
  b->homophily_exact = HomophilyFor(c, d, mask);
  if (b->cond.threshold < mask.size()) {
    b->kappa_h = fse::ChanceCorrected(b->homophily_exact, s);
  }
  for (const std::size_t kl : c.local_density_k) {
    b->zero_fraction.push_back(fse::ZeroFraction(d.unfiltered, mask, kl));
  }
  if (!attr.cluster.empty()) {
    b->partial_clusters = PartialClusters(attr, mask);
    b->cells_spanned = CellsSpanned(attr, mask);
  }
}

bool BuildAndCheckCondition(const Config& c, const Data& d,
                            const fse::RankAttribute& attr, double s,
                            std::vector<char>* prev, Outputs* out, Built* b) {
  const std::size_t n = d.base.rows;
  b->cond = fse::MakeFilterCondition(attr, s, d.base_hash);
  const fse::FilterCondition& f = b->cond;

  const auto count = static_cast<std::size_t>(std::count_if(
      f.mask.begin(), f.mask.end(), [](char x) { return x != 0; }));
  const double err = f.achieved_selectivity - s;
  bool nested = true;
  for (std::size_t i = 0; i < n; ++i) {
    nested = nested && ((*prev)[i] == 0 || f.mask[i] != 0);
  }
  *prev = f.mask;
  const bool sel_ok = count == f.threshold &&
                      std::abs(err) <= 0.5 / static_cast<double>(n) && nested;

  const std::uint64_t qh = HashMatrix(d.queries);

  const fse::NeighborTable gt =
      fse::FilteredGroundTruthMask(d.base, d.queries, f.mask, c.gt_k);
  b->identity = {f.condition_id, f.mask_hash, qh};
  b->gt_path =
      c.cache_dir + "/" + f.name + "_" + fse::Hex64(f.condition_id) + ".gt";
  fse::WriteConditionGroundTruth(b->gt_path, b->identity, gt);
  const fse::NeighborTable stored =
      fse::ReadConditionGroundTruth(b->gt_path, b->identity);
  const fse::NeighborTable again =
      c.gt_recompute_check
          ? fse::FilteredGroundTruthMask(d.base, d.queries, f.mask, c.gt_k)
          : gt;
  const bool fresh_eq = stored.ids == gt.ids &&
                        stored.distances == gt.distances &&
                        again.ids == gt.ids && again.distances == gt.distances;
  b->gt_hash = HashTable(gt);

  std::size_t violations = 0;
  for (const std::int64_t id : gt.ids) {
    violations += (id < 0 || f.mask[static_cast<std::size_t>(id)] == 0) ? 1 : 0;
  }

  const auto fx = FaissCrossCheck(d.base, d.queries, f.mask, gt);

  b->homophily =
      fse::SampledHomophily(d.base_knn, d.sample, f.mask, c.homophily_k);
  RecordCorrectedStructure(c, d, attr, s, b);
  std::ostringstream ld_cols;
  for (const std::size_t kl : c.local_density_k) {
    const auto ld = fse::LocalFilteredDensity(d.unfiltered, f.mask, kl);
    const DensityStats st = Summarise(ld, f.achieved_selectivity, kl);
    b->density.push_back(st);
    ld_cols << "," << st.mean << "," << st.var << "," << st.binomial_var << ","
            << st.frac_zero << "," << st.frac_one;
    for (std::size_t q = 0; q < ld.size(); ++q) {
      out->local_density << f.name << "," << q << "," << kl << "," << ld[q]
                         << "\n";
    }
  }
  const bool ok =
      sel_ok && qh == d.query_hash && fresh_eq && violations == 0 && fx[2] == 0;
  out->conditions << f.name << "," << fse::CorrelationName(f.correlation) << ","
                  << s << "," << f.threshold << "," << f.achieved_selectivity
                  << "," << err << "," << count << "," << nested << ","
                  << fse::Hex64(f.mask_hash) << ","
                  << fse::Hex64(f.condition_id) << "," << fse::Hex64(qh) << ","
                  << fse::Hex64(b->gt_hash) << "," << violations << ","
                  << fresh_eq << "," << fx[0] << "," << fx[1] << "," << fx[2]
                  << "," << b->homophily << ld_cols.str() << "\n";
  out->digest_text += fse::Hex64(attr.content_hash) + fse::Hex64(f.mask_hash) +
                      fse::Hex64(b->gt_hash);
  Log(f.name + ": T=" + std::to_string(f.threshold) +
      " homophily=" + std::to_string(b->homophily) + " faiss " +
      std::to_string(fx[0]) + "/" + std::to_string(d.queries.rows) + " (+" +
      std::to_string(fx[1]) + " tie-only) " + (ok ? "ok" : "FAIL"));
  return ok;
}

std::pair<std::size_t, std::size_t> CheckCorrelation(
    const Config& c, const std::vector<Built>& built, std::size_t n,
    std::size_t nq) {
  std::size_t judged = 0;
  std::size_t failed = 0;
  for (const Built& rnd : built) {
    if (rnd.cond.correlation != fse::Correlation::kRandom ||
        rnd.cond.threshold == n) {
      continue;
    }
    for (const Built& clu : built) {
      if (clu.cond.correlation != fse::Correlation::kClustered ||
          clu.cond.requested_selectivity != rnd.cond.requested_selectivity) {
        continue;
      }
      ++judged;
      const std::string fails =
          CorrelationFailures(rnd, clu, c.local_density_k, nq);
      failed += fails.empty() ? 0 : 1;
      std::ostringstream m;
      m << "correlation s=" << rnd.cond.requested_selectivity << ":";
      for (std::size_t i = 0; i < c.local_density_k.size(); ++i) {
        m << " D_rand(K=" << c.local_density_k[i]
          << ")=" << rnd.density[i].var / rnd.density[i].binomial_var
          << " D_clu=" << clu.density[i].var / clu.density[i].binomial_var;
      }
      m << " homophily rand=" << rnd.homophily << " clu=" << clu.homophily
        << (fails.empty() ? " ok" : " FAIL:" + fails);
      Log(m.str());
    }
  }
  return {judged, failed};
}

struct NullLevel {
  double s = 0.0;
  std::vector<std::vector<double>> zero_fraction;
  std::vector<double> kappa_h;
};

std::vector<NullLevel> ComputeNull(const Config& c, const Data& d,
                                   const std::vector<double>& levels) {
  const std::size_t n = d.base.rows;
  const std::size_t r_total = c.null_draws;
  std::vector<NullLevel> out;
  for (const double s : levels) {
    if (fse::SelectivityThreshold(n, s) == n) {
      continue;
    }
    NullLevel l;
    l.s = s;
    l.zero_fraction.assign(c.local_density_k.size(),
                           std::vector<double>(r_total));
    l.kappa_h.resize(r_total);
    out.push_back(std::move(l));
  }
  const auto r_signed = static_cast<std::int64_t>(r_total);
#pragma omp parallel for schedule(dynamic, 16)
  for (std::int64_t r = 0; r < r_signed; ++r) {
    const auto ru = static_cast<std::size_t>(r);
    const fse::RankAttribute a = fse::RandomRankAttribute(n, c.null_seed + ru);
    std::vector<char> mask(n);
    for (NullLevel& l : out) {
      const std::size_t t = fse::SelectivityThreshold(n, l.s);
      for (std::size_t i = 0; i < n; ++i) {
        mask[i] = static_cast<char>(static_cast<std::size_t>(a.rank[i]) < t);
      }
      for (std::size_t ki = 0; ki < c.local_density_k.size(); ++ki) {
        l.zero_fraction[ki][ru] =
            fse::ZeroFraction(d.unfiltered, mask, c.local_density_k[ki]);
      }
      l.kappa_h[ru] = fse::ChanceCorrected(HomophilyFor(c, d, mask), l.s);
    }
  }
  return out;
}

double Quantile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  const auto idx = static_cast<std::size_t>(
      std::floor(p * static_cast<double>(v.size() - 1)));
  return v[idx];
}

struct NullMoments {
  double mean = 0.0;
  double sd = 0.0;
};

NullMoments Moments(const std::vector<double>& v) {
  NullMoments m;
  for (const double x : v) {
    m.mean += x / static_cast<double>(v.size());
  }
  for (const double x : v) {
    m.sd += (x - m.mean) * (x - m.mean) / static_cast<double>(v.size() - 1);
  }
  m.sd = std::sqrt(m.sd);
  return m;
}

std::pair<bool, bool> JudgeStatistic(const std::string& level_name,
                                     const std::string& stat,
                                     const std::vector<double>& null,
                                     double random_obs, double clustered_obs,
                                     double alpha, std::ofstream* csv) {
  const NullMoments m = Moments(null);
  const double p_rand_hi = fse::EmpiricalPUpper(null, random_obs);
  const double p_rand_lo = fse::EmpiricalPLower(null, random_obs);
  const double p_clu = fse::EmpiricalPUpper(null, clustered_obs);
  const bool random_ok = p_rand_hi > alpha / 2 && p_rand_lo > alpha / 2;
  const bool clustered_ok = p_clu <= alpha;
  *csv << level_name << "," << stat << "," << null.size() << "," << m.mean
       << "," << m.sd << "," << Quantile(null, alpha / 2) << ","
       << Quantile(null, 1.0 - alpha / 2) << "," << Quantile(null, 1.0 - alpha)
       << "," << random_obs << "," << p_rand_lo << "," << p_rand_hi << ","
       << random_ok << "," << clustered_obs << "," << p_clu << ","
       << clustered_ok << "\n";
  return {random_ok, clustered_ok};
}

std::pair<const Built*, const Built*> FindPair(const std::vector<Built>& built,
                                               double s) {
  const Built* rnd = nullptr;
  const Built* clu = nullptr;
  for (const Built& b : built) {
    if (b.cond.requested_selectivity != s) {
      continue;
    }
    if (b.cond.correlation == fse::Correlation::kRandom) {
      rnd = &b;
    } else {
      clu = &b;
    }
  }
  if (rnd == nullptr || clu == nullptr) {
    throw std::runtime_error("corrected check: missing condition");
  }
  return {rnd, clu};
}

std::string UnchangedSubcriteria(const Config& c, const Built& rnd,
                                 const Built& clu, std::size_t nq) {
  std::ostringstream fails;
  const double s = rnd.cond.achieved_selectivity;
  for (std::size_t i = 0; i < c.local_density_k.size(); ++i) {
    const DensityStats& r = rnd.density[i];
    const double dr = r.var / r.binomial_var;
    if (dr < 0.5 || dr > 2.0) {
      fails << " random D(K=" << c.local_density_k[i] << ")=" << dr;
    }
    if (std::abs(r.mean - s) >
        4.0 * std::sqrt(r.binomial_var / static_cast<double>(nq))) {
      fails << " random mean(K=" << c.local_density_k[i] << ")";
    }
    if (c.local_density_k[i] == 100) {
      const DensityStats& k = clu.density[i];
      const double dc = k.var / k.binomial_var;
      if (dc < 3.0 || dc < 2.0 * dr) {
        fails << " clustered D(K=100)=" << dc;
      }
    }
  }
  if (std::abs(rnd.homophily_exact - s) > 0.1) {
    fails << " random exact homophily=" << rnd.homophily_exact;
  }
  return fails.str();
}

std::string ReplacementCriteria(const Config& c, const NullLevel& nl,
                                const Built& rnd, const Built& clu,
                                const std::string& level, double alpha,
                                std::ofstream* csv) {
  std::ostringstream fails;
  for (std::size_t i = 0; i < c.local_density_k.size(); ++i) {
    const auto [r_ok, c_ok] = JudgeStatistic(
        level, "zero_fraction_K" + std::to_string(c.local_density_k[i]),
        nl.zero_fraction[i], rnd.zero_fraction[i], clu.zero_fraction[i], alpha,
        csv);
    if (!r_ok) {
      fails << " random f0(K=" << c.local_density_k[i] << ") outside null";
    }
    if (!c_ok) {
      fails << " clustered f0(K=" << c.local_density_k[i]
            << ")=" << clu.zero_fraction[i] << " not above null";
    }
  }
  const auto [r_ok, c_ok] = JudgeStatistic(
      level, "kappa_h", nl.kappa_h, rnd.kappa_h, clu.kappa_h, alpha, csv);
  if (!r_ok) {
    fails << " random kappa_h outside null";
  }
  if (!c_ok || clu.kappa_h < c.kappa_min) {
    fails << " clustered kappa_h=" << clu.kappa_h;
  }
  return fails.str();
}

struct CorrectedResult {
  std::size_t levels = 0;
  std::size_t failed = 0;
  double alpha_per_test = 0.0;
  double min_clustered_kappa = 1.0;
};

CorrectedResult CheckCorrelationCorrected(const Config& c,
                                          const std::vector<Built>& built,
                                          const std::vector<NullLevel>& nulls,
                                          std::size_t nq,
                                          const std::string& csv_path) {
  if (std::find(c.local_density_k.begin(), c.local_density_k.end(), 100) ==
      c.local_density_k.end()) {
    throw std::invalid_argument("check 2 requires local_density_k to hold 100");
  }
  CorrectedResult res;
  const std::size_t tests = nulls.size() * (c.local_density_k.size() + 1);
  res.alpha_per_test = c.alpha_family / static_cast<double>(tests);
  std::ofstream csv(csv_path);
  csv << "level,statistic,null_draws,null_mean,null_sd,null_q_lo,null_q_hi,"
         "null_q_upper,random_obs,random_p_lower,random_p_upper,random_ok,"
         "clustered_obs,clustered_p_upper,clustered_ok\n";
  for (const NullLevel& nl : nulls) {
    const auto [rnd, clu] = FindPair(built, nl.s);
    ++res.levels;
    std::ostringstream fails;
    const std::string level = rnd->cond.name.substr(rnd->cond.name.find("_s"));
    fails << UnchangedSubcriteria(c, *rnd, *clu, nq);
    fails << ReplacementCriteria(c, nl, *rnd, *clu, level, res.alpha_per_test,
                                 &csv);
    res.min_clustered_kappa = std::min(res.min_clustered_kappa, clu->kappa_h);

    if (clu->partial_clusters > 1) {
      fails << " partial clusters=" << clu->partial_clusters;
    }
    res.failed += fails.str().empty() ? 0 : 1;
    std::ostringstream m;
    m << "CORRECTED check 2" << level << ": kappa_h rand=" << rnd->kappa_h
      << " clu=" << clu->kappa_h << " f0(K=10) rand=" << rnd->zero_fraction[0]
      << " clu=" << clu->zero_fraction[0]
      << " partial=" << clu->partial_clusters
      << (fails.str().empty() ? " PASS" : " FAIL:" + fails.str());
    Log(m.str());
  }
  return res;
}

std::pair<std::size_t, std::size_t> CheckSwaps(
    const std::vector<Built>& built) {
  std::size_t pairs = 0;
  std::size_t rejected = 0;
  for (const Built& a : built) {
    for (const Built& other : built) {
      if (&a == &other) {
        continue;
      }
      ++pairs;
      try {
        (void)fse::ReadConditionGroundTruth(a.gt_path, other.identity);
      } catch (const std::runtime_error&) {
        ++rejected;
      }
    }
  }
  return {pairs, rejected};
}

std::string JsonStrings(const std::vector<std::string>& v) {
  std::string out = "[";
  for (std::size_t i = 0; i < v.size(); ++i) {
    out += (i > 0 ? ", \"" : "\"") + fse::JsonEscape(v[i]) + "\"";
  }
  return out + "]";
}

void WriteStructureExtra(const std::vector<Built>& built, std::size_t nq,
                         const std::string& path) {
  std::ofstream out(path);
  out << "condition,f0_k10,f0_k100,queries_hit_k10,queries_hit_k100,"
         "kappa_h,cells_spanned,partial_clusters\n";
  for (const Built& b : built) {
    const auto hits = [&](double f0) {
      return std::llround((1.0 - f0) * static_cast<double>(nq));
    };
    out << b.cond.name << "," << b.zero_fraction.front() << ","
        << b.zero_fraction.back() << "," << hits(b.zero_fraction.front()) << ","
        << hits(b.zero_fraction.back()) << "," << b.kappa_h << ","
        << b.cells_spanned << "," << b.partial_clusters << "\n";
  }
}

int RunCalibration(const Config& c, const Data& d, const std::string& dir) {
  if (c.calibration_candidates.empty()) {
    throw std::invalid_argument("calibration: no candidates");
  }
  const auto levels = fse::LogSpacedSelectivities(c.s_min, c.s_max, c.levels);
  const std::size_t n = d.base.rows;
  std::ofstream csv(dir + "/calibration.csv");
  csv << "C,s,kappa_h,D_k100,f0_k10,f0_k100,cells_spanned,partial_clusters,"
         "kmeans_seconds\n";
  struct Candidate {
    std::size_t c;
    bool eligible;
    double min_kappa;
  };
  std::vector<Candidate> results;
  std::ostringstream verdicts;
  for (const std::size_t cand : c.calibration_candidates) {
    fse::ClusteredParams p = c.clustered;
    p.num_clusters = cand;
    const auto t0 = std::chrono::steady_clock::now();
    const fse::RankAttribute a = fse::ClusteredRankAttribute(d.base, p);
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    bool eligible = true;
    double min_kappa = 1.0;
    for (const double s : levels) {
      const fse::FilterCondition f = fse::MakeFilterCondition(a, s, 0);
      if (f.threshold == n) {
        continue;
      }
      const double kappa = fse::ChanceCorrected(HomophilyFor(c, d, f.mask), s);
      const auto ld = fse::LocalFilteredDensity(d.unfiltered, f.mask, 100);
      const DensityStats st = Summarise(ld, f.achieved_selectivity, 100);
      const double d100 = st.var / st.binomial_var;
      const double f0_10 = fse::ZeroFraction(d.unfiltered, f.mask, 10);
      const double f0_100 = fse::ZeroFraction(d.unfiltered, f.mask, 100);
      const std::size_t spanned = CellsSpanned(a, f.mask);
      const std::size_t partial = PartialClusters(a, f.mask);
      csv << cand << "," << s << "," << kappa << "," << d100 << "," << f0_10
          << "," << f0_100 << "," << spanned << "," << partial << "," << secs
          << "\n";
      eligible = eligible && kappa >= c.kappa_min && d100 >= 3.0;
      min_kappa = std::min(min_kappa, kappa);
    }
    verdicts << " C=" << cand << (eligible ? " eligible" : " ineligible")
             << " min_kappa=" << min_kappa << ";";
    Log("calibration C=" + std::to_string(cand) +
        (eligible ? " eligible" : " NOT eligible") +
        " min kappa_h=" + std::to_string(min_kappa));
    results.push_back({cand, eligible, min_kappa});
  }

  double best = -1.0;
  for (const auto& [cand, eligible, min_kappa] : results) {
    if (eligible) {
      best = std::max(best, min_kappa);
    }
  }
  std::size_t chosen = 0;
  for (const auto& [cand, eligible, min_kappa] : results) {
    if (eligible && min_kappa >= best - 0.01) {
      chosen = std::max(chosen, cand);
    }
  }
  fse::JsonObject r;
  r.Str("experiment_name", c.experiment_name)
      .Str("mode", "calibration")
      .Str("source_fingerprint_sha256",
           fse::SourceFingerprint(fse::GetBuildInfo().source_dir))
      .Raw("build", fse::BuildInfoJson())
      .Raw("hardware", fse::HardwareJson())
      .Str("base_hash", fse::Hex64(d.base_hash))
      .Str("query_hash", fse::Hex64(d.query_hash))
      .Str("verdicts", verdicts.str())
      .Int("selected_C", static_cast<std::int64_t>(chosen));
  std::ofstream(dir + "/calibration_summary.json") << r.Render() << "\n";
  Log(chosen == 0 ? std::string("NO ELIGIBLE CANDIDATE -- stop and report")
                  : "selected C = " + std::to_string(chosen));
  return chosen == 0 ? 1 : 0;
}

int Run(const Config& c) {
  const std::string dir = c.output_dir + "/" + c.experiment_name;
  fs::create_directories(dir);
  fs::create_directories(c.cache_dir);
  std::ofstream(dir + "/config.yaml") << c.text;
  const fse::BuildInfo binfo = fse::GetBuildInfo();
  const Data d = LoadData(c);
  if (c.mode == "calibration") {
    return RunCalibration(c, d, dir);
  }
  if (c.mode != "validation") {
    throw std::invalid_argument("unknown mode " + c.mode);
  }
  const std::vector<fse::RankAttribute> attrs = BuildAttributes(c, d);

  std::vector<std::string> attr_files;
  for (const fse::RankAttribute& a : attrs) {
    const std::string path = c.cache_dir + "/attr_" +
                             fse::CorrelationName(a.correlation) + "_" +
                             fse::Hex64(a.content_hash) + ".bin";
    fse::WriteRankAttribute(path, a);
    attr_files.push_back(path);
  }
  const auto levels = fse::LogSpacedSelectivities(c.s_min, c.s_max, c.levels);

  Outputs out{std::ofstream(dir + "/conditions.csv"),
              std::ofstream(dir + "/local_density.csv"), ""};
  WriteHeaders(c, &out);
  bool all_ok = true;
  std::vector<Built> built;
  built.reserve(attrs.size() * levels.size());
  for (const fse::RankAttribute& attr : attrs) {
    std::vector<char> prev(d.base.rows, 0);
    for (const double s : levels) {
      Built b;
      all_ok = BuildAndCheckCondition(c, d, attr, s, &prev, &out, &b) && all_ok;
      built.push_back(std::move(b));
    }
  }

  Log("ORIGINAL check-2 criteria (superseded by section 9, record only):");
  const auto [corr_levels, corr_fail] =
      CheckCorrelation(c, built, d.base.rows, d.queries.rows);

  Log("computing independence null: " + std::to_string(c.null_draws) +
      " draws");
  const std::vector<NullLevel> nulls = ComputeNull(c, d, levels);
  const CorrectedResult corrected = CheckCorrelationCorrected(
      c, built, nulls, d.queries.rows, dir + "/check2_corrected.csv");
  all_ok = all_ok && corrected.levels > 0 && corrected.failed == 0;
  WriteStructureExtra(built, d.queries.rows, dir + "/structure_extra.csv");
  const auto [pairs, rejected] = CheckSwaps(built);
  all_ok = all_ok && rejected == pairs;
  Log("wrong-condition ground truth rejected: " + std::to_string(rejected) +
      "/" + std::to_string(pairs) + " pairs");

  const std::string digest = fse::Hex64(fse::Fnv1a64(
      out.digest_text + fse::Hex64(d.query_hash) + fse::Hex64(d.base_hash)));
  fse::JsonObject r;
  r.Str("experiment_name", c.experiment_name)
      .Str("run_id", fse::MakeExperimentId(c.text))
      .Str("config_path", c.path)
      .Str("git_commit", fse::GetGitInfo(binfo.source_dir).commit)
      .Str("source_fingerprint_sha256",
           fse::SourceFingerprint(binfo.source_dir))
      .Raw("build", fse::BuildInfoJson())
      .Raw("hardware", fse::HardwareJson())
      .Str("base_hash", fse::Hex64(d.base_hash))
      .Str("query_hash", fse::Hex64(d.query_hash))
      .Int("conditions", static_cast<std::int64_t>(built.size()))
      .Int("original_check2_levels_judged",
           static_cast<std::int64_t>(corr_levels))
      .Int("original_check2_levels_failed",
           static_cast<std::int64_t>(corr_fail))
      .Int("corrected_check2_levels_judged",
           static_cast<std::int64_t>(corrected.levels))
      .Int("corrected_check2_levels_failed",
           static_cast<std::int64_t>(corrected.failed))
      .Num("corrected_alpha_per_test", corrected.alpha_per_test)
      .Int("null_draws", static_cast<std::int64_t>(c.null_draws))
      .Int("null_seed", static_cast<std::int64_t>(c.null_seed))
      .Num("kappa_min", c.kappa_min)
      .Num("min_clustered_kappa_h", corrected.min_clustered_kappa)
      .Int("swap_pairs_tested", static_cast<std::int64_t>(pairs))
      .Int("swap_pairs_rejected", static_cast<std::int64_t>(rejected))
      .Str("determinism_digest", digest)
      .Raw("attribute_files", JsonStrings(attr_files))
      .Int("ground_truth_k", static_cast<std::int64_t>(c.gt_k))
      .Bool("homophily_exact", c.homophily_exact)
      .Bool("all_checks_pass", all_ok);
  std::ofstream(dir + "/summary.json") << r.Render() << "\n";
  Log("determinism digest " + digest);
  Log(std::string("ALL CHECKS ") + (all_ok ? "PASS" : "FAIL") + " -> " + dir);
  return all_ok ? 0 : 1;
}

}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: fse_filter_conditions <config.yaml>\n";
    return 2;
  }
  try {
    return Run(LoadConfig(argv[1]));
  } catch (const std::exception& e) {
    std::cerr << "fse_filter_conditions: error: " << e.what() << "\n";
    return 1;
  }
}
