#include <faiss/IndexFlat.h>
#include <faiss/impl/ACORN.h>
#include <omp.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/build_info.h"
#include "common/run_metadata.h"
#include "common/vecs_io.h"
#include "filter_generator/uniform_attributes.h"
#include "ground_truth/filtered_ground_truth.h"
#include "methods/ingraph/acorn_index.h"
#include "metrics/recall.h"

namespace fs = std::filesystem;

namespace {

struct MethodConfig {
  std::string name;
  fse::AcornParams params;
};

struct Config {
  std::string path;
  std::string text;
  std::string experiment_name;
  std::string base_path;
  std::string query_path;
  std::string official_gt_path;
  std::size_t max_base = 0;
  std::size_t max_queries = 0;
  std::int32_t attr_min = 0;
  std::int32_t attr_max = 0;
  std::uint64_t base_seed = 0;
  std::uint64_t query_seed = 0;
  std::string draw_label;
  std::string build_label;
  bool reuse_cached_index = false;
  int build_threads = 0;
  std::vector<MethodConfig> methods;
  std::size_t k = 0;
  std::vector<int> per_query_efs;
  std::vector<int> timing_efs;
  int timing_threads = 0;
  int timing_warmup = 0;
  int timing_trials = 0;
  bool official_gt_check = false;
  std::size_t faiss_crosscheck_queries = 0;
  std::vector<int> determinism_efs;
  std::vector<int> counter_check_efs;
  bool reload_check = false;
  std::string cache_dir;
  std::string output_dir;
};

std::vector<int> Range(const YAML::Node& n) {
  const int from = n["from"].as<int>();
  const int to = n["to"].as<int>();
  const int step = n["step"].as<int>();
  if (step <= 0 || to < from) {
    throw std::invalid_argument("bad efs range");
  }
  std::vector<int> out;
  for (int v = from; v <= to; v += step) {
    out.push_back(v);
  }
  return out;
}

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
  c.official_gt_path = y["dataset"]["official_groundtruth"].as<std::string>();
  c.max_base = y["dataset"]["max_base"].as<std::size_t>();
  c.max_queries = y["dataset"]["max_queries"].as<std::size_t>();
  c.attr_min = y["attributes"]["min"].as<std::int32_t>();
  c.attr_max = y["attributes"]["max"].as<std::int32_t>();
  c.base_seed = y["attributes"]["base_seed"].as<std::uint64_t>();
  c.query_seed = y["attributes"]["query_seed"].as<std::uint64_t>();
  c.draw_label = y["attributes"]["draw_label"].as<std::string>();
  c.build_label = y["index"]["build_label"].as<std::string>();
  c.reuse_cached_index = y["index"]["reuse_cached"].as<bool>();
  c.build_threads = y["index"]["build_threads"].as<int>();
  for (const auto& m : y["index"]["methods"]) {
    MethodConfig mc;
    mc.name = m["name"].as<std::string>();
    mc.params.gamma = m["gamma"].as<int>();
    mc.params.m = m["m"].as<int>();
    mc.params.m_beta = m["m_beta"].as<int>();
    c.methods.push_back(mc);
  }
  c.k = y["search"]["k"].as<std::size_t>();
  const std::vector<int> dense = Range(y["search"]["dense_efs"]);
  c.timing_efs = Range(y["search"]["paper_efs"]);
  std::set<int> all(dense.begin(), dense.end());
  all.insert(c.timing_efs.begin(), c.timing_efs.end());
  c.per_query_efs.assign(all.begin(), all.end());
  c.timing_threads = y["timing"]["threads"].as<int>();
  c.timing_warmup = y["timing"]["warmup_trials"].as<int>();
  c.timing_trials = y["timing"]["trials"].as<int>();
  c.official_gt_check = y["checks"]["official_gt_check"].as<bool>();
  c.faiss_crosscheck_queries =
      y["checks"]["faiss_gt_crosscheck_queries"].as<std::size_t>();
  c.determinism_efs = y["checks"]["determinism_efs"].as<std::vector<int>>();
  c.counter_check_efs = y["checks"]["counter_check_efs"].as<std::vector<int>>();
  c.reload_check = y["checks"]["reload_check"].as<bool>();
  c.cache_dir = y["paths"]["cache_dir"].as<std::string>();
  c.output_dir = y["paths"]["output_dir"].as<std::string>();
  return c;
}

double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void Log(const std::string& msg) {
  std::cout << "[fse_acorn_repro] " << msg << '\n' << std::flush;
}

std::string ReadFirstLineStartingWith(const std::string& path,
                                      const std::string& key) {
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.starts_with(key)) {
      return line;
    }
  }
  return "";
}

std::string HardwareJson() {
  std::string model = ReadFirstLineStartingWith("/proc/cpuinfo", "model name");
  const auto colon = model.find(':');
  model = colon == std::string::npos ? model : model.substr(colon + 2);
  return fse::JsonObject()
      .Str("cpu_model", model)
      .Int("omp_num_procs", omp_get_num_procs())
      .Str("mem_total", ReadFirstLineStartingWith("/proc/meminfo", "MemTotal"))
      .Render();
}

std::string BuildInfoJson() {
  const fse::BuildInfo b = fse::GetBuildInfo();
  return fse::JsonObject()
      .Str("compiler", b.compiler_id + " " + b.compiler_version)
      .Str("build_type", b.build_type)
      .Str("cxx_flags", b.cxx_flags)
      .Int("cxx_standard", b.cxx_standard)
      .Int("openmp_version", b.openmp_version)
      .Bool("with_acorn", b.with_acorn)
      .Str("acorn_commit", b.acorn_commit)
      .Str("acorn_opt_level", b.acorn_opt_level)
      .Render();
}

template <typename T>
std::string JsonArray(const std::vector<T>& v) {
  std::ostringstream os;
  os.precision(10);
  os << "[";
  for (std::size_t i = 0; i < v.size(); ++i) {
    os << (i ? ", " : "") << v[i];
  }
  os << "]";
  return os.str();
}

std::uint64_t HashIds(const std::vector<std::int64_t>& ids) {
  std::uint64_t h = 1469598103934665603ULL;
  for (const std::int64_t id : ids) {
    h ^= static_cast<std::uint64_t>(id);
    h *= 1099511628211ULL;
  }
  return h;
}

struct Workload {
  fse::FloatMatrix base;
  fse::FloatMatrix queries;
  std::vector<std::int32_t> base_attr;
  std::vector<std::int32_t> query_attr;
  fse::NeighborTable gt_full;
  std::vector<std::int64_t> gt_ids;
  double gt_seconds = 0.0;
};

std::string GroundTruthCachePath(const Config& c, std::size_t n,
                                 std::size_t nq) {
  const std::string key =
      c.base_path + "|" + c.query_path + "|" + std::to_string(n) + "|" +
      std::to_string(nq) + "|" + std::to_string(c.attr_min) + "|" +
      std::to_string(c.attr_max) + "|" + std::to_string(c.base_seed) + "|" +
      std::to_string(c.query_seed) + "|k=" + std::to_string(c.k + 1);
  return c.cache_dir + "/gt_" + fse::Hex64(fse::Fnv1a64(key)) + ".bin";
}

Workload LoadWorkload(const Config& c) {
  Workload w;
  w.base =
      fse::ReadFvecs(c.base_path, c.max_base == 0 ? fse::kAllRows : c.max_base);
  w.queries = fse::ReadFvecs(
      c.query_path, c.max_queries == 0 ? fse::kAllRows : c.max_queries);
  const std::size_t n = w.base.rows;
  const std::size_t nq = w.queries.rows;
  w.base_attr =
      fse::UniformIntAttributes(n, c.attr_min, c.attr_max, c.base_seed);
  w.query_attr =
      fse::UniformIntAttributes(nq, c.attr_min, c.attr_max, c.query_seed);
  const std::string gt_path = GroundTruthCachePath(c, n, nq);
  if (fs::exists(gt_path)) {
    Log("loading cached filtered ground truth " + gt_path);
    w.gt_full = fse::ReadNeighborTable(gt_path);
  } else {
    Log("computing filtered ground truth (brute force)");
    const double t0 = Now();
    w.gt_full = fse::FilteredGroundTruthEquals(w.base, w.queries, w.base_attr,
                                               w.query_attr, c.k + 1);
    w.gt_seconds = Now() - t0;
    fse::WriteNeighborTable(gt_path, w.gt_full);
  }
  if (w.gt_full.nq != nq || w.gt_full.k != c.k + 1) {
    throw std::runtime_error("cached ground truth shape mismatch");
  }
  w.gt_ids.resize(nq * c.k);
  for (std::size_t q = 0; q < nq; ++q) {
    std::copy_n(w.gt_full.Ids(q), c.k,
                w.gt_ids.begin() + static_cast<std::ptrdiff_t>(q * c.k));
  }
  return w;
}

void FillFilterRow(const Workload& w, std::size_t q, std::vector<char>* row) {
  row->resize(w.base.rows);
  for (std::size_t i = 0; i < w.base.rows; ++i) {
    (*row)[i] = static_cast<char>(w.base_attr[i] == w.query_attr[q]);
  }
}

std::vector<char> FullFilterMap(const Workload& w) {
  const std::size_t n = w.base.rows;
  const std::size_t nq = w.queries.rows;
  Log("building filter map " + std::to_string(nq) + " x " + std::to_string(n));
  std::vector<char> map(nq * n);
#pragma omp parallel for
  for (std::int64_t q = 0; q < static_cast<std::int64_t>(nq); ++q) {
    for (std::size_t i = 0; i < n; ++i) {
      map[q * n + i] = static_cast<char>(
          w.base_attr[i] == w.query_attr[static_cast<std::size_t>(q)]);
    }
  }
  return map;
}

const MethodConfig& FindMethod(const Config& c, const std::string& name) {
  for (const MethodConfig& m : c.methods) {
    if (m.name == name) {
      return m;
    }
  }
  throw std::invalid_argument("method not in config: " + name);
}

std::string IndexPath(const Config& c, const MethodConfig& mc) {
  return c.cache_dir + "/acorn_" + mc.name + "_build" + c.build_label +
         ".index";
}

fse::AcornIndex LoadIndex(const Config& c, const MethodConfig& mc,
                          const Workload& w) {
  const std::string path = IndexPath(c, mc);
  if (!fs::exists(path)) {
    throw std::runtime_error("index not built yet: " + path +
                             " (run the build stage first)");
  }
  return fse::AcornIndex::Load(path, w.base_attr);
}

fse::JsonObject StageRecord(const Config& c, const std::string& stage,
                            const std::string& method) {
  const fse::BuildInfo binfo = fse::GetBuildInfo();
  const fse::GitInfo git = fse::GetGitInfo(binfo.source_dir);
  fse::JsonObject r;
  r.Str("experiment_name", c.experiment_name)
      .Str("stage", stage)
      .Str("method", method)
      .Str("stage_run_id", fse::MakeExperimentId(c.text + stage + method))
      .Str("config_path", c.path)
      .Str("git_commit", git.commit)
      .Bool("git_dirty", git.dirty)
      .Str("source_fingerprint_sha256",
           fse::SourceFingerprint(binfo.source_dir))
      .Raw("build", BuildInfoJson())
      .Raw("hardware", HardwareJson())
      .Int("base_seed", static_cast<std::int64_t>(c.base_seed))
      .Int("query_seed", static_cast<std::int64_t>(c.query_seed))
      .Str("draw_label", c.draw_label)
      .Str("build_label", c.build_label);
  return r;
}

void WriteJson(const std::string& path, const fse::JsonObject& r) {
  std::ofstream out(path);
  out << r.Render() << "\n";
  if (!out) {
    throw std::runtime_error("cannot write " + path);
  }
}

void RecordGtSanity(const Config& c, const Workload& w,
                    const std::string& exp_dir, fse::JsonObject* r) {
  const std::size_t n = w.base.rows;
  const std::size_t nq = w.queries.rows;
  const std::size_t k = c.k;
  std::map<std::int32_t, std::size_t> counts;
  for (const std::int32_t a : w.base_attr) {
    ++counts[a];
  }
  double mean_sel = 0.0;
  for (const std::int32_t a : w.query_attr) {
    mean_sel += static_cast<double>(counts[a]) / static_cast<double>(n);
  }
  mean_sel /= static_cast<double>(nq);
  std::vector<std::size_t> counts_vec;
  counts_vec.reserve(counts.size());
  for (const auto& [value, cnt] : counts) {
    counts_vec.push_back(cnt);
  }
  r->Num("mean_query_selectivity", mean_sel)
      .Raw("base_attr_value_counts", JsonArray(counts_vec));
  std::ofstream qa(exp_dir + "/query_attributes.csv");
  qa << "query,attr,selectivity\n";
  for (std::size_t q = 0; q < nq; ++q) {
    qa << q << "," << w.query_attr[q] << ","
       << static_cast<double>(counts[w.query_attr[q]]) / static_cast<double>(n)
       << "\n";
  }
  std::size_t ties = 0;
  std::size_t violations = 0;
  for (std::size_t q = 0; q < nq; ++q) {
    ties += w.gt_full.Distances(q)[k - 1] == w.gt_full.Distances(q)[k] ? 1 : 0;
    for (std::size_t j = 0; j < k; ++j) {
      const std::int64_t id = w.gt_ids[q * k + j];
      violations += (id < 0 || w.base_attr[id] != w.query_attr[q]) ? 1 : 0;
    }
  }
  r->Int("gt_queries_with_tie_at_rank_k", static_cast<std::int64_t>(ties))
      .Int("gt_filter_violations", static_cast<std::int64_t>(violations));
}

void CheckOfficialGt(const Config& c, const Workload& w, fse::JsonObject* r) {
  const std::size_t nq = w.queries.rows;
  const std::size_t k = c.k;
  Log("unfiltered ground truth vs official sift_groundtruth.ivecs");
  const fse::IdMatrix official = fse::ReadIvecs(c.official_gt_path, nq);
  const fse::NeighborTable ours =
      fse::UnfilteredGroundTruth(w.base, w.queries, k + 1);
  std::size_t exact = 0;
  std::size_t tie_only = 0;
  std::size_t mismatched = 0;
  for (std::size_t q = 0; q < nq; ++q) {
    bool same = true;
    bool tie_explained = true;
    for (std::size_t j = 0; j < k; ++j) {
      const std::int32_t off = official.Row(q)[j];
      same = same && ours.Ids(q)[j] == off;
      tie_explained =
          tie_explained && fse::SquaredL2(w.queries.Row(q), w.base.Row(off),
                                          w.base.dim) == ours.Distances(q)[j];
    }
    if (same) {
      ++exact;
    } else if (tie_explained) {
      ++tie_only;
    } else {
      ++mismatched;
    }
  }
  r->Int("official_gt_rows_identical", static_cast<std::int64_t>(exact))
      .Int("official_gt_rows_differ_only_by_ties",
           static_cast<std::int64_t>(tie_only))
      .Int("official_gt_rows_mismatched",
           static_cast<std::int64_t>(mismatched));
  Log("official GT: identical " + std::to_string(exact) + ", tie-only " +
      std::to_string(tie_only) + ", mismatched " + std::to_string(mismatched));
}

void CheckFaissGt(const Config& c, const Workload& w, fse::JsonObject* r) {
  const std::size_t n = w.base.rows;
  const std::size_t k = c.k;
  const std::size_t m = std::min(c.faiss_crosscheck_queries, w.queries.rows);
  std::size_t identical = 0;
  std::size_t tie_only = 0;
  std::size_t mismatched = 0;
  for (std::size_t q = 0; q < m; ++q) {
    std::vector<std::int64_t> subset;
    for (std::size_t i = 0; i < n; ++i) {
      if (w.base_attr[i] == w.query_attr[q]) {
        subset.push_back(static_cast<std::int64_t>(i));
      }
    }
    faiss::IndexFlatL2 flat(static_cast<faiss::idx_t>(w.base.dim));
    std::vector<float> sub(subset.size() * w.base.dim);
    for (std::size_t s = 0; s < subset.size(); ++s) {
      std::copy_n(w.base.Row(subset[s]), w.base.dim,
                  sub.begin() + static_cast<std::ptrdiff_t>(s * w.base.dim));
    }
    flat.add(static_cast<faiss::idx_t>(subset.size()), sub.data());
    std::vector<float> d(k);
    std::vector<faiss::idx_t> l(k);
    flat.search(1, w.queries.Row(q), static_cast<faiss::idx_t>(k), d.data(),
                l.data());
    bool ids_same = true;
    bool dist_same = true;
    for (std::size_t j = 0; j < k; ++j) {
      ids_same = ids_same && subset[l[j]] == w.gt_ids[q * k + j];
      dist_same = dist_same && d[j] == w.gt_full.Distances(q)[j];
    }
    if (ids_same) {
      ++identical;
    } else if (dist_same) {
      ++tie_only;
    } else {
      ++mismatched;
    }
  }
  r->Int("faiss_crosscheck_queries", static_cast<std::int64_t>(m))
      .Int("faiss_crosscheck_identical", static_cast<std::int64_t>(identical))
      .Int("faiss_crosscheck_differ_only_by_ties",
           static_cast<std::int64_t>(tie_only))
      .Int("faiss_crosscheck_mismatched",
           static_cast<std::int64_t>(mismatched));
  Log("FAISS cross-check: identical " + std::to_string(identical) + "/" +
      std::to_string(m) + ", tie-only " + std::to_string(tie_only) +
      ", mismatched " + std::to_string(mismatched));
}

void StageGtChecks(const Config& c, const std::string& exp_dir) {
  const Workload w = LoadWorkload(c);
  fse::JsonObject r = StageRecord(c, "gt_checks", "");
  r.Num("ground_truth_seconds", w.gt_seconds);
  RecordGtSanity(c, w, exp_dir, &r);
  if (c.official_gt_check && c.max_base == 0) {
    CheckOfficialGt(c, w, &r);
  }
  CheckFaissGt(c, w, &r);
  WriteJson(exp_dir + "/gt_checks.json", r);
}

void StageBuild(const Config& c, const MethodConfig& mc,
                const std::string& dir) {
  const Workload w = LoadWorkload(c);
  const std::string path = IndexPath(c, mc);
  fse::JsonObject r = StageRecord(c, "build", mc.name);
  const bool cached = fs::exists(path);
  if (cached && !c.reuse_cached_index) {
    throw std::runtime_error(
        "index already exists and reuse_cached is "
        "false: " +
        path);
  }
  bool built_here = false;
  fse::AcornIndex index = cached ? fse::AcornIndex::Load(path, w.base_attr)
                                 : fse::AcornIndex(static_cast<int>(w.base.dim),
                                                   mc.params, w.base_attr);
  if (!cached) {
    omp_set_num_threads(c.build_threads);
    Log(mc.name + ": building (" + std::to_string(c.build_threads) +
        " threads)");
    const double tti = index.Add(w.base.rows, w.base.data.data());
    Log(mc.name + ": TTI " + std::to_string(tti) + " s");
    r.Num("tti_seconds", tti).Int("build_threads", c.build_threads);
    index.Save(path);
    built_here = true;
  } else {
    Log(mc.name + ": using cached index " + path);
    r.Str("loaded_from", path);
  }
  r.Bool("built_this_run", built_here)
      .Int("gamma", mc.params.gamma)
      .Int("m", mc.params.m)
      .Int("m_beta", mc.params.m_beta)
      .Int("ef_construction", index.EfConstruction())
      .Raw("avg_out_degree_per_level",
           JsonArray(index.AverageOutDegreePerLevel()))
      .Raw("nodes_per_level", JsonArray(index.NodesPerLevel()))
      .Int("memory_bytes", static_cast<std::int64_t>(index.MemoryBytes()))
      .Int("file_bytes", static_cast<std::int64_t>(fs::file_size(path)));

  if (c.reload_check && built_here) {
    const std::vector<char> map = FullFilterMap(w);
    fse::AcornIndex reloaded = fse::AcornIndex::Load(path, w.base_attr);
    const int efs = c.determinism_efs.front();
    index.SetEfSearch(efs);
    reloaded.SetEfSearch(efs);
    omp_set_num_threads(c.timing_threads);
    const std::size_t nq = w.queries.rows;
    std::vector<std::int64_t> a(nq * c.k);
    std::vector<std::int64_t> b(nq * c.k);
    std::vector<float> da(nq * c.k);
    std::vector<float> db(nq * c.k);
    index.SearchBatch(nq, w.queries.data.data(), c.k, map.data(), a.data(),
                      da.data());
    reloaded.SearchBatch(nq, w.queries.data.data(), c.k, map.data(), b.data(),
                         db.data());
    const bool same = a == b && da == db;
    r.Bool("reload_identical", same);
    Log(mc.name + ": reload identical: " + (same ? "yes" : "NO"));
  }
  WriteJson(dir + "/build.json", r);
}

void StageSweep(const Config& c, const MethodConfig& mc,
                const std::string& dir) {
  const Workload w = LoadWorkload(c);
  fse::AcornIndex index = LoadIndex(c, mc, w);
  fse::JsonObject r = StageRecord(c, "sweep", mc.name);
  const std::size_t nq = w.queries.rows;
  const std::size_t k = c.k;
  omp_set_num_threads(1);

  std::ofstream per_query(dir + "/per_query.csv");
  per_query << "method,efs,query,recall,ndis,latency_us,n_valid,"
               "filter_violations,distance_mismatches\n";
  std::ofstream aggregate(dir + "/aggregate.csv");
  aggregate << "method,efs,mean_recall,mean_ndis,total_ndis,"
               "filter_violations,distance_mismatches,max_abs_dist_err,"
               "mean_latency_us,ids_hash\n";
  const std::set<int> paper(c.timing_efs.begin(), c.timing_efs.end());
  std::vector<char> row;
  std::size_t total_violations = 0;
  std::size_t total_mismatches = 0;
  const double t_start = Now();
  for (const int efs : c.per_query_efs) {
    index.SetEfSearch(efs);
    std::vector<std::int64_t> all_ids(nq * k);
    std::vector<float> all_d(nq * k);
    double recall_sum = 0.0;
    double lat_sum = 0.0;
    std::uint64_t ndis_total = 0;
    std::size_t viol_total = 0;
    std::size_t mism_total = 0;
    double max_err = 0.0;
    for (std::size_t q = 0; q < nq; ++q) {
      FillFilterRow(w, q, &row);
      const fse::AcornQueryResult res =
          index.SearchOne(w.queries.Row(q), k, row.data());
      std::size_t valid = 0;
      std::size_t viol = 0;
      std::size_t mism = 0;
      for (std::size_t j = 0; j < k; ++j) {
        const std::int64_t id = res.ids[j];
        if (id < 0) {
          continue;
        }
        ++valid;
        viol += w.base_attr[id] != w.query_attr[q] ? 1 : 0;
        const float d =
            fse::SquaredL2(w.queries.Row(q), w.base.Row(id), w.base.dim);
        const double err = std::abs(static_cast<double>(d) -
                                    static_cast<double>(res.distances[j]));
        max_err = std::max(max_err, err);
        mism += err > 1e-3 * std::max(1.0, static_cast<double>(d)) ? 1 : 0;
      }
      const double rec = fse::RecallAtK(res.ids.data(), &w.gt_ids[q * k], k);
      recall_sum += rec;
      lat_sum += res.seconds;
      ndis_total += res.distance_computations;
      viol_total += viol;
      mism_total += mism;
      const auto off = static_cast<std::ptrdiff_t>(q * k);
      std::copy(res.ids.begin(), res.ids.end(), all_ids.begin() + off);
      std::copy(res.distances.begin(), res.distances.end(),
                all_d.begin() + off);
      per_query << mc.name << "," << efs << "," << q << "," << rec << ","
                << res.distance_computations << "," << res.seconds * 1e6 << ","
                << valid << "," << viol << "," << mism << "\n";
    }
    const auto dnq = static_cast<double>(nq);
    aggregate << mc.name << "," << efs << "," << recall_sum / dnq << ","
              << static_cast<double>(ndis_total) / dnq << "," << ndis_total
              << "," << viol_total << "," << mism_total << "," << max_err << ","
              << lat_sum / dnq * 1e6 << "," << fse::Hex64(HashIds(all_ids))
              << "\n";
    per_query.flush();
    aggregate.flush();
    total_violations += viol_total;
    total_mismatches += mism_total;
    if (paper.contains(efs)) {
      fse::NeighborTable raw{nq, k, all_ids, all_d};
      fse::WriteNeighborTable(dir + "/raw_efs" + std::to_string(efs) + ".bin",
                              raw);
    }
    Log(mc.name + " efs=" + std::to_string(efs) +
        " recall=" + std::to_string(recall_sum / dnq) +
        " ndis=" + std::to_string(static_cast<double>(ndis_total) / dnq) +
        " violations=" + std::to_string(viol_total));
  }
  r.Int("efs_values", static_cast<std::int64_t>(c.per_query_efs.size()))
      .Int("filter_violations", static_cast<std::int64_t>(total_violations))
      .Int("distance_mismatches", static_cast<std::int64_t>(total_mismatches))
      .Num("wall_seconds", Now() - t_start);
  WriteJson(dir + "/sweep.json", r);
}

struct SweepRecord {
  std::map<int, std::string> ids_hash;
  std::map<int, std::uint64_t> total_ndis;
  std::map<int, std::vector<std::uint64_t>> ndis;
};

SweepRecord ReadSweep(const std::string& dir, const std::set<int>& want,
                      std::size_t nq) {
  SweepRecord s;
  std::ifstream agg(dir + "/aggregate.csv");
  std::string line;
  std::getline(agg, line);
  while (std::getline(agg, line)) {
    std::vector<std::string> f;
    std::stringstream ls(line);
    std::string cell;
    while (std::getline(ls, cell, ',')) {
      f.push_back(cell);
    }
    if (f.size() < 10) {
      continue;
    }
    const int efs = std::stoi(f[1]);
    s.total_ndis[efs] = std::stoull(f[4]);
    s.ids_hash[efs] = f[9];
  }
  std::ifstream pq(dir + "/per_query.csv");
  std::getline(pq, line);
  while (std::getline(pq, line)) {
    std::stringstream ls(line);
    std::string method;
    std::string efs_s;
    std::string q_s;
    std::string recall_s;
    std::string ndis_s;
    std::getline(ls, method, ',');
    std::getline(ls, efs_s, ',');
    const int efs = std::stoi(efs_s);
    if (!want.contains(efs)) {
      continue;
    }
    std::getline(ls, q_s, ',');
    std::getline(ls, recall_s, ',');
    std::getline(ls, ndis_s, ',');
    auto& v = s.ndis[efs];
    v.resize(nq);
    v[std::stoull(q_s)] = std::stoull(ndis_s);
  }
  return s;
}

void StageVerify(const Config& c, const MethodConfig& mc,
                 const std::string& dir) {
  const Workload w = LoadWorkload(c);
  fse::AcornIndex index = LoadIndex(c, mc, w);
  fse::JsonObject r = StageRecord(c, "verify", mc.name);
  const std::size_t nq = w.queries.rows;
  const std::size_t k = c.k;
  std::set<int> want(c.determinism_efs.begin(), c.determinism_efs.end());
  const SweepRecord sweep = ReadSweep(dir, want, nq);
  if (sweep.ids_hash.empty()) {
    throw std::runtime_error("no sweep output in " + dir);
  }

  omp_set_num_threads(1);
  std::vector<char> row;
  bool all_same = true;
  std::ostringstream det;
  for (const int efs : c.determinism_efs) {
    index.SetEfSearch(efs);
    std::vector<std::int64_t> ids(nq * k);
    std::size_t ndis_diffs = 0;
    for (std::size_t q = 0; q < nq; ++q) {
      FillFilterRow(w, q, &row);
      const fse::AcornQueryResult res =
          index.SearchOne(w.queries.Row(q), k, row.data());
      std::copy(res.ids.begin(), res.ids.end(),
                ids.begin() + static_cast<std::ptrdiff_t>(q * k));
      ndis_diffs += res.distance_computations != sweep.ndis.at(efs)[q] ? 1 : 0;
    }
    const bool ids_same = fse::Hex64(HashIds(ids)) == sweep.ids_hash.at(efs);
    all_same = all_same && ids_same && ndis_diffs == 0;
    det << "efs" << efs << ": ids " << (ids_same ? "identical" : "DIFFERENT")
        << ", ndis diffs " << ndis_diffs << "; ";
  }
  r.Bool("per_query_pass_deterministic", all_same)
      .Str("determinism_detail", det.str());
  Log(mc.name + " determinism: " + det.str());

  const std::vector<char> map = FullFilterMap(w);
  omp_set_num_threads(c.timing_threads);
  bool counters_equal = true;
  bool batch_ids_equal = true;
  std::ostringstream cnt;
  for (const int efs : c.counter_check_efs) {
    index.SetEfSearch(efs);
    std::vector<std::int64_t> ids(nq * k);
    std::vector<float> d(nq * k);
    const std::size_t before = faiss::acorn_stats.n3;
    index.SearchBatch(nq, w.queries.data.data(), k, map.data(), ids.data(),
                      d.data());
    const std::size_t batch = faiss::acorn_stats.n3 - before;
    counters_equal = counters_equal && batch == sweep.total_ndis.at(efs);
    batch_ids_equal =
        batch_ids_equal && fse::Hex64(HashIds(ids)) == sweep.ids_hash.at(efs);
    cnt << "efs" << efs << ": batch=" << batch
        << " per_query_sum=" << sweep.total_ndis.at(efs) << "; ";
  }
  r.Bool("counter_batch_equals_per_query_sum", counters_equal)
      .Bool("batch_ids_equal_per_query_ids", batch_ids_equal)
      .Str("counter_detail", cnt.str());
  Log(mc.name + " counter consistency: " + cnt.str() +
      (batch_ids_equal ? " batch ids == per-query ids" : " BATCH IDS DIFFER"));
  WriteJson(dir + "/verify.json", r);
}

void StageTiming(const Config& c, const MethodConfig& mc,
                 const std::string& dir) {
  const Workload w = LoadWorkload(c);
  fse::AcornIndex index = LoadIndex(c, mc, w);
  fse::JsonObject r = StageRecord(c, "timing", mc.name);
  const std::size_t nq = w.queries.rows;
  const std::size_t k = c.k;
  const std::vector<char> map = FullFilterMap(w);
  omp_set_num_threads(c.timing_threads);
  std::ofstream timing(dir + "/timing_trials.csv");
  timing << "method,efs,threads,trial,seconds,qps,batch_recall\n";
  for (const int efs : c.timing_efs) {
    index.SetEfSearch(efs);
    std::vector<std::int64_t> ids(nq * k);
    std::vector<float> d(nq * k);
    for (int t = -c.timing_warmup; t < c.timing_trials; ++t) {
      const double t0 = Now();
      index.SearchBatch(nq, w.queries.data.data(), k, map.data(), ids.data(),
                        d.data());
      const double secs = Now() - t0;
      if (t < 0) {
        continue;
      }
      std::string rec_s;
      if (t == 0) {
        double rec = 0.0;
        for (std::size_t q = 0; q < nq; ++q) {
          rec += fse::RecallAtK(&ids[q * k], &w.gt_ids[q * k], k);
        }
        rec_s = std::to_string(rec / static_cast<double>(nq));
      }
      timing << mc.name << "," << efs << "," << c.timing_threads << "," << t
             << "," << secs << "," << static_cast<double>(nq) / secs << ","
             << rec_s << "\n";
    }
    timing.flush();
    Log(mc.name + " timed efs=" + std::to_string(efs));
  }
  r.Int("threads", c.timing_threads)
      .Int("warmup_trials", c.timing_warmup)
      .Int("trials", c.timing_trials);
  WriteJson(dir + "/timing.json", r);
}

}

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: fse_acorn_repro <config.yaml> gt_checks\n"
                 "       fse_acorn_repro <config.yaml> "
                 "{build|sweep|verify|timing} <method>\n";
    return 2;
  }
  try {
    const Config c = LoadConfig(argv[1]);
    const std::string stage = argv[2];
    const std::string exp_dir = c.output_dir + "/" + c.experiment_name;
    fs::create_directories(exp_dir);
    fs::create_directories(c.cache_dir);
    std::ofstream(exp_dir + "/config.yaml") << c.text;
    if (stage == "gt_checks") {
      StageGtChecks(c, exp_dir);
      return 0;
    }
    if (argc != 4) {
      throw std::invalid_argument("stage " + stage + " needs a method");
    }
    const std::string method_name = argv[3];
    const MethodConfig& mc = FindMethod(c, method_name);
    const std::string dir = exp_dir + "/" + mc.name;
    fs::create_directories(dir);
    if (stage == "build") {
      StageBuild(c, mc, dir);
    } else if (stage == "sweep") {
      StageSweep(c, mc, dir);
    } else if (stage == "verify") {
      StageVerify(c, mc, dir);
    } else if (stage == "timing") {
      StageTiming(c, mc, dir);
    } else {
      throw std::invalid_argument("unknown stage " + stage);
    }
    Log("stage " + stage + " done");
  } catch (const std::exception& e) {
    std::cerr << "fse_acorn_repro: error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
