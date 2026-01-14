// Runs one stage of the effort matrix: index builds, the ACORN audit, a sweep
// or the timing pass. Usage: fse_matrix <config.yaml> <stage> [method
// condition]

#include <faiss/IndexACORN.h>
#include <faiss/impl/ACORN.h>
#include <omp.h>
#include <sched.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
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
#include "methods/filtered_result.h"
#include "methods/ingraph/acorn_accounting.h"
#include "methods/ingraph/acorn_counting_storage.h"
#include "methods/ingraph/acorn_index.h"
#include "methods/postfilter/hnsw_index.h"
#include "methods/postfilter/postfilter.h"
#include "methods/prefilter/prefilter.h"
#include "metrics/recall.h"

namespace fs = std::filesystem;

namespace {

struct Config {
  std::string path;
  std::string text;
  std::string experiment_name;
  std::string base_path;
  std::string query_path;
  std::size_t k = 0;
  std::size_t gt_k = 0;
  double s_min = 0.0;
  double s_max = 0.0;
  int levels = 0;
  std::map<std::string, std::uint64_t> attribute_hashes;
  std::string manifest;
  std::string generator_cache_dir;
  std::vector<std::size_t> budgets;

  fse::HnswParams hnsw;
  int post_build_threads = 1;
  fse::PostfilterParams post;
  std::size_t replay_queries = 0;
  std::uint64_t replay_seed = 0;

  bool post_reuse_inert = false;

  fse::AcornParams acorn;
  int acorn_build_threads = 16;
  std::size_t audit_base = 0;
  std::size_t audit_queries = 0;
  std::string acorn_index_path;
  std::string acorn_index_hash;

  std::size_t subset_queries = 0;
  std::uint64_t subset_seed = 0;
  bool subset_first_n = true;
  std::vector<std::string> only_budgets_note;
  int sweep_threads = 16;

  std::size_t timing_queries = 0;
  std::uint64_t timing_seed = 0;
  int timing_repeats = 0;
  std::string timing_mode = "all_methods";
  int timing_pin_core = -1;
  std::string index_dir;
  std::string output_dir;
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
  c.k = y["search"]["k"].as<std::size_t>();
  c.gt_k = y["search"]["gt_k"].as<std::size_t>();
  c.s_min = y["selectivity"]["min"].as<double>();
  c.s_max = y["selectivity"]["max"].as<double>();
  c.levels = y["selectivity"]["levels"].as<int>();
  if (y["generator"]["attribute_hashes"]) {
    for (const auto& kv : y["generator"]["attribute_hashes"]) {
      c.attribute_hashes[kv.first.as<std::string>()] =
          std::stoull(kv.second.as<std::string>(), nullptr, 16);
    }
  }
  if (y["generator"]["manifest"]) {
    c.manifest = y["generator"]["manifest"].as<std::string>();
  }
  c.generator_cache_dir = y["generator"]["cache_dir"].as<std::string>();
  c.budgets = y["budgets"].as<std::vector<std::size_t>>();
  const YAML::Node p = y["methods"]["postfilter"];
  c.hnsw.m = p["m"].as<std::size_t>();
  c.hnsw.ef_construction = p["ef_construction"].as<std::size_t>();
  c.hnsw.seed = p["seed"].as<std::uint64_t>();
  c.post_build_threads = p["build_threads"].as<int>();
  c.post.overfetch_factor = p["overfetch_factor"].as<double>();
  c.post.growth_factor = p["growth_factor"].as<double>();
  c.post.max_rounds = p["max_rounds"].as<std::uint32_t>();
  c.replay_queries = p["replay_queries"].as<std::size_t>();
  c.replay_seed = p["replay_seed"].as<std::uint64_t>();
  if (p["reuse_inert_budgets"]) {
    c.post_reuse_inert = p["reuse_inert_budgets"].as<bool>();
  }
  const YAML::Node a = y["methods"]["acorn"];
  c.acorn.gamma = a["gamma"].as<int>();
  c.acorn.m = a["m"].as<int>();
  c.acorn.m_beta = a["m_beta"].as<int>();
  c.acorn_build_threads = a["build_threads"].as<int>();
  c.audit_base = a["audit_base"].as<std::size_t>();
  c.audit_queries = a["audit_queries"].as<std::size_t>();
  if (a["index_path"]) {
    c.acorn_index_path = a["index_path"].as<std::string>();
    c.acorn_index_hash = a["index_hash"].as<std::string>();
  }
  const YAML::Node q = y["query_subset"];
  c.subset_queries = q["count"].as<std::size_t>();
  c.subset_seed = q["seed"].as<std::uint64_t>();
  c.subset_first_n = q["first_n"].as<bool>();
  c.sweep_threads = y["sweep_threads"].as<int>();
  const YAML::Node t = y["timing"];
  c.timing_queries = t["queries"].as<std::size_t>();
  c.timing_seed = t["seed"].as<std::uint64_t>();
  c.timing_repeats = t["repeats"].as<int>();
  if (t["mode"]) {
    c.timing_mode = t["mode"].as<std::string>();
  }
  if (c.timing_mode != "all_methods" && c.timing_mode != "acorn_audit") {
    throw std::invalid_argument("unknown timing.mode " + c.timing_mode);
  }
  if (t["pin_core"]) {
    c.timing_pin_core = t["pin_core"].as<int>();
  }
  c.index_dir = y["paths"]["index_dir"].as<std::string>();
  c.output_dir = y["paths"]["output_dir"].as<std::string>();
  return c;
}

void Log(const std::string& msg) {
  std::cout << "[fse_matrix] " << msg << '\n' << std::flush;
}

double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::uint64_t HashMatrix(const fse::FloatMatrix& m) {
  return fse::Fnv1a64Bytes(m.data.data(), m.data.size() * sizeof(float));
}

std::uint64_t HashFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  std::vector<char> buf(1 << 22);
  std::uint64_t h = 1469598103934665603ULL;
  while (in) {
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    h = fse::Fnv1a64Bytes(buf.data(), static_cast<std::size_t>(in.gcount()), h);
  }
  return h;
}

std::string PostIndexPath(const Config& c) {
  return c.index_dir + "/hnsw_post_m" + std::to_string(c.hnsw.m) + "_efc" +
         std::to_string(c.hnsw.ef_construction) + "_seed" +
         std::to_string(c.hnsw.seed) + ".bin";
}

std::string AcornIndexPath(const Config& c) {
  if (!c.acorn_index_path.empty()) {
    return c.acorn_index_path;
  }
  return c.index_dir + "/acorn_g" + std::to_string(c.acorn.gamma) + "_m" +
         std::to_string(c.acorn.m) + "_mb" + std::to_string(c.acorn.m_beta) +
         ".index";
}

fse::JsonObject RecordHeader(const Config& c, const std::string& stage) {
  const fse::BuildInfo b = fse::GetBuildInfo();
  const fse::GitInfo g = fse::GetGitInfo(b.source_dir);
  fse::JsonObject r;
  r.Str("experiment_name", c.experiment_name)
      .Str("stage", stage)
      .Str("run_id", fse::MakeExperimentId(c.text + stage))
      .Str("config_path", c.path)
      .Str("git_commit", g.commit)
      .Bool("git_dirty", g.dirty)
      .Str("source_fingerprint_sha256", fse::SourceFingerprint(b.source_dir))
      .Raw("build", fse::BuildInfoJson())
      .Raw("hardware", fse::HardwareJson());
  return r;
}

std::vector<std::size_t> QueryIds(const Config& c, std::size_t nq) {
  std::vector<std::size_t> ids;
  if (c.subset_queries == 0 || c.subset_queries >= nq) {
    ids.resize(nq);
    for (std::size_t i = 0; i < nq; ++i) {
      ids[i] = i;
    }
    return ids;
  }
  if (c.subset_first_n) {
    for (std::size_t i = 0; i < c.subset_queries; ++i) {
      ids.push_back(i);
    }
    return ids;
  }
  const auto perm = fse::SeededPermutation(nq, c.subset_seed);
  for (std::size_t i = 0; i < c.subset_queries; ++i) {
    ids.push_back(static_cast<std::size_t>(perm[i]));
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

struct Conditions {
  std::vector<fse::FilterCondition> list;
  std::uint64_t base_hash = 0;
};

Conditions LoadManifestConditions(const Config& c, Conditions out) {
  const YAML::Node m = YAML::LoadFile(c.manifest);
  if (m["base_hash"].as<std::string>() != fse::Hex64(out.base_hash)) {
    throw std::runtime_error("manifest base hash mismatch");
  }
  std::map<std::string, fse::RankAttribute> attrs;
  for (const auto& e : m["conditions"]) {
    const auto path = e["attribute"].as<std::string>();
    if (attrs.find(path) == attrs.end()) {
      attrs.emplace(path,
                    fse::ReadRankAttribute(
                        path, std::stoull(e["attribute_hash"].as<std::string>(),
                                          nullptr, 16)));
    }
    fse::FilterCondition f = fse::MakeFilterCondition(
        attrs.at(path), e["s"].as<double>(), out.base_hash);
    f.name = e["name"].as<std::string>();
    if (fse::Hex64(f.condition_id) != e["condition_id"].as<std::string>() ||
        fse::Hex64(f.mask_hash) != e["mask_hash"].as<std::string>()) {
      throw std::runtime_error("manifest condition mismatch: " + f.name);
    }
    out.list.push_back(std::move(f));
  }
  return out;
}

Conditions LoadConditions(const Config& c, const fse::FloatMatrix& base) {
  Conditions out;
  out.base_hash = HashMatrix(base);
  if (!c.manifest.empty()) {
    return LoadManifestConditions(c, std::move(out));
  }
  const auto levels = fse::LogSpacedSelectivities(c.s_min, c.s_max, c.levels);
  for (const auto& [corr, hash] : c.attribute_hashes) {
    const std::string path = c.generator_cache_dir + "/attr_" + corr + "_" +
                             fse::Hex64(hash) + ".bin";
    const fse::RankAttribute a = fse::ReadRankAttribute(path, hash);
    if (fse::CorrelationName(a.correlation) != corr ||
        a.rank.size() != base.rows) {
      throw std::runtime_error("attribute does not match: " + path);
    }
    for (const double s : levels) {
      out.list.push_back(fse::MakeFilterCondition(a, s, out.base_hash));
    }
  }
  return out;
}

const fse::FilterCondition& FindCondition(const Conditions& cs,
                                          const std::string& name) {
  for (const auto& f : cs.list) {
    if (f.name == name) {
      return f;
    }
  }
  throw std::invalid_argument("unknown condition " + name);
}

int BuildPost(const Config& c) {
  const fse::FloatMatrix base = fse::ReadFvecs(c.base_path);
  fs::create_directories(c.index_dir);
  fse::JsonObject r = RecordHeader(c, "build_post");
  fse::HnswIndex index(base.dim, base.rows, c.hnsw);
  const double t0 = Now();
  index.Add(base, 0, c.post_build_threads);
  const double secs = Now() - t0;
  const std::string path = PostIndexPath(c);
  index.Save(path);
  r.Num("build_seconds", secs)
      .Int("build_threads", c.post_build_threads)
      .Int("m", static_cast<std::int64_t>(c.hnsw.m))
      .Int("ef_construction", static_cast<std::int64_t>(c.hnsw.ef_construction))
      .Int("seed", static_cast<std::int64_t>(c.hnsw.seed))
      .Str("index_path", path)
      .Str("index_file_hash", fse::Hex64(HashFile(path)))
      .Int("file_bytes", static_cast<std::int64_t>(fs::file_size(path)));
  std::ofstream(c.index_dir + "/build_post.json") << r.Render() << "\n";
  Log("post-filter HNSW built in " + std::to_string(secs) + " s -> " + path);
  return 0;
}

int BuildAcorn(const Config& c) {
  if (!c.acorn_index_path.empty()) {
    throw std::invalid_argument(
        "build_acorn refused: index_path is a frozen "
        "index override");
  }
  const fse::FloatMatrix base = fse::ReadFvecs(c.base_path);
  fs::create_directories(c.index_dir);
  fse::JsonObject r = RecordHeader(c, "build_acorn");

  fse::AcornIndex index(static_cast<int>(base.dim), c.acorn,
                        std::vector<std::int32_t>(base.rows, 0));
  omp_set_num_threads(c.acorn_build_threads);
  const double secs = index.Add(base.rows, base.data.data());
  const std::string path = AcornIndexPath(c);
  index.Save(path);
  std::ostringstream deg;
  for (const double x : index.AverageOutDegreePerLevel()) {
    deg << x << " ";
  }
  r.Num("build_seconds", secs)
      .Int("build_threads", c.acorn_build_threads)
      .Int("gamma", c.acorn.gamma)
      .Int("m", c.acorn.m)
      .Int("m_beta", c.acorn.m_beta)
      .Int("ef_construction", index.EfConstruction())
      .Str("avg_out_degree_per_level", deg.str())
      .Int("memory_bytes", static_cast<std::int64_t>(index.MemoryBytes()))
      .Str("index_path", path)
      .Str("index_file_hash", fse::Hex64(HashFile(path)))
      .Int("file_bytes", static_cast<std::int64_t>(fs::file_size(path)));
  std::ofstream(c.index_dir + "/build_acorn.json") << r.Render() << "\n";
  Log("ACORN built in " + std::to_string(secs) + " s -> " + path);
  return 0;
}

int AuditAcorn(const Config& c) {
  const fse::FloatMatrix base = fse::ReadFvecs(c.base_path, c.audit_base);
  const fse::FloatMatrix queries =
      fse::ReadFvecs(c.query_path, c.audit_queries);
  const auto n = static_cast<faiss::idx_t>(base.rows);
  const auto d = static_cast<faiss::idx_t>(base.dim);
  std::vector<int> metadata(base.rows, 0);
  omp_set_num_threads(1);
  fse::CountingFlatL2 storage(d);
  faiss::IndexACORN counted(&storage, c.acorn.m, c.acorn.gamma, metadata,
                            c.acorn.m_beta);
  counted.add(n, base.data.data());
  faiss::IndexACORNFlat plain(static_cast<int>(d), c.acorn.m, c.acorn.gamma,
                              metadata, c.acorn.m_beta);
  plain.add(n, base.data.data());
  const auto attr = fse::RandomRankAttribute(base.rows, 20261010);
  const auto levels = fse::LogSpacedSelectivities(c.s_min, c.s_max, c.levels);
  std::size_t checked = 0;
  std::size_t offset_errors = 0;
  std::size_t behaviour_diffs = 0;
  for (const double s : levels) {
    const fse::FilterCondition f = fse::MakeFilterCondition(attr, s, 0);
    std::vector<char> row = f.mask;
    for (const std::size_t b : c.budgets) {
      counted.acorn.efSearch = static_cast<int>(b);
      plain.acorn.efSearch = static_cast<int>(b);
      for (std::size_t q = 0; q < queries.rows; ++q) {
        std::vector<faiss::idx_t> ic(c.k);
        std::vector<faiss::idx_t> ip(c.k);
        std::vector<float> dc(c.k);
        std::vector<float> dp(c.k);
        storage.count = 0;
        const std::size_t n3c = faiss::acorn_stats.n3;
        counted.search(1, queries.Row(q), static_cast<faiss::idx_t>(c.k),
                       dc.data(), ic.data(), row.data());
        const std::uint64_t native = faiss::acorn_stats.n3 - n3c;
        const std::size_t n3p = faiss::acorn_stats.n3;
        plain.search(1, queries.Row(q), static_cast<faiss::idx_t>(c.k),
                     dp.data(), ip.data(), row.data());
        const std::uint64_t native_p = faiss::acorn_stats.n3 - n3p;
        behaviour_diffs += (ic != ip || native != native_p) ? 1 : 0;
        offset_errors +=
            fse::AcornExactDistanceComputations(native) != storage.count ? 1
                                                                         : 0;
        ++checked;
      }
    }
  }
  const bool ok = offset_errors == 0 && behaviour_diffs == 0 && checked > 0;
  fse::JsonObject r = RecordHeader(c, "audit_acorn");
  r.Int("audit_base", static_cast<std::int64_t>(base.rows))
      .Int("audit_queries", static_cast<std::int64_t>(queries.rows))
      .Int("gamma", c.acorn.gamma)
      .Int("query_searches_checked", static_cast<std::int64_t>(checked))
      .Int("offset_errors", static_cast<std::int64_t>(offset_errors))
      .Int("behaviour_differences", static_cast<std::int64_t>(behaviour_diffs))
      .Bool("pass", ok);
  fs::create_directories(c.output_dir);
  std::ofstream(c.output_dir + "/audit_acorn.json") << r.Render() << "\n";
  std::ostringstream msg;
  msg << "ACORN +1 audit: " << checked << " searches, offset errors "
      << offset_errors << ", behaviour diffs " << behaviour_diffs
      << (ok ? " PASS" : " FAIL");
  Log(msg.str());
  return ok ? 0 : 1;
}

struct Row {
  std::size_t budget = 0;
  std::size_t query = 0;
  std::size_t effective_list = 0;
  double recall = 0.0;
  std::uint64_t dist_exact = 0;
  std::uint64_t dist_native = 0;
  std::uint64_t filter_checks = 0;
  std::uint32_t rounds = 0;
  std::size_t last_fetch = 0;
  double latency_us = 0.0;
  std::size_t n_valid = 0;
  std::size_t violations = 0;
  std::size_t mismatches = 0;
  bool tie = false;
  std::uint64_t n_scanned = 0;
  std::int64_t seed0 = -1;
};

void Score(const fse::FilteredSearchResult& r, const fse::FloatMatrix& base,
           const float* query, const std::vector<char>& mask,
           const fse::NeighborTable& gt, std::size_t q, std::size_t k,
           Row* row) {
  row->recall = fse::RecallAtK(r.ids.data(), gt.Ids(q), k);
  row->tie = gt.Distances(q)[k - 1] == gt.Distances(q)[k];
  for (std::size_t j = 0; j < k; ++j) {
    const std::int64_t id = r.ids[j];
    if (id < 0) {
      continue;
    }
    ++row->n_valid;
    row->violations += mask[static_cast<std::size_t>(id)] == 0 ? 1 : 0;
    const double dd = fse::SquaredL2(query, base.Row(id), base.dim);
    row->mismatches +=
        std::abs(dd - r.distances[j]) > 1e-3 * std::max(1.0, dd) ? 1 : 0;
  }
}

struct SweepContext {
  const Config& c;
  const fse::FloatMatrix& base;
  const fse::FloatMatrix& queries;
  const fse::FilterCondition& f;
  const std::vector<std::size_t>& qids;
  double s;
};

void RunPrefilter(const SweepContext& x,
                  std::vector<fse::FilteredSearchResult>* res,
                  std::vector<Row>* br) {
  const auto n = static_cast<std::int64_t>(x.qids.size());
#pragma omp parallel for schedule(dynamic, 8) num_threads(x.c.sweep_threads)
  for (std::int64_t i = 0; i < n; ++i) {
    const auto iu = static_cast<std::size_t>(i);
    const double t0 = Now();
    (*res)[iu] = fse::PrefilterSearch(x.base, x.queries.Row(x.qids[iu]), x.c.k,
                                      x.f.mask.data());
    (*br)[iu].latency_us = (Now() - t0) * 1e6;
    (*br)[iu].effective_list = 0;
    (*br)[iu].dist_native = (*res)[iu].distance_computations;
    (*br)[iu].dist_exact = (*res)[iu].distance_computations;
  }
}

std::pair<std::size_t, std::size_t> RunPostfilter(
    const SweepContext& x, fse::HnswIndex* post, std::size_t b,
    std::vector<fse::FilteredSearchResult>* res, std::vector<Row>* br) {
  fse::PostfilterParams p = x.c.post;
  p.ef_search = b;
  post->SetEf(b);
  const auto n = static_cast<std::int64_t>(x.qids.size());
#pragma omp parallel for schedule(dynamic, 8) num_threads(x.c.sweep_threads)
  for (std::int64_t i = 0; i < n; ++i) {
    const auto iu = static_cast<std::size_t>(i);
    const double t0 = Now();
    (*res)[iu] = fse::PostfilterSearch(*post, x.queries.Row(x.qids[iu]), x.c.k,
                                       x.f.mask.data(), x.s, p);
    (*br)[iu].latency_us = (Now() - t0) * 1e6;
    (*br)[iu].effective_list = std::max(b, (*res)[iu].last_fetch);
    (*br)[iu].dist_native = (*res)[iu].distance_computations;
    (*br)[iu].dist_exact = (*res)[iu].distance_computations;
  }

  const auto perm = fse::SeededPermutation(x.qids.size(), x.c.replay_seed);
  const std::size_t m = std::min(x.c.replay_queries, x.qids.size());
  std::size_t bad = 0;
  for (std::size_t j = 0; j < m; ++j) {
    const auto iu = static_cast<std::size_t>(perm[j]);
    std::uint64_t replay = 0;
    for (std::uint32_t r = 1; r <= (*res)[iu].rounds; ++r) {
      replay += post->SearchAtCurrentEf(
                        x.queries.Row(x.qids[iu]),
                        fse::PostfilterFetchSize(x.c.k, x.s, p, r, x.base.rows))
                    .distance_computations;
    }
    bad += replay != (*res)[iu].distance_computations ? 1 : 0;
  }
  return {m, bad};
}

void RunAcorn(const SweepContext& x, fse::AcornIndex* acorn, std::size_t b,
              std::vector<fse::FilteredSearchResult>* res,
              std::vector<Row>* br) {
  omp_set_num_threads(1);
  acorn->SetEfSearch(static_cast<int>(b));
  for (std::size_t iu = 0; iu < x.qids.size(); ++iu) {
    const fse::AcornQueryResult a =
        acorn->SearchOne(x.queries.Row(x.qids[iu]), x.c.k, x.f.mask.data());
    (*res)[iu].ids = a.ids;
    (*res)[iu].distances = a.distances;
    (*res)[iu].distance_computations = a.distance_computations;
    (*br)[iu].latency_us = a.seconds * 1e6;
    (*br)[iu].effective_list = std::max(b, x.c.k);
    (*br)[iu].dist_native = a.distance_computations;
    (*br)[iu].n_scanned = a.entries_scanned;
    (*br)[iu].seed0 = a.level0_seed;
    (*br)[iu].dist_exact =
        fse::AcornExactDistanceComputations(a.distance_computations);
  }
}

void CollectBudget(const SweepContext& x, const fse::NeighborTable& gt,
                   std::size_t b,
                   const std::vector<fse::FilteredSearchResult>& res,
                   std::vector<Row>* br, std::vector<Row>* rows,
                   const std::string& dir) {
  fse::NeighborTable raw{x.qids.size(), x.c.k, std::vector<std::int64_t>(),
                         std::vector<float>()};
  raw.ids.reserve(x.qids.size() * x.c.k);
  raw.distances.reserve(x.qids.size() * x.c.k);
  for (std::size_t iu = 0; iu < x.qids.size(); ++iu) {
    Row& row = (*br)[iu];
    row.budget = b;
    row.query = x.qids[iu];
    row.filter_checks = res[iu].filter_checks;
    row.rounds = res[iu].rounds;
    row.last_fetch = res[iu].last_fetch;
    Score(res[iu], x.base, x.queries.Row(x.qids[iu]), x.f.mask, gt, x.qids[iu],
          x.c.k, &row);
    raw.ids.insert(raw.ids.end(), res[iu].ids.begin(), res[iu].ids.end());
    raw.distances.insert(raw.distances.end(), res[iu].distances.begin(),
                         res[iu].distances.end());
    rows->push_back(row);
  }
  std::string path = dir;
  path += "/raw_b";
  path += std::to_string(b);
  path += ".bin";
  fse::WriteNeighborTable(path, raw);
}

int Sweep(const Config& c, const std::string& method,
          const std::string& cond_name) {
  const double t_start = Now();
  const fse::FloatMatrix base = fse::ReadFvecs(c.base_path);
  const fse::FloatMatrix queries = fse::ReadFvecs(c.query_path);
  const std::uint64_t query_hash = HashMatrix(queries);
  const Conditions cs = LoadConditions(c, base);
  const fse::FilterCondition& f = FindCondition(cs, cond_name);
  const fse::GroundTruthIdentity identity{f.condition_id, f.mask_hash,
                                          query_hash};
  const std::string gt_path = c.generator_cache_dir + "/" + f.name + "_" +
                              fse::Hex64(f.condition_id) + ".gt";
  const fse::NeighborTable gt =
      fse::ReadConditionGroundTruth(gt_path, identity);
  if (gt.k != c.gt_k || gt.nq != queries.rows) {
    throw std::runtime_error("ground truth shape mismatch: " + gt_path);
  }
  const std::vector<std::size_t> qids = QueryIds(c, queries.rows);
  const auto nsub = static_cast<std::int64_t>(qids.size());
  const double s = f.achieved_selectivity;
  const std::string dir = c.output_dir + "/" + method + "/" + cond_name;
  fs::create_directories(dir);
  fse::JsonObject rec = RecordHeader(c, "sweep");
  rec.Str("method", method)
      .Str("condition", f.name)
      .Str("condition_id", fse::Hex64(f.condition_id))
      .Str("mask_hash", fse::Hex64(f.mask_hash))
      .Str("query_hash", fse::Hex64(query_hash))
      .Str("base_hash", fse::Hex64(cs.base_hash))
      .Num("s_achieved", s)
      .Int("threshold", static_cast<std::int64_t>(f.threshold))
      .Int("queries", nsub);

  std::vector<std::size_t> budgets = c.budgets;
  if (method == "prefilter") {
    budgets = {0};
  }
  std::vector<Row> rows;
  rows.reserve(budgets.size() * qids.size());
  std::size_t replay_mismatch = 0;
  std::size_t replay_checked = 0;

  std::unique_ptr<fse::HnswIndex> post;
  std::unique_ptr<fse::AcornIndex> acorn;
  if (method == "postfilter") {
    post = std::make_unique<fse::HnswIndex>(
        fse::HnswIndex::Load(PostIndexPath(c), base.dim));
    rec.Str("index_file_hash", fse::Hex64(HashFile(PostIndexPath(c))));
  } else if (method == "acorn") {
    acorn = std::make_unique<fse::AcornIndex>(fse::AcornIndex::Load(
        AcornIndexPath(c), std::vector<std::int32_t>(base.rows, 0)));
    const std::string ih = fse::Hex64(HashFile(AcornIndexPath(c)));
    if (!c.acorn_index_hash.empty() && ih != c.acorn_index_hash) {
      throw std::runtime_error("ACORN index hash " + ih + " != expected " +
                               c.acorn_index_hash);
    }
    if (acorn->Gamma() != c.acorn.gamma || acorn->M() != c.acorn.m ||
        acorn->MBeta() != c.acorn.m_beta) {
      throw std::runtime_error("ACORN index parameters differ from config");
    }
    rec.Str("index_file_hash", ih)
        .Int("acorn_gamma", acorn->Gamma())
        .Int("acorn_m", acorn->M())
        .Int("acorn_m_beta", acorn->MBeta());
  } else if (method != "prefilter") {
    throw std::invalid_argument("unknown method " + method);
  }

  const SweepContext ctx{c, base, queries, f, qids, s};
  const std::size_t first_fetch =
      fse::PostfilterFetchSize(c.k, s, c.post, 1, base.rows);
  std::vector<Row> inert_br;
  std::vector<fse::FilteredSearchResult> inert_res;
  std::string reused;
  for (const std::size_t b : budgets) {
    std::vector<Row> br(qids.size());
    std::vector<fse::FilteredSearchResult> res(qids.size());
    // Budgets up to the first fetch size run exactly the same post-filter
    // search, so we compute them once and reuse the result.
    const bool inert = c.post_reuse_inert && b <= first_fetch;
    if (method == "prefilter") {
      RunPrefilter(ctx, &res, &br);
    } else if (method == "postfilter" && inert && !inert_res.empty()) {
      res = inert_res;
      br = inert_br;
      reused += (reused.empty() ? "" : " ") + std::to_string(b);
    } else if (method == "postfilter") {
      const auto [checked, bad] = RunPostfilter(ctx, post.get(), b, &res, &br);
      replay_checked += checked;
      replay_mismatch += bad;
      if (inert) {
        inert_res = res;
        inert_br = br;
      }
    } else {
      RunAcorn(ctx, acorn.get(), b, &res, &br);
    }
    CollectBudget(ctx, gt, b, res, &br, &rows, dir);
    std::string msg = method;
    msg += " " + cond_name;
    msg += " b=" + std::to_string(b) + " done";
    Log(msg);
  }

  std::ofstream csv(dir + "/per_query.csv");
  csv << "query_id,condition,condition_id,correlation,s_requested,s_achieved,"
         "method,budget,effective_list,recall,dist_exact,dist_native,"
         "filter_checks,rounds,last_fetch,latency_us,n_valid,"
         "filter_violations,distance_mismatches,gt_tie_at_k,n_scanned,seed0\n";
  std::size_t violations = 0;
  std::size_t mismatches = 0;
  for (const Row& r : rows) {
    violations += r.violations;
    mismatches += r.mismatches;
    csv << r.query << "," << f.name << "," << fse::Hex64(f.condition_id) << ","
        << fse::CorrelationName(f.correlation) << "," << f.requested_selectivity
        << "," << s << "," << method << "," << r.budget << ","
        << r.effective_list << "," << r.recall << "," << r.dist_exact << ","
        << r.dist_native << "," << r.filter_checks << "," << r.rounds << ","
        << r.last_fetch << "," << r.latency_us << "," << r.n_valid << ","
        << r.violations << "," << r.mismatches << "," << (r.tie ? 1 : 0) << ","
        << r.n_scanned << "," << r.seed0 << "\n";
  }
  rec.Int("rows", static_cast<std::int64_t>(rows.size()))
      .Int("filter_violations", static_cast<std::int64_t>(violations))
      .Int("distance_mismatches", static_cast<std::int64_t>(mismatches))
      .Int("replay_checked", static_cast<std::int64_t>(replay_checked))
      .Str("post_reused_inert_budgets", reused)
      .Int("post_first_fetch", static_cast<std::int64_t>(first_fetch))
      .Int("replay_mismatches", static_cast<std::int64_t>(replay_mismatch))
      .Num("wall_seconds", Now() - t_start);
  std::ofstream(dir + "/sweep.json") << rec.Render() << "\n";
  Log(method + " " + cond_name + ": " + std::to_string(rows.size()) +
      " rows, violations " + std::to_string(violations) + ", mismatches " +
      std::to_string(mismatches) + ", replay mismatches " +
      std::to_string(replay_mismatch));
  return 0;
}

int Timing(const Config& c) {
  const fse::FloatMatrix base = fse::ReadFvecs(c.base_path);
  const fse::FloatMatrix queries = fse::ReadFvecs(c.query_path);
  const Conditions cs = LoadConditions(c, base);
  fse::HnswIndex post = fse::HnswIndex::Load(PostIndexPath(c), base.dim);
  fse::AcornIndex acorn = fse::AcornIndex::Load(
      AcornIndexPath(c), std::vector<std::int32_t>(base.rows, 0));
  omp_set_num_threads(1);
  const auto perm = fse::SeededPermutation(queries.rows, c.timing_seed);
  std::vector<std::size_t> qids;
  for (std::size_t i = 0; i < std::min(c.timing_queries, queries.rows); ++i) {
    qids.push_back(static_cast<std::size_t>(perm[i]));
  }
  std::sort(qids.begin(), qids.end());
  fs::create_directories(c.output_dir);
  std::ofstream csv(c.output_dir + "/timing.csv");
  csv << "condition,method,budget,query_id,median_us\n";
  auto time_one = [&](auto&& fn) {
    fn();
    std::vector<double> t;
    for (int r = 0; r < c.timing_repeats; ++r) {
      const double t0 = Now();
      fn();
      t.push_back((Now() - t0) * 1e6);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
  };
  const std::size_t k = c.k;
  for (const auto& f : cs.list) {
    for (const std::size_t q : qids) {
      csv << f.name << ",prefilter,0," << q << "," << time_one([&] {
        (void)fse::PrefilterSearch(base, queries.Row(q), k, f.mask.data());
      }) << "\n";
    }
    for (const std::size_t b : c.budgets) {
      fse::PostfilterParams p = c.post;
      p.ef_search = b;
      post.SetEf(b);
      acorn.SetEfSearch(static_cast<int>(b));
      for (const std::size_t q : qids) {
        csv << f.name << ",postfilter," << b << "," << q << ","
            << time_one([&] {
                 (void)fse::PostfilterSearch(post, queries.Row(q), k,
                                             f.mask.data(),
                                             f.achieved_selectivity, p);
               })
            << "\n";
        csv << f.name << ",acorn," << b << "," << q << "," << time_one([&] {
          (void)acorn.SearchOne(queries.Row(q), k, f.mask.data());
        }) << "\n";
      }
    }
    csv.flush();
    Log("timed " + f.name);
  }
  fse::JsonObject r = RecordHeader(c, "timing");
  r.Int("queries", static_cast<std::int64_t>(qids.size()))
      .Int("repeats", c.timing_repeats)
      .Int("seed", static_cast<std::int64_t>(c.timing_seed));
  std::ofstream(c.output_dir + "/timing.json") << r.Render() << "\n";
  return 0;
}

std::string LoadAvg() {
  std::ifstream in("/proc/loadavg");
  std::string line;
  std::getline(in, line);
  return line;
}

int ProcessThreads() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.starts_with("Threads:")) {
      return std::stoi(line.substr(8));
    }
  }
  return -1;
}

std::string PinToCore(int core) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    throw std::runtime_error("sched_setaffinity failed for core " +
                             std::to_string(core));
  }
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0) {
    throw std::runtime_error("sched_getaffinity failed");
  }
  std::string cpus;
  for (int i = 0; i < CPU_SETSIZE; ++i) {
    if (CPU_ISSET(i, &set)) {
      cpus += (cpus.empty() ? "" : " ") + std::to_string(i);
    }
  }
  if (cpus != std::to_string(core)) {
    throw std::runtime_error("affinity not applied: " + cpus);
  }
  return cpus;
}

bool SameResult(const fse::AcornQueryResult& a,
                const fse::AcornQueryResult& b) {
  return a.ids == b.ids && a.distances == b.distances &&
         a.distance_computations == b.distance_computations;
}

struct AuditCounts {
  std::size_t rows = 0;
  std::size_t repeat_mismatches = 0;
  std::size_t off_core = 0;
};

void AuditBlock(const Config& c, fse::AcornIndex* acorn,
                const fse::FloatMatrix& queries, const fse::FilterCondition& f,
                const fse::NeighborTable& gt, std::size_t b,
                const std::vector<std::size_t>& qids, std::size_t order,
                std::ofstream* csv, AuditCounts* n) {
  acorn->SetEfSearch(static_cast<int>(b));
  for (const std::size_t q : qids) {
    const fse::AcornQueryResult warm =
        acorn->SearchOne(queries.Row(q), c.k, f.mask.data());
    const double recall = fse::RecallAtK(warm.ids.data(), gt.Ids(q), c.k);
    for (int r = 1; r <= c.timing_repeats; ++r) {
      const double t0 = Now();
      const fse::AcornQueryResult a =
          acorn->SearchOne(queries.Row(q), c.k, f.mask.data());
      const double outer_us = (Now() - t0) * 1e6;
      const int cpu = sched_getcpu();
      const bool same = SameResult(a, warm);
      n->repeat_mismatches += same ? 0 : 1;
      n->off_core += cpu == c.timing_pin_core ? 0 : 1;
      ++n->rows;
      *csv << order << "," << f.name << "," << fse::Hex64(f.condition_id) << ","
           << fse::CorrelationName(f.correlation) << ","
           << f.achieved_selectivity << "," << b << "," << q << "," << r << ","
           << a.seconds * 1e6 << "," << outer_us << ","
           << a.distance_computations << ","
           << fse::AcornExactDistanceComputations(a.distance_computations)
           << "," << recall << "," << (same ? 1 : 0) << "," << cpu << "\n";
    }
  }
}

int TimingAcornAudit(const Config& c) {
  if (c.timing_pin_core < 0) {
    throw std::invalid_argument("acorn_audit requires timing.pin_core");
  }
  const std::string affinity = PinToCore(c.timing_pin_core);
  omp_set_num_threads(1);
  const std::string load_start = LoadAvg();
  const double t_start = Now();
  const fse::FloatMatrix base = fse::ReadFvecs(c.base_path);
  const fse::FloatMatrix queries = fse::ReadFvecs(c.query_path);
  const std::uint64_t query_hash = HashMatrix(queries);
  const Conditions cs = LoadConditions(c, base);
  std::vector<fse::NeighborTable> gts;
  for (const auto& f : cs.list) {
    const fse::GroundTruthIdentity id{f.condition_id, f.mask_hash, query_hash};
    gts.push_back(fse::ReadConditionGroundTruth(
        c.generator_cache_dir + "/" + f.name + "_" +
            fse::Hex64(f.condition_id) + ".gt",
        id));
  }
  const std::string index_hash = fse::Hex64(HashFile(AcornIndexPath(c)));
  fse::AcornIndex acorn = fse::AcornIndex::Load(
      AcornIndexPath(c), std::vector<std::int32_t>(base.rows, 0));

  const auto perm = fse::SeededPermutation(queries.rows, c.timing_seed);
  std::vector<std::size_t> qids;
  for (std::size_t i = 0; i < std::min(c.timing_queries, queries.rows); ++i) {
    qids.push_back(static_cast<std::size_t>(perm[i]));
  }
  std::sort(qids.begin(), qids.end());
  const std::size_t nb = c.budgets.size();
  const std::size_t nblocks = cs.list.size() * nb;
  const auto block_order = fse::SeededPermutation(nblocks, c.timing_seed);

  fs::create_directories(c.output_dir);
  std::ofstream csv(c.output_dir + "/timing_rows.csv");
  csv << "order,condition,condition_id,correlation,s_achieved,budget,"
         "query_id,repeat,latency_us,outer_us,dist_native,dist_exact,recall,"
         "same_as_warmup,cpu\n";
  std::ofstream blocks(c.output_dir + "/blocks.csv");
  blocks << "order,condition,budget,t_start_s,t_end_s,loadavg_start,"
            "threads\n";
  const int threads_start = ProcessThreads();
  AuditCounts n;
  for (std::size_t o = 0; o < nblocks; ++o) {
    const auto blk = static_cast<std::size_t>(block_order[o]);
    const std::size_t ci = blk / nb;
    const std::size_t b = c.budgets[blk % nb];
    const fse::FilterCondition& f = cs.list[ci];
    const double t0 = Now() - t_start;
    const std::string load = LoadAvg();
    AuditBlock(c, &acorn, queries, f, gts[ci], b, qids, o, &csv, &n);
    blocks << o << "," << f.name << "," << b << "," << t0 << ","
           << Now() - t_start << ",\"" << load << "\"," << ProcessThreads()
           << "\n";
    csv.flush();
    blocks.flush();
    Log("block " + std::to_string(o + 1) + "/" + std::to_string(nblocks) + " " +
        f.name + " b=" + std::to_string(b));
  }

  fse::JsonObject r = RecordHeader(c, "timing_acorn_audit");
  std::ostringstream conds;
  for (const auto& f : cs.list) {
    conds << f.name << ":" << fse::Hex64(f.condition_id) << ":"
          << fse::Hex64(f.mask_hash) << " ";
  }
  r.Int("queries", static_cast<std::int64_t>(qids.size()))
      .Int("repeats", c.timing_repeats)
      .Int("warmup_calls_per_query", 1)
      .Int("seed", static_cast<std::int64_t>(c.timing_seed))
      .Int("pin_core_requested", c.timing_pin_core)
      .Str("affinity_actual", affinity)
      .Int("omp_max_threads", omp_get_max_threads())
      .Int("process_threads_start", threads_start)
      .Int("process_threads_end", ProcessThreads())
      .Str("loadavg_start", load_start)
      .Str("loadavg_end", LoadAvg())
      .Num("wall_seconds", Now() - t_start)
      .Str("index_file_hash", index_hash)
      .Str("query_hash", fse::Hex64(query_hash))
      .Str("conditions", conds.str())
      .Int("blocks", static_cast<std::int64_t>(nblocks))
      .Int("rows", static_cast<std::int64_t>(n.rows))
      .Int("repeat_mismatches", static_cast<std::int64_t>(n.repeat_mismatches))
      .Int("off_core_rows", static_cast<std::int64_t>(n.off_core));
  std::ofstream(c.output_dir + "/timing.json") << r.Render() << "\n";
  Log("audit done: rows " + std::to_string(n.rows) + ", repeat mismatches " +
      std::to_string(n.repeat_mismatches) + ", off-core rows " +
      std::to_string(n.off_core));
  return n.repeat_mismatches == 0 && n.off_core == 0 ? 0 : 1;
}

}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: fse_matrix <config.yaml> "
                 "{build_post|build_acorn|audit_acorn|timing|sweep <method> "
                 "<condition>}\n";
    return 2;
  }
  try {
    const Config c = LoadConfig(argv[1]);
    const std::string stage = argv[2];
    if (stage == "build_post") {
      return BuildPost(c);
    }
    if (stage == "build_acorn") {
      return BuildAcorn(c);
    }
    if (stage == "audit_acorn") {
      return AuditAcorn(c);
    }
    if (stage == "timing") {
      return c.timing_mode == "acorn_audit" ? TimingAcornAudit(c) : Timing(c);
    }
    if (stage == "sweep" && argc == 5) {
      const std::string method = argv[3];
      const std::string cond = argv[4];
      return Sweep(c, method, cond);
    }
    throw std::invalid_argument("bad arguments");
  } catch (const std::exception& e) {
    std::cerr << "fse_matrix: error: " << e.what() << "\n";
    return 1;
  }
}
