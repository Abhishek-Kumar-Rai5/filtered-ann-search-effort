#include <omp.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/build_info.h"
#include "common/run_metadata.h"
#include "common/vecs_io.h"
#include "filter_generator/uniform_attributes.h"
#include "ground_truth/filtered_ground_truth.h"
#include "methods/filtered_result.h"
#include "methods/postfilter/hnsw_index.h"
#include "methods/postfilter/postfilter.h"
#include "methods/prefilter/prefilter.h"
#include "metrics/recall.h"

namespace fs = std::filesystem;

namespace {

struct Condition {
  std::string name;
  std::int32_t values = 0;
  std::uint64_t base_seed = 0;
  std::uint64_t query_seed = 0;
};

struct Config {
  std::string path;
  std::string text;
  std::string experiment_name;
  std::string base_path;
  std::string query_path;
  std::size_t max_base = 0;
  std::size_t max_queries = 0;
  std::vector<Condition> conditions;
  std::size_t k = 0;
  fse::HnswParams hnsw;
  int build_threads = 1;
  std::vector<std::size_t> ef_grid;
  double overfetch_factor = 1.0;
  double growth_factor = 2.0;
  std::uint32_t max_rounds = 1;
  int parallel_threads = 1;
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
  c.max_base = y["dataset"]["max_base"].as<std::size_t>();
  c.max_queries = y["dataset"]["max_queries"].as<std::size_t>();
  for (const auto& n : y["conditions"]) {
    c.conditions.push_back({n["name"].as<std::string>(),
                            n["values"].as<std::int32_t>(),
                            n["base_seed"].as<std::uint64_t>(),
                            n["query_seed"].as<std::uint64_t>()});
  }
  c.k = y["search"]["k"].as<std::size_t>();
  c.hnsw.m = y["postfilter"]["hnsw_m"].as<std::size_t>();
  c.hnsw.ef_construction =
      y["postfilter"]["hnsw_ef_construction"].as<std::size_t>();
  c.hnsw.seed = y["postfilter"]["hnsw_seed"].as<std::uint64_t>();
  c.build_threads = y["postfilter"]["build_threads"].as<int>();
  c.ef_grid = y["postfilter"]["ef_search"].as<std::vector<std::size_t>>();
  c.overfetch_factor = y["postfilter"]["overfetch_factor"].as<double>();
  c.growth_factor = y["postfilter"]["growth_factor"].as<double>();
  c.max_rounds = y["postfilter"]["max_rounds"].as<std::uint32_t>();
  c.parallel_threads = y["checks"]["parallel_threads"].as<int>();
  c.output_dir = y["paths"]["output_dir"].as<std::string>();
  return c;
}

void Log(const std::string& msg) {
  std::cout << "[fse_baselines_check] " << msg << '\n' << std::flush;
}

struct Tally {
  std::size_t queries = 0;
  double recall_sum = 0.0;
  double recall_min = 1.0;
  std::uint64_t ndis_sum = 0;
  std::uint64_t filter_checks_sum = 0;
  std::uint64_t rounds_sum = 0;
  std::uint32_t rounds_max = 0;
  std::size_t padded = 0;
  std::size_t filter_violations = 0;
  std::size_t distance_mismatches = 0;
  std::size_t unsorted = 0;
  std::size_t accounting_errors = 0;
  std::size_t gt_id_mismatch = 0;
  std::size_t gt_tie_only = 0;
  std::size_t parallel_mismatch = 0;

  [[nodiscard]] bool Clean() const {
    return filter_violations == 0 && distance_mismatches == 0 &&
           unsorted == 0 && accounting_errors == 0 && gt_id_mismatch == 0 &&
           parallel_mismatch == 0;
  }
  [[nodiscard]] std::string Json(const std::string& method,
                                 const std::string& setting) const {
    const auto d = static_cast<double>(queries);
    return fse::JsonObject()
        .Str("method", method)
        .Str("setting", setting)
        .Int("queries", static_cast<std::int64_t>(queries))
        .Num("mean_recall", recall_sum / d)
        .Num("min_recall", recall_min)
        .Num("mean_distance_computations", static_cast<double>(ndis_sum) / d)
        .Num("mean_filter_checks", static_cast<double>(filter_checks_sum) / d)
        .Num("mean_rounds", static_cast<double>(rounds_sum) / d)
        .Int("max_rounds_used", rounds_max)
        .Int("padded_slots", static_cast<std::int64_t>(padded))
        .Int("filter_violations", static_cast<std::int64_t>(filter_violations))
        .Int("distance_mismatches",
             static_cast<std::int64_t>(distance_mismatches))
        .Int("unsorted_results", static_cast<std::int64_t>(unsorted))
        .Int("accounting_errors", static_cast<std::int64_t>(accounting_errors))
        .Int("gt_id_mismatches", static_cast<std::int64_t>(gt_id_mismatch))
        .Int("gt_differ_only_by_ties", static_cast<std::int64_t>(gt_tie_only))
        .Int("parallel_mismatches",
             static_cast<std::int64_t>(parallel_mismatch))
        .Bool("all_checks_pass", Clean())
        .Render();
  }
};

void CheckResult(const fse::FilteredSearchResult& r, const float* query,
                 const fse::FloatMatrix& base, const char* mask,
                 const std::int64_t* gt_ids, std::size_t k, Tally* t) {
  ++t->queries;
  const double rec = fse::RecallAtK(r.ids.data(), gt_ids, k);
  t->recall_sum += rec;
  t->recall_min = std::min(t->recall_min, rec);
  t->ndis_sum += r.distance_computations;
  t->filter_checks_sum += r.filter_checks;
  t->rounds_sum += r.rounds;
  t->rounds_max = std::max(t->rounds_max, r.rounds);
  for (std::size_t j = 0; j < k; ++j) {
    const std::int64_t id = r.ids[j];
    if (id < 0) {
      ++t->padded;
      continue;
    }
    t->filter_violations += mask[id] == 0 ? 1 : 0;
    const float d = fse::SquaredL2(query, base.Row(id), base.dim);
    const double err = std::abs(static_cast<double>(d) - r.distances[j]);
    t->distance_mismatches +=
        err > 1e-4 * std::max(1.0, static_cast<double>(d)) ? 1 : 0;
    if (j > 0 && r.ids[j - 1] >= 0 && r.distances[j - 1] > r.distances[j]) {
      ++t->unsorted;
    }
  }
}

void CompareToGt(const fse::FilteredSearchResult& r,
                 const fse::NeighborTable& gt, std::size_t q, std::size_t k,
                 Tally* t) {
  bool same = true;
  bool tie_only = true;
  for (std::size_t j = 0; j < k; ++j) {
    same = same && r.ids[j] == gt.Ids(q)[j];
    tie_only = tie_only && r.distances[j] == gt.Distances(q)[j];
  }
  if (!same) {
    ++(tie_only ? t->gt_tie_only : t->gt_id_mismatch);
  }
}

struct ConditionData {
  fse::NeighborTable gt;
  std::vector<std::vector<char>> masks;
  std::vector<double> selectivity;
  std::vector<std::uint64_t> passing;
  double mean_selectivity = 0.0;
};

ConditionData MakeConditionData(const Config& c, const Condition& cond,
                                const fse::FloatMatrix& base,
                                const fse::FloatMatrix& queries) {
  const std::size_t n = base.rows;
  const std::size_t nq = queries.rows;
  const auto base_attr =
      fse::UniformIntAttributes(n, 1, cond.values, cond.base_seed);
  const auto query_attr =
      fse::UniformIntAttributes(nq, 1, cond.values, cond.query_seed);
  ConditionData d;
  d.gt =
      fse::FilteredGroundTruthEquals(base, queries, base_attr, query_attr, c.k);
  d.masks.resize(nq);
  d.selectivity.resize(nq);
  d.passing.resize(nq);
  for (std::size_t q = 0; q < nq; ++q) {
    d.masks[q].resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      d.masks[q][i] = static_cast<char>(base_attr[i] == query_attr[q]);
      d.passing[q] += d.masks[q][i] != 0 ? 1 : 0;
    }
    d.selectivity[q] =
        static_cast<double>(d.passing[q]) / static_cast<double>(n);
    d.mean_selectivity += d.selectivity[q] / static_cast<double>(nq);
  }
  return d;
}

void WritePerQuery(std::ofstream* out, const std::string& cond,
                   const std::string& method, const std::string& setting,
                   std::size_t q, double selectivity, double recall,
                   const fse::FilteredSearchResult& r) {
  *out << cond << "," << method << "," << setting << "," << q << ","
       << selectivity << "," << recall << "," << r.distance_computations << ","
       << r.filter_checks << "," << r.rounds << "\n";
}

Tally CheckPrefilter(const Config& c, const Condition& cond,
                     const ConditionData& d, const fse::FloatMatrix& base,
                     const fse::FloatMatrix& queries,
                     std::ofstream* per_query) {
  const std::size_t k = c.k;
  Tally t;
  for (std::size_t q = 0; q < queries.rows; ++q) {
    const auto r =
        fse::PrefilterSearch(base, queries.Row(q), k, d.masks[q].data());
    CheckResult(r, queries.Row(q), base, d.masks[q].data(), d.gt.Ids(q), k, &t);
    CompareToGt(r, d.gt, q, k, &t);

    const bool exact_effort =
        r.distance_computations == d.passing[q] && r.filter_checks == base.rows;
    t.accounting_errors += exact_effort ? 0 : 1;
    WritePerQuery(per_query, cond.name, "prefilter", "exact", q,
                  d.selectivity[q],
                  fse::RecallAtK(r.ids.data(), d.gt.Ids(q), k), r);
  }

  t.gt_id_mismatch += t.gt_tie_only;
  return t;
}

struct Setting {
  std::string name;
  fse::PostfilterParams params;
  bool exhaustive = false;
};

Tally CheckPostfilterSetting(const Config& c, const Condition& cond,
                             const ConditionData& d, const Setting& st,
                             const fse::FloatMatrix& base,
                             const fse::FloatMatrix& queries,
                             const fse::HnswIndex& index,
                             std::ofstream* per_query) {
  const std::size_t n = base.rows;
  const std::size_t nq = queries.rows;
  const std::size_t k = c.k;
  Tally t;
  std::vector<fse::FilteredSearchResult> serial(nq);
  for (std::size_t q = 0; q < nq; ++q) {
    serial[q] =
        fse::PostfilterSearch(index, queries.Row(q), k, d.masks[q].data(),
                              d.selectivity[q], st.params);
    const auto& r = serial[q];
    CheckResult(r, queries.Row(q), base, d.masks[q].data(), d.gt.Ids(q), k, &t);
    if (st.exhaustive) {
      CompareToGt(r, d.gt, q, k, &t);
    }

    std::uint64_t replay = 0;
    for (std::uint32_t round = 1; round <= r.rounds; ++round) {
      const std::size_t fetch =
          fse::PostfilterFetchSize(k, d.selectivity[q], st.params, round, n);
      replay +=
          index.SearchAtCurrentEf(queries.Row(q), fetch).distance_computations;
    }
    t.accounting_errors += replay != r.distance_computations ? 1 : 0;
    WritePerQuery(per_query, cond.name, "postfilter", st.name, q,
                  d.selectivity[q],
                  fse::RecallAtK(r.ids.data(), d.gt.Ids(q), k), r);
  }
  std::vector<fse::FilteredSearchResult> parallel(nq);
  const auto nq_signed = static_cast<std::int64_t>(nq);
#pragma omp parallel for schedule(dynamic, 4) num_threads(c.parallel_threads)
  for (std::int64_t q = 0; q < nq_signed; ++q) {
    const auto qu = static_cast<std::size_t>(q);
    parallel[qu] =
        fse::PostfilterSearch(index, queries.Row(qu), k, d.masks[qu].data(),
                              d.selectivity[qu], st.params);
  }
  for (std::size_t q = 0; q < nq; ++q) {
    const bool same =
        parallel[q].ids == serial[q].ids &&
        parallel[q].distance_computations == serial[q].distance_computations &&
        parallel[q].filter_checks == serial[q].filter_checks;
    t.parallel_mismatch += same ? 0 : 1;
  }
  return t;
}

std::string JoinJson(const std::vector<std::string>& items) {
  std::string out = "[";
  for (std::size_t i = 0; i < items.size(); ++i) {
    out += (i > 0 ? ", " : "") + items[i];
  }
  return out + "]";
}

std::string RunCondition(const Config& c, const Condition& cond,
                         const fse::FloatMatrix& base,
                         const fse::FloatMatrix& queries, fse::HnswIndex& index,
                         bool* all_clean, std::ofstream* per_query) {
  const ConditionData d = MakeConditionData(c, cond, base, queries);
  std::vector<std::string> rows;

  const Tally pre = CheckPrefilter(c, cond, d, base, queries, per_query);
  *all_clean = *all_clean && pre.Clean();
  rows.push_back(pre.Json("prefilter", "exact"));
  Log(cond.name + " prefilter: " + (pre.Clean() ? "all checks pass" : "FAIL"));

  std::vector<Setting> settings;
  settings.reserve(c.ef_grid.size() + 1);
  for (const std::size_t ef : c.ef_grid) {
    settings.push_back({"ef" + std::to_string(ef),
                        {ef, c.overfetch_factor, c.growth_factor, c.max_rounds},
                        false});
  }
  settings.push_back(
      {"exhaustive",
       {base.rows, static_cast<double>(base.rows), c.growth_factor, 1},
       true});
  for (const Setting& st : settings) {
    index.SetEf(st.params.ef_search);
    const Tally t =
        CheckPostfilterSetting(c, cond, d, st, base, queries, index, per_query);

    const bool ok = t.Clean() && (!st.exhaustive || t.recall_min == 1.0);
    *all_clean = *all_clean && ok;
    rows.push_back(t.Json("postfilter", st.name));
    const auto dnq = static_cast<double>(queries.rows);
    std::ostringstream msg;
    msg << cond.name << " postfilter " << st.name
        << ": recall=" << t.recall_sum / dnq
        << " ndis=" << static_cast<double>(t.ndis_sum) / dnq
        << " rounds(max)=" << t.rounds_max << " "
        << (ok ? "all checks pass" : "FAIL");
    Log(msg.str());
  }
  return fse::JsonObject()
      .Str("condition", cond.name)
      .Int("attribute_values", cond.values)
      .Int("base_seed", static_cast<std::int64_t>(cond.base_seed))
      .Int("query_seed", static_cast<std::int64_t>(cond.query_seed))
      .Num("mean_selectivity", d.mean_selectivity)
      .Raw("results", JoinJson(rows))
      .Render();
}

}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: fse_baselines_check <config.yaml>\n";
    return 2;
  }
  try {
    const Config c = LoadConfig(argv[1]);
    const std::string dir = c.output_dir + "/" + c.experiment_name;
    fs::create_directories(dir);
    std::ofstream(dir + "/config.yaml") << c.text;
    const fse::BuildInfo binfo = fse::GetBuildInfo();
    const fse::GitInfo git = fse::GetGitInfo(binfo.source_dir);

    const fse::FloatMatrix base = fse::ReadFvecs(
        c.base_path, c.max_base == 0 ? fse::kAllRows : c.max_base);
    const fse::FloatMatrix queries = fse::ReadFvecs(
        c.query_path, c.max_queries == 0 ? fse::kAllRows : c.max_queries);
    Log("base " + std::to_string(base.rows) + " x " + std::to_string(base.dim) +
        ", queries " + std::to_string(queries.rows));

    fse::HnswIndex index(base.dim, base.rows, c.hnsw);
    index.Add(base, 0, c.build_threads);

    std::ofstream per_query(dir + "/per_query.csv");
    per_query << "condition,method,setting,query,selectivity,recall,"
                 "distance_computations,filter_checks,rounds\n";
    bool all_clean = true;
    std::vector<std::string> cond_json;
    cond_json.reserve(c.conditions.size());
    for (const Condition& cond : c.conditions) {
      cond_json.push_back(
          RunCondition(c, cond, base, queries, index, &all_clean, &per_query));
    }
    const std::string conds = JoinJson(cond_json);

    fse::JsonObject r;
    r.Str("experiment_name", c.experiment_name)
        .Str("run_id", fse::MakeExperimentId(c.text))
        .Str("config_path", c.path)
        .Str("git_commit", git.commit)
        .Bool("git_dirty", git.dirty)
        .Str("source_fingerprint_sha256",
             fse::SourceFingerprint(binfo.source_dir))
        .Raw("build", fse::BuildInfoJson())
        .Raw("hardware", fse::HardwareJson())
        .Int("k", static_cast<std::int64_t>(c.k))
        .Raw("conditions", conds)
        .Bool("all_checks_pass", all_clean);
    std::ofstream(dir + "/summary.json") << r.Render() << "\n";
    Log(std::string("ALL CHECKS ") + (all_clean ? "PASS" : "FAIL") + " -> " +
        dir);
    return all_clean ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "fse_baselines_check: error: " << e.what() << "\n";
    return 1;
  }
}
