// Phase 5 live feature extraction (docs/phase5_predictor.md §2).
//
// Usage: fse_features <config.yaml>
//
// One unfiltered probe per query on the shared post-filter hnswlib index
// (frozen: k = 20, ef = 20), its exact distance computations counted. From it:
// score concentration (d_1/d_10), LID (MLE, k = 20) and, per filter condition,
// the live local-density proxy (fraction of the 20 probe results passing the
// condition's mask). Centroid distance is ||q - base mean|| (one distance).
// Filter masks are regenerated from the hash-verified Phase 4 attributes.
// Ground truth is never read: no feature here can see the true filtered or
// unfiltered neighbours.

#include <omp.h>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/build_info.h"
#include "common/run_metadata.h"
#include "common/vecs_io.h"
#include "feature_extraction/live_features.h"
#include "filter_generator/filter_conditions.h"
#include "ground_truth/filtered_ground_truth.h"
#include "methods/postfilter/hnsw_index.h"

namespace fs = std::filesystem;

namespace {

struct Config {
  std::string path;
  std::string text;
  std::string experiment_name;
  std::string base_path;
  std::string query_path;
  double s_min = 0.0;
  double s_max = 0.0;
  int levels = 0;
  std::map<std::string, std::uint64_t> attribute_hashes;
  std::string generator_cache_dir;
  std::string index_path;
  std::string index_hash;
  std::size_t probe_k = 0;
  std::size_t probe_ef = 0;
  std::size_t concentration_k = 0;
  std::size_t lid_k = 0;
  int threads = 1;
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
  c.s_min = y["selectivity"]["min"].as<double>();
  c.s_max = y["selectivity"]["max"].as<double>();
  c.levels = y["selectivity"]["levels"].as<int>();
  for (const auto& kv : y["generator"]["attribute_hashes"]) {
    c.attribute_hashes[kv.first.as<std::string>()] =
        std::stoull(kv.second.as<std::string>(), nullptr, 16);
  }
  c.generator_cache_dir = y["generator"]["cache_dir"].as<std::string>();
  const YAML::Node p = y["probe"];
  c.index_path = p["index"].as<std::string>();
  c.index_hash = p["index_hash"].as<std::string>();
  c.probe_k = p["k"].as<std::size_t>();
  c.probe_ef = p["ef"].as<std::size_t>();
  c.threads = p["threads"].as<int>();
  c.concentration_k = y["features"]["score_concentration_k"].as<std::size_t>();
  c.lid_k = y["features"]["lid_k"].as<std::size_t>();
  if (c.concentration_k > c.probe_k || c.lid_k > c.probe_k) {
    throw std::invalid_argument("feature k must not exceed probe k");
  }
  c.output_dir = y["output_dir"].as<std::string>();
  return c;
}

double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
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

// The frozen Phase 4 conditions, regenerated from the hash-verified
// attributes (same procedure as fse_matrix).
std::vector<fse::FilterCondition> LoadConditions(const Config& c,
                                                 const fse::FloatMatrix& base,
                                                 std::uint64_t base_hash) {
  std::vector<fse::FilterCondition> out;
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
      out.push_back(fse::MakeFilterCondition(a, s, base_hash));
    }
  }
  return out;
}

struct Probe {
  std::vector<fse::HnswSearchResult> results;
  double seconds = 0.0;
};

Probe RunProbe(const Config& c, const fse::FloatMatrix& queries) {
  fse::HnswIndex index = fse::HnswIndex::Load(c.index_path, queries.dim);
  index.SetEf(c.probe_ef);
  Probe p;
  p.results.resize(queries.rows);
  const double t0 = Now();
#pragma omp parallel for num_threads(c.threads) schedule(dynamic, 64)
  for (std::size_t q = 0; q < queries.rows; ++q) {
    p.results[q] = index.SearchAtCurrentEf(queries.Row(q), c.probe_k);
  }
  p.seconds = Now() - t0;
  for (const auto& r : p.results) {
    if (r.labels.size() != c.probe_k) {
      throw std::runtime_error("probe returned fewer than k results");
    }
  }
  return p;
}

void WriteQueryFeatures(const Config& c, const fse::FloatMatrix& queries,
                        const std::vector<double>& centroid, const Probe& p) {
  std::ofstream csv(c.output_dir + "/query_features.csv");
  csv.precision(17);
  csv << "query_id,probe_dc,centroid_dc,centroid_dist,score_concentration,"
         "lid\n";
  for (std::size_t q = 0; q < queries.rows; ++q) {
    const auto& r = p.results[q];
    csv << q << "," << r.distance_computations << ",1,"
        << fse::CentroidDistance(queries.Row(q), centroid) << ","
        << fse::ScoreConcentration(r.dists, c.concentration_k) << ","
        << fse::LidMle(r.dists, c.lid_k) << "\n";
  }
}

void WriteProxy(const Config& c, const std::vector<fse::FilterCondition>& cs,
                const Probe& p) {
  std::ofstream csv(c.output_dir + "/proxy.csv");
  csv.precision(17);
  csv << "condition,query_id,rho_hat\n";
  for (const auto& f : cs) {
    for (std::size_t q = 0; q < p.results.size(); ++q) {
      csv << f.name << "," << q << ","
          << fse::PassingFraction(p.results[q].labels, f.mask) << "\n";
    }
  }
}

void WriteProbeTable(const Config& c, const Probe& p) {
  fse::NeighborTable t{p.results.size(), c.probe_k, {}, {}};
  for (const auto& r : p.results) {
    for (std::size_t i = 0; i < c.probe_k; ++i) {
      t.ids.push_back(static_cast<std::int64_t>(r.labels[i]));
      t.distances.push_back(r.dists[i]);
    }
  }
  fse::WriteNeighborTable(c.output_dir + "/probe_k20_ef20.bin", t);
}

int Run(const Config& c) {
  const fse::FloatMatrix base = fse::ReadFvecs(c.base_path);
  const fse::FloatMatrix queries = fse::ReadFvecs(c.query_path);
  const std::uint64_t base_hash =
      fse::Fnv1a64Bytes(base.data.data(), base.data.size() * sizeof(float));
  const std::uint64_t query_hash = fse::Fnv1a64Bytes(
      queries.data.data(), queries.data.size() * sizeof(float));
  const std::string index_hash = fse::Hex64(HashFile(c.index_path));
  if (index_hash != c.index_hash) {
    throw std::runtime_error("probe index hash " + index_hash +
                             " != expected " + c.index_hash);
  }
  const auto cs = LoadConditions(c, base, base_hash);
  const std::vector<double> centroid = fse::ComputeCentroid(base);
  const Probe p = RunProbe(c, queries);

  fs::create_directories(c.output_dir);
  WriteQueryFeatures(c, queries, centroid, p);
  WriteProxy(c, cs, p);
  WriteProbeTable(c, p);

  std::uint64_t probe_dc = 0;
  for (const auto& r : p.results) {
    probe_dc += r.distance_computations;
  }
  const fse::BuildInfo b = fse::GetBuildInfo();
  const fse::GitInfo g = fse::GetGitInfo(b.source_dir);
  std::ostringstream conds;
  for (const auto& f : cs) {
    conds << f.name << ":" << fse::Hex64(f.condition_id) << ":"
          << fse::Hex64(f.mask_hash) << " ";
  }
  fse::JsonObject r;
  r.Str("experiment_name", c.experiment_name)
      .Str("run_id", fse::MakeExperimentId(c.text))
      .Str("config_path", c.path)
      .Str("git_commit", g.commit)
      .Bool("git_dirty", g.dirty)
      .Str("source_fingerprint_sha256", fse::SourceFingerprint(b.source_dir))
      .Raw("build", fse::BuildInfoJson())
      .Raw("hardware", fse::HardwareJson())
      .Str("base_hash", fse::Hex64(base_hash))
      .Str("query_hash", fse::Hex64(query_hash))
      .Str("probe_index_hash", index_hash)
      .Int("probe_k", static_cast<std::int64_t>(c.probe_k))
      .Int("probe_ef", static_cast<std::int64_t>(c.probe_ef))
      .Int("queries", static_cast<std::int64_t>(queries.rows))
      .Num("probe_mean_dc",
           static_cast<double>(probe_dc) / static_cast<double>(queries.rows))
      .Num("probe_seconds", p.seconds)
      .Int("threads", c.threads)
      .Str("conditions", conds.str());
  std::ofstream(c.output_dir + "/features.json") << r.Render() << "\n";
  std::cout << "[fse_features] " << queries.rows << " queries, mean probe dc "
            << static_cast<double>(probe_dc) / static_cast<double>(queries.rows)
            << ", " << cs.size() << " conditions -> " << c.output_dir << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: fse_features <config.yaml>\n";
    return 2;
  }
  try {
    return Run(LoadConfig(argv[1]));
  } catch (const std::exception& e) {
    std::cerr << "fse_features: error: " << e.what() << "\n";
    return 1;
  }
}
