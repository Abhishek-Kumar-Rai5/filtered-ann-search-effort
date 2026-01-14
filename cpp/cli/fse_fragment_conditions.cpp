// Structural study: controlled predicate fragmentation conditions
// (docs/structural_design.md §3.3 items 3-4, §5 V-S5).
//
// Usage: fse_fragment_conditions <config.yaml>
//
// For every fragmentation level (clustered with k-means C clusters, or
// random) and every realization, builds the rank attribute with the frozen
// Phase 3 generator (unchanged): clustered levels use ONE k-means seed per C
// (fixed partition) and realizations differ only in the cluster-order seed;
// random realizations differ in the permutation seed. For every selectivity
// it writes the exact filtered ground truth into the verified store and a
// manifest recording every hash. Checks: exact T, nested levels within a
// realization, fixed partition per C, distinct realizations, and optional
// expected attribute hashes (Phase 4 reproduction).

#include <yaml-cpp/yaml.h>

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
#include "filter_generator/filter_conditions.h"
#include "ground_truth/filtered_ground_truth.h"

namespace fs = std::filesystem;

namespace {

struct Level {
  std::string name;
  bool random = false;
  fse::ClusteredParams clustered;  // num_clusters, kmeans seed/iter, cap
};

struct Config {
  std::string path;
  std::string text;
  std::string experiment_name;
  std::string base_path;
  std::string query_path;
  std::size_t gt_k = 0;
  std::vector<double> selectivities;
  std::vector<Level> levels;
  std::vector<std::uint64_t> order_seeds;              // clustered realizations
  std::vector<std::uint64_t> random_seeds;             // random realizations
  std::map<std::string, std::uint64_t> expected_hash;  // "<level>_r<i>"
  std::string cache_dir;
  std::string output_dir;
};

std::uint64_t ParseHex(const std::string& s) {
  return std::stoull(s, nullptr, 16);
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
  c.gt_k = y["ground_truth"]["k"].as<std::size_t>();
  c.selectivities = y["selectivities"].as<std::vector<double>>();
  const YAML::Node f = y["fragmentation"];
  const YAML::Node km = f["kmeans"];
  for (const auto& l : f["levels"]) {
    Level lv;
    lv.name = l["name"].as<std::string>();
    lv.random = l["random"] && l["random"].as<bool>();
    if (!lv.random) {
      lv.clustered.num_clusters = l["num_clusters"].as<std::size_t>();
      lv.clustered.kmeans_seed = l["kmeans_seed"].as<std::uint64_t>();
      lv.clustered.kmeans_iterations = km["iterations"].as<int>();
      lv.clustered.max_points_per_centroid =
          km["max_points_per_centroid"].as<int>();
    }
    c.levels.push_back(lv);
  }
  c.order_seeds = f["order_seeds"].as<std::vector<std::uint64_t>>();
  c.random_seeds = f["random_seeds"].as<std::vector<std::uint64_t>>();
  if (f["expected_attribute_hashes"]) {
    for (const auto& kv : f["expected_attribute_hashes"]) {
      c.expected_hash[kv.first.as<std::string>()] =
          ParseHex(kv.second.as<std::string>());
    }
  }
  c.cache_dir = y["paths"]["cache_dir"].as<std::string>();
  c.output_dir = y["paths"]["output_dir"].as<std::string>();
  return c;
}

std::string SelName(double s) {
  std::ostringstream o;
  o.setf(std::ios::fixed);
  o.precision(4);
  o << s;
  return o.str();
}

struct Entry {
  std::string name;
  std::string level;
  std::size_t realization = 0;
  std::uint64_t seed = 0;
  std::size_t num_clusters = 0;
  std::uint64_t kmeans_seed = 0;
  std::string attribute_path;
  std::uint64_t attribute_hash = 0;
  fse::FilterCondition f;
  std::size_t passing_clusters = 0;  // distinct clusters touched (designed)
  std::size_t partial_clusters = 0;  // clusters only partly passing
  std::string gt_path;
};

void CountClusters(const fse::RankAttribute& a, Entry* e) {
  if (a.cluster.empty()) {
    return;
  }
  std::map<std::int32_t, std::size_t> pass;
  std::map<std::int32_t, std::size_t> size;
  for (std::size_t i = 0; i < a.cluster.size(); ++i) {
    ++size[a.cluster[i]];
    if (e->f.mask[i] != 0) {
      ++pass[a.cluster[i]];
    }
  }
  e->passing_clusters = pass.size();
  for (const auto& [c, n] : pass) {
    e->partial_clusters += n < size[c] ? 1 : 0;
  }
}

void WriteManifest(const Config& c, const std::vector<Entry>& es,
                   std::uint64_t base_hash, std::uint64_t query_hash) {
  std::ofstream y(c.output_dir + "/manifest.yaml");
  y << "# Structural conditions manifest (fse_fragment_conditions).\n"
    << "base_hash: \"" << fse::Hex64(base_hash) << "\"\n"
    << "query_hash: \"" << fse::Hex64(query_hash) << "\"\n"
    << "gt_k: " << c.gt_k << "\n"
    << "conditions:\n";
  for (const Entry& e : es) {
    y << "  - {name: " << e.name << ", level: " << e.level
      << ", realization: " << e.realization << ", seed: " << e.seed
      << ", num_clusters: " << e.num_clusters
      << ", kmeans_seed: " << e.kmeans_seed
      << ", s: " << e.f.requested_selectivity
      << ", threshold: " << e.f.threshold << ", attribute: " << e.attribute_path
      << ", attribute_hash: \"" << fse::Hex64(e.attribute_hash)
      << "\", condition_id: \"" << fse::Hex64(e.f.condition_id)
      << "\", mask_hash: \"" << fse::Hex64(e.f.mask_hash)
      << "\", passing_clusters: " << e.passing_clusters
      << ", partial_clusters: " << e.partial_clusters << ", gt: " << e.gt_path
      << "}\n";
  }
  std::ofstream csv(c.output_dir + "/conditions.csv");
  csv << "name,level,realization,seed,num_clusters,kmeans_seed,s,threshold,"
         "achieved_s,attribute_hash,condition_id,mask_hash,passing_clusters,"
         "partial_clusters\n";
  for (const Entry& e : es) {
    csv << e.name << "," << e.level << "," << e.realization << "," << e.seed
        << "," << e.num_clusters << "," << e.kmeans_seed << ","
        << e.f.requested_selectivity << "," << e.f.threshold << ","
        << e.f.achieved_selectivity << "," << fse::Hex64(e.attribute_hash)
        << "," << fse::Hex64(e.f.condition_id) << ","
        << fse::Hex64(e.f.mask_hash) << "," << e.passing_clusters << ","
        << e.partial_clusters << "\n";
  }
}

struct Inputs {
  const Config& c;
  const fse::FloatMatrix& base;
  const fse::FloatMatrix& queries;
  std::uint64_t base_hash = 0;
  std::uint64_t query_hash = 0;
};

// Attribute of one realization of a level; enforces the fixed partition per
// C, distinct realizations and expected (Phase 4) hashes.
fse::RankAttribute BuildAttribute(const Inputs& in, const Level& lv,
                                  std::size_t r, std::uint64_t seed,
                                  std::vector<std::int32_t>* partition,
                                  std::set<std::uint64_t>* attr_hashes,
                                  std::vector<std::string>* checks) {
  fse::RankAttribute a;
  if (lv.random) {
    a = fse::RandomRankAttribute(in.base.rows, seed);
  } else {
    fse::ClusteredParams p = lv.clustered;
    p.order_seed = seed;
    a = fse::ClusteredRankAttribute(in.base, p);
    if (r == 0) {
      *partition = a.cluster;
    } else if (a.cluster != *partition) {
      throw std::runtime_error(lv.name +
                               ": k-means partition differs across "
                               "realizations");
    }
  }
  if (!attr_hashes->insert(a.content_hash).second) {
    throw std::runtime_error(lv.name + ": duplicate realization");
  }
  const std::string key = lv.name + "_r" + std::to_string(r);
  if (auto it = in.c.expected_hash.find(key); it != in.c.expected_hash.end()) {
    if (it->second != a.content_hash) {
      throw std::runtime_error(key + ": attribute hash " +
                               fse::Hex64(a.content_hash) + " != expected " +
                               fse::Hex64(it->second));
    }
    checks->push_back(key + " attribute hash matches expected " +
                      fse::Hex64(it->second));
  }
  return a;
}

// One condition: exact T, nesting against the previous level, cluster
// counts, verified ground truth.
Entry MakeEntry(const Inputs& in, const Level& lv, std::size_t r,
                std::uint64_t seed, const fse::RankAttribute& a,
                const std::string& attr_path, double s,
                std::vector<char>* prev) {
  Entry e;
  e.f = fse::MakeFilterCondition(a, s, in.base_hash);
  e.level = lv.name;
  e.realization = r;
  e.seed = seed;
  e.num_clusters = lv.random ? 0 : lv.clustered.num_clusters;
  e.kmeans_seed = lv.random ? 0 : lv.clustered.kmeans_seed;
  e.name = lv.name + "_r" + std::to_string(r) + "_s" + SelName(s);
  e.f.name = e.name;
  e.attribute_path = attr_path;
  e.attribute_hash = a.content_hash;
  std::size_t passing = 0;
  for (const char m : e.f.mask) {
    passing += m != 0 ? 1 : 0;
  }
  if (passing != e.f.threshold) {
    throw std::runtime_error(e.name + ": |P| != T");
  }
  for (std::size_t i = 0; i < prev->size(); ++i) {
    if ((*prev)[i] != 0 && e.f.mask[i] == 0) {
      throw std::runtime_error(e.name + ": levels not nested");
    }
  }
  *prev = e.f.mask;
  CountClusters(a, &e);
  const fse::NeighborTable gt =
      fse::FilteredGroundTruthMask(in.base, in.queries, e.f.mask, in.c.gt_k);
  e.gt_path = in.c.cache_dir + "/" + e.name + "_" +
              fse::Hex64(e.f.condition_id) + ".gt";
  fse::WriteConditionGroundTruth(
      e.gt_path, {e.f.condition_id, e.f.mask_hash, in.query_hash}, gt);
  std::cout << "[fse_fragment_conditions] " << e.name << ": T " << e.f.threshold
            << ", passing clusters " << e.passing_clusters << " (partial "
            << e.partial_clusters << "), attr " << fse::Hex64(a.content_hash)
            << "\n"
            << std::flush;
  return e;
}

int Run(const Config& c) {
  const fse::FloatMatrix base = fse::ReadFvecs(c.base_path);
  const fse::FloatMatrix queries = fse::ReadFvecs(c.query_path);
  const std::uint64_t base_hash =
      fse::Fnv1a64Bytes(base.data.data(), base.data.size() * sizeof(float));
  const std::uint64_t query_hash = fse::Fnv1a64Bytes(
      queries.data.data(), queries.data.size() * sizeof(float));
  const Inputs in{c, base, queries, base_hash, query_hash};
  fs::create_directories(c.cache_dir);
  fs::create_directories(c.output_dir);
  std::vector<Entry> entries;
  std::vector<std::string> checks;
  for (const Level& lv : c.levels) {
    const auto& seeds = lv.random ? c.random_seeds : c.order_seeds;
    std::vector<std::int32_t> partition;
    std::set<std::uint64_t> attr_hashes;
    for (std::size_t r = 0; r < seeds.size(); ++r) {
      const fse::RankAttribute a = BuildAttribute(
          in, lv, r, seeds[r], &partition, &attr_hashes, &checks);
      const std::string attr_path = c.cache_dir + "/attr_" + lv.name + "_r" +
                                    std::to_string(r) + "_" +
                                    fse::Hex64(a.content_hash) + ".bin";
      fse::WriteRankAttribute(attr_path, a);
      std::vector<char> prev;
      for (const double s : c.selectivities) {
        entries.push_back(
            MakeEntry(in, lv, r, seeds[r], a, attr_path, s, &prev));
      }
    }
  }
  WriteManifest(c, entries, base_hash, query_hash);
  const fse::BuildInfo b = fse::GetBuildInfo();
  const fse::GitInfo g = fse::GetGitInfo(b.source_dir);
  std::ostringstream ck;
  for (const auto& s : checks) {
    ck << s << "; ";
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
      .Int("conditions", static_cast<std::int64_t>(entries.size()))
      .Str("checks", ck.str());
  std::ofstream(c.output_dir + "/summary.json") << r.Render() << "\n";
  std::ofstream(c.output_dir + "/config.yaml") << c.text;
  std::cout << "[fse_fragment_conditions] " << entries.size()
            << " conditions -> " << c.output_dir << "/manifest.yaml\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: fse_fragment_conditions <config.yaml>\n";
    return 2;
  }
  try {
    return Run(LoadConfig(argv[1]));
  } catch (const std::exception& e) {
    std::cerr << "fse_fragment_conditions: error: " << e.what() << "\n";
    return 1;
  }
}
