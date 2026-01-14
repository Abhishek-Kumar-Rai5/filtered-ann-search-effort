// For every query, which true targets can the search reach at all?
// Usage: fse_reach <config.yaml> <method>

#include <omp.h>
#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
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
#include "methods/ingraph/acorn_index.h"
#include "methods/postfilter/hnsw_index.h"
#include "reachability/reach.h"

namespace fs = std::filesystem;

namespace {

struct Graph {
  std::string method;
  std::string kind;
  std::string index_path;
  std::string index_hash;
  std::string sweep_dir;
  int gamma = 0;
  int m = 0;
  int m_beta = 0;
};

struct Config {
  std::string path;
  std::string text;
  std::string experiment_name;
  std::string base_path;
  std::string query_path;
  std::string manifest;
  std::size_t k = 10;
  std::vector<std::size_t> budgets;
  std::map<std::string, Graph> graphs;
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
  c.manifest = y["manifest"].as<std::string>();
  c.k = y["k"].as<std::size_t>();
  c.budgets = y["budgets"].as<std::vector<std::size_t>>();
  for (const auto& kv : y["graphs"]) {
    Graph g;
    g.method = kv.first.as<std::string>();
    const YAML::Node n = kv.second;
    g.kind = n["kind"].as<std::string>();
    g.index_path = n["index_path"].as<std::string>();
    g.index_hash = n["index_hash"].as<std::string>();
    g.sweep_dir = n["sweep_dir"].as<std::string>();
    if (g.kind == "acorn") {
      g.gamma = n["gamma"].as<int>();
      g.m = n["m"].as<int>();
      g.m_beta = n["m_beta"].as<int>();
    }
    c.graphs[g.method] = g;
  }
  c.output_dir = y["output_dir"].as<std::string>();
  return c;
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

struct Cond {
  fse::FilterCondition f;
  std::string gt_path;
};

std::vector<Cond> LoadManifest(const Config& c, std::uint64_t base_hash) {
  const YAML::Node m = YAML::LoadFile(c.manifest);
  if (m["base_hash"].as<std::string>() != fse::Hex64(base_hash)) {
    throw std::runtime_error("manifest base hash mismatch");
  }
  std::map<std::string, fse::RankAttribute> attrs;
  std::vector<Cond> out;
  for (const auto& e : m["conditions"]) {
    const auto path = e["attribute"].as<std::string>();
    if (attrs.find(path) == attrs.end()) {
      attrs.emplace(path,
                    fse::ReadRankAttribute(
                        path, std::stoull(e["attribute_hash"].as<std::string>(),
                                          nullptr, 16)));
    }
    Cond cd{fse::MakeFilterCondition(attrs.at(path), e["s"].as<double>(),
                                     base_hash),
            e["gt"].as<std::string>()};
    cd.f.name = e["name"].as<std::string>();
    if (fse::Hex64(cd.f.condition_id) != e["condition_id"].as<std::string>() ||
        fse::Hex64(cd.f.mask_hash) != e["mask_hash"].as<std::string>()) {
      throw std::runtime_error("manifest condition mismatch: " + cd.f.name);
    }
    out.push_back(std::move(cd));
  }
  return out;
}

std::uint32_t TargetBits(fse::ReachAnalyzer* ra, std::int64_t seed,
                         const fse::NeighborTable& gt, std::size_t q,
                         std::size_t k) {
  std::uint32_t bits = 0;
  for (std::size_t i = 0; i < k; ++i) {
    const std::int64_t t = gt.Ids(q)[i];
    if (t >= 0 && ra->Reachable(seed, t)) {
      bits |= 1U << i;
    }
  }
  return bits;
}

std::string StatsJson(const fse::ReachStats& s) {
  std::ostringstream o;
  o << "{\"nodes\": " << s.nodes << ", \"edges\": " << s.edges
    << ", \"sccs\": " << s.sccs << ", \"largest_scc\": " << s.largest_scc
    << ", \"wccs\": " << s.wccs << ", \"largest_wcc\": " << s.largest_wcc
    << "}";
  return o.str();
}

std::size_t SoundnessViolations(const Config& c, const Graph& gc,
                                const Cond& cd,
                                const std::vector<std::int64_t>& seeds,
                                fse::ReachAnalyzer* primary,
                                std::size_t* checked) {
  std::size_t bad = 0;
  for (const std::size_t b : c.budgets) {
    const std::string p =
        gc.sweep_dir + "/" + cd.f.name + "/raw_b" + std::to_string(b) + ".bin";
    const fse::NeighborTable raw = fse::ReadNeighborTable(p);
    if (raw.nq != seeds.size()) {
      throw std::runtime_error("raw table size mismatch: " + p);
    }
    for (std::size_t q = 0; q < raw.nq; ++q) {
      for (std::size_t i = 0; i < raw.k; ++i) {
        const std::int64_t id = raw.Ids(q)[i];
        if (id < 0 || cd.f.mask[static_cast<std::size_t>(id)] == 0) {
          continue;
        }
        ++*checked;
        bad += primary->Reachable(seeds[q], id) ? 0 : 1;
      }
    }
  }
  return bad;
}

struct Ctx {
  const Config& c;
  const Graph& gc;
  const fse::FloatMatrix& queries;
  std::uint64_t query_hash = 0;
  fse::AcornIndex* acorn = nullptr;
  const fse::CsrGraph& graph;
  fse::ReachAnalyzer* unfiltered = nullptr;
  std::string out_dir;
};

struct Analyzers {
  std::unique_ptr<fse::ReachAnalyzer> graph;
  std::unique_ptr<fse::ReachAnalyzer> sem;
  std::unique_ptr<fse::ReachAnalyzer> cap;
  fse::ReachAnalyzer* primary = nullptr;
};

Analyzers Prepare(const Ctx& x, const Cond& cd,
                  std::vector<std::int64_t>* seeds) {
  Analyzers a;
  if (x.acorn == nullptr) {
    seeds->assign(x.queries.rows, x.unfiltered->LargestSccMember());
    a.primary = x.unfiltered;
    return a;
  }
  for (std::size_t q = 0; q < x.queries.rows; ++q) {
    (*seeds)[q] = x.acorn->SearchOne(x.queries.Row(q), x.c.k, cd.f.mask.data())
                      .level0_seed;
  }
  const auto mk = [&](fse::Semantics s) {
    return std::make_unique<fse::ReachAnalyzer>(
        x.graph, cd.f.mask,
        fse::SemanticsParams{s, x.gc.gamma, x.gc.m, x.gc.m_beta});
  };
  a.graph = mk(fse::Semantics::kGraph);
  a.sem = mk(fse::Semantics::kSem);
  a.cap = mk(fse::Semantics::kCap);
  a.primary = a.sem.get();
  return a;
}

std::pair<std::size_t, std::size_t> WriteReachCsv(
    const Ctx& x, const Cond& cd, const fse::NeighborTable& gt,
    const std::vector<std::int64_t>& seeds, const Analyzers& a) {
  std::ofstream csv(x.out_dir + "/" + cd.f.name + "_reach.csv");
  csv << "query_id,seed0,seed_passes,n_targets,graph,sem,cap\n";
  std::set<std::int64_t> distinct;
  std::size_t seed_fail = 0;
  for (std::size_t q = 0; q < x.queries.rows; ++q) {
    std::size_t nt = 0;
    for (std::size_t i = 0; i < x.c.k; ++i) {
      nt += gt.Ids(q)[i] >= 0 ? 1 : 0;
    }
    const bool passes =
        seeds[q] >= 0 && cd.f.mask[static_cast<std::size_t>(seeds[q])] != 0;
    distinct.insert(seeds[q]);
    seed_fail += passes ? 0 : 1;
    csv << q << "," << seeds[q] << "," << (passes ? 1 : 0) << "," << nt << ",";
    if (x.acorn != nullptr) {
      csv << TargetBits(a.graph.get(), seeds[q], gt, q, x.c.k) << ","
          << TargetBits(a.sem.get(), seeds[q], gt, q, x.c.k) << ","
          << TargetBits(a.cap.get(), seeds[q], gt, q, x.c.k) << "\n";
    } else {
      csv << "-1," << TargetBits(a.primary, seeds[q], gt, q, x.c.k) << ",-1\n";
    }
  }
  return {distinct.size(), seed_fail};
}

std::string ProcessCondition(const Ctx& x, const Cond& cd, std::size_t* bad,
                             std::size_t* checked) {
  const fse::NeighborTable gt = fse::ReadConditionGroundTruth(
      cd.gt_path, {cd.f.condition_id, cd.f.mask_hash, x.query_hash});
  std::vector<std::int64_t> seeds(x.queries.rows, -1);
  const Analyzers a = Prepare(x, cd, &seeds);
  const auto [distinct, seed_fail] = WriteReachCsv(x, cd, gt, seeds, a);
  *checked = 0;
  *bad = x.gc.sweep_dir.empty()
             ? 0
             : SoundnessViolations(x.c, x.gc, cd, seeds, a.primary, checked);
  std::ostringstream j;
  j << "\"" << cd.f.name << R"(": {"distinct_seeds": )" << distinct
    << R"(, "seeds_failing_filter": )" << seed_fail
    << R"(, "soundness_checked": )" << *checked
    << R"(, "soundness_violations": )" << *bad;
  if (x.acorn != nullptr) {
    j << R"(, "graph": )" << StatsJson(a.graph->Stats()) << R"(, "sem": )"
      << StatsJson(a.sem->Stats()) << R"(, "cap": )"
      << StatsJson(a.cap->Stats());
  } else {
    j << R"(, "unfiltered": )" << StatsJson(a.primary->Stats());
  }
  j << "}";
  std::cout << "[fse_reach] " << x.gc.method << " " << cd.f.name
            << ": distinct seeds " << distinct << ", soundness " << *bad << "/"
            << *checked << "\n"
            << std::flush;
  return j.str();
}

int Run(const Config& c, const std::string& method) {
  const auto git = c.graphs.find(method);
  if (git == c.graphs.end()) {
    throw std::invalid_argument("no graph configured for " + method);
  }
  const Graph& gc = git->second;
  const fse::FloatMatrix base = fse::ReadFvecs(c.base_path);
  const fse::FloatMatrix queries = fse::ReadFvecs(c.query_path);
  const std::uint64_t base_hash =
      fse::Fnv1a64Bytes(base.data.data(), base.data.size() * sizeof(float));
  const std::uint64_t query_hash = fse::Fnv1a64Bytes(
      queries.data.data(), queries.data.size() * sizeof(float));
  const auto conds = LoadManifest(c, base_hash);
  const std::string ih = fse::Hex64(HashFile(gc.index_path));
  if (ih != gc.index_hash) {
    throw std::runtime_error("index hash " + ih + " != " + gc.index_hash);
  }
  std::unique_ptr<fse::AcornIndex> acorn;
  fse::CsrGraph graph;
  if (gc.kind == "acorn") {
    acorn = std::make_unique<fse::AcornIndex>(fse::AcornIndex::Load(
        gc.index_path, std::vector<std::int32_t>(base.rows, 0)));
    if (acorn->Gamma() != gc.gamma || acorn->M() != gc.m ||
        acorn->MBeta() != gc.m_beta) {
      throw std::runtime_error("ACORN parameters differ from config");
    }
    graph = acorn->Level0Graph();
    acorn->SetEfSearch(10);
  } else if (gc.kind == "hnsw") {
    graph = fse::HnswIndex::Load(gc.index_path, base.dim).Level0Graph();
  } else {
    throw std::invalid_argument("unknown graph kind " + gc.kind);
  }
  omp_set_num_threads(1);
  const std::vector<char> no_mask;
  std::unique_ptr<fse::ReachAnalyzer> unfiltered;
  if (gc.kind == "hnsw") {
    unfiltered = std::make_unique<fse::ReachAnalyzer>(
        graph, no_mask, fse::SemanticsParams{fse::Semantics::kUnfiltered});
  }
  const Ctx x{
      c,           gc,    queries,          query_hash,
      acorn.get(), graph, unfiltered.get(), c.output_dir + "/" + method};
  fs::create_directories(x.out_dir);
  std::ostringstream conds_json;
  std::size_t total_bad = 0;
  std::size_t total_checked = 0;
  for (const Cond& cd : conds) {
    std::size_t bad = 0;
    std::size_t checked = 0;
    conds_json << (conds_json.tellp() > 0 ? ", " : "")
               << ProcessCondition(x, cd, &bad, &checked);
    total_bad += bad;
    total_checked += checked;
  }
  const std::string& out_dir = x.out_dir;
  const fse::BuildInfo b = fse::GetBuildInfo();
  const fse::GitInfo g = fse::GetGitInfo(b.source_dir);
  fse::JsonObject r;
  r.Str("experiment_name", c.experiment_name)
      .Str("method", method)
      .Str("run_id", fse::MakeExperimentId(c.text + method))
      .Str("config_path", c.path)
      .Str("git_commit", g.commit)
      .Bool("git_dirty", g.dirty)
      .Str("source_fingerprint_sha256", fse::SourceFingerprint(b.source_dir))
      .Raw("build", fse::BuildInfoJson())
      .Raw("hardware", fse::HardwareJson())
      .Str("index_file_hash", ih)
      .Str("query_hash", fse::Hex64(query_hash))
      .Int("graph_nodes", static_cast<std::int64_t>(graph.Nodes()))
      .Int("graph_edges", static_cast<std::int64_t>(graph.ids.size()))
      .Int("soundness_checked", static_cast<std::int64_t>(total_checked))
      .Int("soundness_violations", static_cast<std::int64_t>(total_bad))
      .Raw("conditions", "{" + conds_json.str() + "}");
  std::ofstream(out_dir + "/reach.json") << r.Render() << "\n";
  return total_bad == 0 ? 0 : 1;
}

}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: fse_reach <config.yaml> <method>\n";
    return 2;
  }
  try {
    return Run(LoadConfig(argv[1]), argv[2]);
  } catch (const std::exception& e) {
    std::cerr << "fse_reach: error: " << e.what() << "\n";
    return 1;
  }
}
