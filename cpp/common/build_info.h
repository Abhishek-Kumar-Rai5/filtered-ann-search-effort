#pragma once

#include <string>

namespace fse {

// Compile-time build metadata, logged into every run's metadata file
// (CLAUDE.md reproducibility rules).
struct BuildInfo {
  std::string compiler_id;
  std::string compiler_version;
  std::string build_type;
  std::string cxx_flags;
  long cxx_standard;
  int openmp_version;
  std::string hnswlib_commit;   // pinned submodule commit, set at configure
  bool with_acorn;              // whether ACORN was built and linked
  std::string acorn_commit;     // pinned submodule commit, set at configure
  std::string acorn_opt_level;  // ACORN's FAISS_OPT_LEVEL
  std::string source_dir;       // repo root, for runtime git commit lookup
};

BuildInfo GetBuildInfo();

}  // namespace fse
