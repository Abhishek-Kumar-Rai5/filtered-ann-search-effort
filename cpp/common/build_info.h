#pragma once

#include <string>

namespace fse {

struct BuildInfo {
  std::string compiler_id;
  std::string compiler_version;
  std::string build_type;
  std::string cxx_flags;
  long cxx_standard;
  int openmp_version;
  std::string hnswlib_commit;
  bool with_acorn;
  std::string acorn_commit;
  std::string acorn_opt_level;
  std::string source_dir;
};

BuildInfo GetBuildInfo();

}
