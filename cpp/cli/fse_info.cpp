#include <iostream>

#include "common/build_info.h"

int main() {
  const fse::BuildInfo info = fse::GetBuildInfo();
  std::cout << "compiler: " << info.compiler_id << " " << info.compiler_version
            << "\n"
            << "build_type: " << info.build_type << "\n"
            << "cxx_flags: \"" << info.cxx_flags << "\"\n"
            << "cxx_standard: " << info.cxx_standard << "\n"
            << "openmp_version: " << info.openmp_version << "\n"
            << "hnswlib_commit: " << info.hnswlib_commit << "\n"
            << "with_acorn: " << (info.with_acorn ? "true" : "false") << "\n"
            << "acorn_commit: " << info.acorn_commit << "\n"
            << "acorn_opt_level: " << info.acorn_opt_level << "\n";
  return 0;
}
