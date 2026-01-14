#include "common/build_info.h"

#include <omp.h>

namespace fse {

BuildInfo GetBuildInfo() {
  return BuildInfo{
      .compiler_id = FSE_COMPILER_ID,
      .compiler_version = FSE_COMPILER_VERSION,
      .build_type = FSE_BUILD_TYPE,
      .cxx_flags = FSE_CXX_FLAGS,
      .cxx_standard = __cplusplus,
      .openmp_version = _OPENMP,
      .hnswlib_commit = FSE_HNSWLIB_COMMIT,
#ifdef FSE_WITH_ACORN
      .with_acorn = true,
#else
      .with_acorn = false,
#endif
      .acorn_commit = FSE_ACORN_COMMIT,
      .acorn_opt_level = FSE_ACORN_OPT_LEVEL,
      .source_dir = FSE_SOURCE_DIR,
  };
}

}
