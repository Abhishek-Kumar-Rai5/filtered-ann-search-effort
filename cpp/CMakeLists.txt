add_library(fse_core
  common/build_info.cpp
  common/run_metadata.cpp
  common/vecs_io.cpp
  filter_generator/uniform_attributes.cpp
  metrics/local_density.cpp
  metrics/structure_stats.cpp
  ground_truth/filtered_ground_truth.cpp
  metrics/recall.cpp
  methods/prefilter/prefilter.cpp
  methods/postfilter/hnsw_index.cpp
  methods/postfilter/postfilter.cpp
  feature_extraction/live_features.cpp
  reachability/acorn_semantics.cpp
  reachability/reach.cpp
)
if(FSE_WITH_ACORN)
  target_sources(fse_core PRIVATE methods/ingraph/acorn_index.cpp
    filter_generator/filter_conditions.cpp)
endif()
target_include_directories(fse_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fse_core PUBLIC hnswlib OpenMP::OpenMP_CXX
  PRIVATE yaml-cpp::yaml-cpp)
if(FSE_WITH_ACORN)
  target_link_libraries(fse_core PUBLIC acorn::faiss)
  target_compile_definitions(fse_core PUBLIC FSE_WITH_ACORN=1)
endif()
target_compile_options(fse_core PRIVATE -Wall -Wextra -Wpedantic)
string(TOUPPER "${CMAKE_BUILD_TYPE}" FSE_BUILD_TYPE_UPPER)
target_compile_definitions(fse_core PRIVATE
  FSE_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
  FSE_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
  FSE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  FSE_CXX_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${FSE_BUILD_TYPE_UPPER}}"
  FSE_HNSWLIB_COMMIT="${FSE_HNSWLIB_COMMIT}"
  FSE_ACORN_COMMIT="${FSE_ACORN_COMMIT}"
  FSE_ACORN_OPT_LEVEL="${FSE_ACORN_OPT_LEVEL}"
  FSE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
if(FSE_CLANG_TIDY)
  set_target_properties(fse_core PROPERTIES CXX_CLANG_TIDY "${FSE_CLANG_TIDY}")
endif()

add_executable(fse_info cli/fse_info.cpp)
add_executable(fse_baselines_check cli/fse_baselines_check.cpp)
target_link_libraries(fse_baselines_check PRIVATE yaml-cpp::yaml-cpp)
set(FSE_APPS fse_info fse_baselines_check)
if(FSE_WITH_ACORN)
  add_executable(fse_acorn_repro cli/fse_acorn_repro.cpp)
  target_link_libraries(fse_acorn_repro PRIVATE yaml-cpp::yaml-cpp)
  add_executable(fse_filter_conditions cli/fse_filter_conditions.cpp)
  target_link_libraries(fse_filter_conditions PRIVATE yaml-cpp::yaml-cpp)
  add_executable(fse_matrix cli/fse_matrix.cpp)
  target_link_libraries(fse_matrix PRIVATE yaml-cpp::yaml-cpp)
  add_executable(fse_features cli/fse_features.cpp)
  target_link_libraries(fse_features PRIVATE yaml-cpp::yaml-cpp)
  add_executable(fse_fragment_conditions cli/fse_fragment_conditions.cpp)
  target_link_libraries(fse_fragment_conditions PRIVATE yaml-cpp::yaml-cpp)
  add_executable(fse_reach cli/fse_reach.cpp)
  target_link_libraries(fse_reach PRIVATE yaml-cpp::yaml-cpp)
  list(APPEND FSE_APPS fse_acorn_repro fse_filter_conditions fse_matrix
    fse_features fse_fragment_conditions fse_reach)
endif()
foreach(app ${FSE_APPS})
  target_link_libraries(${app} PRIVATE fse_core)
  target_compile_options(${app} PRIVATE -Wall -Wextra -Wpedantic)
  if(FSE_CLANG_TIDY)
    set_target_properties(${app} PROPERTIES CXX_CLANG_TIDY "${FSE_CLANG_TIDY}")
  endif()
endforeach()
