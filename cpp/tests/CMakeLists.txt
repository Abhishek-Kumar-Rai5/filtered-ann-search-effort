include(GoogleTest)

add_executable(fse_tests
  test_toolchain.cpp
  test_vecs_io.cpp
  test_uniform_attributes.cpp
  test_ground_truth.cpp
  test_recall.cpp
  test_prefilter.cpp
  test_hnsw_index.cpp
  test_postfilter.cpp
  test_structure_stats.cpp
  test_live_features.cpp
  test_reachability.cpp
)
if(FSE_WITH_ACORN)
  target_sources(fse_tests PRIVATE test_acorn_index.cpp
    test_distance_accounting.cpp test_filter_conditions.cpp)
endif()
target_link_libraries(fse_tests PRIVATE fse_core yaml-cpp::yaml-cpp
  GTest::gtest_main)
target_compile_options(fse_tests PRIVATE -Wall -Wextra)
gtest_discover_tests(fse_tests)
