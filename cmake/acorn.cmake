# ACORN (SIGMOD 2024) -- in-graph filtered-search baseline.
#
# ACORN is not a library that builds *against* FAISS: it is a fork of FAISS
# 1.7.3 with IndexACORN added. Its libfaiss therefore IS this project's FAISS;
# linking a second, upstream FAISS alongside it would give two conflicting
# definitions of the faiss:: namespace.
#
# It is built as an isolated ExternalProject, with the exact flags from its
# README, rather than via add_subdirectory: its top-level CMakeLists sets
# CMAKE_CXX_STANDARD 11, include(CTest), and FetchContent at configure time,
# none of which should leak into (or inherit from) this project's C++20 build.
# Its sources are never modified (CLAUDE.md: black box).

find_package(BLAS)
find_package(LAPACK)
if(NOT BLAS_FOUND OR NOT LAPACK_FOUND)
  message(FATAL_ERROR
    "ACORN/FAISS needs BLAS and LAPACK, which were not found. On Ubuntu: "
    "sudo apt install libopenblas-dev. (Or configure with "
    "-DFSE_WITH_ACORN=OFF to build the rest of the project without ACORN.)")
endif()

include(ExternalProject)

set(FSE_ACORN_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/acorn)
set(FSE_ACORN_BINARY_DIR ${CMAKE_BINARY_DIR}/acorn)
set(FSE_ACORN_LIB
  ${FSE_ACORN_BINARY_DIR}/faiss/${CMAKE_SHARED_LIBRARY_PREFIX}faiss${CMAKE_SHARED_LIBRARY_SUFFIX})

# FAISS_OPT_LEVEL=generic matches ACORN's README build; any change to it is a
# Phase 1 reproduction decision and must be logged.
set(FSE_ACORN_OPT_LEVEL "generic" CACHE STRING "ACORN FAISS_OPT_LEVEL")

ExternalProject_Add(acorn_build
  SOURCE_DIR ${FSE_ACORN_SOURCE_DIR}
  BINARY_DIR ${FSE_ACORN_BINARY_DIR}
  CMAKE_ARGS
    -DFAISS_ENABLE_GPU=OFF
    -DFAISS_ENABLE_PYTHON=OFF
    -DBUILD_TESTING=OFF
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_BUILD_TYPE=Release
    -DFAISS_OPT_LEVEL=${FSE_ACORN_OPT_LEVEL}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    # ACORN adds its FetchContent'd nlohmann_json include dir to faiss's
    # PUBLIC includes; CMake rejects that at generate time if the dir lies
    # inside ACORN's own build tree. Fetching to a sibling dir avoids it
    # without touching ACORN's sources.
    -DFETCHCONTENT_BASE_DIR=${CMAKE_BINARY_DIR}/acorn_deps
  # Only libfaiss (and, from Phase 1, ACORN's own benchmark driver).
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target faiss
                --parallel
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${FSE_ACORN_LIB}
  # Rebuild when the submodule's sources change (e.g. bumping the pin).
  BUILD_ALWAYS OFF
  STEP_TARGETS build
)

add_library(acorn::faiss SHARED IMPORTED GLOBAL)
set_target_properties(acorn::faiss PROPERTIES IMPORTED_LOCATION ${FSE_ACORN_LIB})
target_include_directories(acorn::faiss SYSTEM INTERFACE ${FSE_ACORN_SOURCE_DIR})
target_link_libraries(acorn::faiss INTERFACE OpenMP::OpenMP_CXX)
add_dependencies(acorn::faiss acorn_build)
