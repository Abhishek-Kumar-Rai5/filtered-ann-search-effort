cmake_minimum_required(VERSION 3.25)  # add_subdirectory(... SYSTEM)
project(filter_selectivity_effort LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FSE_BUILD_TESTS "Build GoogleTest unit tests" ON)
option(FSE_ENABLE_CLANG_TIDY "Run clang-tidy on project sources" ON)
# -march=native enables hnswlib's AVX/AVX-512 distance kernels. The effective
# flags are logged into every run's metadata, so results stay attributable.
option(FSE_NATIVE_ARCH "Compile with -march=native" ON)
# ACORN (in-graph filtering baseline). Built as an isolated external project;
# see cmake/acorn.cmake. Turning this OFF is only for working on code that
# does not touch ACORN -- it is NOT the hnswlib fallback decision, which must
# be made and logged explicitly in Phase 1 (CLAUDE.md).
option(FSE_WITH_ACORN "Build and link ACORN (FAISS fork)" ON)

if(FSE_NATIVE_ARCH)
  string(APPEND CMAKE_CXX_FLAGS " -march=native")
endif()

# clang-tidy: on by default, but degrade gracefully if not installed.
if(FSE_ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    set(FSE_CLANG_TIDY "${CLANG_TIDY_EXE}")
    # Point clang-tidy at the libstdc++ of the GCC that actually builds the
    # project. Otherwise clang picks the newest /usr/lib/gcc/<triple>/<ver>,
    # which may lack headers (e.g. a gcc-14 runtime without libstdc++-14-dev).
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-libgcc-file-name
        OUTPUT_VARIABLE FSE_LIBGCC_FILE
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      get_filename_component(FSE_GCC_INSTALL_DIR "${FSE_LIBGCC_FILE}" DIRECTORY)
      list(APPEND FSE_CLANG_TIDY
        "--extra-arg=--gcc-install-dir=${FSE_GCC_INSTALL_DIR}")
    endif()
  else()
    message(WARNING "clang-tidy not found; static analysis disabled for this build")
  endif()
endif()

find_package(OpenMP REQUIRED)

# hnswlib: header-only, used strictly as a black box (CLAUDE.md scope rule).
# Post-filter baseline and the documented in-graph fallback (native filter
# callback) both go through it.
add_library(hnswlib INTERFACE)
target_include_directories(hnswlib SYSTEM INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/hnswlib)

# yaml-cpp: pinned submodule, parses the per-run YAML experiment configs.
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/yaml-cpp EXCLUDE_FROM_ALL SYSTEM)

if(FSE_WITH_ACORN)
  include(cmake/acorn.cmake)
endif()

# Pinned submodule commits, recorded into every run's build metadata.
function(fse_submodule_commit out_var path)
  execute_process(
    COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/${path} rev-parse HEAD
    OUTPUT_VARIABLE commit
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  set(${out_var} "${commit}" PARENT_SCOPE)
endfunction()
fse_submodule_commit(FSE_HNSWLIB_COMMIT third_party/hnswlib)
fse_submodule_commit(FSE_ACORN_COMMIT third_party/acorn)

add_subdirectory(cpp)

if(FSE_BUILD_TESTS)
  enable_testing()
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL SYSTEM)
  add_subdirectory(cpp/tests)
endif()
