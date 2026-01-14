cmake_minimum_required(VERSION 3.25)
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
option(FSE_NATIVE_ARCH "Compile with -march=native" ON)
option(FSE_WITH_ACORN "Build and link ACORN (FAISS fork)" ON)

if(FSE_NATIVE_ARCH)
  string(APPEND CMAKE_CXX_FLAGS " -march=native")
endif()

if(FSE_ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    set(FSE_CLANG_TIDY "${CLANG_TIDY_EXE}")
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

add_library(hnswlib INTERFACE)
target_include_directories(hnswlib SYSTEM INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/hnswlib)

set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/yaml-cpp EXCLUDE_FROM_ALL SYSTEM)

if(FSE_WITH_ACORN)
  include(cmake/acorn.cmake)
endif()

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
