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
    -DFETCHCONTENT_BASE_DIR=${CMAKE_BINARY_DIR}/acorn_deps
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target faiss
                --parallel
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${FSE_ACORN_LIB}
  BUILD_ALWAYS OFF
  STEP_TARGETS build
)

add_library(acorn::faiss SHARED IMPORTED GLOBAL)
set_target_properties(acorn::faiss PROPERTIES IMPORTED_LOCATION ${FSE_ACORN_LIB})
target_include_directories(acorn::faiss SYSTEM INTERFACE ${FSE_ACORN_SOURCE_DIR})
target_link_libraries(acorn::faiss INTERFACE OpenMP::OpenMP_CXX)
add_dependencies(acorn::faiss acorn_build)
