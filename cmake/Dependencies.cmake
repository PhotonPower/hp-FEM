# All third-party dependencies in one place. Prefer system packages when found,
# fall back to FetchContent with pinned versions. FetchContent deps are declared
# SYSTEM so their headers are included via -isystem and exempt from our strict
# -Werror warning set, exactly like find_package() imported targets (e.g. the
# system Eigen used in CI). Keep this list in sync with
# README.md "Dependencies" and docs/adr/0001-language-and-stack.md.
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# --- Eigen (dense + sparse linear algebra) -----------------------------------
find_package(Eigen3 3.4 QUIET CONFIG)
if(NOT Eigen3_FOUND)
  FetchContent_Declare(eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0 GIT_SHALLOW TRUE
    SYSTEM)
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(eigen)
endif()

# --- fmt + spdlog (logging) ---------------------------------------------------
FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 11.1.4 GIT_SHALLOW TRUE
  SYSTEM)
FetchContent_Declare(spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.1 GIT_SHALLOW TRUE
  SYSTEM)
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(fmt spdlog)

# --- nlohmann_json (configs, results) ----------------------------------------
FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3 GIT_SHALLOW TRUE
  SYSTEM)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

# --- Spectra (header-only eigensolvers on top of Eigen) -----------------------
# SOURCE_SUBDIR points at a directory without CMakeLists.txt so that MakeAvailable only
# downloads; the target is declared by hand because Spectra's own CMake insists on
# find_package(Eigen3), which the fetched Eigen does not provide.
FetchContent_Declare(spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra.git
  GIT_TAG v1.2.0 GIT_SHALLOW TRUE
  SOURCE_SUBDIR cmake_disabled)
FetchContent_MakeAvailable(spectra)
add_library(Spectra INTERFACE)
add_library(Spectra::Spectra ALIAS Spectra)
target_include_directories(Spectra SYSTEM INTERFACE ${spectra_SOURCE_DIR}/include)
target_link_libraries(Spectra INTERFACE Eigen3::Eigen)

# --- Boost.Math (special functions, only where the standard library lacks them) -------
# The C++17 mathematical special functions (std::cyl_bessel_j, std::sph_bessel, ...) exist in
# libstdc++ and the MSVC STL but not in libc++ (Apple Clang on macOS). There the header-only
# Boost.Math in standalone mode provides them with the same signatures; elsewhere it is not
# downloaded (core/special_functions.hpp, HPFEM_USE_BOOST_MATH).
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_QUIET ON)
check_cxx_source_compiles([=[
#include <cmath>
int main() { return std::cyl_bessel_j(0.5, 1.0) + std::sph_neumann(1u, 1.0) > 0.0 ? 0 : 1; }
]=] HPFEM_HAVE_STD_SPECIAL_MATH)
unset(CMAKE_REQUIRED_QUIET)
if(NOT HPFEM_HAVE_STD_SPECIAL_MATH)
  message(STATUS "std special math functions missing (libc++): using Boost.Math standalone")
  FetchContent_Declare(boost_math
    GIT_REPOSITORY https://github.com/boostorg/math.git
    GIT_TAG boost-1.86.0 GIT_SHALLOW TRUE
    SOURCE_SUBDIR cmake_disabled)
  FetchContent_MakeAvailable(boost_math)
  add_library(hpfem_boost_math INTERFACE)
  add_library(hpfem::boost_math ALIAS hpfem_boost_math)
  target_include_directories(hpfem_boost_math SYSTEM INTERFACE ${boost_math_SOURCE_DIR}/include)
  target_compile_definitions(hpfem_boost_math INTERFACE BOOST_MATH_STANDALONE=1
    HPFEM_USE_BOOST_MATH)
endif()

# --- Catch2 (tests) -----------------------------------------------------------
if(HPFEM_BUILD_TESTS)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1 GIT_SHALLOW TRUE
    SYSTEM)
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
endif()

# --- pybind11 (Python bindings) ----------------------------------------------
if(HPFEM_BUILD_PYTHON)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6 GIT_SHALLOW TRUE
    SYSTEM)
  FetchContent_MakeAvailable(pybind11)
endif()

# --- optional heavy deps ------------------------------------------------------
if(HPFEM_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()
if(HPFEM_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if(NOT OpenMP_CXX_FOUND)
    message(STATUS "OpenMP not found: assembly runs serially")
  endif()
endif()
if(HPFEM_ENABLE_MUMPS)
  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
  find_package(MUMPS REQUIRED)  # cmake/FindMUMPS.cmake: sequential complex double build
  message(STATUS "MUMPS backend enabled (version '${MUMPS_VERSION}', DLL dir '${MUMPS_BIN_DIR}')")
endif()
