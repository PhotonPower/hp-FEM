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
  GIT_TAG 11.0.2 GIT_SHALLOW TRUE
  SYSTEM)
FetchContent_Declare(spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.0 GIT_SHALLOW TRUE
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
if(HPFEM_ENABLE_MUMPS)
  message(STATUS "MUMPS backend requested: implement FindMUMPS.cmake (milestone M6)")
endif()
