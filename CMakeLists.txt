cmake_minimum_required(VERSION 3.25)

project(hpfem
  VERSION 0.1.0
  DESCRIPTION "Adaptive hp-FEM suite for nano-optics (time-harmonic Maxwell)"
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---- options -----------------------------------------------------------------
option(HPFEM_BUILD_TESTS      "Build unit/convergence/regression tests" ON)
option(HPFEM_BUILD_PYTHON     "Build pybind11 Python bindings"           OFF)
option(HPFEM_BUILD_BENCHMARKS "Build performance benchmarks"             OFF)
option(HPFEM_BUILD_EXAMPLES   "Build the example programs"               ON)
option(HPFEM_BUILD_DOCS       "Add 'doxygen' target"                     OFF)
option(HPFEM_ENABLE_MPI       "Enable MPI (distributed solvers)"         OFF)
option(HPFEM_ENABLE_MUMPS     "Enable MUMPS direct solver backend"       OFF)
option(HPFEM_ENABLE_OPENMP    "Parallel assembly / estimation with OpenMP" ON)
option(HPFEM_ENABLE_ASSERTS   "Keep HPFEM_ASSERT active in Release"      ON)
option(HPFEM_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"      ON)
# MinGW only: link libstdc++/libgcc/winpthread statically so that executables and the
# Python module do not depend on which libstdc++-6.dll happens to be first on PATH
# (Git for Windows ships an older copy in Git/mingw64/bin, see CLAUDE.md §4a).
option(HPFEM_STATIC_RUNTIME   "MinGW: link the GCC runtime statically"   ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(CompilerWarnings)
include(Dependencies)

# ---- version header ----------------------------------------------------------
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/hpfem/version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/hpfem/version.hpp" @ONLY)

# ---- library -----------------------------------------------------------------
add_subdirectory(src)

# ---- tests -------------------------------------------------------------------
if(HPFEM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(HPFEM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(HPFEM_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(HPFEM_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(HPFEM_BUILD_DOCS)
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    add_custom_target(doxygen
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Generating API documentation")
  endif()
endif()

# ---- install -----------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS hpfem EXPORT hpfemTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/hpfem DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/hpfem/version.hpp"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hpfem)
