# hpfem_gpu — the GPU direct-solver DLL of hp-FEM (cuDSS), docs/adr/0008-gpu-backend.md.
#
# A separate CMake project: the library is built with MinGW GCC (or GCC / Clang on Linux),
# which cannot link nvcc / MSVC objects, so this DLL is built on its own and loaded by the
# library at run time (option HPFEM_ENABLE_CUDA, see gpu/README.md). Only the C interface in
# include/hpfem_gpu.h crosses the boundary.
#
#   cmake -S gpu -B build/gpu -G "Visual Studio 17 2022" -A x64 \
#         -Dcudss_DIR="C:/Program Files/NVIDIA cuDSS/v0.8/lib/13/cmake/cudss"
#   cmake --build build/gpu --config Release
#   ctest --test-dir build/gpu -C Release          # self-test on the installed GPU
#
# Linux: cmake -S gpu -B build/gpu -Dcudss_DIR=/usr/lib/x86_64-linux-gnu/cmake/cudss …
cmake_minimum_required(VERSION 3.25)
# must precede project(): the CUDA compiler check fixes the architecture list
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures (86 = Ampere / RTX 30xx)")
endif()
project(hpfem_gpu VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# static MSVC runtime: the DLL then needs no Visual C++ redistributable at run time
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

option(HPFEM_GPU_BUILD_SELFTEST "Build the stand-alone self-test executable" ON)

find_package(CUDAToolkit 12.0 REQUIRED)
find_package(cudss 0.6 REQUIRED CONFIG)

add_library(hpfem_gpu SHARED src/cudss_solver.cu src/matrix.cu src/stepper.cu src/arnoldi.cu
  src/internal.hpp include/hpfem_gpu.h)
target_include_directories(hpfem_gpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(hpfem_gpu PRIVATE HPFEM_GPU_BUILD)
target_link_libraries(hpfem_gpu PRIVATE CUDA::cudart_static cudss)  # only the cuDSS DLLs remain run-time dependencies
set_target_properties(hpfem_gpu PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  OUTPUT_NAME hpfem_gpu
  PREFIX ""                       # hpfem_gpu.dll / hpfem_gpu.so on every platform
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
if(MSVC)
  target_compile_options(hpfem_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4 -Xcompiler=/wd4100>)
endif()

# directory of the cuDSS runtime DLLs for the self-test (Windows); lib/<cuda>/cmake/cudss
# → bin/<cuda> in the same install tree
set(HPFEM_CUDSS_BIN_DIR "" CACHE PATH "Directory of the cuDSS runtime DLLs (Windows)")
if(WIN32 AND NOT HPFEM_CUDSS_BIN_DIR AND cudss_DIR)
  get_filename_component(_cudss_root "${cudss_DIR}/../../../.." ABSOLUTE)
  get_filename_component(_cudss_cuda "${cudss_DIR}/../.." ABSOLUTE)
  get_filename_component(_cudss_cuda "${_cudss_cuda}" NAME)
  if(EXISTS "${_cudss_root}/bin/${_cudss_cuda}")
    set(HPFEM_CUDSS_BIN_DIR "${_cudss_root}/bin/${_cudss_cuda}" CACHE PATH "" FORCE)
  endif()
endif()

if(HPFEM_GPU_BUILD_SELFTEST)
  enable_testing()
  add_executable(hpfem_gpu_selftest selftest/selftest.cpp)
  target_link_libraries(hpfem_gpu_selftest PRIVATE hpfem_gpu)
  set_target_properties(hpfem_gpu_selftest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  add_test(NAME hpfem_gpu_selftest COMMAND hpfem_gpu_selftest 20000)
  # the same cases with the factors forced into host memory (hybrid memory mode)
  add_test(NAME hpfem_gpu_selftest_hybrid COMMAND hpfem_gpu_selftest 20000)
  set_tests_properties(hpfem_gpu_selftest_hybrid PROPERTIES ENVIRONMENT "HPFEM_GPU_HYBRID=1")
  if(HPFEM_CUDSS_BIN_DIR)
    set_tests_properties(hpfem_gpu_selftest hpfem_gpu_selftest_hybrid PROPERTIES
      ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${HPFEM_CUDSS_BIN_DIR}")
  endif()
endif()
