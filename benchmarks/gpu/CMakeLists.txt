# GPU micro-benchmark (docs/adr/0008-gpu-backend.md). Built on its own with nvcc and the
# MSVC host compiler, independent of the MinGW build of the library:
#
#   cmake -S benchmarks/gpu -B build/gpu -G "Visual Studio 17 2022" -A x64 \
#         -DEIGEN_INCLUDE_DIR=build/release/_deps/eigen-src \
#         -Dcudss_DIR="C:/Program Files/NVIDIA cuDSS/v0.8/lib/13/cmake/cudss"
#   cmake --build build/gpu --config Release
#   build/gpu/Release/spmv_solve_bench.exe <matrix.mtx> [results.json]
#
# The matrices come from benchmarks/gpu/export_matrices.py.
cmake_minimum_required(VERSION 3.25)
project(hpfem_gpu_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures")

find_package(CUDAToolkit REQUIRED)
find_package(cudss REQUIRED)
find_package(OpenMP REQUIRED)
set(EIGEN_INCLUDE_DIR "" CACHE PATH "Eigen include directory (header only)")
if(NOT EIGEN_INCLUDE_DIR)
  message(FATAL_ERROR "set -DEIGEN_INCLUDE_DIR=<eigen source dir>")
endif()

add_executable(spmv_solve_bench spmv_solve_bench.cpp gpu_kernels.cu bench.hpp)
target_include_directories(spmv_solve_bench SYSTEM PRIVATE "${EIGEN_INCLUDE_DIR}")
target_link_libraries(spmv_solve_bench PRIVATE CUDA::cusparse CUDA::cudart cudss OpenMP::OpenMP_CXX)
target_compile_options(spmv_solve_bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2 /openmp>)
