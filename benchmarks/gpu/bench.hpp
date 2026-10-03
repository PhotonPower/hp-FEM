// Shared declarations of the GPU micro-benchmark: the CPU side (Eigen, OpenMP, main) is
// compiled by the host compiler, the CUDA side by nvcc (Eigen and nvcc do not mix).
#pragma once
#include <complex>
#include <vector>

struct Csr {
  int n = 0;
  std::vector<int> row_ptr;
  std::vector<int> col;
  std::vector<std::complex<double>> val;
};

struct SolveTimes {
  double factorize = 0;
  double solve = 0;       ///< one right-hand side
  double solve_8 = 0;     ///< eight right-hand sides at once
  double residual = 0;    ///< relative residual of the single solve (filled by the caller)
  long long pivots = -1;  ///< cuDSS: pivots perturbed during the factorisation (CPU: -1)
};

/// Median seconds of `reps` cuSPARSE SpMV products y = A x (upload excluded).
double gpu_spmv(const Csr& a, const std::vector<std::complex<double>>& x,
                std::vector<std::complex<double>>& y, int reps);
/// cuDSS LU: analysis + factorisation, then `reps` single solves and reps/4 solves with
/// eight right-hand sides; the solution of the single solve is returned in x. With
/// `scale` != 1 the matrix scale * A is factorised (cuDSS judges tiny pivots by an absolute
/// threshold, so SI-scaled systems are scaled to max |a_ij| = 1 as the hpfem_gpu library
/// does) and the returned solution is rescaled; `pivots` reports the perturbed pivots.
SolveTimes gpu_solve(const Csr& a, const std::vector<std::complex<double>>& b,
                     std::vector<std::complex<double>>& x, int reps, double scale = 1.0);
/// Name of device 0.
std::string gpu_name();
