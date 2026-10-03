// GPU micro-benchmark for the decision in docs/adr/0008-gpu-backend.md: complex double
// sparse matrix-vector products and direct solves (factorise once, solve many) of hp-FEM
// system matrices on the CPU (OpenMP CSR loop, Eigen SparseLU) and on the GPU (cuSPARSE
// generic SpMV, cuDSS LU, in gpu_kernels.cu). Built separately with nvcc + MSVC (see
// CMakeLists.txt here); not part of the library.
// Usage: spmv_solve_bench <matrix.mtx> [results.json]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <omp.h>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include "bench.hpp"

namespace {

using Complex = std::complex<double>;

/// Matrix Market coordinate file (complex or real, general or symmetric) as CSR.
Csr read_matrix_market(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::string line;
  std::getline(in, line);
  const bool complex = line.find("complex") != std::string::npos;
  const bool symmetric = line.find("symmetric") != std::string::npos;
  while (std::getline(in, line) && !line.empty() && line[0] == '%') {
  }
  int rows = 0, cols = 0;
  long long nnz = 0;
  std::istringstream header(line);
  header >> rows >> cols >> nnz;
  std::vector<Eigen::Triplet<Complex>> triplets;
  triplets.reserve(static_cast<std::size_t>(symmetric ? 2 * nnz : nnz));
  for (long long k = 0; k < nnz; ++k) {
    int i = 0, j = 0;
    double re = 0, im = 0;
    in >> i >> j >> re;
    if (complex) in >> im;
    triplets.emplace_back(i - 1, j - 1, Complex(re, im));
    if (symmetric && i != j) triplets.emplace_back(j - 1, i - 1, Complex(re, im));
  }
  Eigen::SparseMatrix<Complex, Eigen::RowMajor, int> m(rows, cols);
  m.setFromTriplets(triplets.begin(), triplets.end());
  m.makeCompressed();
  Csr csr;
  csr.n = rows;
  csr.row_ptr.assign(m.outerIndexPtr(), m.outerIndexPtr() + rows + 1);
  csr.col.assign(m.innerIndexPtr(), m.innerIndexPtr() + m.nonZeros());
  csr.val.assign(m.valuePtr(), m.valuePtr() + m.nonZeros());
  return csr;
}

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

std::vector<Complex> random_vector(int n) {
  std::vector<Complex> x(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    x[static_cast<std::size_t>(i)] = Complex(std::sin(0.37 * i), std::cos(0.11 * i));
  }
  return x;
}

/// CPU SpMV y = A x with OpenMP over rows; median seconds of `reps` runs.
double cpu_spmv(const Csr& a, const std::vector<Complex>& x, std::vector<Complex>& y, int reps) {
  std::vector<double> times;
  for (int r = 0; r < reps; ++r) {
    const double t0 = now();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < a.n; ++i) {
      Complex sum = 0;
      for (int k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) sum += a.val[k] * x[a.col[k]];
      y[i] = sum;
    }
    times.push_back(now() - t0);
  }
  return median(times);
}

double relative_residual(const Csr& a, const std::vector<Complex>& x,
                         const std::vector<Complex>& b) {
  double num = 0, den = 0;
  for (int i = 0; i < a.n; ++i) {
    Complex sum = 0;
    for (int k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) sum += a.val[k] * x[a.col[k]];
    num += std::norm(sum - b[i]);
    den += std::norm(b[i]);
  }
  return std::sqrt(num / den);
}

SolveTimes cpu_solve(const Csr& a, const std::vector<Complex>& b, int reps) {
  Eigen::SparseMatrix<Complex, Eigen::ColMajor, int> m(a.n, a.n);
  std::vector<Eigen::Triplet<Complex>> triplets;
  for (int i = 0; i < a.n; ++i) {
    for (int k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
      triplets.emplace_back(i, a.col[k], a.val[k]);
    }
  }
  m.setFromTriplets(triplets.begin(), triplets.end());
  m.makeCompressed();
  SolveTimes out;
  double t0 = now();
  Eigen::SparseLU<Eigen::SparseMatrix<Complex, Eigen::ColMajor, int>> lu;
  lu.compute(m);
  out.factorize = now() - t0;
  if (lu.info() != Eigen::Success) {
    std::fprintf(stderr, "SparseLU failed\n");
    std::exit(1);
  }
  const Eigen::Map<const Eigen::VectorXcd> rhs(b.data(), a.n);
  Eigen::VectorXcd x;
  std::vector<double> times;
  for (int r = 0; r < reps; ++r) {
    t0 = now();
    x = lu.solve(rhs);
    times.push_back(now() - t0);
  }
  out.solve = median(times);
  const Eigen::MatrixXcd rhs8 = rhs.replicate(1, 8);
  times.clear();
  for (int r = 0; r < std::max(1, reps / 4); ++r) {
    t0 = now();
    const Eigen::MatrixXcd x8 = lu.solve(rhs8);
    times.push_back(now() - t0);
  }
  out.solve_8 = median(times);
  const std::vector<Complex> xv(x.data(), x.data() + a.n);
  out.residual = relative_residual(a, xv, b);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: spmv_solve_bench <matrix.mtx> [results.json]\n");
    return 1;
  }
  const std::string path = argv[1];
  const Csr a = read_matrix_market(path);
  const int nnz = static_cast<int>(a.val.size());
  const std::string gpu = gpu_name();
  std::printf("%s: n = %d, nnz = %d, GPU %s, %d OpenMP threads\n", path.c_str(), a.n, nnz,
              gpu.c_str(), omp_get_max_threads());

  const std::vector<Complex> x = random_vector(a.n);
  std::vector<Complex> y_cpu(static_cast<std::size_t>(a.n));
  std::vector<Complex> y_gpu(static_cast<std::size_t>(a.n));
  const int reps = 50;
  const double t_cpu = cpu_spmv(a, x, y_cpu, reps);
  const double t_gpu = gpu_spmv(a, x, y_gpu, reps);
  double diff = 0, norm = 0;
  for (int i = 0; i < a.n; ++i) {
    diff += std::norm(y_cpu[i] - y_gpu[i]);
    norm += std::norm(y_cpu[i]);
  }
  // values, column indices, row pointers, x, y
  const double bytes = 16.0 * nnz + 4.0 * nnz + 4.0 * (a.n + 1) + 32.0 * a.n;
  std::printf(
      "SpMV  CPU %.3e s (%.1f GB/s)   GPU %.3e s (%.1f GB/s)   speedup %.1fx   rel. diff %.1e\n",
      t_cpu, bytes / t_cpu * 1e-9, t_gpu, bytes / t_gpu * 1e-9, t_cpu / t_gpu,
      std::sqrt(diff / norm));

  const std::vector<Complex> b = random_vector(a.n);
  const SolveTimes cpu = cpu_solve(a, b, 20);
  std::printf("Solve CPU SparseLU: factorise %.3e s, solve %.3e s, 8 rhs %.3e s, residual %.1e\n",
              cpu.factorize, cpu.solve, cpu.solve_8, cpu.residual);
  // unscaled first: documents how many pivots cuDSS perturbs on the SI-scaled matrix
  std::vector<Complex> x_gpu;
  SolveTimes unscaled = gpu_solve(a, b, x_gpu, 4);
  unscaled.residual = relative_residual(a, x_gpu, b);
  std::printf(
      "Solve GPU cuDSS (unscaled): factorise %.3e s, residual %.1e, %lld perturbed pivots\n",
      unscaled.factorize, unscaled.residual, unscaled.pivots);
  double max_abs = 0;
  for (const Complex& v : a.val) max_abs = std::max(max_abs, std::abs(v));
  const double scale = 1.0 / max_abs;
  SolveTimes gpu_times = gpu_solve(a, b, x_gpu, 20, scale);
  gpu_times.residual = relative_residual(a, x_gpu, b);
  std::printf(
      "Solve GPU cuDSS   : factorise %.3e s, solve %.3e s, 8 rhs %.3e s, residual %.1e, %lld "
      "perturbed pivots (scale %.1e)\n",
      gpu_times.factorize, gpu_times.solve, gpu_times.solve_8, gpu_times.residual, gpu_times.pivots,
      scale);
  std::printf("      speedup factorise %.1fx, solve %.1fx, 8 rhs %.1fx\n",
              cpu.factorize / gpu_times.factorize, cpu.solve / gpu_times.solve,
              cpu.solve_8 / gpu_times.solve_8);
  if (argc > 2) {
    std::ofstream out(argv[2], std::ios::app);
    const std::string name = path.substr(path.find_last_of("/\\") + 1);
    out << "{\"matrix\": \"" << name << "\", \"n\": " << a.n << ", \"nnz\": " << nnz
        << ", \"gpu\": \"" << gpu << "\", \"threads\": " << omp_get_max_threads()
        << ", \"spmv_cpu_s\": " << t_cpu << ", \"spmv_gpu_s\": " << t_gpu
        << ", \"factorize_cpu_s\": " << cpu.factorize << ", \"solve_cpu_s\": " << cpu.solve
        << ", \"solve8_cpu_s\": " << cpu.solve_8 << ", \"factorize_gpu_s\": " << gpu_times.factorize
        << ", \"solve_gpu_s\": " << gpu_times.solve << ", \"solve8_gpu_s\": " << gpu_times.solve_8
        << ", \"residual_cpu\": " << cpu.residual << ", \"residual_gpu\": " << gpu_times.residual
        << ", \"pivots_gpu\": " << gpu_times.pivots << ", \"scale\": " << scale
        << ", \"factorize_gpu_unscaled_s\": " << unscaled.factorize
        << ", \"residual_gpu_unscaled\": " << unscaled.residual
        << ", \"pivots_gpu_unscaled\": " << unscaled.pivots << "}\n";
  }
  return 0;
}
